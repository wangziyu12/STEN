import sys
import torch
from torch import optim
from torch.utils import data
from ptflops import get_model_complexity_info
from src.args import args_main
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.loss import AVGZSLLoss, L2Loss, SquaredL2Loss, ClsContrastiveLoss, APN_Loss, CJMELoss
from src.metrics import DetailedLosses, MeanClassAccuracy, PercentOverlappingClasses, TargetDifficulty
from src.model import AVGZSLNet, DeviseModel, APN, CJME
from src.sampler import SamplerFactory
from src.model_improvements import AVCA
from src.utils_improvements import get_model_params
from src.train import train
from src.utils import fix_seeds, setup_experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from src.monitor_training import TrainingMonitor


def main():
    # 解析命令行参数
    args = args_main()
    
    # # 设置 GPU 设备
    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
    #     if args.gpu_id >= num_gpus:
    #         print(f"错误: GPU {args.gpu_id} 不存在，仅有 {num_gpus} 张 GPU 可用")
    #         exit(1)
    #     torch.cuda.set_device(args.gpu_id)
    #     torch.device(f"cuda:{args.gpu_id}")
    #     print(f"使用 GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    # else:
    #     torch.device("cpu")
    #     print("警告: CUDA 不可用，自动切换到 CPU")

    # 如果 `input_size` 统一指定，则同时设定 `input_size_audio` 和 `input_size_video`
    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size
    
    # 设置随机种子，确保实验可复现
    fix_seeds(args.seed)

    # 初始化日志记录工具，包括日志文件夹 `log_dir`
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")

    # 初始化训练监控器
    monitor = TrainingMonitor(exp_name=args.exp_name, log_dir=log_dir)

    # 选择数据集，根据 `args.dataset_name`，加载训练集、验证集、训练+验证集、完整验证集
    if args.dataset_name == "AudioSetZSL":
        train_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="seen",
        )

        val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="seen",
        )

        train_val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode="seen",
        )

        val_all_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode="all",
        )

    elif args.dataset_name == "VGGSound":
        train_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        train_dataset = UCFDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = UCFDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        train_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train",
            zero_shot_mode="train",
        )
        val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )

        train_val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    # 将数据集转换为对比学习格式
    contrastive_train_dataset = ContrastiveDataset(train_dataset)
    contrastive_val_dataset = ContrastiveDataset(val_dataset)
    contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    # 训练数据集的采样器，使用 `SamplerFactory` 根据类别索引创建随机采样器
    train_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_dataset.target_to_indices.values()),
        batch_size=args.bs,     # 批量大小
        n_batches=args.n_batches,       # 训练批次数
        alpha=1,
        kind='random'
    )

    # 验证数据集的采样器
    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    # 训练+验证数据集的采样器
    train_val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    # 完整验证数据集的采样器
    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    # 使用 `DataLoader` 将数据集和相应的采样器包装成可迭代的数据加载器
    # 训练数据加载器
    train_loader = data.DataLoader(
        dataset=contrastive_train_dataset,
        batch_sampler=train_sampler,
        num_workers=8       # 设定使用的CPU线程数，提高数据加载速度
    )

    # 验证数据加载器
    val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        batch_sampler=val_sampler,
        num_workers=8
    )

    # 训练+验证数据加载器
    train_val_loader = data.DataLoader(
        dataset=contrastive_train_val_dataset,
        batch_sampler=train_val_sampler,
        num_workers=8
    )

    # 完整验证数据加载器
    val_all_loader = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        num_workers=8
    )

    # 如果使用 AVCA 模型，则获取其所需的参数
    if args.AVCA==True:
        model_params = get_model_params(args.lr, args.first_additional_triplet, args.second_additional_triplet, \
                                        args.reg_loss, args.additional_triplets_loss, args.embedding_dropout, \
                                        args.decoder_dropout, args.additional_dropout, args.embeddings_hidden_size, \
                                        args.decoder_hidden_size, args.depth_transformer, args.momentum, \
                                        args.T, args.scheduler, args.eta_min, args.epochs, args.use_bn, args.use_dropout)


    # 根据命令行参数选择合适的模型
    if args.ale==True or args.devise==True or args.sje==True:
        model= DeviseModel(args)    # 选择 DeViSE 模型
    elif args.apn==True:
        model=APN(args)    # 选择 APN 模型
    elif args.cjme==True:
        model=CJME(args)    # 选择 CJME 模型
    elif args.AVCA==True:
        model = AVCA(model_params, input_size_audio=args.input_size_audio, input_size_video=args.input_size_video)    # 选择 AVCA 模型
    else:
        model = AVGZSLNet(args)    # 选择 AVGZSLNet 模型

    # 将模型移动到指定的计算设备（CPU 或 GPU）
    model.to(args.device)
    
    # 获取距离计算函数
    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()

    # 选择合适的损失函数
    if args.ale==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="weighted")
    elif args.devise==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="sum")
    elif args.sje==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=True, topk=1, reduction="sum")
    elif args.apn==True:
        criterion=APN_Loss()
    elif args.cjme==True:
        criterion=CJMELoss(margin=args.margin, distance_fn=distance_fn)
    elif args.AVCA==True:
        criterion=None  # AVCA 不使用标准损失函数
    else:
        criterion = AVGZSLLoss(margin=args.margin, distance_fn=distance_fn)

    # 选择 Adam 优化器，并设定学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 选择学习率调度器
    if args.scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        # print(f"使用 CosineAnnealingLR: T_max={args.epochs}, eta_min={args.eta_min}")
    elif args.scheduler == "reduce":
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, verbose=True)
        # print("使用 ReduceLROnPlateau: mode=max, patience=3")
    else:
        lr_scheduler = None
    # lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True) if args.lr_scheduler else None

    # 定义评估指标
    metrics = [
        MeanClassAccuracy(model=model, dataset=val_all_dataset, device=args.device, distance_fn=distance_fn,
                          model_devise=args.ale or args.sje or args.devise,
                          new_model_attention=args.AVCA,
                          apn=args.apn,
                          args=args)
    ]



    # 记录日志信息
    logger.info(model)  # 记录模型的结构信息，方便后续分析和复现
    logger.info(criterion)  # 记录所使用的损失函数
    logger.info(optimizer)  # 记录优化器的配置信息，包括学习率等超参数
    logger.info(lr_scheduler)   # 记录学习率调度器的信息（如果使用）
    logger.info([metric.__class__.__name__ for metric in metrics])  # 记录所使用的评估指标

    # 选择合适的验证数据加载器
    if args.val_all_loss:
        v_loader = val_all_loader
    elif args.retrain_all:
        v_loader = train_val_loader
    else:
        v_loader = val_loader

    # 训练模型
    best_loss, best_score = train(
        train_loader=train_val_loader if args.retrain_all else train_loader,    # 选择训练数据加载器
        val_loader=v_loader,    # 选择验证数据加载器
        model=model,    # 选择模型
        criterion=criterion,    # 选择损失函数
        optimizer=optimizer,    # 选择优化器
        lr_scheduler=lr_scheduler,    # 选择学习率调度器
        epochs=args.epochs,    # 设定训练的轮数
        device=args.device,    # 设定计算设备(GPU 或 CPU)
        writer=writer,    # 训练过程中的日志记录工具
        metrics=metrics,    # 训练过程中计算的指标
        train_stats=train_stats,    # 训练统计数据
        new_model_attention=args.AVCA,  # 是否使用 AVCA 模型
        val_stats=val_stats,    # 验证统计数据
        log_dir=log_dir,    # 训练日志保存路径
        model_devise=args.ale or args.sje or args.devise,    # 是否使用 DeViSE 模型
        apn=args.apn,    # 是否使用 APN 模型
        cjme=args.cjme,    # 是否使用 CJME 模型
        args=args,    # 命令行参数
        monitor=monitor  # 添加监控器参数
    )

    # 训练结束后记录日志
    logger.info(f"FINISHED. Run is stored at {log_dir}")


if __name__ == '__main__':
    main()

import logging

import torch
import os

from src.metrics import MeanClassAccuracy
from src.utils import check_best_loss, check_best_score, evaluate_dataset, save_best_model

# 添加早停类
class EarlyStopping:
    """提前停止训练的工具类"""
    
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        初始化早停对象
        :param patience: 在触发提前停止前，能够容忍的验证指标未改善的epoch数量
        :param delta: 判断指标是否改善的最小变化量
        :param verbose: 是否打印早停相关信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
    
    def __call__(self, val_metric):
        """
        检查是否应该提前停止训练
        :param val_metric: 当前epoch的验证指标，这里我们期望越高越好（如GZSL）
        :return: True如果应该提前停止，否则False
        """
        score = val_metric
        
        if self.best_score is None:
            # 首次调用
            self.best_score = score
            self.best_epoch = 1
        elif score <= self.best_score + self.delta:
            # 验证指标未改善
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 验证指标改善
            self.best_score = score
            self.best_epoch = self.counter + 1
            self.counter = 0
        
        return self.early_stop

# def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
#           train_stats, val_stats, log_dir, new_model_attention=False, model_devise=False, apn=False, cjme=False, args=None):
#     best_loss = None
#     best_score = None

#     for epoch in range(epochs):
#         train_loss = train_step(train_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics,
#                                 train_stats, new_model_attention, model_devise, apn, cjme, args)
#         val_loss, val_hm = val_step(val_loader, model, criterion, epoch, epochs, writer, device, metrics, val_stats,
#                                      new_model_attention, model_devise,apn,cjme, args)

#         best_loss = check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir)
#         best_score = check_best_score(epoch, best_score, val_hm, model, optimizer, log_dir)

#         if args.save_checkpoints:
#             # save_best_model(epoch, val_loss, model, optimizer, log_dir / "checkpoints", metric="loss", checkpoint=True)
#             save_best_model(epoch, val_hm, model, optimizer, log_dir / "checkpoints", metric="score", checkpoint=True)

#         if lr_scheduler:
#             lr_scheduler.step(val_hm)
#         if  new_model_attention==True:
#             model.optimize_scheduler(val_hm)
#     return best_loss, best_score

# 修改后的train函数
def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
          train_stats, val_stats, log_dir, new_model_attention=False, model_devise=False, apn=False, cjme=False, args=None, monitor=None):
    best_loss = None
    best_score = None
    
    # 初始化早停对象(如果启用)
    early_stopper = None
    if args and args.early_stopping:
        early_stopper = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(epochs):
        # 训练阶段
        train_loss = train_step(train_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics,
                                train_stats, new_model_attention, model_devise, apn, cjme, args)
        
        # 获取当前学习率
        if new_model_attention:
            current_lr = model.optimizer_gen.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
            
        # 如果存在监控器，记录训练指标
        if monitor:
            monitor.log_train_metrics(epoch+1, train_loss, current_lr)  # epoch+1使显示从1开始计数
            
        # 验证阶段
        val_loss, val_hm = val_step(val_loader, model, criterion, epoch, epochs, writer, device, metrics, val_stats,
                                     new_model_attention, model_devise, apn, cjme, args)
        
        # 获取各项性能指标
        seen_acc = 0
        unseen_acc = 0
        zsl_score = 0
        if metrics and len(metrics) > 0:
            metric_values = metrics[0].value()
            for key, value in metric_values.items():
                if "seen_acc" in key:
                    seen_acc = value
                elif "unseen_acc" in key:
                    unseen_acc = value
                elif "both_zsl" in key:
                    zsl_score = value
        
        # 如果存在监控器，记录验证指标
        if monitor:
            is_best = monitor.log_val_metrics(epoch+1, val_loss, seen_acc, unseen_acc, val_hm, zsl_score)
            if is_best and args and args.save_checkpoints:
                save_checkpoint(model, optimizer, epoch+1, log_dir, args.exp_name)

        # 保存传统的最佳模型
        best_loss = check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir)
        best_score = check_best_score(epoch, best_score, val_hm, model, optimizer, log_dir)

        if args and args.save_checkpoints:
            # save_best_model(epoch, val_loss, model, optimizer, log_dir / "checkpoints", metric="loss", checkpoint=True)
            save_best_model(epoch, val_hm, model, optimizer, log_dir / "checkpoints", metric="score", checkpoint=True)

        # 学习率调度
        if lr_scheduler:
            lr_scheduler.step(val_hm)
        if new_model_attention:
            model.optimize_scheduler(val_hm)
            
        # 早停检查
        if early_stopper and val_hm is not None:
            if early_stopper(val_hm):
                print(f"提前停止训练! 最佳GZSL: {early_stopper.best_score:.2f}% 在第 {early_stopper.best_epoch} 轮")
                break
    
    # 训练结束后关闭监控器
    if monitor:
        monitor.close()
        
    return best_loss, best_score

# 增加保存检查点函数
def save_checkpoint(model, optimizer, epoch, log_dir, exp_name):
    """保存模型检查点"""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    
    # 对于AVCA模型，需要特殊处理
    if hasattr(model, 'optimizer_gen'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer_gen.state_dict(),
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        }
    
    torch.save(checkpoint, checkpoint_path)
    
    # 保存一个明确标识的最佳模型副本
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    torch.save(checkpoint, best_model_path)
    
    print(f"已保存模型检查点: {checkpoint_path}")
    print(f"已更新最佳模型: {best_model_path}")
    
    return best_model_path

def train_step(data_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics, stats,
               new_model_attention,model_devise, apn, cjme, args):
    logger = logging.getLogger()
    model.train()

    for metric in metrics:
        metric.reset()

    batch_loss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        model.train()
        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(device)
        x_p_v = p["video"].to(device)
        x_p_t = p["text"].to(device)
        x_p_num = target["positive"].to(device)

        x_q_a = q["audio"].to(device)
        x_q_v = q["video"].to(device)
        x_q_t = q["text"].to(device)

        if new_model_attention==False and model_devise==False and apn==False:
            inputs = (
                x_p_a, x_p_v, x_p_t,
                x_q_a, x_q_v, x_q_t
            )
        elif new_model_attention==True:
            inputs = (
                x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t
            )
        else:
            inputs=(
                x_p_a, x_p_v, x_p_num, x_p_t
            )

        if args.z_score_inputs:
            inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

        if new_model_attention==False and model_devise==False and apn==False:
            if cjme==True:
                outputs=model(*inputs)
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                embeddings_projected=model.get_classes_embedding(embeddings)
                loss, loss_details=criterion(*outputs, embeddings_projected.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                outputs = model(*inputs)
                loss, loss_details = criterion(*outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif apn==True:
            embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
            input_features = torch.cat((inputs[1], inputs[0]), 1)
            output_final, pre_attri, attention, pre_class, attribute = model(input_features, embeddings)
            loss, loss_details=criterion(model, output_final, pre_attri, pre_class, inputs[3] , inputs[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif model_devise==True:
            embeddings, mapping_dict =data_loader.dataset.zsl_dataset.map_embeddings_target
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
            input_features=torch.cat((inputs[1], inputs[0]), 1)
            outputs, _, _=model(input_features,embeddings)
            loss, loss_details=criterion(outputs, inputs[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif new_model_attention==True:
            loss, loss_details = model.optimize_params(*inputs, optimize=True)
            audio_emb, video_emb, emb_cls=model.get_embeddings(inputs[0], inputs[1], inputs[3])
            outputs=torch.stack([video_emb, emb_cls], dim=0)

        batch_loss += loss.item()

        p_target = target["positive"].to(device)
        q_target = target["negative"].to(device)

        # stats
        iteration = len(data_loader) * epoch + batch_idx


    batch_loss /= (batch_idx + 1)
    stats.update((epoch, batch_loss, None))

    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss



def val_step(data_loader, model, criterion, epoch, epochs, writer, device, metrics, stats,
             new_model_attention,model_devise,apn,cjme, args=None):

    logger = logging.getLogger()
    model.eval()

    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        batch_loss = 0
        hm_score = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(device)
            x_p_v = p["video"].to(device)
            x_p_t = p["text"].to(device)
            x_p_num = target["positive"].to(device)

            x_q_a = q["audio"].to(device)
            x_q_v = q["video"].to(device)
            x_q_t = q["text"].to(device)

            if new_model_attention==False and model_devise==False and apn==False:
                inputs = (
                    x_p_a, x_p_v, x_p_t,
                    x_q_a, x_q_v, x_q_t
                )
            elif new_model_attention==True:
                inputs = (
                    x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v,x_q_t
                )
            else:
                inputs = (
                    x_p_a, x_p_v, x_p_num, x_p_t
                )

            if args.z_score_inputs:
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

            if new_model_attention==False and model_devise==False and apn==False:
                if cjme==True:
                    outputs = model(*inputs)
                    embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                    embeddings_projected = model.get_classes_embedding(embeddings)
                    loss, loss_details = criterion(*outputs, embeddings_projected)
                else:
                    outputs = model(*inputs)
                    loss, loss_details = criterion(*outputs)
            elif model_devise==True:
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
                input_features=torch.cat((inputs[1], inputs[0]), 1)
                outputs, _, _=model(input_features, embeddings)
                loss, loss_details=criterion(outputs, inputs[2])
            elif apn == True:
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
                input_features = torch.cat((inputs[1], inputs[0]), 1)
                output_final, pre_attri, attention, pre_class, attribute = model(input_features, embeddings)
                loss, loss_details = criterion(model, output_final, pre_attri, pre_class, inputs[3], inputs[2])
                outputs=output_final
            elif new_model_attention==True:
                loss, loss_details = model.optimize_params(*inputs)
                audio_emb, video_emb, emb_cls = model.get_embeddings(inputs[0], inputs[1], inputs[3])
                outputs = (video_emb, emb_cls)

            batch_loss += loss.item()

            p_target = target["positive"].to(device)
            q_target = target["negative"].to(device)

            # stats
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                for metric in metrics:
                    metric(outputs, (p_target, q_target), (loss, loss_details))
                    for key, value in metric.value().items():
                        if "recall" in key:
                            continue
                        if "both_hm" in key:
                            hm_score = value
                        if "both_zsl" in key:
                            zsl_score=value
                        writer.add_scalar(
                            f"val_{key}", value, iteration
                        )

        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, hm_score))

        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
            f"ZSL score: {zsl_score:.4f}\t"
            f"HM: {hm_score:.4f}"
        )
    return batch_loss, hm_score

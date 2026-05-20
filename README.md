# Spiking Temporal-Enhanced Network for Zero-Shot Audio-Visual Learning

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue)](https://ieeexplore.ieee.org/document/11464107)
[![Conference](https://img.shields.io/badge/ICASSP-2026-green)](https://ieeexplore.ieee.org/document/11464107)

Official research implementation of the ICASSP 2026 paper:

**Spiking Temporal-Enhanced Network for Zero-Shot Audio-Visual Learning**

This repository provides the training and evaluation code for **STEN**, a spiking temporal-enhanced framework for generalized zero-shot audio-visual learning. The code supports multiple audio-visual zero-shot benchmarks and includes comparisons with representative baseline methods.

---

## News

- **2026**: This work was accepted by **ICASSP 2026**.
- The paper is available on [IEEE Xplore](https://ieeexplore.ieee.org/document/11464107).

---

## Overview

Zero-shot audio-visual learning aims to recognize both seen and unseen categories by aligning audio, visual, and semantic representations. STEN introduces spiking temporal modeling into audio-visual zero-shot learning, enhancing cross-modal representation learning through temporal spike-based dynamics and audio-visual semantic alignment.

The repository includes:

- The proposed STEN/AVCA model with spiking temporal-enhanced branches.
- Training and evaluation pipelines for generalized zero-shot learning.
- Dataset loaders for AudioSetZSL, VGGSound, UCF, and ActivityNet.
- Baseline implementations including AVGZSLNet, ALE, DeViSE, SJE, APN, CJME, and f-VAEGAN-D2.
- Experiment scripts for reproducing the reported settings.

---

## Repository Structure

```text
.
├── main.py                         # Main training entry point
├── get_evaluation.py               # Final evaluation entry point
├── vae_gan_d2_xu_fsl.py            # f-VAEGAN-D2 baseline
├── src/
│   ├── args.py                     # Command-line arguments
│   ├── dataset.py                  # Dataset definitions and preprocessing
│   ├── train.py                    # Training and validation loops
│   ├── test.py                     # Test-time evaluation
│   ├── metrics.py                  # Zero-shot/GZSL metrics
│   ├── model.py                    # Baseline models
│   ├── model_improvements.py       # Proposed STEN/AVCA model
│   ├── loss.py                     # Loss functions
│   └── Qtrick_architecture/        # Spiking neural network components
├── run_scripts/                    # Reproduction scripts
├── cls_feature_extraction/         # Class-level feature extraction utilities
├── selavi_feature_extraction/      # SeLaVi feature extraction utilities
├── audioset_vggish_tensorflow_to_pytorch/
├── w2v_features/                   # Word2Vec-related resources
├── requirements.txt
├── environment.yaml
└── LICENSE.md
```

---

## Environment

We recommend using Conda.

```bash
conda env create -f environment.yaml
conda activate STEN
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The code is implemented with PyTorch and uses CUDA by default. To run on CPU, pass:

```bash
--device cpu
```

---

## Data Preparation

The training code expects pre-extracted audio, video, and text features. The expected dataset directory is specified by `--root_dir`, and the feature folder is specified by `--feature_extraction_method`.

The general structure is:

```text
<root_dir>/
├── class-split/
│   └── <split_name>/
│       ├── stage_1_train.txt
│       ├── stage_1_val_seen.txt
│       ├── stage_1_val_unseen.txt
│       ├── stage_2_train.txt
│       ├── stage_2_test_seen.txt
│       └── stage_2_test_unseen.txt
└── features/
    └── <feature_extraction_method>/
        ├── audio/
        ├── video/
        └── text/
```

Supported datasets:

- `AudioSetZSL`
- `VGGSound`
- `UCF`
- `ActivityNet`

The repository also contains feature extraction utilities for commonly used audio-visual representations. Please follow the corresponding dataset and feature extraction scripts when preparing new data.

---

## Training

The main training entry point is:

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_val
```

For two-stage generalized zero-shot evaluation, first train on the validation split:

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_val
```

Then retrain using the train+validation data:

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --retrain_all \
  --save_checkpoints \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_all
```

Ready-to-run examples are provided in:

```text
run_scripts/
├── ActivityNet-GZSL/
├── UCF-GZSL/
└── VGGSound-GZSL/
```

For example:

```bash
bash run_scripts/VGGSound-GZSL/avca.sh
```

---

## Evaluation

After completing the two-stage training procedure, run:

```bash
python get_evaluation.py \
  --load_path_stage_A runs/sten_vggsound_val \
  --load_path_stage_B runs/sten_vggsound_all \
  --dataset_name VGGSound \
  --AVCA
```

The evaluation code reports generalized zero-shot metrics on seen and unseen classes.

---

## Baselines

The repository includes scripts for several comparison methods:

- `AVGZSLNet`
- `ALE`
- `DeViSE`
- `SJE`
- `APN`
- `CJME`
- `f-VAEGAN-D2`
- `AVCA/STEN`

Baseline scripts are available under `run_scripts/<dataset>-GZSL/`.

---

## Important Arguments

| Argument | Description |
| --- | --- |
| `--root_dir` | Path to the dataset directory |
| `--feature_extraction_method` | Feature folder name |
| `--dataset_name` | Dataset name: `AudioSetZSL`, `VGGSound`, `UCF`, or `ActivityNet` |
| `--zero_shot_split` | Zero-shot split name, e.g. `main_split` or `cls_split` |
| `--input_size_audio` | Dimension of audio features |
| `--input_size_video` | Dimension of video features |
| `--AVCA` | Use the proposed STEN/AVCA model |
| `--T` | Number of temporal spiking steps |
| `--depth_transformer` | Number of Transformer layers |
| `--retrain_all` | Retrain with train+validation data |
| `--save_checkpoints` | Save checkpoints during training |

---

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{sten2026,
  title     = {Spiking Temporal-Enhanced Network for Zero-Shot Audio-Visual Learning},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

Please check the [IEEE Xplore page](https://ieeexplore.ieee.org/document/11464107) for the official citation metadata.

---

## License

This project is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

## Acknowledgements

This implementation builds on common audio-visual zero-shot learning pipelines and includes adapted components for feature extraction, baseline comparison, and spiking neural network modeling. We thank the authors of the related open-source projects and benchmark datasets.

---

# 中文说明

## 项目简介

本仓库是 ICASSP 2026 论文 **《Spiking Temporal-Enhanced Network for Zero-Shot Audio-Visual Learning》** 的官方研究代码实现。

论文链接：[IEEE Xplore](https://ieeexplore.ieee.org/document/11464107)

本项目面向 **音视频零样本学习** 与 **广义零样本学习** 任务。模型通过学习音频特征、视频特征和文本语义特征之间的共享表示空间，使模型能够识别训练阶段未见过的类别。

---

## 方法概述

STEN 在音视频零样本学习中引入脉冲神经网络的时间建模机制，通过脉冲时间动态增强音频和视频表示，并结合跨模态注意力机制完成音频、视频和语义特征的对齐。

本仓库包含：

- 本文提出的 STEN/AVCA 模型实现。
- 广义零样本学习训练与测试流程。
- AudioSetZSL、VGGSound、UCF、ActivityNet 数据集加载代码。
- AVGZSLNet、ALE、DeViSE、SJE、APN、CJME、f-VAEGAN-D2 等对比方法。
- 用于复现实验设置的脚本。

---

## 代码结构

```text
.
├── main.py                         # 训练入口
├── get_evaluation.py               # 最终评估入口
├── vae_gan_d2_xu_fsl.py            # f-VAEGAN-D2 baseline
├── src/
│   ├── args.py                     # 命令行参数
│   ├── dataset.py                  # 数据集读取与预处理
│   ├── train.py                    # 训练和验证流程
│   ├── test.py                     # 测试阶段评估
│   ├── metrics.py                  # 零样本/GZSL 指标
│   ├── model.py                    # baseline 模型
│   ├── model_improvements.py       # 本文提出的 STEN/AVCA 模型
│   ├── loss.py                     # 损失函数
│   └── Qtrick_architecture/        # 脉冲神经网络相关组件
├── run_scripts/                    # 实验脚本
├── cls_feature_extraction/         # 类别级特征提取工具
├── selavi_feature_extraction/      # SeLaVi 特征提取工具
├── audioset_vggish_tensorflow_to_pytorch/
├── w2v_features/                   # Word2Vec 相关资源
├── requirements.txt
├── environment.yaml
└── LICENSE.md
```

---

## 环境配置

推荐使用 Conda 创建环境：

```bash
conda env create -f environment.yaml
conda activate STEN
```

也可以使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

代码默认使用 CUDA。如需使用 CPU，请添加：

```bash
--device cpu
```

---

## 数据准备

本代码默认使用预提取好的音频、视频和文本语义特征。数据根目录通过 `--root_dir` 指定，特征类型通过 `--feature_extraction_method` 指定。

推荐的数据组织形式如下：

```text
<root_dir>/
├── class-split/
│   └── <split_name>/
│       ├── stage_1_train.txt
│       ├── stage_1_val_seen.txt
│       ├── stage_1_val_unseen.txt
│       ├── stage_2_train.txt
│       ├── stage_2_test_seen.txt
│       └── stage_2_test_unseen.txt
└── features/
    └── <feature_extraction_method>/
        ├── audio/
        ├── video/
        └── text/
```

支持的数据集包括：

- `AudioSetZSL`
- `VGGSound`
- `UCF`
- `ActivityNet`

仓库中也包含部分特征提取相关工具，可根据具体数据集和特征类型进行使用。

---

## 模型训练

主训练入口为 `main.py`。示例命令如下：

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_val
```

广义零样本评估通常采用两阶段流程。

第一阶段：在验证划分上训练并选择模型：

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_val
```

第二阶段：使用训练集和验证集重新训练，并保存 checkpoint：

```bash
python main.py \
  --root_dir avgzsl_benchmark_datasets/VGGSound/ \
  --feature_extraction_method main_features \
  --input_size_audio 512 \
  --input_size_video 512 \
  --dataset_name VGGSound \
  --zero_shot_split main_split \
  --AVCA \
  --retrain_all \
  --save_checkpoints \
  --epochs 50 \
  --lr 0.001 \
  --n_batches 500 \
  --exp_name sten_vggsound_all
```

更多复现实验脚本位于：

```text
run_scripts/
├── ActivityNet-GZSL/
├── UCF-GZSL/
└── VGGSound-GZSL/
```

例如：

```bash
bash run_scripts/VGGSound-GZSL/avca.sh
```

---

## 模型评估

完成两阶段训练后，可使用如下命令进行评估：

```bash
python get_evaluation.py \
  --load_path_stage_A runs/sten_vggsound_val \
  --load_path_stage_B runs/sten_vggsound_all \
  --dataset_name VGGSound \
  --AVCA
```

评估代码会在 seen 和 unseen 类别上计算广义零样本学习指标。

---

## 对比方法

仓库中包含以下方法的训练和评估脚本：

- `AVGZSLNet`
- `ALE`
- `DeViSE`
- `SJE`
- `APN`
- `CJME`
- `f-VAEGAN-D2`
- `AVCA/STEN`

相关脚本位于 `run_scripts/<dataset>-GZSL/`。

---

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--root_dir` | 数据集根目录 |
| `--feature_extraction_method` | 特征文件夹名称 |
| `--dataset_name` | 数据集名称：`AudioSetZSL`、`VGGSound`、`UCF` 或 `ActivityNet` |
| `--zero_shot_split` | 零样本划分名称，如 `main_split` 或 `cls_split` |
| `--input_size_audio` | 音频特征维度 |
| `--input_size_video` | 视频特征维度 |
| `--AVCA` | 使用本文提出的 STEN/AVCA 模型 |
| `--T` | 脉冲时间步数 |
| `--depth_transformer` | Transformer 层数 |
| `--retrain_all` | 使用训练集和验证集重新训练 |
| `--save_checkpoints` | 训练过程中保存 checkpoint |

---

## 引用

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@inproceedings{sten2026,
  title     = {Spiking Temporal-Enhanced Network for Zero-Shot Audio-Visual Learning},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

正式 BibTeX 信息请以 [IEEE Xplore 页面](https://ieeexplore.ieee.org/document/11464107) 为准。

---

## 许可证



---

## 致谢

本代码基于音视频零样本学习的常用实验流程，并结合特征提取、baseline 对比和脉冲神经网络建模模块完成实现。感谢相关开源项目和公开数据集的贡献。

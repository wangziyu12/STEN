#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

# user defined
import src.utils_improvements

# 定义 PreNorm 类，用于在某个操作（如 Attention 或 FeedForward）前进行 Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 定义层归一化（LayerNorm），对输入进行归一化处理
        self.fn = fn  # 传入的模块（如 Attention 或 FeedForward）

    def forward(self, x, **kwargs):
        """
        前向传播：
        1. 对输入 x 进行 LayerNorm 归一化处理。
        2. 将归一化后的输入传递给指定的函数 fn，并返回其输出。
        :param x: 输入张量
        :param kwargs: 传递给 fn 的额外参数
        :return: 经过 fn 处理后的输出
        """
        return self.fn(self.norm(x), **kwargs)

# 定义 FeedForward 类，用于实现前馈神经网络（MLP）
# 这个类实现了一个具有两层全连接网络的前馈模块，用于特征变换。
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        初始化前馈神经网络模块
        :param dim: 输入特征的维度
        :param hidden_dim: 隐藏层的特征维度
        :param dropout: Dropout 的比例，用于正则化，防止过拟合
        """
        super().__init__()
        # 定义一个前馈网络的顺序模块
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 第一个线性层：将输入特征从 dim 映射到 hidden_dim
            nn.GELU(),  # 使用 GELU 激活函数，增加模型的非线性能力
            nn.Dropout(dropout),  # 添加 Dropout，用于随机丢弃部分神经元，提高模型的泛化能力
            nn.Linear(hidden_dim, dim),  # 第二个线性层：将特征从 hidden_dim 映射回 dim
            nn.Dropout(dropout)  # 再次添加 Dropout，进一步防止过拟合
        )
    
    def forward(self, x):
        """
        前向传播：
        输入数据通过定义好的前馈神经网络，返回输出结果
        :param x: 输入张量
        :return: 经过前馈网络处理后的张量
        """
        return self.net(x)  # 输入 x 依次经过定义好的顺序模块，并返回输出

# 定义 Attention 类，实现多头自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        初始化多头自注意力模块
        :param dim: 输入特征维度
        :param heads: 注意力头的数量，默认为 8
        :param dim_head: 每个注意力头的维度，即QKV的维度，默认为 64
        :param dropout: Dropout 的比例，用于正则化
        """
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度是每个头的维度乘以头的数量
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要投影输出

        self.heads = heads  # 多头数量
        self.scale = dim_head ** -0.5  # 缩放因子，防止点积值过大导致梯度不稳定

        self.attend = nn.Softmax(dim=-1)  # 用于归一化注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 将输入映射为查询 (Q)、键 (K) 和值 (V)

        # 定义输出投影层，如果不需要投影，则用 Identity 占位
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出维度映射回原始输入维度
            nn.Dropout(dropout)  # 添加 Dropout，防止过拟合
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, sequence_length, feature_dim)
        :return: 输出张量，形状为 (batch_size, sequence_length, feature_dim)
        """
        # 将输入 x 映射为 Q, K, V，然后沿最后一维分割成三部分
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 调整 Q, K, V 的形状为 (batch_size, heads, sequence_length, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数矩阵，Q 和 K 的点积，并乘以缩放因子 self.scale
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 对注意力分数应用 Softmax，计算注意力权重
        attn = self.attend(dots)

        # 根据注意力权重对 V 加权求和，得到注意力结果
        out = torch.matmul(attn, v)
        # 将多头的结果合并回原始形状 (batch_size, sequence_length, heads * dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 通过输出投影层返回最终结果
        return self.to_out(out)


# 定义 Transformer 类，实现多层 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        初始化 Transformer 模块
        :param dim: 输入特征的维度
        :param depth: Transformer 堆叠的层数
        :param heads: 多头注意力中头的数量
        :param dim_head: 每个头的特征维度
        :param mlp_dim: 前馈网络中隐藏层的特征维度
        :param dropout: Dropout 的比例，用于正则化
        """
        super().__init__()
        self.layers = nn.ModuleList([])  # 用于存储多层 Transformer 模块

        for _ in range(depth):  # 按层数依次构建 Transformer 模块
            self.layers.append(nn.ModuleList([  # 每层由两个子模块组成
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前馈网络模块
            ]))

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, sequence_length, feature_dim)
        :return: 输出张量，形状与输入相同
        """
        for attn, ff in self.layers:  # 遍历每层的注意力模块和前馈网络
            x = attn(x) + x  # 应用注意力模块，并添加残差连接
            x = ff(x) + x  # 应用前馈网络，并添加残差连接
        return x  # 返回最终输出


# 定义 EmbeddingNet 类，用于实现嵌入网络
class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum, hidden_size=None):
        """
        初始化嵌入网络
        :param input_size: 输入特征的维度
        :param output_size: 输出特征的维度（最终嵌入的维度）
        :param dropout: Dropout 的比例，用于正则化
        :param use_bn: 是否使用批归一化（Batch Normalization）
        :param momentum: 批归一化的动量参数
        :param hidden_size: 隐藏层的特征维度（如果为 None，则无隐藏层）
        """
        super(EmbeddingNet, self).__init__()
        modules = []  # 用于存储网络层的模块

        # 如果指定了隐藏层大小
        if hidden_size:
            # 添加输入层到隐藏层的全连接层
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:  # 如果使用批归一化，添加 BatchNorm 层
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层

            # 添加隐藏层到输出层的全连接层
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))  # 添加 BatchNorm 层
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层
        else:
            # 如果没有隐藏层，直接从输入层到输出层
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))  # 添加 BatchNorm 层
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层

        # 使用 nn.Sequential 将所有模块组合成一个网络
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        output = self.fc(x)  # 输入通过全连接网络
        return output

    def get_embedding(self, x):
        """
        获取嵌入表示
        :param x: 输入张量
        :return: 嵌入表示（网络的前向传播结果）
        """
        return self.forward(x)  # 调用 forward 方法


# Inspired from https://github.com/AnjanDutta/sem-pcyc

# 定义 AVCA 类（用于音频和视频联合嵌入对齐的模型）
class AVCA(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        """
        初始化 AVCA 模型
        :param params_model: 包含模型超参数的字典
        :param input_size_audio: 音频输入特征的维度
        :param input_size_video: 视频输入特征的维度
        """
        super(AVCA, self).__init__()

        print('Initializing model variables...', end='')

        # 嵌入向量的维度（最终投影后的维度）
        self.dim_out = params_model['dim_out']

        # 编码器和解码器的隐藏层维度
        self.hidden_size_encoder = params_model['encoder_hidden_size']
        self.hidden_size_decoder = params_model['decoder_hidden_size']

        # Dropout 比例，用于正则化
        self.r_enc = params_model['dropout_encoder']  # 编码器的 Dropout
        self.r_proj = params_model['dropout_decoder']  # 解码器的 Dropout
        self.r_dec = params_model['additional_dropout']  # 附加模块的 Dropout

        # Transformer 模块的深度（层数）
        self.depth_transformer = params_model['depth_transformer']

        # 附加损失的控制变量
        self.additional_triplets_loss = params_model['additional_triplets_loss']  # 是否使用附加三元组损失
        self.reg_loss = params_model['reg_loss']  # 是否使用正则化损失

        # 批归一化的动量参数
        self.momentum = params_model['momentum']

        # 附加三元组损失的权重系数
        self.first_additional_triplet = params_model['first_additional_triplet']
        self.second_additional_triplet = params_model['second_additional_triplet']

        print('Initializing trainable models...', end='')

        # 定义音频编码器
        self.A_enc = EmbeddingNet(
            input_size=input_size_audio,  # 音频特征输入维度
            hidden_size=self.hidden_size_encoder,  # 编码器隐藏层维度
            output_size=300,  # 输出的嵌入维度
            dropout=self.r_enc,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义视频编码器
        self.V_enc = EmbeddingNet(
            input_size=input_size_video,  # 视频特征输入维度
            hidden_size=self.hidden_size_encoder,  # 编码器隐藏层维度
            output_size=300,  # 输出的嵌入维度
            dropout=self.r_enc,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义跨模态的 Transformer 注意力机制
        self.cross_attention = Transformer(
            dim=300,  # 输入特征维度
            depth=self.depth_transformer,  # Transformer 的层数
            heads=3,  # 注意力头的数量
            dim_head=100,  # 每个注意力头的特征维度
            mlp_dim=64,  # 前馈网络隐藏层的维度
            dropout=self.r_enc  # Dropout 比例
        )

        # 定义词嵌入的投影模块
        self.W_proj = EmbeddingNet(
            input_size=300,  # 输入的嵌入维度
            output_size=self.dim_out,  # 输出的最终投影维度
            dropout=self.r_dec,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义重建模块，用于重构嵌入
        self.D = EmbeddingNet(
            input_size=self.dim_out,  # 输入的投影维度
            output_size=300,  # 输出的嵌入维度
            dropout=self.r_dec,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义音频和视频的投影模块
        self.A_proj = EmbeddingNet(
            input_size=300,  # 输入嵌入的维度
            hidden_size=self.hidden_size_decoder,  # 解码器隐藏层维度
            output_size=self.dim_out,  # 输出的最终投影维度
            dropout=self.r_proj,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )
        self.V_proj = EmbeddingNet(
            input_size=300,  # 输入嵌入的维度
            hidden_size=self.hidden_size_decoder,  # 解码器隐藏层维度
            output_size=self.dim_out,  # 输出的最终投影维度
            dropout=self.r_proj,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义音频和视频的重建模块
        self.A_rec = EmbeddingNet(
            input_size=self.dim_out,  # 输入的投影维度
            output_size=300,  # 输出的嵌入维度
            dropout=self.r_dec,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )
        self.V_rec = EmbeddingNet(
            input_size=self.dim_out,  # 输入的投影维度
            output_size=300,  # 输出的嵌入维度
            dropout=self.r_dec,  # Dropout 比例
            momentum=self.momentum,  # 批归一化的动量
            use_bn=True  # 是否使用批归一化
        )

        # 定义 1D 的位置嵌入参数（音频和视频各一个）
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))

        # 定义优化器
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']  # 学习率
        self.optimizer_gen = optim.Adam(
            list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
            list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
            list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
            list(self.cross_attention.parameters()) + list(self.D.parameters()) +
            list(self.W_proj.parameters()),
            lr=self.lr,  # 学习率
            weight_decay=1e-5  # L2 正则化权重衰减
        )

        # 定义学习率调度器（基于指标的动态调整）
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen,
            mode='max',  # 调整目标是最大化某个指标
            patience=3,  # 连续 3 个周期没有提升时调整学习率
            verbose=True  # 输出调整日志
        )

        print('Done')

        # 定义损失函数
        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()  # 均方误差损失，用于重建模块
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)  # 三元组损失，用于学习区分嵌入
        print('Done')

    def optimize_scheduler(self, value):
        """
        调整学习率调度器
        :param value: 被监控的指标值（例如验证集的准确率或损失）
        """
        self.scheduler_gen.step(value)  # 根据指标调整学习率

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
        """
        前向传播函数
        :param audio: 音频输入特征
        :param image: 视频输入特征
        :param negative_audio: 负样本音频输入
        :param negative_image: 负样本视频输入
        :param word_embedding: 词嵌入输入
        :param negative_word_embedding: 负样本词嵌入输入
        """

        # 1. 编码音频和视频特征
        self.phi_a = self.A_enc(audio)  # 编码音频输入，生成音频嵌入
        self.phi_v = self.V_enc(image)  # 编码视频输入，生成视频嵌入

        self.phi_a_neg = self.A_enc(negative_audio)  # 编码负样本音频
        self.phi_v_neg = self.V_enc(negative_image)  # 编码负样本视频

        # 2. 提取词嵌入和负样本词嵌入
        self.w = word_embedding  # 词嵌入
        self.w_neg = negative_word_embedding  # 负样本词嵌入

        # 3. 通过投影模块映射词嵌入
        self.theta_w = self.W_proj(word_embedding)  # 投影正样本词嵌入
        self.theta_w_neg = self.W_proj(negative_word_embedding)  # 投影负样本词嵌入

        # 4. 重建词嵌入
        self.rho_w = self.D(self.theta_w)  # 重建正样本词嵌入
        self.rho_w_neg = self.D(self.theta_w_neg)  # 重建负样本词嵌入

        # 5. 构造输入张量（音频+视频嵌入与位置嵌入相加）
        self.positive_input = torch.stack(
            (self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1
        )  # 正样本输入
        self.negative_input = torch.stack(
            (self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1
        )  # 负样本输入

        # 6. 通过跨模态注意力模块
        self.phi_attn = self.cross_attention(self.positive_input)  # 正样本的注意力输出
        self.phi_attn_neg = self.cross_attention(self.negative_input)  # 负样本的注意力输出

        # 7. 加入注意力后得到的音频和视频嵌入
        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]  # 融合注意力的音频嵌入
        self.video_fe_attn = self.phi_v + self.phi_attn[:, 1, :]  # 融合注意力的视频嵌入

        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]  # 负样本音频嵌入
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]  # 负样本视频嵌入

        # 8. 将音频和视频嵌入投影到最终共享空间
        self.theta_v = self.V_proj(self.video_fe_attn)  # 正样本视频投影
        self.theta_v_neg = self.V_proj(self.video_fe_neg_attn)  # 负样本视频投影
        self.theta_a = self.A_proj(self.audio_fe_attn)  # 正样本音频投影
        self.theta_a_neg = self.A_proj(self.audio_fe_neg_attn)  # 负样本音频投影

        # 9. 重建视频和音频嵌入
        self.phi_v_rec = self.V_rec(self.theta_v)  # 视频嵌入的重建
        self.phi_a_rec = self.A_rec(self.theta_a)  # 音频嵌入的重建

        # 10. 投影重建结果
        self.se_em_hat1 = self.A_proj(self.phi_a_rec)  # 重建音频后的投影
        self.se_em_hat2 = self.V_proj(self.phi_v_rec)  # 重建视频后的投影

        # 11. 通过 D 模块进一步重构嵌入
        self.rho_a = self.D(self.theta_a)  # 重构音频嵌入
        self.rho_a_neg = self.D(self.theta_a_neg)  # 重构负样本音频嵌入
        self.rho_v = self.D(self.theta_v)  # 重构视频嵌入
        self.rho_v_neg = self.D(self.theta_v_neg)  # 重构负样本视频嵌入



    def backward(self, optimize):
        """
        反向传播函数：计算损失、执行反向传播和优化步骤
        :param optimize: 是否执行优化器步骤（True 表示更新参数）
        :return: 数值化的损失值和损失字典
        """

        # 1. 计算附加三元组损失（如果启用）
        if self.additional_triplets_loss == True:
            # 第一组三元组损失：正样本音频/视频与词嵌入的对比学习
            first_pair = self.first_additional_triplet * (
                self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) +  # 音频-词嵌入
                self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg)    # 视频-词嵌入
            )
            # 第二组三元组损失：词嵌入与音频/视频的对比学习
            second_pair = self.second_additional_triplet * (
                self.triplet_loss(self.theta_w, self.theta_a, self.theta_w_neg) +  # 词嵌入-音频
                self.triplet_loss(self.theta_w, self.theta_v, self.theta_w_neg)    # 词嵌入-视频
            )

            # 总的附加三元组损失
            l_t = first_pair + second_pair

        # 2. 计算正则化损失（如果启用）
        if self.reg_loss == True:
            l_r = (
                self.criterion_reg(self.phi_v_rec, self.phi_v) +  # 重建的视频嵌入与原始视频嵌入的差异
                self.criterion_reg(self.phi_a_rec, self.phi_a) +  # 重建的音频嵌入与原始音频嵌入的差异
                self.criterion_reg(self.theta_v, self.theta_w) +  # 视频投影与词嵌入投影的差异
                self.criterion_reg(self.theta_a, self.theta_w)    # 音频投影与词嵌入投影的差异
            )

        # 3. 计算重建损失
        l_rec = (
            self.criterion_reg(self.w, self.rho_v) +  # 正样本词嵌入与视频的重建嵌入的差异
            self.criterion_reg(self.w, self.rho_a) +  # 正样本词嵌入与音频的重建嵌入的差异
            self.criterion_reg(self.w, self.rho_w)    # 正样本词嵌入与自身重建嵌入的差异
        )

        # 4. 计算跨模态三元组损失
        l_ctv = self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg)  # 词嵌入-视频正负样本的三元组损失
        l_cta = self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg)  # 词嵌入-音频正负样本的三元组损失
        l_ct = l_cta + l_ctv  # 总的跨模态三元组损失
        l_cmd = l_rec + l_ct   # 跨模态对比学习总损失

        # 5. 计算投影三元组损失
        l_tv = self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)  # 词嵌入-视频
        l_ta = self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg)  # 词嵌入-音频
        l_at = self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg)  # 音频-词嵌入
        l_vt = self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)  # 视频-词嵌入
        l_w = l_ta + l_at + l_tv + l_vt  # 投影总三元组损失

        # 6. 汇总生成损失
        loss_gen = l_cmd + l_w  # 生成损失：跨模态对比损失 + 投影损失
        if self.additional_triplets_loss == True:
            loss_gen += l_t  # 加入附加的三元组损失
        if self.reg_loss == True:
            loss_gen += l_r  # 加入正则化损失

        # 7. 执行优化步骤（如果 optimize=True）
        if optimize == True:
            self.optimizer_gen.zero_grad()  # 清空梯度
            loss_gen.backward()             # 计算梯度
            self.optimizer_gen.step()       # 更新参数

        # 8. 定义损失字典
        loss = {
            'aut_enc': 0,       # 该字段可能是未来扩展用，当前未使用
            'gen_cyc': 0,       # 周期损失，未使用
            'gen_reg': 0,       # 正则化损失，未直接使用
            'gen': loss_gen     # 总生成损失
        }

        # 9. 返回数值化的损失值和损失字典
        loss_numeric = loss['gen_cyc'] + loss['gen']
        return loss_numeric, loss


    def optimize_params(self, audio, video, cls_numeric, cls_embedding, audio_negative, video_negative, negative_cls_embedding, optimize=False):
        """
        参数优化函数：执行前向传播、损失计算和（可选）优化步骤
        :param audio: 音频输入特征
        :param video: 视频输入特征
        :param cls_numeric: 类别编号（通常用于标签或辅助信息）
        :param cls_embedding: 词嵌入输入（正样本的词嵌入）
        :param audio_negative: 负样本音频输入特征
        :param video_negative: 负样本视频输入特征
        :param negative_cls_embedding: 负样本词嵌入
        :param optimize: 是否执行优化步骤（True 表示更新参数，False 仅计算损失）
        :return:
            - loss_numeric: 数值化的总损失值
            - loss: 各损失项组成的损失字典
        """

        # 1. 执行前向传播：计算音频、视频、词嵌入的投影和重建
        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)

        # 2. 执行反向传播和（可选的）优化步骤
        loss_numeric, loss = self.backward(optimize)

        # 3. 返回总损失值和损失字典
        return loss_numeric, loss

    def get_embeddings(self, audio, video, embedding):
        """
        获取音频、视频和词嵌入的投影表示
        :param audio: 音频输入特征
        :param video: 视频输入特征
        :param embedding: 词嵌入（目标语义的嵌入表示）
        :return:
            - theta_a: 投影到共享空间的音频嵌入
            - theta_v: 投影到共享空间的视频嵌入
            - theta_w: 投影到共享空间的词嵌入
        """

        # 1. 编码音频和视频特征
        phi_a = self.A_enc(audio)  # 通过音频编码器生成音频嵌入
        phi_v = self.V_enc(video)  # 通过视频编码器生成视频嵌入

        # 2. 将词嵌入映射到共享空间
        theta_w = self.W_proj(embedding)  # 通过词嵌入投影模块生成词嵌入

        # 3. 构造跨模态输入张量
        input_concatenated = torch.stack(
            (phi_a + self.pos_emb1D[0, :], phi_v + self.pos_emb1D[1, :]), dim=1
        )  # 将音频和视频嵌入与位置嵌入相加，沿第 1 维堆叠

        # 4. 跨模态注意力机制
        phi_attn = self.cross_attention(input_concatenated)  # 通过 Transformer 计算注意力输出

        # 5. 融合注意力输出
        phi_a = phi_a + phi_attn[:, 0, :]  # 更新音频嵌入
        phi_v = phi_v + phi_attn[:, 1, :]  # 更新视频嵌入

        # 6. 将音频和视频嵌入投影到共享空间
        theta_v = self.V_proj(phi_v)  # 视频投影
        theta_a = self.A_proj(phi_a)  # 音频投影

        # 7. 返回投影结果
        return theta_a, theta_v, theta_w

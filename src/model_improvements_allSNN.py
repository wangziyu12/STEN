#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional
# from spikingjelly.clock_driven.neuron import (
#     MultiStepParametricLIFNode,
#     MultiStepLIFNode,
# )
# user defined
import src.utils_improvements
# from .Qtrick_architecture.clock_driven import neuron
from .Qtrick_architecture.clock_driven import surrogate, layer
from .submodules.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul
# import tensorly as tl
# from tensorly.tenalg import inner as tl_inner
# from tensorly.decomposition import tucker
# import math

# torch.pi = math.pi
# tl.set_backend('pytorch')

# 定义 PreNorm 类，用于在某个操作（如 Attention 或 FeedForward）前进行 Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 定义 FeedForward 类，用于实现前馈神经网络（MLP）
# 这个类实现了一个具有两层全连接网络的前馈模块，用于特征变换。
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        '''纯SNN'''
        self.lif = LIFAct(step=2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            self.lif,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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
        # print(f"初始维度x.shape: {x.shape}") 
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print(f"维度qkv[0].shape, qkv[1].shape, qkv[2].shape: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}")
        # 调整 Q, K, V 的形状为 (batch_size, heads, sequence_length, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # print(f"维度q.shape，k.shape，v.shape: {q.shape}, {k.shape}, {v.shape}")

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

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,  # 输入通道数
        out_channel,  # 输出通道数
        bias=False,  # 是否使用偏置
    ):
        """
        初始化 RepConv 模块：
        1. 由 1x1 卷积、BatchNorm 以及 3x3 深度卷积 + 1x1 逐点卷积 组成
        2. 这种结构可以在推理时转换为普通的 3x3 卷积，提高效率
        """
        super().__init__()

        # **1x1 卷积层**（用于调整通道信息）
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)

        # **BatchNorm 和 Pad 层**
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)

        # **3x3 深度可分离卷积 + 1x1 逐点卷积**
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),  # 深度卷积
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(out_channel),  # 归一化
        )

        # **构建完整的 RepConv 结构**
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        """
        前向传播：
        - 输入: x.shape = (B, C_in, H, W)
        - 输出: x.shape = (B, C_out, H, W)
        """
        return self.body(x)
    

# Spike-Driven Self-Attention 计算 Q, K, V 并使用 LIF 神经元
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,  # 输入特征维度（即通道数 C）
        num_heads=8,  # 多头注意力的头数（默认为 8）
        qkv_bias=False,  # 是否对 QKV 计算添加偏置
        qk_scale=None,  # 额外的缩放因子（默认使用 `1 / sqrt(dim_head)`）
        attn_drop=0.0,  # 注意力矩阵的 Dropout
        proj_drop=0.0,  # 输出投影的 Dropout
        sr_ratio=1,  # 采样率（默认为 1）
    ):
        super().__init__()
        # 确保 dim 可以被 num_heads 整除
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # 特征维度
        self.num_heads = num_heads  # 头数
        # self.scale = 0.125  # 缩放因子（相当于 `1 / sqrt(dim_head)`）
        self.scale = (dim // num_heads) ** -0.5  # sqrt(dim_head)

        # **全局 LIF 神经元**：用于模拟输入的时间序列信息
        # self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.head_lif = LIFAct(step=2)

        # **使用 RepConv 计算 Q, K, V**
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        # **LIF 神经元处理 Q, K, V**
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.q_lif = LIFAct(step=2)
        self.k_lif = LIFAct(step=2)
        self.v_lif = LIFAct(step=2)
        
        # **LIF 神经元处理注意力计算后的输出**
        # self.attn_lif = MultiStepLIFNode(
        #     tau=2.0, v_threshold=0.5, detach_reset=True
        # )
        self.attn_lif = LIFAct(step=2)

         # **最终投影层（RepConv）**
        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        """
        前向传播流程：
        1. LIF 神经元处理输入
        2. 计算 Q, K, V
        3. LIF 处理 Q, K, V
        4. 计算注意力分数 QK^T
        5. 计算加权 V 并通过 LIF
        6. 进行投影并输出

        输入：
        - x: 形状为 (T, B, C, H, W)，其中：
            - T = 时间步
            - B = 批次大小
            - C = 通道数
            - H, W = 空间维度

        输出：
        - x: 形状为 (T, B, C, H, W)，经过 SNN 处理的特征
        """
        T, B, C, H, W = x.shape  # 获取输入形状
        N = H * W  # 计算 Flatten 后的 token 数量

        # **1. 通过 LIF 神经元处理输入**
        x = self.head_lif(x)

        # **2. 计算 Q, K, V**
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

         # **3. 通过 LIF 神经元处理 Q, K, V**
        q = self.q_lif(q).flatten(3)  # (T, B, C, N)
        q = (
            q.transpose(-1, -2)  # (T, B, N, C)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # 分割多头
            .permute(0, 1, 3, 2, 4)  # 变换维度 (T, B, num_heads, N, head_dim)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # **4. 计算 QK^T（注意力分数）**
        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        # **5. 计算注意力权重的加权 V**
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)

        # **6. 进行最终投影**
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x

# 定义 Attention 类，实现多头自注意力机制
'''ronghe6'''
class Attention_new_SDSA(nn.Module):
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

        # self.attend = nn.Softmax(dim=-1)  # 用于归一化注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 将输入映射为查询 (Q)、键 (K) 和值 (V)

        self.q_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))
        
        '''new: ronghe9'''
        self.conv = nn.Sequential(nn.Conv1d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=1, padding=0, bias=False), 
                                  nn.BatchNorm1d(inner_dim))
        self.sigmoid = nn.Sigmoid()

        # self.q_lif = neuron.IFNode()
        # self.k_lif = neuron.IFNode()
        # self.v_lif = neuron.IFNode()
        # self.attn_lif = neuron.IFNode()

        self.head_spike = LIFAct(step=2)
        self.q_lif = LIFAct(step=2)
        self.k_lif = LIFAct(step=2)
        self.v_lif = LIFAct(step=2) 
        self.attn_lif = LIFAct(step=2)


        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        # self.attn_lif = MultiStepLIFNode(
        #     tau=2.0, v_threshold=0.5, detach_reset=True
        # )

        # self.head_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.q_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.k_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.v_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
        # self.attn_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

        self.proj_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

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
        print(f"初始维度x.shape: {x.shape}")
        # B, N, D = x.shape
        B, D = x.shape
        x = self.head_spike(x.clone())
        functional.reset_net(self.head_spike)
        # print(f"初始维度x.shape: {x.shape}") 
        # 将输入 x 映射为 Q, K, V，然后沿最后一维分割成三部分
        # qkv = (self.to_qkv(x).chunk(3, dim=-1))
        # 输出形状为 (B, N, inner_dim*3)，其中 inner_dim = dim_head * heads
        qkv = self.to_qkv(x)
        # 分离出 Q, K, V，每个张量形状为 (B, N, inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        '''new: ronghe9'''
        q = self.q_conv(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.k_conv(k.permute(0, 2, 1)).permute(0, 2, 1)
        v = self.v_conv(v.permute(0, 2, 1)).permute(0, 2, 1)

        # 将每个张量 reshape 为 (B, N, heads, d_head) ，d_head = inner_dim / heads，然后转置为 (B, heads, N, d_head)
        q = q.reshape(B, N, self.heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.heads, -1).permute(0, 2, 1, 3)

        # 调整 Q, K, V 的形状为 (batch_size, heads, sequence_length, dim_head)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # print(f"维度q.shape，k.shape，v.shape: {q.shape}, {k.shape}, {v.shape}")

        # # 计算注意力分数矩阵，Q 和 K 的点积，并乘以缩放因子 self.scale
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # # 对注意力分数应用 Softmax，计算注意力权重
        # attn = self.attend(dots)

        # # 根据注意力权重对 V 加权求和，得到注意力结果
        # out = torch.matmul(attn, v)

        q = self.q_lif(q)
        k = self.k_lif(k)
        v = self.v_lif(v)
        # print(f"LIF后的维度q.shape，k.shape，v.shape: {q.shape}, {k.shape}, {v.shape}")
        functional.reset_net(self.q_lif)
        functional.reset_net(self.k_lif)
        functional.reset_net(self.v_lif)
        

        # attn = k.transpose(-2, -1) @ v
        attn = torch.matmul(k.transpose(-2, -1) , v)
        # print(f"维度attn.shape: {attn.shape}")
        out = torch.matmul(q, attn) * self.scale
        # print(f"维度out.shape: {out.shape}")
        out = self.attn_lif(out)
        # print(f"LIF后的维度out.shape: {out.shape}")
        functional.reset_net(self.attn_lif)
        # out = self.proj_conv(out.clone()).contiguous()
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)

        '''new: ronghe10 no'''
        # out = self.q_conv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # 将多头的结果合并回原始形状 (batch_size, sequence_length, heads * dim_head)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # print(f"rearrange后的维度out.shape: {out.shape}")

        # 通过输出投影层返回最终结果
        return self.to_out(out)


# class Attention_new_SDSA(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         """
#         初始化多头自注意力模块
#         :param dim: 输入特征维度
#         :param heads: 注意力头的数量，默认为 8
#         :param dim_head: 每个注意力头的维度，即QKV的维度，默认为 64
#         :param dropout: Dropout 的比例，用于正则化
#         """
#         super().__init__()
#         inner_dim = dim_head * heads  # 内部维度是每个头的维度乘以头的数量
#         project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要投影输出

#         self.heads = heads  # 多头数量
#         self.scale = dim_head ** -0.5  # 缩放因子，防止点积值过大导致梯度不稳定

#         # self.attend = nn.Softmax(dim=-1)  # 用于归一化注意力权重
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 将输入映射为查询 (Q)、键 (K) 和值 (V)

#         self.q_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))
#         self.k_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))
#         self.v_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm1d(dim))

#         # self.q_lif = neuron.IFNode()
#         # self.k_lif = neuron.IFNode()
#         # self.v_lif = neuron.IFNode()
#         # self.attn_lif = neuron.IFNode()

#         self.head_spike = LIFAct(step=2)
#         self.q_lif = LIFAct(step=2)
#         self.k_lif = LIFAct(step=2)
#         self.v_lif = LIFAct(step=2) 
#         self.attn_lif = LIFAct(step=2)


#         # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
#         # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
#         # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
#         # self.attn_lif = MultiStepLIFNode(
#         #     tau=2.0, v_threshold=0.5, detach_reset=True
#         # )

#         # self.head_spike = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
#         # self.q_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
#         # self.k_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
#         # self.v_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())
#         # self.attn_lif = neuron.Q_IFNode(surrogate_function=surrogate.Quant())

#         self.proj_conv = nn.Sequential(
#             nn.Conv1d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
#         )

#         # 定义输出投影层，如果不需要投影，则用 Identity 占位
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),  # 输出维度映射回原始输入维度
#             nn.Dropout(dropout)  # 添加 Dropout，防止过拟合
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入张量，形状为 (batch_size, sequence_length, feature_dim)
#         :return: 输出张量，形状为 (batch_size, sequence_length, feature_dim)
#         """
#         x = self.head_spike(x.clone())
#         functional.reset_net(self.head_spike)
#         # print(f"初始维度x.shape: {x.shape}") 
#         # 将输入 x 映射为 Q, K, V，然后沿最后一维分割成三部分
#         qkv = (self.to_qkv(x).chunk(3, dim=-1))
#         # qkv = list(self.to_qkv(x).chunk(3, dim=-1))
#         # print(f"维度qkv[0].shape, qkv[1].shape, qkv[2].shape: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}")
#         # qkv[0] = qkv[0].permute(0, 2, 1).contiguous()
#         # qkv[1] = qkv[1].permute(0, 2, 1).contiguous()
#         # qkv[2] = qkv[2].permute(0, 2, 1).contiguous()
#         # qkv[0] = self.q_conv(qkv[0].clone()).contiguous()
#         # qkv[1] = self.k_conv(qkv[1].clone()).contiguous()
#         # qkv[2] = self.v_conv(qkv[2].clone()).contiguous()
#         # qkv[0] = qkv[0].permute(0, 2, 1).contiguous()
#         # qkv[1] = qkv[1].permute(0, 2, 1).contiguous()
#         # qkv[2] = qkv[2].permute(0, 2, 1).contiguous()
#         # qkv = tuple(qkv) 
#         # 调整 Q, K, V 的形状为 (batch_size, heads, sequence_length, dim_head)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         # print(f"维度q.shape，k.shape，v.shape: {q.shape}, {k.shape}, {v.shape}")

#         # # 计算注意力分数矩阵，Q 和 K 的点积，并乘以缩放因子 self.scale
#         # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         # # 对注意力分数应用 Softmax，计算注意力权重
#         # attn = self.attend(dots)

#         # # 根据注意力权重对 V 加权求和，得到注意力结果
#         # out = torch.matmul(attn, v)

#         # q = q.permute(0, 3, 2, 1).contiguous()  # (B, H, N, D) → (B, D, N, H)
#         # k = k.permute(0, 3, 2, 1).contiguous()
#         # v = v.permute(0, 3, 2, 1).contiguous()

#         # q = self.q_conv(q.clone()).contiguous()
#         # k = self.k_conv(k.clone()).contiguous()
#         # v = self.v_conv(v.clone()).contiguous()

#         # q = q.permute(0, 3, 2, 1).contiguous()  
#         # k = k.permute(0, 3, 2, 1).contiguous()
#         # v = v.permute(0, 3, 2, 1).contiguous()

#         q = self.q_lif(q).contiguous()
#         k = self.k_lif(k).contiguous()
#         v = self.v_lif(v).contiguous()
#         # print(f"LIF后的维度q.shape，k.shape，v.shape: {q.shape}, {k.shape}, {v.shape}")
#         functional.reset_net(self.q_lif)
#         functional.reset_net(self.k_lif)
#         functional.reset_net(self.v_lif)
        

#         # attn = k.transpose(-2, -1) @ v
#         attn = torch.matmul(k.transpose(-2, -1) , v).contiguous()
#         # print(f"维度attn.shape: {attn.shape}")
#         out = torch.matmul(q, attn) * self.scale
#         # print(f"维度out.shape: {out.shape}")
#         out = self.attn_lif(out).contiguous()
#         # print(f"LIF后的维度out.shape: {out.shape}")
#         functional.reset_net(self.attn_lif)
#         # out = self.proj_conv(out.clone()).contiguous()

#         # 将多头的结果合并回原始形状 (batch_size, sequence_length, heads * dim_head)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         # print(f"rearrange后的维度out.shape: {out.shape}")

#         # 通过输出投影层返回最终结果
#         return self.to_out(out)


# 定义 Transformer 类，实现多层 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., attention = Attention):
    # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
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
        self.layers = nn.ModuleList([]) # 用于存储多层 Transformer 模块
        for _ in range(depth):  # 按层数依次构建 Transformer 模块
            self.layers.append(nn.ModuleList([  # 每层由两个子模块组成
                PreNorm(dim, attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 注意力模块
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
        '''纯SNN'''
        functional.reset_net(self.layers)
        return x

# '''ronghe7 ? 或许因为下面两个残差连接导致Spikeformer的输出不是完全脉冲 ? '''
# class Transformer_SDSA(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#     # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         """
#         初始化 Transformer 模块
#         :param dim: 输入特征的维度
#         :param depth: Transformer 堆叠的层数
#         :param heads: 多头注意力中头的数量
#         :param dim_head: 每个头的特征维度
#         :param mlp_dim: 前馈网络中隐藏层的特征维度
#         :param dropout: Dropout 的比例，用于正则化
#         """
#         super().__init__()
#         self.layers = nn.ModuleList([]) # 用于存储多层 Transformer 模块
#         for _ in range(depth):  # 按层数依次构建 Transformer 模块
#             self.layers.append(nn.ModuleList([  # 每层由两个子模块组成
#                 PreNorm(dim, Attention_new_SDSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 注意力模块
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前馈网络模块
#             ]))
#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入张量，形状为 (batch_size, sequence_length, feature_dim)
#         :return: 输出张量，形状与输入相同
#         """
#         self.lif = LIFAct(step=2)
#         for attn, ff in self.layers:  # 遍历每层的注意力模块和前馈网络
#             x = attn(x) + self.lif(x)  # 应用注意力模块，并添加残差连接
#             '''ronghe8'''
#             # x = ff(x) + self.lif(x)  # 应用前馈网络，并添加残差连接
#         # functional.reset_net(self.layers)
#         return x

# 定义 EmbeddingNet 类，用于实现嵌入网络，即将输入特征映射到输出特征空间
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
            if use_bn:  # 如果使用批归一化，添加 BatchNorm 批归一化层
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层

            # 添加隐藏层到输出层的全连接层
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))  # 添加 BatchNorm 批归一化层
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层
        else:
            # 如果没有隐藏层，直接从输入层到输出层
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))  # 添加 BatchNorm 批归一化层
            modules.append(nn.ReLU())  # 添加 ReLU 激活函数
            modules.append(nn.Dropout(dropout))  # 添加 Dropout 层

        # 使用 nn.Sequential 将所有模块组合成一个网络
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)



def MPR(s, thresh):
    """
    MPR（膜电位重置）函数，根据膜电位的不同值范围对输入 `s` 进行处理，以模拟神经元膜电位的变化。
    - s (Tensor): 输入的膜电位张量，表示神经元的膜电位。
    - thresh (float): 阈值，用于在膜电位范围 [0, 1] 之间平滑地控制膜电位的变化。
    返回：
    Tensor: 处理后的膜电位张量。
    """
    # 对大于 1 的膜电位进行立方根处理
    s[s > 1.] = s[s > 1.] ** (1.0 / 3)
    # 对小于 0 的膜电位进行调整：先减去 1 再取立方根，最后加回 1
    s[s < 0.] = -(-(s[s < 0.] - 1.)) ** (1.0 / 3) + 1.
    # 对介于 0 和 1 之间的膜电位进行平滑处理，使用双曲正切函数控制平滑度
    s[(0. < s) & (s < 1.)] = 0.5 * torch.tanh(3. * (s[(0. < s) & (s < 1.)] - thresh)) / np.tanh(3. * (thresh)) + 0.5
    return s


def gradient_scale(x, scale):
    """
    对输入张量 `x` 进行梯度缩放，用于调整梯度的大小，防止梯度消失或梯度爆炸。
    - x (Tensor): 输入张量，通常是神经网络中的输出。
    - scale (float): 缩放因子，用于调整梯度的大小。
    返回：
    Tensor: 返回经过梯度缩放后的张量。
    """
    # 保存原始输出张量
    yout = x
    # 计算梯度（即将输入张量乘以缩放因子）
    ygrad = x * scale
    # 返回经过梯度缩放后的张量，但通过 .detach() 防止梯度的传播
    y = (yout - ygrad).detach() + ygrad
    return y



class SpikeModule(nn.Module):
    """
    SpikeModule 类继承自 nn.Module，是一个基础的脉冲神经元模块。
    用于控制是否处于脉冲状态，并处理输入信号。
    """
    def __init__(self):
        super().__init__()
        self._spiking = False  # 默认不处于脉冲状态

    # 设置脉冲状态，控制当前模块是否进入脉冲状态。
    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        self._spiking = use_spike  # 如果 use_spike 为 True，则开启脉冲状态，否则关闭

    # 如果不是脉冲状态（_spiking 为 False），并且输入是5维数据（表示时间序列数据），
    # 那么会对输入数据沿时间维度进行均值计算，来实现形状修正
    # 将输入张量从五维压缩到四维，从而符合非脉冲计算模式的需求，即 [批量大小, 通道数, 高度, 宽度]。
    def forward(self, x):
        if self._spiking is not True and len(x.shape) == 5:  # 判断是否为脉冲状态并且输入是五维张量
            x = x.mean([0])  # 沿着时间维度对输入数据求均值
        return x  # 返回处理后的数据
    

def spike_activation(x, binary=False, temp=1.0):
    """
    脉冲激活函数，用于根据膜电位决定是否发放脉冲。
    - `binary` 参数控制是否使用二值脉冲
    - `temp` 控制温度，通常用于模拟发放脉冲的“平滑性”
    """
    if binary:
        # 二值脉冲激活：只输出 0 或 1
        out_s = torch.gt(x, 0.5)  # 实际的脉冲输出张量，如果 x > 0.5，则输出 1；否则输出 0
        out_bp = torch.clamp(x, 0, 1)  # 为反向传播设计的代理张量，将输入 x 限制在 [0, 1] 范围内
        return (out_s.float() - out_bp).detach() + out_bp
    else:
        # 三值脉冲激活：输出 -1、0 或 1
        out_s = torch.sign(x)  # 生成三值脉冲：如果 x > 0 输出 1，x < 0 输出 -1，x = 0 输出 0
        out_s[torch.abs(x) < 0.5] = torch.tensor(0.)  # 如果膜电位的绝对值小于 0.5，则输出 0
        out_bp = torch.clamp(x, -1, 1)  # 将输入 x 限制在 [-1, 1] 范围内
        # 原始的平滑处理，在调整时可以选择性启用
        # out_bp[out_bp > 0.] = (torch.tanh(temp * (out_bp[out_bp > 0.] - 0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        # out_bp[out_bp <= 0.] = (torch.tanh(temp * (out_bp[out_bp <= 0.] + 0.5)) - np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        return (out_s.float() - out_bp).detach() + out_bp  # 返回脉冲的激活值

def mem_update(x_in, mem, V_th, decay, fire_ratio, grad_scale=1., temp=1.0):
    """
    计算神经元的膜电位更新，并决定是否发放脉冲（spike）。
    - x_in: 输入信号（当前时间步输入）
    - mem:  神经元的当前膜电位
    - V_th:  脉冲发放的阈值（可训练参数）
    - decay: 膜电位衰减因子（通常小于1，用于模拟生物神经元的泄漏效应）
    - fire_ratio:  脉冲发放比例（可训练参数，影响脉冲的幅度）
    - grad_scale:  梯度缩放因子，用于调整梯度的大小，防止梯度消失或爆炸
    - temp: 温度参数，控制 `spike_activation` 的平滑度
    """
    mem = mem * decay + x_in    # 计算新的膜电位
    #if mem.shape[1]==256:
    #    embed()

    '''new: ronghe5'''
    V_th = gradient_scale(V_th, grad_scale)
    mem = MPR(mem, 0.5)

    spike = spike_activation(mem / V_th, temp=temp) # 归一化膜电位后进行激活，得到 {-1, 0, 1}
    mem = mem * (1 - torch.abs(spike))  # 发生脉冲后膜电位清零
    #mem = mem - spike
    spike = spike * fire_ratio  # 乘以可训练的发放比例
    return mem, spike

class LIFAct(SpikeModule):
    """
    LIF 神经元的实现，它可以生成脉冲信号，并用于替代 ReLU 激活函数。
    - 通过时间步进行计算
    - 具有可训练的阈值 `V_th` 和脉冲幅度 `fire_ratio`
    - 采用 `mem_update` 进行膜电位计算，并使用 `spike_activation` 计算脉冲信号
    """

    def __init__(self, step):
        """
        - step: 时间步长，即神经元在 `forward` 传播时要处理的时间步数量
        """
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = nn.Parameter(torch.tensor(1.))  # 训练过程中可学习的脉冲发放阈值
        # self.V_th = 1.0

        '''new: ronghe5'''
        self.tau = nn.Parameter(torch.tensor(-1.1))   # 可训练的时间常数（控制膜电位衰减速度）

        '''new: ronghe5'''
        # self.temp = 3.0     # 温度参数，影响 `spike_activation` 的平滑性
        self.temp = nn.Parameter(torch.tensor(1.))
        
        self.grad_scale = 0.1   # 梯度缩放因子，用于调整梯度的大小，防止梯度消失或爆炸
        
        #self.fire_ratio = nn.Parameter(torch.ones(1,512,1,1), requires_grad=True)
        self.fire_ratio = nn.Parameter(torch.tensor(1.))    # 训练过程中可调整的脉冲强度
        # self.fire_ratio = 1


    def forward(self, x):
        """
        前向传播，计算神经元的膜电位，并发放脉冲。
        - out: 计算得到的脉冲张量（多时间步堆叠后的输出）
        """
        # 如果当前不处于 SNN 计算模式，则默认使用 ReLU 激活
        if self._spiking is not True:
            return F.relu(x)
        # 如果 `grad_scale` 未被初始化，则自动计算
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        # 初始化膜电位 `u`，其形状与 `x[0]` 相同
        u = torch.zeros_like(x[0])
        # 定义一个列表来保存所有时间步的输出
        out = []
        T, B, C, H, W = x.shape     # 获取输入数据的维度
        # 遍历每个时间步，计算膜电位并发放脉冲
        for i in range(self.step):
            # 调用 `mem_update` 函数计算膜电位和脉冲
            '''new: ronghe5'''
            # u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,fire_ratio=self.fire_ratio,
            #                       grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,fire_ratio=self.fire_ratio,
                                  grad_scale=self.grad_scale, decay=self.tau, temp=self.temp)
            
            out += [out_i]
        # 将所有时间步的脉冲堆叠成一个张量，输出形状为 `[T, B, C, H, W]`
        out = torch.stack(out)
        return out

'''new: ronghe9'''
# class TCJA(nn.Module):
#     def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
#         super().__init__()

#         self.conv = nn.Conv1d(in_channels=T, out_channels=T,
#                               kernel_size=kernel_size_t, padding='same', bias=False)
#         self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
#                                 kernel_size=kernel_size_c, padding='same', bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x_seq: torch.Tensor):
#         x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
#         x_c = x.permute(0, 2, 1)
#         conv_t_out = self.conv(x).permute(1, 0, 2)
#         conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
#         out = self.sigmoid(conv_c_out * conv_t_out)
#         y_seq = x_seq * out[:, :, :, None, None]
#         return y_seq
class TCJA(nn.Module):
    """
    Temporal-Channel Joint Attention (TCJA) 模块
    该模块用于同时处理时间和通道维度的注意力机制，适用于脉冲神经网络（SNN）。
    """
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        """
        :param kernel_size_t: 时间维度的卷积核大小，默认为 2
        :param kernel_size_c: 通道维度的卷积核大小，默认为 1
        :param T: 时间步长，默认为 8
        :param channel: 通道数，默认为 128
        """
        super().__init__()
        # 定义时间维度的1D卷积层
        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding=(kernel_size_t - 1) // 2, bias=False)
        # 定义通道维度的1D卷积层
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding=(kernel_size_t - 1) // 2, bias=False)
        # 定义Sigmoid激活函数，用于将输出限制在0到1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        """
        :param x_seq: 输入张量，形状为 (T, batch_size, channel, H, W)
        :return: 输出张量，形状与输入相同
        """
        # 1. 对输入张量进行空间维度的压缩
        # 将输入张量的维度从 (T, batch_size, channel, H, W) 调整为 (batch_size, T, channel)
        # x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x = x_seq.permute(1, 0, 2)
        
        # 2. 对通道维度进行调整，以便进行1D卷积
        x_c = x.permute(0, 2, 1)  # 调整维度为 (batch_size, channel, T)
        
        # 3. 对时间维度进行1D卷积
        conv_t_out = self.conv(x).permute(1, 0, 2)  # 调整维度为 (batch_size, T, channel)
        
        # 4. 对通道维度进行1D卷积
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)  # 调整维度为 (T, batch_size, channel)
        
        # 5. 计算时间和通道维度的联合注意力权重
        out = self.sigmoid(conv_c_out * conv_t_out)  # 使用Sigmoid激活函数将输出限制在0到1之间
        
        # 6. 将注意力权重应用到输入张量上
        y_seq = x_seq * out # 注意力权重的形状为 (T, batch_size, channel, 1, 1)
        
        return y_seq

class SNNBranch(nn.Module):
    """
    SNN 分支模块（SNNBranch），用于构建包含 LIF（Leaky Integrate-and-Fire）神经元的神经网络分支。
    该模块类似于标准的多层感知机（MLP），但使用 LIFAct 作为激活函数，适用于 SNN 计算。
    """
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        """
        - input_size: 输入特征维度
        - output_size: 输出特征维度
        - dropout:  预留的 dropout 率（当前未启用）
        - use_bn: 是否使用批量归一化（BatchNorm），当前未启用
        - momentum: 批量归一化的动量参数（如果启用 BN）
        - hidden_size: 隐藏层维度，如果为 None，则不使用隐藏层
        """
        '''
        Initial input shape: torch.Size([256, 512])
        After module 0 (Linear): torch.Size([256, 512])
        After module 1 (LIFAct): torch.Size([256, 512])
        After module 2 (Linear): torch.Size([256, 300])     
        After module 3 (LIFAct): torch.Size([256, 300])
        '''
        super(SNNBranch, self).__init__()
        modules = []    # 存储模型层的列表
        if hidden_size:
            # 第一层：全连接层，将输入大小 `input_size` 映射到 `hidden_size`
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            '''new: ronghe11'''
            ## 如果启用了 BN，则使用批量归一化
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
           
            # 添加 LIF 脉冲神经元层，step=4 表示计算 4 个时间步
            modules.append(LIFAct(step=2))

            '''new: ronghe9'''
            # modules.append(TCJA(kernel_size_t=2, kernel_size_c=1, T=2, channel=hidden_size))
            # modules.append(neuron.IFNode())
            '''new: ronghe12, ronghe11√'''
            ## 可选的 Dropout 层
            modules.append(nn.Dropout(dropout))
            
            # 第二层：全连接层，将 `hidden_size` 映射到 `output_size`
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            '''new: ronghe11'''
            ## 如果启用了 BN，则使用批量归一化
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))

            modules.append(LIFAct(step=2))
            # modules.append(neuron.IFNode())
            '''new: ronghe12, ronghe11√'''
            ## 可选的 Dropout 层
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        """
        输入/输出维度：[batch_size, feature_dim] [批次，特征维度]
        """
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        """
        获取 SNN 分支的嵌入特征。
        该方法调用 `forward()`，返回模型的输出结果，适用于获取特征表示。
        """
        return self.forward(x)



class AVCA(nn.Module):
    """
    AVCA 模型：音频-视频跨模态匹配模型，使用 SNN 进行时间序列处理，并结合跨模态注意力融合音频和视频特征。
    """
    def __init__(self, params_model, input_size_audio, input_size_video):
        """
        - params_model: 字典，包含超参数（如隐藏层大小、dropout、学习率等）
        - input_size_audio: 音频输入特征的维度
        - input_size_video: 视频输入特征的维度
        """
        super(AVCA, self).__init__()

        print('Initializing model variables...', end='')
        # 读取超参数
        self.dim_out = params_model['dim_out']  # 输出嵌入维度
        # Number of classes
        self.hidden_size_encoder = params_model['encoder_hidden_size']  # 编码器隐藏层维度
        self.hidden_size_decoder = params_model['decoder_hidden_size']  # 解码器隐藏层维度
        self.r_enc = params_model['dropout_encoder']  # 编码器 dropout
        self.r_proj = params_model['dropout_decoder']  # 投影层 dropout
        self.depth_transformer = params_model['depth_transformer']  # Transformer 层数
        self.additional_triplets_loss = params_model['additional_triplets_loss']  # 是否使用额外的 triplet loss
        self.reg_loss = params_model['reg_loss']  # 是否使用正则化损失
        self.r_dec = params_model['additional_dropout']  # 额外 dropout
        self.momentum = params_model['momentum']  # BN 动量参数

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']

        self.T=params_model['T']
        self.scheduler=params_model['scheduler']
        self.eta_min=params_model['eta_min']
        self.epochs=params_model['epochs']




        print('Initializing trainable models...', end='')

        # **音频和视频编码器**（提取特征）
        self.A_enc = EmbeddingNet(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.V_enc = EmbeddingNet(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )


        # **跨模态注意力机制**（融合音频和视频）
        self.cross_attention = Transformer(300, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)
        self.cross_attention_SNN = Transformer(300, self.depth_transformer, 3, 100, 64, dropout=self.r_enc, attention=Attention_new_SDSA)

        # **投影层**（将特征映射到输出空间）
        self.W_proj= EmbeddingNet(
            input_size=300,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )
        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        # **SNN 分支**（音频和视频分别用 LIF 神经元处理）
        self.SNNbranchaudio = SNNBranch(input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True)
        self.SNNbranchvideo = SNNBranch(input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True)
        
        '''new: ronghe12, ronghe11√'''
        self.TCJA = TCJA(kernel_size_t=1, kernel_size_c=1, T=self.T, channel=300)

        # **投影层**
        self.A_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)
        self.V_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        # **重构模块**（尝试恢复原始特征）
        self.A_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)
        self.V_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        # **位置编码**（跨模态注意力使用）
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))
        self.pos_emb1D_t = torch.nn.Parameter(torch.randn(2, 300))
        
        
        # **优化器**
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.optimizer_gen = optim.Adam(list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.cross_attention.parameters()) + list(self.D.parameters()) +
                                        list(self.W_proj.parameters()) +
                                        list(self.SNNbranchaudio.parameters()) +  # 确保优化器能更新 LIFAct 里的参数
                                        list(self.SNNbranchvideo.parameters()) ,
                                        lr=self.lr, weight_decay=1e-5)

        # 选择学习率调度器
        if self.scheduler == "cosine":
            self.scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_gen, T_max=self.epochs, eta_min=self.eta_min)
        elif self.scheduler == "reduce":
            self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        print('Done')

        # 损失函数Loss function
        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        print('Done')

    def optimize_scheduler(self, value):
        """ 调整学习率 """
        self.scheduler_gen.step(value)

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
        """
        前向传播，计算音频、视频的嵌入，并进行跨模态匹配。
        - audio: 音频输入
        - image: 视频输入
        - negative_audio: 负样本（音频）
        - negative_image: 负样本（视频）
        - word_embedding: 词嵌入（用于跨模态匹配）
        - negative_word_embedding: 负样本（词嵌入）
        """
        # **提取音频和视频的初始特征**
        '''纯SNN'''
        # self.phi_a = self.A_enc(audio)
        # self.phi_v = self.V_enc(image)

        # **SNN 处理音频和视频**

        # phi_a1 = self.SNNbranchaudio(audio)
        # for t in range(1, self.T):
        #     phi_a1 += self.SNNbranchaudio(audio)
        # self.phi_a1 = phi_a1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_a1_list = []
        # phi_a1 = self.SNNbranchaudio(audio)  # 第一步计算
        # phi_a1_list.append(phi_a1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_a1_list.append(self.SNNbranchaudio(audio))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_a1_stacked = torch.stack(phi_a1_list, dim=0)  
        # phi_a1_stacked = self.TCJA(phi_a1_stacked) 
        # self.phi_a1 = torch.mean(phi_a1_stacked, dim=0)
        '''纯SNN'''
        phi_a_list = []
        phi_a = self.SNNbranchaudio(audio)  # 第一步计算
        phi_a_list.append(phi_a)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_a_list.append(self.SNNbranchaudio(audio))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_a_stacked = torch.stack(phi_a_list, dim=0)  
        phi_a_stacked = self.TCJA(phi_a_stacked) 
        self.phi_a = torch.mean(phi_a_stacked, dim=0)

        # phi_v1= self.SNNbranchvideo(image)
        # for t in range(1, self.T):
        #     phi_v1 += self.SNNbranchaudio(image) ?
        # self.phi_v1 = phi_v1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_v1_list = []
        # phi_v1 = self.SNNbranchvideo(image)  # 第一步计算
        # phi_v1_list.append(phi_v1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_v1_list.append(self.SNNbranchvideo(image))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_v1_stacked = torch.stack(phi_v1_list, dim=0)  
        # phi_v1_stacked = self.TCJA(phi_v1_stacked) 
        # self.phi_v1 = torch.mean(phi_v1_stacked, dim=0)
        '''纯SNN'''
        phi_v_list = []
        phi_v = self.SNNbranchvideo(image)  # 第一步计算
        phi_v_list.append(phi_v)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_v_list.append(self.SNNbranchvideo(image))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_v_stacked = torch.stack(phi_v_list, dim=0)  
        phi_v_stacked = self.TCJA(phi_v_stacked) 
        self.phi_v = torch.mean(phi_v_stacked, dim=0)
        
        # **跨模态注意力机制**

        '''new: ronghe4 ;5 没有启用'''
        # self.phi_a = 0.5*self.phi_a + 0.5*self.phi_a*nn.functional.softmax(self.phi_a1)

        '''纯SNN'''
        # self.phi_input = torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_a*nn.functional.softmax(self.phi_a1) + self.pos_emb1D[0, :]), dim=1)
        # self.phi_a= self.cross_attention(self.phi_input)[:, 0, :]   # 获取注意力后的音频特征
        self.phi_input = self.phi_a + self.pos_emb1D[0, :]
        self.phi_a= self.cross_attention_SNN(self.phi_input)[:, 0, :]   # 获取注意力后的音频特征

        '''new: ronghe4 ;5 没有启用'''
        # self.phi_v = 0.5*self.phi_v + 0.5*self.phi_v*nn.functional.softmax(self.phi_v1)

        '''纯SNN'''
        # self.phi_vinput = torch.stack((self.phi_v + self.pos_emb1D[1, :], self.phi_v*nn.functional.softmax(self.phi_v1) + self.pos_emb1D[1, :]), dim=1)
        # self.phi_v = self.cross_attention(self.phi_vinput)[:, 1, :] # 获取注意力后的视频特征
        self.phi_vinput = self.phi_v + self.pos_emb1D[1, :]
        self.phi_v = self.cross_attention_SNN(self.phi_vinput)[:, 1, :] # 获取注意力后的视频特征

        # '''new: 正样本 SNN 音视频联合表示'''
        # self.phi_SNNinput = torch.stack((self.phi_a1 + self.pos_emb1D[0, :], self.phi_v1 + self.pos_emb1D[1, :]), dim=1)
        # self.phi_SNN = self.cross_attention_SNN(self.phi_SNNinput)

        '''纯SNN'''
        # **编码负样本（音频和视频）**
        # self.phi_a_neg=self.A_enc(negative_audio)
        # self.phi_v_neg=self.V_enc(negative_image)

        # **SNN处理负样本（音频和视频）**

        # phi_a_neg1=self.SNNbranchaudio(negative_audio)
        # for t in range(1, self.T):
        #     phi_a_neg1 += self.SNNbranchaudio(negative_audio)
        # self.phi_a_neg1 = phi_a_neg1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_a_neg1_list = []
        # phi_a_neg1 = self.SNNbranchaudio(negative_audio)  # 第一步计算
        # phi_a_neg1_list.append(phi_a_neg1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_a_neg1_list.append(self.SNNbranchaudio(negative_audio))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_a_neg1_stacked = torch.stack(phi_a_neg1_list, dim=0)  
        # phi_a_neg1_stacked = self.TCJA(phi_a_neg1_stacked) 
        # self.phi_a_neg1 = torch.mean(phi_a_neg1_stacked, dim=0)
        '''纯SNN'''
        phi_a_neg_list = []
        phi_a_neg = self.SNNbranchaudio(negative_audio)  # 第一步计算
        phi_a_neg_list.append(phi_a_neg)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_a_neg_list.append(self.SNNbranchaudio(negative_audio))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_a_neg_stacked = torch.stack(phi_a_neg_list, dim=0)  
        phi_a_neg_stacked = self.TCJA(phi_a_neg_stacked) 
        self.phi_a_neg = torch.mean(phi_a_neg_stacked, dim=0)
        
        # phi_v_neg1=self.SNNbranchvideo(negative_image)
        # for t in range(1, self.T):
        #     phi_v_neg1 += self.SNNbranchaudio(negative_image)
        # self.phi_v_neg1 = phi_v_neg1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_v_neg1_list = []
        # phi_v_neg1 = self.SNNbranchvideo(negative_image)  # 第一步计算
        # phi_v_neg1_list.append(phi_v_neg1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_v_neg1_list.append(self.SNNbranchvideo(negative_image))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_v_neg1_stacked = torch.stack(phi_v_neg1_list, dim=0)  
        # phi_v_neg1_stacked = self.TCJA(phi_v_neg1_stacked) 
        # self.phi_v_neg1 = torch.mean(phi_v_neg1_stacked, dim=0)
        '''纯SNN'''
        phi_v_neg_list = []
        phi_v_neg = self.SNNbranchvideo(negative_image)  # 第一步计算
        phi_v_neg_list.append(phi_v_neg)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_v_neg_list.append(self.SNNbranchvideo(negative_image))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_v_neg_stacked = torch.stack(phi_v_neg_list, dim=0)  
        phi_v_neg_stacked = self.TCJA(phi_v_neg_stacked) 
        self.phi_v_neg = torch.mean(phi_v_neg_stacked, dim=0)

        # **负样本的跨模态注意力处理**
        '''new: ronghe4 ;5 没有启用'''
        # self.phi_a_neg = 0.5*self.phi_a_neg + 0.5*self.phi_a_neg*nn.functional.softmax(self.phi_a_neg1)

        '''纯SNN'''
        # self.phi_a_neg_input = torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_a_neg*nn.functional.softmax(self.phi_a_neg1) + self.pos_emb1D[0, :]), dim=1)
        # self.phi_a_neg= self.cross_attention(self.phi_a_neg_input)[:, 0, :]
        self.phi_a_neg_input = self.phi_a_neg + self.pos_emb1D[0, :]
        self.phi_a_neg= self.cross_attention_SNN(self.phi_a_neg_input)[:, 0, :]

        '''new: ronghe4 ;5 没有启用'''
        # self.phi_v_neg = 0.5*self.phi_v_neg + 0.5*self.phi_v_neg*nn.functional.softmax(self.phi_v_neg1)

        '''纯SNN'''
        # self.phi_v_neg_input = torch.stack((self.phi_v_neg + self.pos_emb1D[1, :], self.phi_v_neg*nn.functional.softmax(self.phi_v_neg1) + self.pos_emb1D[1, :]), dim=1)
        # self.phi_v_neg= self.cross_attention(self.phi_v_neg_input)[:, 1, :]
        self.phi_v_neg_input = self.phi_v_neg + self.pos_emb1D[1, :]
        self.phi_v_neg= self.cross_attention_SNN(self.phi_v_neg_input)[:, 1, :]

        # '''new: 负样本 SNN 音视频联合表示'''
        # self.phi_SNNinput_neg = torch.stack((self.phi_a_neg1 + self.pos_emb1D[0, :], self.phi_v_neg1 + self.pos_emb1D[1, :]), dim=1)
        # self.phi_SNN_neg = self.cross_attention_SNN(self.phi_SNNinput_neg)

        # **重置 SNN 分支**
        '''清除 SNN在当前批次中的状态，以便正确处理下一个批次的输入。'''
        functional.reset_net(self.SNNbranchvideo)
        functional.reset_net(self.SNNbranchaudio)

        # **计算词嵌入**
        self.w=word_embedding
        self.w_neg=negative_word_embedding

        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg=self.W_proj(negative_word_embedding)
        # functional.reset_net(self.W_proj)
        self.rho_w=self.D(self.theta_w)
        self.rho_w_neg=self.D(self.theta_w_neg)

        # **计算跨模态匹配输入**
        self.positive_input=torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1)
        self.negative_input=torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1)

        # '''new: ANN 与 SNN 的联合表示'''
        # self.positive_input = torch.stack((self.positive_input + self.pos_emb1D, self.positive_input*nn.functional.softmax(self.phi_SNN) + self.pos_emb1D), dim=1)
        # self.negative_input = torch.stack((self.negative_input + self.pos_emb1D, self.negative_input*nn.functional.softmax(self.phi_SNN_neg) + self.pos_emb1D), dim=1)
        # self.positive_input = self.positive_input.flatten(start_dim=1, end_dim=2)
        # self.negative_input = self.negative_input.flatten(start_dim=1, end_dim=2)
        
        '''纯SNN'''
        # **计算跨模态注意力（正样本和负样本）**
        # self.phi_attn= self.cross_attention(self.positive_input)
        # self.phi_attn_neg = self.cross_attention(self.negative_input)
        self.phi_attn= self.cross_attention_SNN(self.positive_input)
        self.phi_attn_neg = self.cross_attention_SNN(self.negative_input)

        # **计算最终的跨模态特征（正样本）**
        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn= self.phi_v + self.phi_attn[:, 1, :]

        # **计算最终的跨模态特征（负样本）**
        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]

        # **投影到最终空间**
        self.theta_v = self.V_proj(self.video_fe_attn)
        self.theta_v_neg=self.V_proj(self.video_fe_neg_attn)
        self.theta_a = self.A_proj(self.audio_fe_attn)
        self.theta_a_neg=self.A_proj(self.audio_fe_neg_attn)

        # **计算重构特征**
        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)
        self.se_em_hat1 = self.A_proj(self.phi_a_rec)
        self.se_em_hat2 = self.V_proj(self.phi_v_rec)

        # **计算最终的跨模态嵌入**
        self.rho_a=self.D(self.theta_a)
        self.rho_a_neg=self.D(self.theta_a_neg)
        self.rho_v=self.D(self.theta_v)
        self.rho_v_neg=self.D(self.theta_v_neg)
        # functional.reset_net(self.D)
        # functional.reset_net(self.V_rec)
        # functional.reset_net(self.A_rec)
        # functional.reset_net(self.A_proj)
        # functional.reset_net(self.V_proj)




    def backward(self, optimize):
        """
        反向传播计算损失，并在 optimize=True 时更新模型参数
        - optimize (bool): 是否执行梯度更新
        """
        # **1. 计算附加的三元组损失**
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

        # **2. 计算正则化损失**
        if self.reg_loss == True:
            l_r = (
                self.criterion_reg(self.phi_v_rec, self.phi_v) +  # 重建的视频嵌入与原始视频嵌入的差异
                self.criterion_reg(self.phi_a_rec, self.phi_a) +  # 重建的音频嵌入与原始音频嵌入的差异
                self.criterion_reg(self.theta_v, self.theta_w) +  # 视频投影与词嵌入投影的差异
                self.criterion_reg(self.theta_a, self.theta_w)    # 音频投影与词嵌入投影的差异
            )


        # **3. 计算重构损失**
        l_rec = (
            self.criterion_reg(self.w, self.rho_v) +  # 正样本词嵌入与视频的重建嵌入的差异
            self.criterion_reg(self.w, self.rho_a) +  # 正样本词嵌入与音频的重建嵌入的差异
            self.criterion_reg(self.w, self.rho_w)    # 正样本词嵌入与自身重建嵌入的差异
        )


        # **4. 计算跨模态三元组损失**
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


    def optimize_params(self, audio, video, cls_numeric, cls_embedding,audio_negative, video_negative, negative_cls_embedding,optimize=False):
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
        # print('audio size = {}'.format(audio.size()))
        # print('video size {}'.format(video.size()))
        # print('embedding size {}'.format(embedding.size()))
        # print("********************")
        # audio= torch.Size([256, 512])
        # video    torch.Size([256, 512])
        # embedding   torch.Size([256, 300])

        '''纯SNN'''
        # **1. 提取音频和视频的嵌入**
        # phi_a = self.A_enc(audio)
        # phi_v = self.V_enc(video)

        # **2. 计算 SNN 处理后的音频特征**
        # phi_a1 = self.SNNbranchaudio(audio)
        # for t in range(1, self.T):
        #     phi_a1 += self.SNNbranchaudio(audio)
        # phi_a1 = phi_a1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_a1_list = []
        # phi_a1 = self.SNNbranchaudio(audio)  # 第一步计算
        # phi_a1_list.append(phi_a1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_a1_list.append(self.SNNbranchaudio(audio))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_a1_stacked = torch.stack(phi_a1_list, dim=0)  
        # phi_a1_stacked = self.TCJA(phi_a1_stacked) 
        # phi_a1 = torch.mean(phi_a1_stacked, dim=0)
        '''纯SNN'''
        phi_a_list = []
        phi_a = self.SNNbranchaudio(audio)  # 第一步计算
        phi_a_list.append(phi_a)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_a_list.append(self.SNNbranchaudio(audio))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_a_stacked = torch.stack(phi_a_list, dim=0)  
        phi_a_stacked = self.TCJA(phi_a_stacked) 
        phi_a = torch.mean(phi_a_stacked, dim=0)

        # **3. 计算 SNN 处理后的视频特征**
        # phi_v1= self.SNNbranchvideo(video)
        # for t in range(1, self.T):
        #     phi_v1 += self.SNNbranchaudio(video)
        # phi_v1 = phi_v1/self.T
        # '''new: ronghe12, ronghe11√'''
        # phi_v1_list = []
        # phi_v1 = self.SNNbranchvideo(video)  # 第一步计算
        # phi_v1_list.append(phi_v1)
        # for t in range(1, self.T):  # 进行 T 个时间步累加
        #     phi_v1_list.append(self.SNNbranchvideo(video))
        # # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        # phi_v1_stacked = torch.stack(phi_v1_list, dim=0)  
        # phi_v1_stacked = self.TCJA(phi_v1_stacked) 
        # phi_v1 = torch.mean(phi_v1_stacked, dim=0)
        '''纯SNN'''
        phi_v_list = []
        phi_v = self.SNNbranchvideo(video)  # 第一步计算
        phi_v_list.append(phi_v)
        for t in range(1, self.T):  # 进行 T 个时间步累加
            phi_v_list.append(self.SNNbranchvideo(video))
        # 堆叠成一个新的张量，形状为 (T, batch_size, channel)
        phi_v_stacked = torch.stack(phi_v_list, dim=0)  
        phi_v_stacked = self.TCJA(phi_v_stacked) 
        phi_v = torch.mean(phi_v_stacked, dim=0)


        '''new: ronghe4 ;5 没有启用'''
        # phi_a = 0.5*phi_a + 0.5*phi_a*nn.functional.softmax(phi_a1)
        # phi_v = 0.5*phi_v + 0.5*phi_v*nn.functional.softmax(phi_v1)
        
        '''纯SNN'''
        # **4. 计算跨模态注意力（音频）**
        # phi_input = torch.stack((phi_a + self.pos_emb1D[0, :], phi_a*nn.functional.softmax(phi_a1) + self.pos_emb1D[0, :]), dim=1)
        # phi_a= self.cross_attention(phi_input)[:, 0, :]
        phi_input = phi_a + self.pos_emb1D[0, :]
        phi_a= self.cross_attention_SNN(phi_input)[:, 0, :]

        # self.phi_v = 0.5*self.phi_v + 0.5*self.phi_v*nn.functional.softmax(self.phi_v1)
        
        '''纯SNN'''
        # **5. 计算跨模态注意力（视频）**
        # phi_vinput = torch.stack((phi_v + self.pos_emb1D[1, :], phi_v*nn.functional.softmax(phi_v1) + self.pos_emb1D[1, :]), dim=1)
        # phi_v = self.cross_attention(phi_vinput)[:, 1, :]
        phi_vinput = phi_v + self.pos_emb1D[1, :]
        phi_v = self.cross_attention_SNN(phi_vinput)[:, 1, :]

        # '''new:  SNN 音视频联合表示'''
        # phi_SNNinput = torch.stack((phi_a1 + self.pos_emb1D[0, :], phi_v1 + self.pos_emb1D[1, :]), dim=1)
        # phi_SNN = self.cross_attention_SNN(phi_SNNinput)

        # **6. 重置 SNN 网络**
        functional.reset_net(self.SNNbranchvideo)
        functional.reset_net(self.SNNbranchaudio)
        
        # **7. 计算文本投影**
        theta_w=self.W_proj(embedding)

        # **8. 计算音视频跨模态融合**
        # input_concatenated = torch.stack((phi_a+self.pos_emb1D[0,:], phi_v+self.pos_emb1D[1,:]), dim=1)
        # print(f"维度phi_a shape = {phi_a.shape}")
        # print(f"维度phi_v shape = {phi_v.shape}")
        # print(f"维度phi_SNN[:,0,:] shape = {(phi_SNN[:,0,:]).shape}")
        # print(f"维度phi_SNN[:,1,:] shape = {(phi_SNN[:,1,:]).shape}")
        # print(f"维度self.pos_emb1D[0,:] shape = {(self.pos_emb1D[0,:]).shape}")
        # print(f"维度self.pos_emb1D[1,:] shape = {(self.pos_emb1D[1,:]).shape}")
        # print(f"维度phi_a+self.pos_emb1D[0,:] shape = {(phi_a+self.pos_emb1D[0,:]).shape}")
        # print(f"维度phi_v+self.pos_emb1D[1,:] shape = {(phi_v+self.pos_emb1D[1,:]).shape}")
        # print(f"维度phi_a*nn.functional.softmax(phi_SNN[:, 0, :]) shape = {(phi_a*nn.functional.softmax(phi_SNN[:, 0, :])).shape}")
        # print(f"维度phi_v*nn.functional.softmax(phi_SNN[:, 1, :]) shape = {(phi_v*nn.functional.softmax(phi_SNN[:, 1, :])).shape}")
        # print(f"维度phi_a*nn.functional.softmax(self.phi_SNN[:, 0, :])+self.pos_emb1D[0,:] shape = {(phi_a*nn.functional.softmax(self.phi_SNN[:, 0, :])+self.pos_emb1D[0,:]).shape}")
        # print(f"维度phi_v*nn.functional.softmax(self.phi_SNN[:, 1, :])+self.pos_emb1D[1,:] shape = {(phi_v*nn.functional.softmax(self.phi_SNN[:, 1, :])+self.pos_emb1D[1,:]).shape}")
        
        '''ronghe1'''
        # input_concatenated = torch.stack((phi_a, phi_v, 
        #                                   phi_a*nn.functional.softmax(phi_SNN[:, 0, :]), 
        #                                   phi_v*nn.functional.softmax(phi_SNN[:, 1, :]))
        #                                   , dim=1)

        '''ronghe2  √'''
        # input_concatenated = torch.stack((phi_a+self.pos_emb1D[0,:], 
        #                                   phi_v+self.pos_emb1D[1,:], 
        #                                   phi_a*nn.functional.softmax(phi_SNN[:, 0, :])+self.pos_emb1D[0,:], 
        #                                   phi_v*nn.functional.softmax(phi_SNN[:, 1, :])+self.pos_emb1D[1,:]), 
        #                                   dim=1)

        '''ronghe3'''
        # input_concatenated = torch.stack((phi_a+self.pos_emb1D[0,:], 
        #                                   phi_v+self.pos_emb1D[1,:], 
        #                                   phi_a*nn.functional.softmax(phi_SNN[:, 0, :]), 
        #                                   phi_v*nn.functional.softmax(phi_SNN[:, 1, :])), 
        #                                   dim=1)
        
        '''纯SNN'''
        input_concatenated = torch.stack((phi_a+self.pos_emb1D[0,:], phi_v+self.pos_emb1D[1,:]), dim=1)

        # input_concatenated = torch.stack((input_concatenated , input_concatenated*nn.functional.softmax(self.phi_SNN)), dim=1)
        # print(f"维度input_concatenated size = {input_concatenated.size()}")
        # input_concatenated = input_concatenated.flatten(start_dim=1, end_dim=2)
        phi_attn= self.cross_attention_SNN(input_concatenated)
        
        # **9. 计算最终的音频、视频嵌入**
        phi_a = phi_a + phi_attn[:,0,:]
        phi_v = phi_v + phi_attn[:,1,:]


        theta_v = self.V_proj(phi_v)
        theta_a = self.A_proj(phi_a)
        # functional.reset_net(self.A_proj)
        # functional.reset_net(self.V_proj)
        return theta_a, theta_v, theta_w

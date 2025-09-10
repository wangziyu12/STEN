#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    """训练监控器，用于监控训练过程中的关键指标"""
    
    def __init__(self, exp_name, log_dir='./logs'):
        """
        初始化训练监控器
        :param exp_name: 实验名称
        :param log_dir: 日志保存目录
        """
        self.exp_name = exp_name
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 创建实验专属目录
        self.exp_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 初始化TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        
        # 初始化存储指标的字典
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'seen_acc': [],
            'unseen_acc': [],
            'gzsl': [],
            'zsl': [],
            'learning_rates': []
        }
        
        # 初始化最佳模型相关信息
        self.best_gzsl = 0
        self.best_epoch = 0
        self.best_model_path = None
        
        print(f"训练监控器初始化完成，日志将保存在: {self.exp_dir}")
    
    def log_train_metrics(self, epoch, train_loss, learning_rate):
        """
        记录训练指标
        :param epoch: 当前训练轮次
        :param train_loss: 训练损失
        :param learning_rate: 当前学习率
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['learning_rates'].append(learning_rate)
        
        # 写入TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('LearningRate', learning_rate, epoch)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, LR: {learning_rate:.6f}")
    
    def log_val_metrics(self, epoch, val_loss, seen_acc, unseen_acc, gzsl, zsl):
        """
        记录验证指标
        :param epoch: 当前训练轮次
        :param val_loss: 验证损失
        :param seen_acc: 已知类别准确率
        :param unseen_acc: 未知类别准确率
        :param gzsl: 广义零样本学习指标
        :param zsl: 零样本学习指标
        """
        self.metrics['val_loss'].append(val_loss)
        self.metrics['seen_acc'].append(seen_acc)
        self.metrics['unseen_acc'].append(unseen_acc)
        self.metrics['gzsl'].append(gzsl)
        self.metrics['zsl'].append(zsl)
        
        # 写入TensorBoard
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Seen', seen_acc, epoch)
        self.writer.add_scalar('Accuracy/Unseen', unseen_acc, epoch)
        self.writer.add_scalar('Performance/GZSL', gzsl, epoch)
        self.writer.add_scalar('Performance/ZSL', zsl, epoch)
        
        print(f"Validation: Loss: {val_loss:.4f}, Seen: {seen_acc:.2f}%, Unseen: {unseen_acc:.2f}%, GZSL: {gzsl:.2f}%, ZSL: {zsl:.2f}%")
        
        # 检查是否是最佳模型
        if gzsl > self.best_gzsl:
            self.best_gzsl = gzsl
            self.best_epoch = epoch
            print(f"New best model at epoch {epoch} with GZSL: {gzsl:.2f}%")
            return True
        return False
    
    def save_learning_curves(self):
        """保存学习曲线图"""
        # 创建图表目录
        plots_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 绘制训练/验证损失曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss')
        if self.metrics['val_loss']:
            plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
        plt.close()
        
        # 绘制性能指标曲线
        if self.metrics['gzsl']:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.metrics['gzsl']) + 1)
            plt.plot(epochs, self.metrics['seen_acc'], 'g-', label='Seen Acc')
            plt.plot(epochs, self.metrics['unseen_acc'], 'b-', label='Unseen Acc')
            plt.plot(epochs, self.metrics['gzsl'], 'r-', label='GZSL')
            plt.plot(epochs, self.metrics['zsl'], 'y-', label='ZSL')
            plt.title('Performance Metrics')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'performance_curve.png'))
            plt.close()
            
        # 绘制学习率曲线
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics['learning_rates']) + 1)
        plt.plot(epochs, self.metrics['learning_rates'], 'g-')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'lr_curve.png'))
        plt.close()
        
        print(f"学习曲线已保存至: {plots_dir}")
    
    def save_metrics_to_file(self):
        """将指标保存到文件"""
        metrics_file = os.path.join(self.exp_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Best GZSL: {self.best_gzsl:.2f}% at epoch {self.best_epoch}\n")
            f.write("\nAll Metrics:\n")
            f.write("Epoch\tTrain Loss\tVal Loss\tSeen Acc\tUnseen Acc\tGZSL\tZSL\tLR\n")
            
            for i in range(len(self.metrics['train_loss'])):
                epoch = i + 1
                train_loss = self.metrics['train_loss'][i] if i < len(self.metrics['train_loss']) else '-'
                val_loss = self.metrics['val_loss'][i] if i < len(self.metrics['val_loss']) else '-'
                seen_acc = self.metrics['seen_acc'][i] if i < len(self.metrics['seen_acc']) else '-'
                unseen_acc = self.metrics['unseen_acc'][i] if i < len(self.metrics['unseen_acc']) else '-'
                gzsl = self.metrics['gzsl'][i] if i < len(self.metrics['gzsl']) else '-'
                zsl = self.metrics['zsl'][i] if i < len(self.metrics['zsl']) else '-'
                lr = self.metrics['learning_rates'][i] if i < len(self.metrics['learning_rates']) else '-'
                
                f.write(f"{epoch}\t{train_loss}\t{val_loss}\t{seen_acc}\t{unseen_acc}\t{gzsl}\t{zsl}\t{lr}\n")
        
        print(f"指标已保存至: {metrics_file}")
    
    def close(self):
        """关闭TensorBoard writer并保存最终结果"""
        self.save_learning_curves()
        self.save_metrics_to_file()
        self.writer.close()
        print("训练监控器已关闭，所有指标已保存")
        
    def __del__(self):
        """析构函数，确保writer被关闭"""
        try:
            self.writer.close()
        except:
            pass

def update_main_script_for_monitoring():
    """
    提供的示例代码说明如何在main.py中集成训练监控器
    """
    # 这里不是实际修改main.py的代码，而是提供一个示例说明如何修改
    
    # 1. 在main.py的导入部分添加:
    # from training_monitor import TrainingMonitor
    
    # 2. 在main函数中初始化监控器:
    # monitor = TrainingMonitor(exp_name=args.exp_name)
    
    # 3. 在训练循环中每个epoch结束后记录训练指标:
    # current_lr = model.optimizer_gen.param_groups[0]['lr']
    # monitor.log_train_metrics(epoch, loss_train, current_lr)
    
    # 4. 在验证步骤后记录验证指标:
    # is_best = monitor.log_val_metrics(epoch, loss_val, seen_acc, unseen_acc, gzsl, zsl)
    # if is_best and args.save_checkpoints:
    #     save_checkpoint(model, epoch, optimizer, args.exp_name)
    
    # 5. 在训练结束后关闭监控器:
    # monitor.close()
    
    pass

def main():
    """示例使用训练监控器的主函数"""
    parser = argparse.ArgumentParser(description="训练监控器示例")
    parser.add_argument('--exp_name', type=str, default='test_monitor', help='实验名称')
    args = parser.parse_args()
    
    # 初始化监控器
    monitor = TrainingMonitor(exp_name=args.exp_name)
    
    # 模拟训练过程
    for epoch in range(1, 11):
        # 模拟训练损失下降
        train_loss = 1.0 / (epoch + 1)
        # 模拟学习率衰减
        lr = 0.001 * (0.9 ** epoch)
        
        # 记录训练指标
        monitor.log_train_metrics(epoch, train_loss, lr)
        
        # 模拟验证结果
        val_loss = train_loss * 1.2
        seen_acc = 50 + epoch * 2
        unseen_acc = 10 + epoch * 1.5
        gzsl = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
        zsl = unseen_acc
        
        # 记录验证指标
        is_best = monitor.log_val_metrics(epoch, val_loss, seen_acc, unseen_acc, gzsl, zsl)
        
        if is_best:
            print(f"保存最佳模型 (epoch {epoch})")
    
    # 关闭监控器并保存结果
    monitor.close()

if __name__ == "__main__":
    main()

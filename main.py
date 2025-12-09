#!/usr/bin/env python3
"""
RGB2HSI 完整训练脚本
支持从命令行配置运行参数
"""

import argparse
import yaml
import sys
from pathlib import Path

import torch
from src.trainers.trainer import Trainer


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RGB2HSI MultimodalCLIP 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认配置训练
  python main.py
  
  # 使用自定义配置文件
  python main.py --config configs/custom.yaml
  
  # 覆盖配置参数
  python main.py --batch-size 64 --epochs 300 --lr 5e-5
  
  # 从检查点恢复训练
  python main.py --resume checkpoints/epoch_50.pth
        '''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='覆盖配置的batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='覆盖配置的训练轮数'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='覆盖配置的学习率'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='从检查点恢复训练'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='使用的设备'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        help='数据加载的工作进程数'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='启用混合精度训练'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        help='梯度累积步数'
    )
    
    return parser.parse_args()


def merge_config_with_args(config, args):
    """将命令行参数合并到配置中"""
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.lr:
        config['training']['lr'] = args.lr
    
    if args.num_workers:
        config['training']['num_workers'] = args.num_workers
    
    if args.mixed_precision:
        config['training']['mixed_precision'] = True
    
    if args.gradient_accumulation_steps:
        config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 检查GPU可用性
    print("=" * 80)
    print("RGB2HSI MultimodalCLIP 训练")
    print("=" * 80)
    
    print(f"\n设备检查:")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
    print()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"加载配置: {config_path}")
    config = load_config(config_path)
    
    # 合并命令行参数
    config = merge_config_with_args(config, args)
    
    # 打印配置摘要
    print(f"\n训练配置:")
    print(f"  模型: {config['project_name']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  学习率: {config['training']['lr']}")
    print(f"  总轮数: {config['training']['epochs']}")
    print(f"  梯度累积: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"  混合精度: {config['training'].get('mixed_precision', False)}")
    print(f"\n模型架构:")
    print(f"  RGB编码器: {config['model']['rgb_encoder']['name']}")
    print(f"  HSI编码器: {config['model']['hsi_encoder']['name']}")
    print(f"  融合模块: {config['model']['fusion_module']['name']}")
    print(f"  输出通道数: {config['model']['decoder']['num_bands']}")
    print()
    
    # 创建训练器
    print("初始化训练器...")
    trainer = Trainer(config)
    
    # 从检查点恢复（如果提供）
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"警告: 检查点文件不存在: {args.resume}")
    
    # 开始训练
    print("\n开始训练...")
    print("=" * 80)
    
    try:
        trainer.train()
        print("\n训练成功完成！")
        print(f"最终模型已保存到: {config['paths']['save_dir']}")
        
    except KeyboardInterrupt:
        print("\n\n训练被中断!")
        print(f"最后的检查点已保存到: {config['paths']['save_dir']}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n错误发生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

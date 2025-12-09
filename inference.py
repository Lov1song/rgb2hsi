#!/usr/bin/env python3
"""
RGB2HSI 推理脚本
用于对新的RGB图像进行HSI预测
"""

import argparse
import yaml
from pathlib import Path
import numpy as np

import torch
from src.models.rgb2hsi_model import RGB2HSIModel
from src.utils.inference import RGB2HSIInference


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RGB2HSI 推理脚本'
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='输入RGB图像路径'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/model_final.pth',
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/output.npy',
        help='输出HSI文件路径'
    )
    
    parser.add_argument(
        '--save-visual',
        action='store_true',
        help='保存可视化结果'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='使用的设备'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='批量推理：输入目录路径'
    )
    
    return parser.parse_args()


def infer_single(image_path, inferencer, output_path):
    """单张图像推理"""
    print(f"\n推理: {image_path}")
    
    try:
        # 执行推理
        pred_hsi, rgb_proj = inferencer.inference(image_path)
        
        print(f"  预测HSI形状: {pred_hsi.shape}")
        print(f"  RGB投影维度: {rgb_proj.shape}")
        
        # 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        inferencer.save_hsi(pred_hsi, str(output_path))
        
        # 统计信息
        print(f"  值范围: [{pred_hsi.min():.4f}, {pred_hsi.max():.4f}]")
        print(f"  平均值: {pred_hsi.mean():.4f}, 标准差: {pred_hsi.std():.4f}")
        
        return pred_hsi, rgb_proj
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def infer_batch(input_dir, inferencer, output_dir):
    """批量推理"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        
        # 生成输出路径
        output_path = output_dir / f"{image_path.stem}.npy"
        
        # 推理
        pred_hsi, rgb_proj = infer_single(str(image_path), inferencer, str(output_path))
        
        if pred_hsi is not None:
            results.append({
                'image': image_path.name,
                'hsi_shape': pred_hsi.shape,
                'output': str(output_path),
            })
    
    print(f"\n完成! 处理了 {len(results)} 张图像")
    return results


def save_visual_example(pred_hsi, output_path):
    """保存可视化示例（RGB伪彩合成）"""
    # 选择三个通道作为RGB进行可视化
    try:
        from PIL import Image
        
        # 选择通道 (例如 10, 20, 30)
        if pred_hsi.shape[2] >= 31:
            r = pred_hsi[..., 10]
            g = pred_hsi[..., 20]
            b = pred_hsi[..., 30]
        else:
            r = pred_hsi[..., 0] if pred_hsi.shape[2] > 0 else pred_hsi[..., 0]
            g = pred_hsi[..., min(1, pred_hsi.shape[2]-1)] 
            b = pred_hsi[..., min(2, pred_hsi.shape[2]-1)]
        
        # 正规化到 [0, 255]
        r = ((r - r.min()) / (r.max() - r.min() + 1e-10) * 255).astype(np.uint8)
        g = ((g - g.min()) / (g.max() - g.min() + 1e-10) * 255).astype(np.uint8)
        b = ((b - b.min()) / (b.max() - b.min() + 1e-10) * 255).astype(np.uint8)
        
        # 合成RGB图像
        rgb_visual = np.stack([r, g, b], axis=2)
        img = Image.fromarray(rgb_visual)
        
        output_path = Path(output_path).with_suffix('.png')
        img.save(str(output_path))
        print(f"✓ 可视化已保存: {output_path}")
        
    except Exception as e:
        print(f"可视化保存失败: {e}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("RGB2HSI 推理")
    print("=" * 80)
    
    # 检查文件存在性
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    # 加载配置
    print(f"\n加载配置: {config_path}")
    config = load_config(config_path)
    
    # 初始化模型
    print("初始化模型...")
    device = torch.device(args.device)
    print(f"设备: {device}")
    
    model = RGB2HSIModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载: {checkpoint_path}")
    
    # 创建推理器
    inferencer = RGB2HSIInference(model, config, device=device)
    
    # 推理
    if args.batch:
        # 批量推理
        results = infer_batch(args.batch, inferencer, args.output)
        print(f"\n结果已保存到: {args.output}")
    else:
        # 单张推理
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"错误: 图像文件不存在: {image_path}")
            return
        
        pred_hsi, rgb_proj = infer_single(
            str(image_path),
            inferencer,
            args.output
        )
        
        if pred_hsi is not None:
            print(f"\n✓ 推理完成!")
            print(f"  输出: {args.output}")
            
            if args.save_visual:
                save_visual_example(pred_hsi, args.output)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

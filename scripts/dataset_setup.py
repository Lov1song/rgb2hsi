#!/usr/bin/env python3
"""
数据集初始化和验证工具
"""

import os
from pathlib import Path


def create_dataset_structure(root_dir):
    """创建数据集目录结构"""
    root = Path(root_dir)
    
    # 创建目录
    directories = [
        'data/real_pair/rgb',
        'data/real_pair/hsi',
        'data/synthetic_pair/rgb',
        'data/synthetic_pair/hsi',
        'data/unpaired_rgb',
        'data/unpaired_hsi',
        'checkpoints',
        'logs',
        'results',
        'experiments',
    ]
    
    for dir_path in directories:
        (root / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    print("\n数据集目录结构创建完成!")
    print("\n请按照以下结构放置数据:")
    print("""
data/
├── real_pair/
│   ├── rgb/          (放置RGB图像: .jpg, .png)
│   └── hsi/          (放置HSI数据: .npy, shape=[H, W, 31])
├── synthetic_pair/
│   ├── rgb/
│   └── hsi/
├── unpaired_rgb/     (只有RGB图像)
└── unpaired_hsi/     (只有HSI数据)
    """)


def verify_dataset(data_dir):
    """验证数据集完整性"""
    data_path = Path(data_dir)
    
    print("\n数据集验证:")
    print("=" * 60)
    
    dataset_types = {
        'real_pair': 'data/real_pair',
        'synthetic_pair': 'data/synthetic_pair',
        'unpaired_rgb': 'data/unpaired_rgb',
        'unpaired_hsi': 'data/unpaired_hsi',
    }
    
    total_pairs = 0
    total_unpaired_rgb = 0
    total_unpaired_hsi = 0
    
    for ds_name, ds_path in dataset_types.items():
        full_path = data_path / ds_path
        
        if not full_path.exists():
            print(f"⚠ {ds_name}: 目录不存在")
            continue
        
        if 'pair' in ds_name:
            rgb_dir = full_path / 'rgb'
            hsi_dir = full_path / 'hsi'
            
            rgb_files = len(list(rgb_dir.glob('*'))) if rgb_dir.exists() else 0
            hsi_files = len(list(hsi_dir.glob('*'))) if hsi_dir.exists() else 0
            
            print(f"✓ {ds_name}:")
            print(f"    RGB文件: {rgb_files}")
            print(f"    HSI文件: {hsi_files}")
            
            if rgb_files != hsi_files:
                print(f"    ⚠ 警告: RGB和HSI文件数量不匹配!")
            
            total_pairs += rgb_files
        
        elif 'unpaired_rgb' in ds_name:
            rgb_files = len(list(full_path.glob('*')))
            print(f"✓ unpaired_rgb: {rgb_files} 文件")
            total_unpaired_rgb = rgb_files
        
        elif 'unpaired_hsi' in ds_name:
            hsi_files = len(list(full_path.glob('*')))
            print(f"✓ unpaired_hsi: {hsi_files} 文件")
            total_unpaired_hsi = hsi_files
    
    print("\n" + "=" * 60)
    print(f"总计:")
    print(f"  成对数据: {total_pairs}")
    print(f"  无配对RGB: {total_unpaired_rgb}")
    print(f"  无配对HSI: {total_unpaired_hsi}")
    
    if total_pairs == 0:
        print("\n⚠ 警告: 没有找到成对数据! 请确保数据已正确放置。")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集初始化工具')
    parser.add_argument('--create', action='store_true', help='创建数据集目录结构')
    parser.add_argument('--verify', type=str, help='验证数据集完整性')
    
    args = parser.parse_args()
    
    if args.create:
        create_dataset_structure('.')
    elif args.verify:
        verify_dataset(args.verify)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

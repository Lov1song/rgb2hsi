# src/datasets/pair_datasets.py
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from pathlib import Path


class PairedRGBHSIDataset(Dataset):
    """
    成对RGB-HSI数据集加载器
    
    目录结构:
        root/
        ├── rgb/       (RGB图像)
        └── hsi/       (HSI .npy文件)
    """
    def __init__(self, root, transform=None, valid_extensions=None):
        self.root = Path(root)
        self.transform = transform
        
        # 支持的图像扩展名
        if valid_extensions is None:
            self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.npy'}
        else:
            self.valid_extensions = valid_extensions
        
        # 加载RGB文件列表
        rgb_dir = self.root / "rgb"
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB目录不存在: {rgb_dir}")
        
        self.rgb_files = sorted([
            f for f in rgb_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        ])
        
        # 加载HSI文件列表
        hsi_dir = self.root / "hsi"
        if not hsi_dir.exists():
            raise FileNotFoundError(f"HSI目录不存在: {hsi_dir}")
        
        self.hsi_files = sorted([
            f for f in hsi_dir.iterdir()
            if f.suffix.lower() == '.npy'
        ])
        
        # 检查配对完整性
        if len(self.rgb_files) == 0:
            raise ValueError(f"找不到RGB图像文件: {rgb_dir}")
        
        if len(self.hsi_files) == 0:
            raise ValueError(f"找不到HSI数据文件: {hsi_dir}")
        
        if len(self.rgb_files) != len(self.hsi_files):
            print(f"警告: RGB文件数 ({len(self.rgb_files)}) != HSI文件数 ({len(self.hsi_files)})")
            # 取最小值
            min_len = min(len(self.rgb_files), len(self.hsi_files))
            self.rgb_files = self.rgb_files[:min_len]
            self.hsi_files = self.hsi_files[:min_len]
        
        print(f"加载数据集: {self.root}")
        print(f"  RGB文件: {len(self.rgb_files)}")
        print(f"  HSI文件: {len(self.hsi_files)}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        try:
            # 加载RGB
            rgb_path = self.rgb_files[idx]
            rgb = Image.open(rgb_path).convert("RGB")
            rgb = np.array(rgb, dtype=np.float32) / 255.0
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

            # 加载HSI
            hsi_path = self.hsi_files[idx]
            hsi = np.load(hsi_path)  # [H, W, C]
            
            # 确保HSI是float32
            hsi = hsi.astype(np.float32)
            hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()
            
            # 应用变换
            if self.transform:
                rgb, hsi = self.transform(rgb, hsi)
            
            return rgb, hsi
            
        except Exception as e:
            print(f"加载数据失败 [{idx}]: {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self))


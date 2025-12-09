# src/datasets/unpaired_hsi.py
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path


class UnpairedHSIDataset(Dataset):
    """
    无配对HSI数据集加载器
    用于光谱先验学习
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # 加载所有.npy文件
        self.files = sorted([
            f for f in self.root.iterdir()
            if f.suffix.lower() == '.npy'
        ])
        
        if len(self.files) == 0:
            print(f"警告: 在 {self.root} 中找不到.npy文件")
        else:
            print(f"加载无配对HSI数据: {len(self.files)} 个样本")

    def __len__(self):
        return len(self.files) if len(self.files) > 0 else 1

    def __getitem__(self, idx):
        try:
            if len(self.files) == 0:
                # 返回空张量作为占位符
                return torch.zeros(31, 256, 256)
            
            hsi_path = self.files[idx % len(self.files)]
            hsi = np.load(hsi_path)  # [H, W, C]
            
            # 确保形状正确
            if hsi.ndim == 2:  # [H, W] 灰度图
                hsi = np.stack([hsi] * 31, axis=2)  # 复制31份
            elif hsi.ndim == 3 and hsi.shape[2] != 31:
                # 调整通道数
                if hsi.shape[2] < 31:
                    # 重复通道
                    hsi = np.tile(hsi, (1, 1, int(np.ceil(31 / hsi.shape[2]))))[:, :, :31]
                else:
                    # 随机选择31个通道
                    indices = np.linspace(0, hsi.shape[2]-1, 31, dtype=int)
                    hsi = hsi[:, :, indices]
            
            hsi = hsi.astype(np.float32)
            hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()
            
            if self.transform:
                hsi = self.transform(hsi)
            
            return hsi
            
        except Exception as e:
            print(f"加载HSI失败 [{idx}]: {e}")
            return torch.zeros(31, 256, 256)


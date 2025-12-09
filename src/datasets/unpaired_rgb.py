# src/datasets/unpaired_rgb.py
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from pathlib import Path


class UnpairedRGBDataset(Dataset):
    """
    无配对RGB数据集加载器
    用于自监督学习
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 加载所有RGB文件
        self.files = sorted([
            f for f in self.root.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if len(self.files) == 0:
            print(f"警告: 在 {self.root} 中找不到图像文件")
        else:
            print(f"加载无配对RGB数据: {len(self.files)} 张图像")

    def __len__(self):
        return len(self.files) if len(self.files) > 0 else 1

    def __getitem__(self, idx):
        try:
            if len(self.files) == 0:
                # 返回空张量作为占位符
                return torch.zeros(3, 256, 256)
            
            img_path = self.files[idx % len(self.files)]
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            
            if self.transform:
                img = self.transform(img)
            
            return img
            
        except Exception as e:
            print(f"加载RGB失败 [{idx}]: {e}")
            return torch.zeros(3, 256, 256)


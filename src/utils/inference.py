# src/utils/inference.py
"""
推理模块 - 用于单个RGB图像的HSI重建
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path


class RGB2HSIInference:
    """
    RGB到HSI的推理封装
    """
    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config = config
        
    def preprocess_rgb(self, rgb_path_or_array):
        """
        预处理RGB图像
        
        参数:
            rgb_path_or_array: 图像路径或numpy数组
            
        返回:
            tensor [1, 3, H, W]
        """
        if isinstance(rgb_path_or_array, str):
            rgb = Image.open(rgb_path_or_array).convert("RGB")
            rgb = np.array(rgb) / 255.0
        else:
            rgb = rgb_path_or_array.astype(np.float32) / 255.0
        
        # 转为张量
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        rgb_tensor = rgb_tensor.to(self.device)
        
        return rgb_tensor
    
    def inference(self, rgb_path_or_array):
        """
        执行推理
        
        返回:
            pred_hsi: [H, W, 31] numpy数组
            rgb_proj: RGB投影特征
        """
        rgb_tensor = self.preprocess_rgb(rgb_path_or_array)
        
        with torch.no_grad():
            # 提取嵌入
            rgb_feat, rgb_proj = self.model.encode_rgb(rgb_tensor)  # [1, D], [1, E]
            
            # 融合和解码
            fused = self.model.fusion(rgb_feat, hsi_feat=None)  # [1, D]
            pred_hsi = self.model.decoder(fused)  # [1, 31, H, W]
        
        # 后处理
        pred_hsi = pred_hsi.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 31]
        rgb_proj = rgb_proj.squeeze(0).cpu().numpy()  # [E]
        
        return pred_hsi, rgb_proj
    
    def batch_inference(self, rgb_paths_or_arrays):
        """
        批量推理
        """
        results = []
        for rgb in rgb_paths_or_arrays:
            pred_hsi, rgb_proj = self.inference(rgb)
            results.append({
                'hsi': pred_hsi,
                'embedding': rgb_proj,
            })
        return results
    
    def save_hsi(self, pred_hsi, save_path):
        """保存HSI为.npy"""
        np.save(save_path, pred_hsi.astype(np.float32))
        print(f"✓ HSI 已保存: {save_path}")

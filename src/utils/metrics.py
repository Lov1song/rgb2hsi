# src/utils/metrics.py
"""
评估指标模块
"""

import numpy as np
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter


def mse(pred, gt):
    """均方误差"""
    return mean_squared_error(pred, gt)


def psnr(pred, gt, data_range=1.0):
    """峰值信噪比"""
    mse_val = mean_squared_error(pred, gt)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse_val)


def ssim(pred, gt, data_range=1.0):
    """结构相似度"""
    if pred.ndim == 3:  # 多通道
        return structural_similarity(pred, gt, channel_axis=2, data_range=data_range)
    else:
        return structural_similarity(pred, gt, data_range=data_range)


def spectral_angle_mapper(pred, gt):
    """
    光谱角制图 (SAM)
    计算预测和真实HSI在每个像素处的光谱向量夹角
    """
    # 确保是numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    # 形状: [H, W, C]
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    
    # 展平空间维度
    pred_flat = pred.reshape(-1, pred.shape[-1])  # [N, C]
    gt_flat = gt.reshape(-1, gt.shape[-1])  # [N, C]
    
    # 计算角度
    dots = np.sum(pred_flat * gt_flat, axis=1)
    norms_pred = np.linalg.norm(pred_flat, axis=1)
    norms_gt = np.linalg.norm(gt_flat, axis=1)
    
    # 避免除零
    norms_prod = norms_pred * norms_gt
    norms_prod[norms_prod == 0] = 1e-10
    
    cos_angles = dots / norms_prod
    # 夹角范围[-1, 1]，需要clip
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    
    # 转为角度制并计算平均
    sam = np.mean(angles) * 180 / np.pi
    return sam


def ergas(pred, gt):
    """
    相对无维数误差总和 (ERGAS)
    用于评估多光谱图像重建质量
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    
    # 按通道计算
    sum_squared_error = 0
    sum_gt_mean_squared = 0
    
    for i in range(pred.shape[-1]):
        mse_i = np.mean((pred[..., i] - gt[..., i]) ** 2)
        mean_gt_i = np.mean(gt[..., i])
        
        sum_squared_error += mse_i / (mean_gt_i ** 2 + 1e-10)
        sum_gt_mean_squared += 1
    
    ergas_val = 100 * np.sqrt(sum_squared_error / sum_gt_mean_squared)
    return ergas_val


class MetricsCalculator:
    """度量计算器"""
    def __init__(self, metrics=None):
        if metrics is None:
            self.metrics = ['mse', 'psnr', 'ssim', 'sam']
        else:
            self.metrics = metrics
    
    def compute(self, pred, gt):
        """
        计算所有指标
        
        参数:
            pred: [H, W, C] 预测或 [B, C, H, W]
            gt: [H, W, C] 真实或 [B, C, H, W]
        
        返回:
            dict 包含各项指标
        """
        # 处理batch维度
        if pred.ndim == 4:  # [B, C, H, W]
            pred = pred.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
            gt = gt.permute(0, 2, 3, 1).cpu().numpy()
            
            # 只取第一个样本
            pred = pred[0]
            gt = gt[0]
        
        results = {}
        
        if 'mse' in self.metrics:
            results['mse'] = mse(pred, gt)
        
        if 'psnr' in self.metrics:
            results['psnr'] = psnr(pred, gt)
        
        if 'ssim' in self.metrics:
            results['ssim'] = ssim(pred, gt)
        
        if 'sam' in self.metrics or 'spectral_angle_mapper' in self.metrics:
            results['sam'] = spectral_angle_mapper(pred, gt)
        
        if 'ergas' in self.metrics:
            results['ergas'] = ergas(pred, gt)
        
        return results

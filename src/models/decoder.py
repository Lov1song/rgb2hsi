# src/models/decoder.py
import torch.nn as nn

class SpectrallDecodeBlock(nn.Module):
    """光谱解码块"""
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        
        layers.extend([
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpectralDecoder(nn.Module):
    """
    改进的高光谱解码器
    从融合特征重建31通道的HSI
    """
    def __init__(self, config):
        super().__init__()
        
        num_bands = config.get("num_bands", 31)
        hidden_dim = config.get("hidden_dim", 256)
        
        # 特征维度扩展
        self.expand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
        )
        
        # 空间维度恢复 (1,1) -> (H/4, W/4)
        self.spatial_expand = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 光谱恢复 - 渐进式上采样
        self.decoder = nn.Sequential(
            SpectrallDecodeBlock(hidden_dim, hidden_dim // 2, upsample=True),
            SpectrallDecodeBlock(hidden_dim // 2, hidden_dim // 4, upsample=True),
            nn.Conv2d(hidden_dim // 4, num_bands, 1),
        )
        
        self.num_bands = num_bands

    def forward(self, x):
        """
        x: [B, D] - 融合特征向量
        返回: [B, num_bands, H, W] - 重建的HSI
        """
        B = x.shape[0]
        
        # 展开特征
        feat = self.expand(x)  # [B, D*4]
        feat = feat.view(B, -1, 1, 1)  # [B, D*4, 1, 1]
        
        # 恢复空间维度
        feat = self.spatial_expand(feat)  # [B, D, H/4, W/4]
        
        # 解码为HSI
        out = self.decoder(feat)  # [B, num_bands, H, W]
        
        return out

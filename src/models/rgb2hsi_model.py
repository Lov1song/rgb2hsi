# src/models/rgb2hsi_model.py

import torch
import torch.nn as nn
from src.models.rgb_encoder import RGBEncoder
from src.models.hsi_encoder import SpectralUNet
from src.modules.fusion import CrossAttentionFusionModule
from src.models.decoder import SpectralDecoder


class ProjectionHead(nn.Module):
    """
    投影头 - 将特征映射到共享的嵌入空间
    类似CLIP的做法
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        # L2正则化
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x


class RGB2HSIModel(nn.Module):
    """
    多模态RGB2HSI模型 - CLIP风格
    
    支持三种工作模式：
    1. 有监督学习: RGB -> HSI (成对数据)
    2. 自监督学习: 利用对比学习约束多模态空间
    3. 重建学习: 重建高光谱图像的细节
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"].get("embedding_dim", 256)
        
        # ========== 编码器 ==========
        # RGB编码器 - 提取RGB特征
        rgb_cfg = config["model"]["rgb_encoder"]
        self.rgb_encoder = RGBEncoder(rgb_cfg)
        
        # HSI编码器 - 提取HSI特征
        hsi_cfg = config["model"]["hsi_encoder"]
        self.hsi_encoder = SpectralUNet(hsi_cfg)
        
        # ========== 投影头 (CLIP-style) ==========
        proj_cfg = config["model"]["projection_head"]
        hidden_dim = rgb_cfg.get("hidden_dim", 512)
        
        self.rgb_proj = ProjectionHead(
            hidden_dim,
            proj_cfg.get("hidden_dim", 512),
            proj_cfg.get("output_dim", 256),
            proj_cfg.get("dropout", 0.1)
        )
        
        self.hsi_proj = ProjectionHead(
            hidden_dim,
            proj_cfg.get("hidden_dim", 512),
            proj_cfg.get("output_dim", 256),
            proj_cfg.get("dropout", 0.1)
        )
        
        # ========== 融合模块 ==========
        fusion_cfg = config["model"]["fusion_module"]
        fusion_cfg["dim"] = hidden_dim
        self.fusion = CrossAttentionFusionModule(fusion_cfg)
        
        # ========== 解码器 ==========
        decoder_cfg = config["model"]["decoder"]
        self.decoder = SpectralDecoder(decoder_cfg)
        
        self.hidden_dim = hidden_dim

    def encode_rgb(self, rgb):
        """
        编码RGB图像
        rgb: [B, 3, H, W]
        返回: RGB特征 [B, hidden_dim], RGB投影 [B, embedding_dim]
        """
        feat = self.rgb_encoder(rgb)  # [B, hidden_dim]
        proj = self.rgb_proj(feat)     # [B, embedding_dim]
        return feat, proj

    def encode_hsi(self, hsi):
        """
        编码HSI图像
        hsi: [B, 31, H, W]
        返回: HSI特征 [B, hidden_dim], HSI投影 [B, embedding_dim]
        """
        feat = self.hsi_encoder(hsi)  # [B, hidden_dim]
        proj = self.hsi_proj(feat)     # [B, embedding_dim]
        return feat, proj

    def forward(self, rgb, hsi=None, return_embedding=False):
        """
        前向传播 - 支持多种输入模式
        
        参数:
            rgb: [B, 3, H, W] - RGB图像
            hsi: [B, 31, H, W] (可选) - GT HSI或引导HSI
            return_embedding: bool - 是否返回投影嵌入
        
        返回:
            - 如果hsi=None: 预测的HSI [B, 31, H, W]
            - 如果hsi不为None: (预测HSI, RGB投影, HSI投影)
            - 如果return_embedding=True: 返回投影向量用于对比学习
        """
        # 编码RGB
        rgb_feat, rgb_proj = self.encode_rgb(rgb)  # [B, D], [B, E]
        
        # 编码HSI (如果提供)
        if hsi is not None:
            hsi_feat, hsi_proj = self.encode_hsi(hsi)  # [B, D], [B, E]
        else:
            hsi_feat, hsi_proj = None, None
        
        # 融合特征
        fused_feat = self.fusion(rgb_feat, hsi_feat)  # [B, D]
        
        # 解码为HSI
        pred_hsi = self.decoder(fused_feat)  # [B, 31, H, W]
        
        # 返回形式取决于是否有GT和是否需要嵌入
        if return_embedding:
            return {
                'pred_hsi': pred_hsi,
                'rgb_proj': rgb_proj,
                'hsi_proj': hsi_proj,
                'rgb_feat': rgb_feat,
                'hsi_feat': hsi_feat,
            }
        
        if hsi is not None:
            return pred_hsi, rgb_proj, hsi_proj
        else:
            return pred_hsi

    def get_embedding(self, rgb, hsi=None):
        """获取投影嵌入用于对比学习"""
        _, rgb_proj = self.encode_rgb(rgb)
        if hsi is not None:
            _, hsi_proj = self.encode_hsi(hsi)
            return rgb_proj, hsi_proj
        return rgb_proj, None

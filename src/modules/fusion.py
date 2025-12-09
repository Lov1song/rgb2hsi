# src/modules/fusion.py
import torch.nn as nn
import torch
import math

class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力模块
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv):
        """
        q: [B, L1, D] - query (RGB特征)
        kv: [B, L2, D] - key/value (HSI特征)
        """
        B, L1, D = q.shape
        
        q = self.to_q(q)  # [B, L1, D]
        k = self.to_k(kv)  # [B, L2, D]
        v = self.to_v(kv)  # [B, L2, D]
        
        # 分割头
        q = q.view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L1, Dh]
        k = k.view(B, L1, self.num_heads, self.head_dim).transpose(1, 2) if L1 == kv.shape[1] else k.view(B, kv.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L2, Dh]
        
        # 注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L1, L2]
        attn = torch.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)  # [B, H, L1, Dh]
        out = out.transpose(1, 2).contiguous()  # [B, L1, H, Dh]
        out = out.view(B, L1, D)  # [B, L1, D]
        out = self.to_out(out)
        
        return out


class CrossAttentionFusionModule(nn.Module):
    """
    跨模态融合模块 - CLIP风格
    融合RGB和HSI的特征为统一的多模态表示
    """
    def __init__(self, config):
        super().__init__()
        
        dim = config.get("dim", 512)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.1)
        
        # 自注意力和交叉注意力层
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fusion_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'cross_attn': MultiHeadCrossAttention(dim, num_heads, dropout),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
                'norm3': nn.LayerNorm(dim),
                'mlp': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))
        
        self.output_proj = nn.Linear(dim * 2, dim)
        self.dim = dim

    def forward(self, rgb_feat, hsi_feat=None):
        """
        rgb_feat: [B, D] - RGB编码特征
        hsi_feat: [B, D] - HSI编码特征 (可选)
        """
        B, D = rgb_feat.shape
        
        # 初始化查询和键值
        q = rgb_feat.unsqueeze(1)  # [B, 1, D]
        
        if hsi_feat is not None:
            kv = hsi_feat.unsqueeze(1)  # [B, 1, D]
        else:
            kv = q
        
        # 多层交叉融合
        for layer in self.fusion_layers:
            # 自注意力
            q_norm = layer['norm1'](q)
            q_attn, _ = layer['self_attn'](q_norm, q_norm, q_norm)
            q = q + q_attn
            
            # 交叉注意力（RGB关注HSI）
            q_norm = layer['norm2'](q)
            kv_norm = layer['norm3'](kv) if hsi_feat is not None else q_norm
            cross = layer['cross_attn'](q_norm, kv_norm)
            q = q + cross
            
            # MLP
            mlp_out = layer['mlp'](q)
            q = q + mlp_out
        
        # 融合多个特征和原始特征
        fused = q.squeeze(1)  # [B, D]
        
        if hsi_feat is not None:
            # 连接RGB和HSI特征
            fused = torch.cat([fused, hsi_feat], dim=1)  # [B, 2D]
            fused = self.output_proj(fused)  # [B, D]
        
        return fused


# 向后兼容
class CrossAttentionFusion(CrossAttentionFusionModule):
    """向后兼容的别名"""
    def __init__(self, dim=256, num_heads=4):
        config = {
            "dim": dim,
            "num_heads": num_heads,
            "num_layers": 1,
            "dropout": 0.1,
        }
        super().__init__(config)

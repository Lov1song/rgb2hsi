# src/models/hsi_encoder.py
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """标准的卷积块：Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpectralUNet(nn.Module):
    """
    改进的高光谱编码器 - U-Net风格
    支持多尺度特征提取和跳跃连接
    """
    def __init__(self, config):
        super().__init__()
        
        in_channels = config.get("in_channels", 31)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 4)
        
        # 编码器路径
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(in_ch, out_ch),
                    ConvBlock(out_ch, out_ch),
                )
            )
            in_ch = out_ch
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(in_ch, in_ch * 2),
            ConvBlock(in_ch * 2, in_ch),
        )
        
        # 解码器路径
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_dim * (2 ** i)
            in_ch_up = hidden_dim * (2 ** (i + 1))
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch_up, out_ch, 2, stride=2),
                    ConvBlock(out_ch * 2, out_ch),  # 与跳跃连接连接后
                    ConvBlock(out_ch, out_ch),
                )
            )
        
        # 全局平均池化和投影头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: [B, C, H, W] where C=31
        
        # 编码路径 - 保存跳跃连接
        skip_connections = []
        out = x
        for encoder_block in self.encoder_blocks:
            out = encoder_block(out)
            skip_connections.append(out)
            out = self.pool(out)
        
        # 瓶颈层
        out = self.bottleneck(out)
        
        # 解码路径 - 使用跳跃连接
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            out = decoder_block[0](out)  # 上采样
            # 连接跳跃连接
            out = torch.cat([out, skip], dim=1)
            out = decoder_block[1:](out)
        
        # 全局池化和投影
        feat = self.gap(out)  # [B, hidden_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, hidden_dim]
        feat = self.proj(feat)  # [B, hidden_dim]
        
        return feat

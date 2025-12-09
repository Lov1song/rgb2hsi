# src/models/rgb_encoder.py
import torch.nn as nn
import torchvision.models as models
import torch

class RGBEncoder(nn.Module):
    """
    增强的RGB编码器，支持多种骨干网络
    """
    def __init__(self, config):
        super().__init__()
        
        name = config.get("name", "ResNet50")
        pretrained = config.get("pretrained", True)
        hidden_dim = config.get("hidden_dim", 512)
        frozen_bn = config.get("frozen_bn", False)
        
        if name == "ResNet18":
            backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif name == "ResNet50":
            backbone = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            feat_dim = 2048
        elif name == "ResNet101":
            backbone = models.resnet101(weights="IMAGENET1K_V2" if pretrained else None)
            feat_dim = 2048
        else:
            raise NotImplementedError(f"Unsupported encoder: {name}")

        # 移除分类器，保留conv到avgpool的特征提取部分
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        
        # 冻结BN层
        if frozen_bn:
            self._freeze_bn()
        
        # 特征投影层，将不同大小的特征投影到统一维度
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

    def _freeze_bn(self):
        """冻结所有BN层"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        # x shape: [B, 3, H, W]
        feat = self.encoder(x)  # [B, feat_dim, 1, 1] 经过avgpool
        feat = feat.view(feat.size(0), -1)  # [B, feat_dim]
        feat = self.proj(feat)  # [B, hidden_dim]
        return feat

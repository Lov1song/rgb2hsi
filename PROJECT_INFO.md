# RGB2HSI MultimodalCLIP

**多模态RGB到高光谱图像转换的大模型项目**

## 项目信息

- **名称**: RGB2HSI MultimodalCLIP
- **版本**: 1.0.0
- **更新时间**: 2025-12-09
- **许可证**: MIT

## 核心特性

✨ **CLIP风格多模态对比学习**
- RGB和HSI特征的跨模态对齐
- NT-Xent对比损失
- 共享的投影嵌入空间

🎯 **灵活的学习策略**
- 有监督学习 (成对真实/合成数据)
- 自监督学习 (无配对数据)
- 混合学习 (成对+无配对)

🚀 **高性能训练**
- 梯度累积
- 混合精度 (AMP)
- 学习率调度 (余弦退火)
- 分布式训练支持

📊 **完整的评估工具**
- PSNR, SSIM, MSE
- 光谱角制图 (SAM)
- ERGAS指标

## 架构设计

```
RGB Input (3 channels)
    ↓
[RGB Encoder: ResNet50]
    ↓
[Shared Feature Space: 512-dim]
    ↓
[Fusion Module: CrossAttentionFusion]  ← [HSI Encoder: SpectralUNet]
    ↓
[Projection Head: 256-dim embedding]
    ↓
[Decoder: SpectralDecoder]
    ↓
HSI Output (31 channels)
```

## 关键组件

### 1. RGB编码器 (RGBEncoder)
- 支持 ResNet18/50/101
- 预训练权重初始化
- 特征投影层

### 2. HSI编码器 (SpectralUNet)
- U-Net架构
- 多尺度特征提取
- 跳跃连接

### 3. 融合模块 (CrossAttentionFusionModule)
- 多头交叉注意力
- 自注意力层
- MLP前馈网络

### 4. 解码器 (SpectralDecoder)
- 空间维度恢复
- 逐步上采样
- 31通道光谱重建

### 5. 损失函数 (HybridCLIPLoss)
- 重建损失 (L1 + MSE)
- 对比损失 (NT-Xent)
- 光谱约束 (平滑性+先验)

## 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python scripts/dataset_setup.py --create
# 按照目录结构放置数据
```

### 3. 训练
```bash
python main.py --epochs 200 --batch-size 32
```

### 4. 推理
```bash
python inference.py path/to/image.jpg --output results/output.npy
```

## 文件结构

```
rgb2hsi_project/
├── main.py                 # 训练脚本
├── inference.py            # 推理脚本
├── requirements.txt        # 依赖包
├── configs/
│   └── default.yaml       # 默认配置
├── src/
│   ├── models/            # 模型组件
│   ├── losses/            # 损失函数
│   ├── datasets/          # 数据加载器
│   ├── trainers/          # 训练器
│   ├── modules/           # 融合等模块
│   └── utils/             # 工具函数
├── scripts/
│   ├── train.sh           # 训练脚本
│   ├── eval.sh            # 评估脚本
│   └── dataset_setup.py   # 数据设置
├── checkpoints/           # 模型检查点
├── logs/                  # 训练日志
├── results/               # 推理结果
└── README.md              # 详细文档
```

## 配置详解

主要配置项在 `configs/default.yaml`:

### 模型配置
- `rgb_encoder`: ResNet50, pretrained, hidden_dim=512
- `hsi_encoder`: SpectralUNet, num_layers=4
- `fusion_module`: 8-head attention, 3 layers
- `embedding_dim`: 256 (共享嵌入空间)

### 训练配置
- `batch_size`: 32
- `epochs`: 200
- `lr`: 1e-4
- `gradient_accumulation_steps`: 4
- `mixed_precision`: false

### 损失权重
- `l1`: 1.0
- `spectral`: 1.0
- `contrastive`: 1.0
- `reconstruction`: 1.0

## 评估指标

| 指标 | 说明 | 更好的值 |
|------|------|---------|
| MSE | 均方误差 | 越小越好 |
| PSNR | 峰值信噪比 (dB) | 越大越好 |
| SSIM | 结构相似度 | 越大越好 (-1~1) |
| SAM | 光谱角制图 (度) | 越小越好 |
| ERGAS | 相对无维数误差 | 越小越好 |

## 高级配置

### 启用混合精度训练
```yaml
training:
  mixed_precision: true
```

### 增加融合层数
```yaml
model:
  fusion_module:
    num_layers: 5
```

### 调整学习率调度
```yaml
optimizer:
  lr_scheduler:
    type: "cosine"
    min_lr: 1e-6
```

## 研究方向

1. **架构改进**
   - Vision Transformer 编码器
   - 动态融合权重
   - 多分辨率金字塔

2. **损失函数**
   - 感知损失 (LPIPS)
   - 对抗性学习 (GAN)
   - 度量学习 (三元组损失)

3. **数据增强**
   - 频谱级增强
   - 混合增强 (MixUp)
   - 自适应增强 (AutoAugment)

4. **训练策略**
   - 课程学习
   - 元学习优化
   - 联邦学习

## 性能基准

(在标准数据集上的预期性能)

| 指标 | PSNR (dB) | SSIM | SAM (°) |
|------|-----------|------|---------|
| RGB→HSI | 30-35 | 0.85-0.90 | 5-10 |

## 常见问题

**Q: 如何处理不同尺寸的输入图像?**
A: 使用自适应池化或补零到固定大小。在推理中可以处理任意尺寸。

**Q: 如何使用GPU加速?**
A: 确保安装了CUDA兼容的PyTorch版本，模型会自动使用GPU。

**Q: 如何从检查点恢复训练?**
A: 使用 `python main.py --resume checkpoints/epoch_50.pth`

**Q: 如何评估预训练模型?**
A: 使用inference.py脚本，提供RGB图像即可获得HSI预测。

## 引用

如果使用该项目，请引用:

```bibtex
@article{rgb2hsi2025,
  title={RGB2HSI: Multimodal Image Transformation with CLIP-style Learning},
  author={Author},
  year={2025}
}
```

## 许可证

MIT License - 详见LICENSE文件

## 贡献

欢迎提交Issue和Pull Request!

## 联系方式

- 提交Issue进行讨论
- 提交PR贡献代码

---

**最后更新**: 2025-12-09  
**维护者**: RGB2HSI项目组

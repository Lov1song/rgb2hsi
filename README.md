# RGB2HSI MultimodalCLIP - 多模态图像转换模型

一个类似CLIP的多模态深度学习项目，用于RGB图像到高光谱图像(HSI)的转换。支持有监督、自监督和无监督学习。

## 📋 项目概述

### 核心功能
- **RGB→HSI转换**: 将普通RGB彩色图像转换为31通道的高光谱图像
- **多模态对比学习**: 类似CLIP的方式学习跨模态表示
- **灵活的学习模式**:
  - 有配对数据的监督学习
  - 无配对数据的自监督学习
  - 混合学习策略

### 应用场景
- 遥感图像处理
- 医学成像
- 材料科学检测
- 农业监测
- 文化遗产保护

## 🏗️ 项目架构

```
RGB2HSI-MultimodalCLIP
├── main.py                    # 训练入口
├── configs/
│   └── default.yaml          # 训练配置
├── src/
│   ├── models/
│   │   ├── rgb_encoder.py    # RGB编码器 (ResNet系列)
│   │   ├── hsi_encoder.py    # HSI编码器 (SpectralUNet)
│   │   ├── decoder.py        # 光谱解码器
│   │   └── rgb2hsi_model.py  # 主模型
│   ├── modules/
│   │   └── fusion.py         # 跨模态融合模块
│   ├── losses/
│   │   └── hybrid_loss.py    # 混合CLIP损失
│   ├── datasets/
│   │   ├── pair_datasets.py  # 成对数据集
│   │   ├── unpaired_rgb.py   # 无配对RGB数据
│   │   └── unpaired_hsi.py   # 无配对HSI数据
│   ├── trainers/
│   │   └── trainer.py        # 训练器
│   └── utils/
│       ├── inference.py      # 推理工具
│       └── metrics.py        # 评估指标
├── checkpoints/              # 模型检查点
├── logs/                      # 训练日志
└── results/                   # 推理结果
```

## 🔧 技术细节

### 编码器设计

#### RGB编码器
- 支持多种骨干网络: ResNet18, ResNet50, ResNet101
- 预训练权重初始化
- 特征投影到统一维度

#### HSI编码器 (SpectralUNet)
- U-Net架构支持多尺度特征提取
- 跳跃连接保留细节信息
- 全局平均池化和投影头

### 融合模块 (CLIP-Style)
- 多头交叉注意力机制
- RGB特征关注HSI特征的引导
- 多层交叉融合堆叠
- 自注意力 + 交叉注意力 + MLP

### 损失函数

#### 1. 重建损失 (Reconstruction Loss)
```
L_recon = 0.7 * L1 + 0.3 * MSE
```

#### 2. 对比学习损失 (NT-Xent)
```
L_contrastive = CrossEntropyLoss(RGB→HSI) + CrossEntropyLoss(HSI→RGB)
```

#### 3. 光谱约束损失
- 光谱平滑性: 鼓励相邻光谱通道的连续性
- 光谱先验: 利用HSI的统计特性

#### 4. 总体损失
```
L_total = α·L_recon + β·L_contrastive + γ·L_spectral + δ·L_prior
```

### 混合精度训练
- 支持自动混合精度(AMP)加速
- 梯度累积处理大批量
- 学习率预热和余弦退火调度

## 🚀 快速开始

### 环境设置

```bash
# 创建虚拟环境
conda create -n rgb2hsi python=3.10
conda activate rgb2hsi

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install PyYAML numpy pillow scikit-image scipy tensorboard
```

### 数据准备

数据目录结构:
```
data/
├── real_pair/
│   ├── rgb/          # RGB JPG/PNG 图像
│   └── hsi/          # HSI .npy 文件 (H, W, 31)
├── synthetic_pair/
│   ├── rgb/
│   └── hsi/
├── unpaired_rgb/     # RGB只有图像
└── unpaired_hsi/     # HSI只有图像
```

### 训练

```bash
# 使用默认配置训练
python main.py

# 使用自定义配置
python main.py --config configs/custom.yaml
```

### 推理

```python
from src.models.rgb2hsi_model import RGB2HSIModel
from src.utils.inference import RGB2HSIInference
import yaml

# 加载配置和模型
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

model = RGB2HSIModel(config)
model.load_state_dict(torch.load('checkpoints/model_final.pth'))

# 创建推理器
inferencer = RGB2HSIInference(model, config)

# 单张图像推理
pred_hsi, rgb_proj = inferencer.inference('path/to/image.jpg')

# 保存结果
inferencer.save_hsi(pred_hsi, 'results/output.npy')
```

## 📊 配置说明

主要配置项在 `configs/default.yaml`:

### 模型配置
```yaml
model:
  rgb_encoder:
    name: "ResNet50"        # 编码器类型
    pretrained: true        # 使用预训练权重
    hidden_dim: 512         # 特征维度
  
  hsi_encoder:
    name: "SpectralUNet"
    hidden_dim: 256
    num_layers: 4           # U-Net层数
  
  fusion_module:
    num_heads: 8            # 注意力头数
    num_layers: 3           # 融合层数
  
  embedding_dim: 256        # 投影嵌入维度
```

### 训练配置
```yaml
training:
  epochs: 200
  batch_size: 32
  lr: 1e-4
  gradient_accumulation_steps: 4  # 梯度累积
  mixed_precision: false          # 混合精度
```

### 损失权重
```yaml
loss:
  weights:
    l1: 1.0                 # L1重建损失
    spectral: 1.0           # 光谱约束
    perceptual: 0.5         # 感知损失
    contrastive: 1.0        # 对比学习
    reconstruction: 1.0     # 重建损失
```

## 📈 评估指标

- **MSE** (Mean Squared Error): 均方误差
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比
- **SSIM** (Structural Similarity): 结构相似度
- **SAM** (Spectral Angle Mapper): 光谱角制图
- **ERGAS** (Erreur Relative Globale Adimensionnelle): 相对无维数误差

```python
from src.utils.metrics import MetricsCalculator

metrics = MetricsCalculator(['mse', 'psnr', 'ssim', 'sam'])
results = metrics.compute(pred_hsi, gt_hsi)
print(results)
```

## 🎯 高级用法

### 自定义编码器

```python
from src.models.rgb_encoder import RGBEncoder

# 使用ResNet101
config = {
    "name": "ResNet101",
    "pretrained": True,
    "hidden_dim": 512,
}
encoder = RGBEncoder(config)
```

### 自定义融合模块

```python
from src.modules.fusion import CrossAttentionFusionModule

config = {
    "dim": 512,
    "num_heads": 8,
    "num_layers": 5,  # 增加更多层
    "dropout": 0.1,
}
fusion = CrossAttentionFusionModule(config)
```

### 从检查点恢复训练

```python
from src.trainers.trainer import Trainer

trainer = Trainer(config)
trainer.load_checkpoint('checkpoints/epoch_50.pth')
trainer.train()  # 从第51个epoch继续
```

## 🔬 研究扩展

### 可能的改进方向

1. **架构优化**
   - 使用Vision Transformer (ViT)替代CNN
   - 引入图像文本对比 (如果有文本标注)
   - 多尺度融合金字塔

2. **损失函数**
   - 引入感知损失 (LPIPS)
   - 对抗性损失 (GAN)
   - 光谱重建的标准化指标

3. **训练策略**
   - 课程学习 (从简单到复杂)
   - 自适应权重调整
   - 元学习优化

4. **数据增强**
   - 光谱角度的随机变换
   - 混合批次增强 (MixUp/CutMix)
   - 自适应增强

**最后更新**: 2025-12-09
**版本**: 1.0.0

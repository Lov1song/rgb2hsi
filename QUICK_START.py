#!/usr/bin/env python3
"""
RGB2HSI MultimodalCLIP - 快速开始指南
"""

# ============================================================================
# 1. 安装和设置
# ============================================================================

"""
# 步骤1: 安装依赖
pip install -r requirements.txt

# 步骤2: 创建数据目录
python scripts/dataset_setup.py --create
"""

# ============================================================================
# 2. 训练模型
# ============================================================================

"""
# 基础训练
python main.py

# 带自定义参数的训练
python main.py \\
    --batch-size 64 \\
    --epochs 300 \\
    --lr 5e-5 \\
    --mixed-precision \\
    --gradient-accumulation-steps 2

# 从检查点恢复
python main.py --resume checkpoints/epoch_50.pth

# 编辑配置文件进行更细致的控制
# 编辑 configs/default.yaml，然后运行：
python main.py --config configs/default.yaml
"""

# ============================================================================
# 3. 推理示例
# ============================================================================

from src.models.rgb2hsi_model import RGB2HSIModel
from src.utils.inference import RGB2HSIInference
import yaml
import torch

def example_inference():
    """单张图像推理示例"""
    
    # 加载配置
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGB2HSIModel(config).to(device)
    
    # 加载检查点
    checkpoint = torch.load('checkpoints/model_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建推理器
    inferencer = RGB2HSIInference(model, config, device=device)
    
    # 推理
    pred_hsi, rgb_proj = inferencer.inference('path/to/image.jpg')
    
    # 保存结果
    inferencer.save_hsi(pred_hsi, 'results/output.npy')
    
    print(f"HSI形状: {pred_hsi.shape}")
    print(f"RGB投影: {rgb_proj.shape}")

# ============================================================================
# 4. 批量推理
# ============================================================================

"""
python inference.py input_dir/ \\
    --output results/ \\
    --batch \\
    --save-visual
"""

# ============================================================================
# 5. 数据集验证
# ============================================================================

"""
# 验证数据集是否正确
python scripts/dataset_setup.py --verify data/
"""

# ============================================================================
# 6. 评估指标计算
# ============================================================================

from src.utils.metrics import MetricsCalculator
import numpy as np

def example_metrics():
    """评估指标示例"""
    
    # 加载预测和真实数据
    pred_hsi = np.load('results/pred_hsi.npy')  # [H, W, 31]
    gt_hsi = np.load('data/real_pair/hsi/gt.npy')  # [H, W, 31]
    
    # 计算指标
    metrics = MetricsCalculator(['mse', 'psnr', 'ssim', 'sam', 'ergas'])
    results = metrics.compute(pred_hsi, gt_hsi)
    
    # 打印结果
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

# ============================================================================
# 7. 自定义训练循环
# ============================================================================

def example_custom_training():
    """自定义训练循环示例"""
    
    import yaml
    from src.trainers.trainer import Trainer
    
    # 加载配置
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # 修改配置
    config['training']['epochs'] = 100
    config['training']['batch_size'] = 16
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 从检查点恢复（可选）
    # trainer.load_checkpoint('checkpoints/epoch_50.pth')
    
    # 开始训练
    trainer.train()
    
    # 访问训练日志
    print(f"训练损失: {trainer.logs['train_loss']}")
    print(f"验证损失: {trainer.logs['val_loss']}")

# ============================================================================
# 8. 模型结构查看
# ============================================================================

def example_model_inspection():
    """模型结构检查示例"""
    
    import yaml
    from src.models.rgb2hsi_model import RGB2HSIModel
    
    # 加载配置
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = RGB2HSIModel(config)
    
    # 打印模型结构
    print(model)
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 计算输入输出形状
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

# ============================================================================
# 9. 配置文件示例
# ============================================================================

"""
# configs/custom.yaml 示例

project_name: RGB2HSI-Custom

paths:
  real_pair: data/real_pair
  synthetic_pair: data/synthetic_pair
  unpaired_rgb: data/unpaired_rgb
  unpaired_hsi: data/unpaired_hsi
  save_dir: checkpoints
  log_dir: logs

training:
  epochs: 300
  batch_size: 64
  lr: 5e-5
  weight_decay: 1e-6
  num_workers: 8
  save_interval: 5
  eval_interval: 2
  gradient_accumulation_steps: 2
  mixed_precision: true
  pin_memory: true

model:
  rgb_encoder:
    name: "ResNet101"
    pretrained: true
    hidden_dim: 512
  
  hsi_encoder:
    name: "SpectralUNet"
    hidden_dim: 256
    num_layers: 5
  
  fusion_module:
    name: "CrossAttentionFusion"
    num_heads: 16
    num_layers: 6
    dropout: 0.15

loss:
  weights:
    l1: 1.0
    spectral: 1.0
    contrastive: 2.0
    reconstruction: 1.0

optimizer:
  type: "adamw"
  lr_scheduler:
    type: "cosine"
    min_lr: 1e-7
"""

# ============================================================================
# 10. 常见问题排查
# ============================================================================

"""
Q1: CUDA内存不足
A: 减小batch_size或启用梯度累积
   python main.py --batch-size 16 --gradient-accumulation-steps 4

Q2: 数据加载失败
A: 检查数据目录结构，运行验证脚本
   python scripts/dataset_setup.py --verify data/

Q3: 模型收敛缓慢
A: 尝试调整学习率或启用混合精度
   python main.py --lr 1e-3 --mixed-precision

Q4: 推理速度慢
A: 使用GPU，并考虑模型量化或蒸馏
   # 确认使用GPU
   python -c "import torch; print(torch.cuda.is_available())"

Q5: 如何使用多GPU
A: 使用DataParallel或DistributedDataParallel (待实现)
"""

# ============================================================================
# 11. 性能优化技巧
# ============================================================================

"""
1. 混合精度训练 (2-4x加速)
   --mixed-precision

2. 梯度累积 (处理大批量，节省内存)
   --gradient-accumulation-steps 4

3. 固定BN层 (加速、节省内存)
   配置中设置: frozen_bn: true

4. 增加num_workers (加速数据加载)
   配置中设置: num_workers: 8

5. Pin内存 (GPU内存充足时加速)
   配置中设置: pin_memory: true

6. 使用pretrained权重 (更快收敛)
   配置中设置: pretrained: true
"""

# ============================================================================
# 12. 扩展和自定义
# ============================================================================

def example_custom_encoder():
    """自定义编码器示例"""
    
    import torch.nn as nn
    from src.models.rgb2hsi_model import RGB2HSIModel
    
    # 创建自定义RGB编码器
    class CustomRGBEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            # 你的自定义实现
            pass
    
    # 在模型中使用自定义编码器
    # 修改 src/models/rgb2hsi_model.py 中的编码器初始化部分

def example_custom_loss():
    """自定义损失函数示例"""
    
    import torch.nn as nn
    from src.losses.hybrid_loss import HybridCLIPLoss
    
    class CustomLoss(HybridCLIPLoss):
        def forward(self, pred, gt, **kwargs):
            # 调用父类损失
            loss, loss_dict = super().forward(pred, gt, **kwargs)
            
            # 添加自定义损失项
            custom_loss = self.compute_custom_loss(pred)
            loss += 0.1 * custom_loss
            
            return loss, loss_dict
        
        def compute_custom_loss(self, pred):
            # 你的自定义损失实现
            pass

# ============================================================================
# 13. 结果分析
# ============================================================================

def example_results_analysis():
    """结果分析示例"""
    
    import json
    import numpy as np
    
    # 加载训练日志
    with open('logs/training_log.json', 'r') as f:
        logs = json.load(f)
    
    # 分析训练曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(logs['train_loss'])
    plt.plot(logs['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'])
    plt.title('Training Loss')
    plt.grid()
    
    # PSNR曲线
    plt.subplot(1, 3, 2)
    plt.plot(logs.get('psnr', []))
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Evolution')
    plt.grid()
    
    # 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(logs.get('lr', []))
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('results/analysis.png')
    print("分析图已保存到: results/analysis.png")

# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n请参考上述示例代码进行使用")

# src/trainers/trainer.py

import torch
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
import torch.nn.functional as F

from src.models.rgb2hsi_model import RGB2HSIModel
from src.datasets.pair_datasets import PairedRGBHSIDataset
from src.datasets.unpaired_rgb import UnpairedRGBDataset
from src.datasets.unpaired_hsi import UnpairedHSIDataset
from src.losses.hybrid_loss import HybridCLIPLoss


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 日志
        self.log_dir = config["paths"]["log_dir"]
        self.save_dir = config["paths"]["save_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.logs = {
            "train_loss": [],
            "val_loss": [],
            "epoch": [],
        }

        # 模型
        self.model = RGB2HSIModel(config).to(self.device)
        
        # 损失函数
        self.loss_fn = HybridCLIPLoss(config)

        # 优化器
        self.setup_optimizer(config)
        
        # 学习率调度器
        self.setup_scheduler(config)

        # 数据集
        self.load_datasets()
        
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        self.mixed_precision = config["training"].get("mixed_precision", False)
        
        # 梯度缩放器（混合精度）
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def setup_optimizer(self, config):
        """设置优化器"""
        lr = config["training"]["lr"]
        weight_decay = config["training"].get("weight_decay", 1e-6)
        
        opt_cfg = config.get("optimizer", {})
        opt_type = opt_cfg.get("type", "adamw").lower()
        
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_cfg.get("betas", [0.9, 0.999]),
            )
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {opt_type} not implemented")

    def setup_scheduler(self, config):
        """设置学习率调度器"""
        opt_cfg = config.get("optimizer", {})
        scheduler_cfg = opt_cfg.get("lr_scheduler", {})
        scheduler_type = scheduler_cfg.get("type", "cosine").lower()
        
        epochs = config["training"]["epochs"]
        
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=scheduler_cfg.get("min_lr", 1e-6)
            )
        elif scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                total_iters=epochs
            )
        else:
            self.scheduler = None

    def load_datasets(self):
        """加载所有数据集"""
        cfg = self.cfg["paths"]
        batch_size = self.cfg["training"]["batch_size"]
        num_workers = self.cfg["training"]["num_workers"]

        # 成对数据集
        self.ds_pair_real = PairedRGBHSIDataset(cfg["real_pair"])
        self.ds_pair_syn = PairedRGBHSIDataset(cfg["synthetic_pair"])
        
        # 无配对数据集
        self.ds_rgb_unpaired = UnpairedRGBDataset(cfg["unpaired_rgb"])
        self.ds_hsi_unpaired = UnpairedHSIDataset(cfg["unpaired_hsi"])

        # 数据加载器
        self.loader_pair_real = DataLoader(
            self.ds_pair_real,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.cfg["training"].get("pin_memory", True),
            prefetch_factor=self.cfg["training"].get("prefetch_factor", 2),
        )
        
        self.loader_pair_syn = DataLoader(
            self.ds_pair_syn,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.cfg["training"].get("pin_memory", True),
        )
        
        self.loader_rgb_unpaired = DataLoader(
            self.ds_rgb_unpaired,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        self.loader_hsi_unpaired = DataLoader(
            self.ds_hsi_unpaired,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def train(self):
        """完整训练循环"""
        epochs = self.cfg["training"]["epochs"]
        
        print(f"开始训练 RGB2HSI 模型")
        print(f"设备: {self.device}")
        print(f"总轮数: {epochs}")
        print(f"批大小: {self.cfg['training']['batch_size']}")
        print("-" * 80)

        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)
            self.logs["train_loss"].append(train_loss)
            self.logs["epoch"].append(epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 保存检查点
            if epoch % self.cfg["training"]["save_interval"] == 0:
                self.save_checkpoint(epoch)
                
            # 评估
            if epoch % self.cfg["training"]["eval_interval"] == 0:
                self.evaluate(epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f}")

        # 保存最终检查点
        self.save_checkpoint(epoch, is_final=True)
        print(f"\n训练完成！最终模型已保存。")

    def train_one_epoch(self, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # --- 有配对数据训练 ---
        # 合成数据
        for batch_idx, (rgb, hsi) in enumerate(self.loader_pair_syn):
            rgb, hsi = rgb.to(self.device), hsi.to(self.device)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                output = self.model(rgb, hsi, return_embedding=True)
                pred_hsi = output['pred_hsi']
                rgb_proj = output['rgb_proj']
                hsi_proj = output['hsi_proj']
                
                # 损失计算
                loss, loss_dict = self.loss_fn(
                    pred_hsi, hsi,
                    rgb_proj=rgb_proj,
                    hsi_proj=hsi_proj
                )
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        # 真实配对数据
        for rgb, hsi in self.loader_pair_real:
            rgb, hsi = rgb.to(self.device), hsi.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                output = self.model(rgb, hsi, return_embedding=True)
                pred_hsi = output['pred_hsi']
                rgb_proj = output['rgb_proj']
                hsi_proj = output['hsi_proj']
                
                loss, _ = self.loss_fn(pred_hsi, hsi, rgb_proj=rgb_proj, hsi_proj=hsi_proj)
                loss = loss / self.gradient_accumulation_steps
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        # --- 无配对RGB自监督训练 ---
        for rgb in self.loader_rgb_unpaired:
            rgb = rgb.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                pred = self.model(rgb)  # RGB only
                loss = self.loss_fn.unpaired_rgb_loss(pred)
                loss = loss / self.gradient_accumulation_steps
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        # --- 无配对HSI光谱先验训练 ---
        for hsi in self.loader_hsi_unpaired:
            hsi = hsi.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss = self.loss_fn.unpaired_hsi_loss(hsi)
                loss = loss / self.gradient_accumulation_steps
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for rgb, hsi in self.loader_pair_real:
                rgb, hsi = rgb.to(self.device), hsi.to(self.device)
                
                output = self.model(rgb, hsi, return_embedding=True)
                pred_hsi = output['pred_hsi']
                rgb_proj = output['rgb_proj']
                hsi_proj = output['hsi_proj']
                
                loss, _ = self.loss_fn(pred_hsi, hsi, rgb_proj=rgb_proj, hsi_proj=hsi_proj)
                val_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:  # 只评估10个batch
                    break
        
        avg_val_loss = val_loss / max(num_batches, 1)
        self.logs["val_loss"].append(avg_val_loss)
        
        return avg_val_loss

    def save_checkpoint(self, epoch, is_final=False):
        """保存检查点"""
        if is_final:
            save_path = os.path.join(self.save_dir, "model_final.pth")
        else:
            save_path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
            "logs": self.logs,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"✓ 检查点已保存: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"✓ 检查点已加载: {checkpoint_path}")

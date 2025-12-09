import torch.nn as nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 对比损失
    类似CLIP的损失函数
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, rgb_proj, hsi_proj):
        """
        rgb_proj: [B, D] - RGB投影
        hsi_proj: [B, D] - HSI投影
        """
        # L2正则化
        rgb_proj = F.normalize(rgb_proj, p=2, dim=-1)
        hsi_proj = F.normalize(hsi_proj, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(rgb_proj, hsi_proj.t()) / self.temperature  # [B, B]
        
        # 正样本对 (对角线)
        labels = torch.arange(rgb_proj.shape[0], device=rgb_proj.device)
        
        # 对称损失：rgb->hsi + hsi->rgb
        loss_rgb_to_hsi = F.cross_entropy(logits, labels)
        loss_hsi_to_rgb = F.cross_entropy(logits.t(), labels)
        loss = (loss_rgb_to_hsi + loss_hsi_to_rgb) / 2
        
        return loss


class SpectralSmoothLoss(nn.Module):
    """光谱平滑损失 - 鼓励光谱曲线平滑"""
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, pred_hsi):
        """
        pred_hsi: [B, C, H, W]
        """
        # 光谱维度的平滑损失
        spec_diff = pred_hsi[:, 1:, :, :] - pred_hsi[:, :-1, :, :]  # [B, C-1, H, W]
        loss = spec_diff.abs().mean()
        return self.weight * loss


class SpectralPriorLoss(nn.Module):
    """光谱先验损失 - 利用高光谱的统计性质"""
    def __init__(self, weight=0.3):
        super().__init__()
        self.weight = weight

    def forward(self, pred_hsi):
        """
        pred_hsi: [B, C, H, W]
        鼓励相邻光谱通道相关性
        """
        B, C, H, W = pred_hsi.shape
        
        # 计算相邻通道的相似性
        for i in range(C - 1):
            diff = (pred_hsi[:, i, :, :] - pred_hsi[:, i+1, :, :]).abs()
            # 平滑的HSI应该有相关的相邻通道
        
        # 简单实现：光谱一致性
        mean_spec = pred_hsi.mean(dim=[2, 3])  # [B, C]
        spec_range = mean_spec.max(dim=1)[0] - mean_spec.min(dim=1)[0]  # [B]
        
        # 鼓励光谱在合理范围内变化
        loss = (spec_range.abs() - 1.0).abs().mean()
        return self.weight * loss


class ReconstructionLoss(nn.Module):
    """重建损失 - L1和L2的组合"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.weight = weight

    def forward(self, pred, gt):
        """
        pred: 预测的HSI [B, C, H, W]
        gt: 真实的HSI [B, C, H, W]
        """
        loss = 0.7 * self.l1(pred, gt) + 0.3 * self.l2(pred, gt)
        return self.weight * loss


class HybridCLIPLoss(nn.Module):
    """
    混合CLIP风格损失
    结合重建、对比、光谱约束
    """
    def __init__(self, config):
        super().__init__()
        w = config["loss"]["weights"]
        
        # 基础损失
        self.w_l1 = w.get("l1", 1.0)
        self.w_spectral = w.get("spectral", 1.0)
        self.w_perceptual = w.get("perceptual", 0.5)
        self.w_contrastive = w.get("contrastive", 1.0)
        self.w_reconstruction = w.get("reconstruction", 1.0)
        
        # 损失模块
        self.reconstruction = ReconstructionLoss(self.w_reconstruction)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # 对比学习
        contrastive_cfg = config["loss"].get("contrastive", {})
        temp = contrastive_cfg.get("temperature", 0.07)
        self.contrastive = ContrastiveLoss(temperature=temp)
        
        # 光谱约束
        spectral_cfg = config["loss"].get("spectral", {})
        self.spectral_smooth = SpectralSmoothLoss(
            weight=spectral_cfg.get("smooth_weight", 0.5)
        )
        self.spectral_prior = SpectralPriorLoss(
            weight=spectral_cfg.get("prior_weight", 0.3)
        )
        # -------------------------------
        # Patch-based shift-invariant loss 配置
        # -------------------------------
        patch_cfg = config["loss"].get("patch", {})
        self.patch_weight = patch_cfg.get("weight", 0.0)
        self.patch_size = patch_cfg.get("patch_size", 16)
        self.patch_search_radius = patch_cfg.get("search_radius", 4)
        self.patch_stride = patch_cfg.get("stride", self.patch_size)
        self.patch_downscale = patch_cfg.get("downscale", 1)

    def forward(self, pred, gt, rgb_proj=None, hsi_proj=None):
        """
        有监督重建损失
        
        参数:
            pred: 预测的HSI [B, C, H, W]
            gt: 真实的HSI [B, C, H, W]
            rgb_proj: RGB投影 [B, D] (用于对比学习)
            hsi_proj: HSI投影 [B, D] (用于对比学习)
        """
        loss = 0
        loss_dict = {}
        
        # 重建损失
        recon_loss = self.reconstruction(pred, gt)
        loss += recon_loss
        loss_dict["recon"] = recon_loss.item()
        
        # L1损失
        l1_loss = self.w_l1 * self.l1_loss(pred, gt)
        loss += l1_loss
        loss_dict["l1"] = l1_loss.item()
        
        # 光谱平滑损失
        spec_smooth = self.spectral_smooth(pred)
        loss += spec_smooth
        loss_dict["spec_smooth"] = spec_smooth.item()
        
        # 光谱先验损失
        spec_prior = self.spectral_prior(pred)
        loss += spec_prior
        loss_dict["spec_prior"] = spec_prior.item()
        
        # 对比学习损失 (如果有投影)
        if rgb_proj is not None and hsi_proj is not None:
            contrastive_loss = self.w_contrastive * self.contrastive(rgb_proj, hsi_proj)
            loss += contrastive_loss
            loss_dict["contrastive"] = contrastive_loss.item()

        # Patch-based shift-invariant loss
        if self.patch_weight > 0 and pred is not None and gt is not None:
            patch_loss = self.shift_invariant_patch_loss(
                pred, gt,
                patch_size=self.patch_size,
                search_radius=self.patch_search_radius,
                stride=self.patch_stride,
                downscale=self.patch_downscale,
            )
            scaled_patch_loss = self.patch_weight * patch_loss
            loss += scaled_patch_loss
            loss_dict['patch'] = scaled_patch_loss.item()
        
        return loss, loss_dict

    def unpaired_rgb_loss(self, pred):
        """
        无配对RGB的自监督损失
        鼓励空间平滑性
        """
        # 梯度平滑约束
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        loss = (dx.abs().mean() + dy.abs().mean()) / 2
        return loss

    def unpaired_hsi_loss(self, hsi):
        """
        无配对HSI的自监督损失
        利用光谱先验
        """
        # 光谱平滑性
        spec = hsi.mean(dim=[2, 3])  # [B, C]
        spec_diff = (spec[:, 1:] - spec[:, :-1]).abs().mean()
        
        # 非负性约束 (可选)
        neg_loss = torch.clamp(hsi, max=0).abs().mean()
        
        loss = spec_diff + 0.1 * neg_loss
        return loss

    def contrastive_loss_unpaired(self, rgb_proj, hsi_proj):
        """
        对无配对数据的对比学习
        RGB和HSI各自的特征约束
        """
        if rgb_proj is not None:
            # RGB特征的自对比
            rgb_proj_norm = F.normalize(rgb_proj, p=2, dim=-1)
            rgb_sim = torch.matmul(rgb_proj_norm, rgb_proj_norm.t())
            
        if hsi_proj is not None:
            # HSI特征的自对比
            hsi_proj_norm = F.normalize(hsi_proj, p=2, dim=-1)
            hsi_sim = torch.matmul(hsi_proj_norm, hsi_proj_norm.t())
        
        return 0

    def shift_invariant_patch_loss(self, pred, gt, patch_size=16, search_radius=4, stride=None, downscale=1):
        """
        Shift-invariant patch loss.
        For each patch in `pred`, search within a local window in `gt` and take the minimal L1 distance.

        Args:
            pred, gt: tensors [B, C, H, W]
            patch_size: int
            search_radius: int (pixels)
            stride: int (patch stride). If None, equals patch_size (non-overlapping)
            downscale: int, factor to downsample images before computing (to save compute)
        Returns:
            scalar tensor
        """
        if stride is None:
            stride = patch_size

        # Basic checks
        B, C, H, W = pred.shape

        # Optionally downscale to reduce computation
        if downscale > 1:
            pred_ds = F.interpolate(pred, scale_factor=1.0 / downscale, mode='bilinear', align_corners=False)
            gt_ds = F.interpolate(gt, scale_factor=1.0 / downscale, mode='bilinear', align_corners=False)
        else:
            pred_ds = pred
            gt_ds = gt

        _, _, Hd, Wd = pred_ds.shape

        # Extract pred patches
        unfold_pred = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
        pred_patches = unfold_pred(pred_ds)  # [B, C*ps*ps, Np]
        Np = pred_patches.shape[-1]
        D = pred_patches.shape[1]
        pred_patches = pred_patches.permute(0, 2, 1)  # [B, Np, D]

        # Extract all gt patches with stride=1 to allow local search
        unfold_gt = torch.nn.Unfold(kernel_size=patch_size, stride=1)
        gt_patches_all = unfold_gt(gt_ds)  # [B, C*ps*ps, Ng]
        Ng = gt_patches_all.shape[-1]
        gt_patches_all = gt_patches_all.permute(0, 2, 1)  # [B, Ng, D]

        # Precompute grid coordinates for gt patches (top-left positions)
        gt_h_patches = Hd - patch_size + 1
        gt_w_patches = Wd - patch_size + 1
        device = pred.device
        ys = torch.arange(0, gt_h_patches, device=device)
        xs = torch.arange(0, gt_w_patches, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)
        # For each pred patch index, compute its top-left coordinate in the downscaled image
        pred_h_patches = (Hd - patch_size) // stride + 1
        pred_w_patches = (Wd - patch_size) // stride + 1
        pred_coords = []
        for i in range(pred_h_patches):
            for j in range(pred_w_patches):
                pred_coords.append((i * stride, j * stride))

        # Sanity: Np should equal pred_h_patches*pred_w_patches
        # If sizes mismatch due to rounding, clamp
        if len(pred_coords) != Np:
            # recompute pred_coords from Np
            pred_coords = []
            # compute indices by enumerating positions
            idx = 0
            for y in range(0, Hd - patch_size + 1, stride):
                for x in range(0, Wd - patch_size + 1, stride):
                    if idx >= Np:
                        break
                    pred_coords.append((y, x))
                    idx += 1

        # Convert gt_patches_all and pred_patches to float32 for distance
        pred_patches = pred_patches.contiguous()
        gt_patches_all = gt_patches_all.contiguous()

        total_loss = 0.0
        # Loop over batch to keep memory reasonable; inside loop vectorize over patches
        for b in range(B):
            pp = pred_patches[b]  # [Np, D]
            gp_all = gt_patches_all[b]  # [Ng, D]
            # for each pred patch compute candidate indices within search radius
            losses_per_patch = []
            for idx_p, (py, px) in enumerate(pred_coords):
                # candidate top-left positions within radius in downscaled coords
                ymin = max(py - search_radius, 0)
                ymax = min(py + search_radius, Hd - patch_size)
                xmin = max(px - search_radius, 0)
                xmax = min(px + search_radius, Wd - patch_size)
                if ymin > ymax or xmin > xmax:
                    # fallback: compare with exact position only
                    cand_idx = (grid_y == py) & (grid_x == px)
                    cand_pos = torch.nonzero(cand_idx, as_tuple=False).squeeze(1)
                else:
                    # mask grid positions inside window
                    mask = (grid_y >= ymin) & (grid_y <= ymax) & (grid_x >= xmin) & (grid_x <= xmax)
                    cand_pos = torch.nonzero(mask, as_tuple=False).squeeze(1)

                if cand_pos.numel() == 0:
                    # use whole gt set as fallback (shouldn't normally happen)
                    candidates = gp_all
                else:
                    candidates = gp_all[cand_pos]

                # Compute L1 distances between pp[idx_p] and candidates
                pvec = pp[idx_p].unsqueeze(0)  # [1, D]
                # Use L1 distance
                dists = (candidates - pvec).abs().mean(dim=1)  # [K]
                min_dist = dists.min()
                losses_per_patch.append(min_dist)

            if len(losses_per_patch) == 0:
                b_loss = torch.tensor(0.0, device=device)
            else:
                b_loss = torch.stack(losses_per_patch).mean()
            total_loss += b_loss

        total_loss = total_loss / B
        return total_loss

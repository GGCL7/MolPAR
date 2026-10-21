"""
losses.py — RA-GLGN 损失函数 (简化版)

保留:
  1. Weighted BCE Loss (类别加权, 解决长尾)
  2. Label-Anchored Contrastive Loss (利用 Label GCN 嵌入作为锚点)

移除 (基于消融实验结论):
  - Pathway Contrastive Loss (SupCon-style) → 贡献有限
  - Batch Prototype Loss → 贡献有限
  - Hierarchy Consistency Loss → 无正向贡献
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===========================================================================
# 1. Weighted BCE Loss (解决长尾分布)
# ===========================================================================

def compute_class_weights(
    label_matrix: np.ndarray,
    method: str = "sqrt_inv",
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    计算类别权重.

    Args:
        label_matrix: [N, L] 二值标签矩阵 (训练集)
        method: "sqrt_inv" | "inv" | "effective"
            - sqrt_inv: weight[l] = sqrt(N_total / N_l) (推荐)
            - inv: weight[l] = N_total / N_l
            - effective: weight[l] = (1 - β^N_l) / (1 - β), β=0.9999
        max_weight: 权重上限 (防止极端值)

    Returns:
        weights: [L] FloatTensor
    """
    N, L = label_matrix.shape
    counts = label_matrix.sum(axis=0).astype(np.float64) + 1.0  # avoid div/0

    if method == "sqrt_inv":
        weights = np.sqrt(N / counts)
    elif method == "inv":
        weights = N / counts
    elif method == "effective":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # 归一化使平均权重 = 1.0
    weights = weights / weights.mean()

    # 裁剪
    weights = np.clip(weights, 0.1, max_weight)

    return torch.tensor(weights, dtype=torch.float32)


class WeightedBCELoss(nn.Module):
    """
    类别加权 BCE Loss.
    尾部标签获得更高权重, 头部标签权重降低.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, L] raw logits
            labels: [B, L] multi-hot

        Returns:
            weighted BCE loss (scalar)
        """
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )  # [B, L]

        # 加权
        loss = loss * self.class_weights.unsqueeze(0)  # [B, L]

        return loss.mean()


# ===========================================================================
# 2. Label-Anchored Contrastive Loss
# ===========================================================================

def label_anchored_contrastive_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    W_label: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    用 Label GCN 输出的标签嵌入作为锚点的对比学习.
    拉近分子表征与所属标签嵌入, 推远与不属于标签的距离.

    相比 SupCon 和 BatchProto:
      - 不依赖 batch 内正对数量 (避免稀疏标签时 loss=0)
      - 锚点来自 Label GCN, 天然利用标签间关系
      - 对 tail labels 更友好 (每个标签都有嵌入锚点)
    """
    device = h.device
    B, L = labels.shape

    if B == 0:
        return torch.tensor(0.0, device=device)

    h_norm = F.normalize(h, dim=-1)
    w_norm = F.normalize(W_label, dim=-1)

    sim = (h_norm @ w_norm.t()) / temperature

    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    logits = sim - sim_max

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    y = (labels > 0.5).float()
    pos_count = y.sum(dim=1)

    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (log_prob * y).sum(dim=1) / (pos_count + eps)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


# ===========================================================================
# 组合损失 (简化版)
# ===========================================================================

class RAGLGNLoss(nn.Module):
    """
    RA-GLGN 的组合损失 (简化版).

    Total = λ_bce * WeightedBCE + λ_anchor * LabelAnchored
    """

    def __init__(self,
                 class_weights: torch.Tensor,
                 lambda_bce: float = 1.0,
                 lambda_anchor: float = 0.3,
                 anchor_temperature: float = 0.1):
        super().__init__()
        self.weighted_bce = WeightedBCELoss(class_weights)
        self.lambda_bce = lambda_bce
        self.lambda_anchor = lambda_anchor
        self.anchor_temperature = anchor_temperature

    def forward(
        self,
        logits_l2: torch.Tensor,
        labels_l2: torch.Tensor,
        h_c: torch.Tensor,
        W_label_L2: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits_l2: [B, 146]
            labels_l2: [B, 146]
            h_c: [B, d] 分子表征
            W_label_L2: [146, d] 标签嵌入

        Returns:
            dict with "total" and per-component losses
        """
        losses = {}

        # 1. Weighted BCE
        loss_bce = self.weighted_bce(logits_l2, labels_l2)
        losses["bce"] = loss_bce

        # 2. Label-Anchored Contrastive
        loss_anchor = label_anchored_contrastive_loss(
            h_c, labels_l2, W_label_L2, self.anchor_temperature
        )
        losses["anchor"] = loss_anchor

        # Total
        total = self.lambda_bce * loss_bce + self.lambda_anchor * loss_anchor
        losses["total"] = total

        return losses

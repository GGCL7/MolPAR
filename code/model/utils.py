"""
utils.py — 工具函数

与 model_fixed/utils.py 功能一致, 保持兼容.
"""

import numpy as np
import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


def set_seed(seed: int = 42):
    """设置全局随机种子, 确保可复现性."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_accuracy(y_true_np: np.ndarray,
                     y_pred_np: np.ndarray) -> float:
    """Sample-based Jaccard accuracy (|Y∩Ŷ| / |Y∪Ŷ|)."""
    assert y_true_np.shape == y_pred_np.shape
    n_samples = y_true_np.shape[0]

    count = 0.0
    for i in range(n_samples):
        yt = y_true_np[i].astype(bool)
        yp = y_pred_np[i].astype(bool)

        inter = np.logical_and(yt, yp).sum()   # |Yi ∩ Ŷi|
        union = np.logical_or(yt, yp).sum()    # |Yi ∪ Ŷi|

        if union == 0:
            continue
        count += inter / union

    acc = count / n_samples
    return float(acc)


def multilabel_metrics(y_true: torch.Tensor,
                       y_pred_prob: torch.Tensor,
                       threshold: float = 0.5):
    """
    多标签分类评估指标.

    Args:
        y_true: [N, L] ground-truth (0/1)
        y_pred_prob: [N, L] 预测概率
        threshold: 二值化阈值

    Returns:
        dict with acc, prec, rec, f1, macro_f1, auprc
    """
    ts = y_true.detach().cpu().numpy().astype(int)
    zs = y_pred_prob.detach().cpu().numpy()
    preds = (zs >= threshold).astype(int)

    # sample-based accuracy
    acc = _sample_accuracy(ts, preds)

    # sample-based precision / recall / F1
    precision = precision_score(ts, preds, average="samples", zero_division=0)
    recall = recall_score(ts, preds, average="samples", zero_division=0)
    f1score = f1_score(ts, preds, average="samples", zero_division=0)

    # macro-F1 (per-label averaged, 对长尾更敏感)
    macro_f1 = f1_score(ts, preds, average="macro", zero_division=0)

    # AUPRC (macro-averaged, 基于概率不受 threshold 影响)
    # 过滤掉全零标签列 (无正样本的标签无法计算 AP)
    col_sums = ts.sum(axis=0)
    valid_cols = col_sums > 0
    if valid_cols.sum() > 0:
        auprc = average_precision_score(
            ts[:, valid_cols], zs[:, valid_cols], average="macro"
        )
    else:
        auprc = 0.0

    return {
        "acc": float(acc),
        "prec": float(precision),
        "rec": float(recall),
        "f1": float(f1score),
        "macro_f1": float(macro_f1),
        "auprc": float(auprc),
    }


def multilabel_metrics_macro(y_true: torch.Tensor,
                             y_pred_prob: torch.Tensor,
                             threshold: float = 0.5):
    """
    多标签分类评估 (macro-averaged, per-label).

    Returns:
        dict with macro_prec, macro_rec, macro_f1, micro_f1
    """
    ts = y_true.detach().cpu().numpy().astype(int)
    zs = y_pred_prob.detach().cpu().numpy()
    preds = (zs >= threshold).astype(int)

    macro_prec = precision_score(ts, preds, average="macro", zero_division=0)
    macro_rec = recall_score(ts, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(ts, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(ts, preds, average="micro", zero_division=0)

    return {
        "macro_prec": float(macro_prec),
        "macro_rec": float(macro_rec),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }

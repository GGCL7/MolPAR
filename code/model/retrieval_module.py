"""
retrieval_module.py — kNN Retrieval Module

两大功能:
1. Inference-time kNN Voting (方案A):
   - 用 FAISS 检索 top-K 最近邻
   - 加权投票得到 retrieval_prior [B, 146]

2. Retrieval-Augmented Label Embedding Update (方案C):
   - 每 epoch 为每个 L2 标签检索代表分子
   - 用化学结构原型温和更新 label_emb
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# FAISS-based kNN Retrieval
# ===========================================================================

class RetrievalIndex:
    """
    FAISS 索引管理器.
    存储训练集的分子嵌入和对应标签, 支持高效 kNN 检索.
    """

    def __init__(self, dim: int = 256, use_gpu: bool = False):
        """
        Args:
            dim: 嵌入维度
            use_gpu: 是否使用 GPU FAISS (如果可用)
        """
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.labels = None  # [N_train, num_labels] 训练集标签矩阵
        self.embeddings = None  # [N_train, dim] 训练集嵌入 (numpy)
        self._built = False

    def build(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        构建 FAISS 索引.

        Args:
            embeddings: [N, dim] float32 numpy array
            labels: [N, num_labels] float32 numpy array (multi-hot)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: "
                "pip install faiss-cpu  (or faiss-gpu for GPU support)"
            )

        N, d = embeddings.shape
        assert d == self.dim, f"Dimension mismatch: expected {self.dim}, got {d}"

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normed = (embeddings / norms).astype(np.float32)

        # 构建索引 (使用 Inner Product = cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(d)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                pass  # fallback to CPU

        self.index.add(embeddings_normed)
        self.labels = labels.astype(np.float32)
        self.embeddings = embeddings_normed
        self._built = True

    def search(self, queries: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        检索 top-K 最近邻.

        Args:
            queries: [B, dim] float32 numpy array
            k: 近邻数

        Returns:
            distances: [B, K] cosine similarity scores
            indices: [B, K] neighbor indices in training set
        """
        assert self._built, "Index not built yet. Call build() first."

        # L2 normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
        queries_normed = (queries / norms).astype(np.float32)

        distances, indices = self.index.search(queries_normed, k)
        return distances, indices

    def knn_vote(self, queries: np.ndarray, k: int = 32, sigma: float = 1.0) -> np.ndarray:
        """
        kNN 加权投票.

        Args:
            queries: [B, dim] float32
            k: 近邻数
            sigma: 距离→权重的温度参数

        Returns:
            retrieval_prior: [B, num_labels] float32 加权投票结果
        """
        distances, indices = self.search(queries, k)

        # distances 是 cosine similarity (越大越好)
        # 转为权重: softmax over similarities
        weights = np.exp(distances * sigma)  # [B, K]
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # normalize

        # 加权投票 (向量化, 无 Python for 循环)
        neighbor_labels = self.labels[indices]  # [B, K, num_labels]
        retrieval_prior = (weights[:, :, None] * neighbor_labels).sum(axis=1)  # [B, num_labels]

        return retrieval_prior

    @property
    def is_built(self) -> bool:
        return self._built


# ===========================================================================
# GPU-native kNN Retrieval (避免 CPU-GPU 数据搬运)
# ===========================================================================

class GPURetrievalIndex:
    """
    全 GPU 原生 kNN 检索.
    训练集嵌入和标签全部驻留在 GPU 上, kNN 搜索使用矩阵乘法实现,
    消除每 batch 的 CPU-GPU 数据传输瓶颈.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda")
        self.embeddings = None   # [N_train, dim] GPU tensor (L2-normalized)
        self.labels = None       # [N_train, num_labels] GPU tensor
        self._built = False

    def build(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        构建 GPU 索引.

        Args:
            embeddings: [N, dim] GPU tensor
            labels: [N, num_labels] GPU tensor (multi-hot)
        """
        # L2 normalize for cosine similarity
        self.embeddings = F.normalize(embeddings.to(self.device), dim=-1)
        self.labels = labels.to(self.device).float()
        self._built = True

    @torch.no_grad()
    def knn_vote(self, queries: torch.Tensor, k: int = 32, sigma: float = 1.0) -> torch.Tensor:
        """
        GPU 原生 kNN 加权投票.

        Args:
            queries: [B, dim] GPU tensor
            k: 近邻数
            sigma: 相似度温度

        Returns:
            retrieval_prior: [B, num_labels] GPU tensor
        """
        assert self._built, "Index not built yet."

        # L2 normalize queries
        queries_normed = F.normalize(queries.to(self.device), dim=-1)

        # Cosine similarity via matrix multiplication: [B, N_train]
        sim = queries_normed @ self.embeddings.t()

        # Top-K
        topk_sim, topk_idx = sim.topk(k, dim=-1)  # [B, k]

        # Softmax weights
        weights = torch.softmax(topk_sim * sigma, dim=-1)  # [B, k]

        # Gather neighbor labels: [B, k, num_labels]
        neighbor_labels = self.labels[topk_idx]

        # Weighted vote: [B, num_labels]
        retrieval_prior = (weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)

        return retrieval_prior

    @property
    def is_built(self) -> bool:
        return self._built


# ===========================================================================
# Retrieval-Augmented Label Embedding Updater (方案C)
# ===========================================================================

class RetrievalAugmentedLabelUpdater:
    """
    每 epoch 用检索到的化学结构原型更新 Label GCN 的 label_emb.

    对每个 L2 标签 l:
      exemplars_l = 该标签正样本中, 嵌入最接近当前 label_emb[l] 的 M 个
      chem_proto_l = mean(h_mol for mol in exemplars_l)
      label_emb[l] = (1-γ) · label_emb[l] + γ · chem_proto_l
    """

    def __init__(self,
                 num_l1: int = 11,
                 num_l2: int = 146,
                 gamma: float = 0.2,
                 num_exemplars: int = 16):
        """
        Args:
            gamma: 化学原型的注入比例 (0.1~0.3 温和注入)
            num_exemplars: 每个标签检索的代表分子数
        """
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.gamma = gamma
        self.num_exemplars = num_exemplars

    @torch.no_grad()
    def update_label_embeddings(
        self,
        label_gcn: nn.Module,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
    ):
        """
        用检索到的化学原型更新 label_emb.

        Args:
            label_gcn: HierarchicalLabelGCN 模块
            train_embeddings: [N_train, dim] 当前 encoder 编码的训练集嵌入
            train_labels: [N_train, 146] 训练集 L2 标签 (multi-hot)
        """
        device = label_gcn.label_emb.device
        current_emb = label_gcn.label_emb.data.clone()  # [157, dim]

        # 只更新 L2 部分 [11:157]
        l2_emb = current_emb[self.num_l1:]  # [146, dim]

        # L2 normalize (仅用于相似度检索)
        l2_emb_normed = F.normalize(l2_emb, dim=-1)
        train_emb_normed = F.normalize(train_embeddings, dim=-1)

        for l in range(self.num_l2):
            # 找到该标签的正样本
            pos_mask = train_labels[:, l] > 0.5
            pos_count = pos_mask.sum().item()

            if pos_count == 0:
                continue

            pos_embeddings_normed = train_emb_normed[pos_mask]  # [n_pos, dim] (归一化，用于检索)
            pos_embeddings_raw = train_embeddings[pos_mask]     # [n_pos, dim] (原始尺度，用于原型计算)

            # 计算与当前 label_emb 的相似度 (在归一化空间)
            sim = pos_embeddings_normed @ l2_emb_normed[l]  # [n_pos]

            # 取 top-M 最近的
            M = min(self.num_exemplars, pos_count)
            _, topk_idx = torch.topk(sim, M)

            # 化学结构原型 (使用原始尺度嵌入，避免尺度不匹配)
            chem_proto = pos_embeddings_raw[topk_idx].mean(dim=0)  # [dim]

            # 将 chem_proto 缩放到与当前 label_emb 相同的模长
            # 防止长期漂移导致 label_emb 模长逐渐变小
            target_norm = l2_emb[l].norm() + 1e-8
            proto_norm = chem_proto.norm() + 1e-8
            chem_proto_scaled = chem_proto * (target_norm / proto_norm)

            # 温和注入 (EMA-style update)
            label_gcn.label_emb.data[self.num_l1 + l] = (
                (1 - self.gamma) * label_gcn.label_emb.data[self.num_l1 + l]
                + self.gamma * chem_proto_scaled.to(device)
            )


# ===========================================================================
# kNN Voting Module (用于 forward pass)
# ===========================================================================

class KNNVotingModule(nn.Module):
    """
    将 kNN 投票结果融入模型的 forward pass.

    改进设计:
      - per-label 可学习门控 (而非单一 β)
      - clamp retrieval_prior 防止极端 logit
      - 温和的 affine 融合而非直接 logit 相加
    """

    def __init__(self, num_labels: int = 146, init_beta: float = 0.3,
                 clamp_range: float = 3.0):
        super().__init__()
        self.num_labels = num_labels
        self.clamp_range = clamp_range

        # Per-label 可学习门控 β_l (初始化为 init_beta)
        self.log_beta = nn.Parameter(
            torch.full((num_labels,), np.log(init_beta))
        )

        # 可选: per-label 偏置修正
        self.retrieval_bias = nn.Parameter(torch.zeros(num_labels))

    @property
    def beta(self):
        """Per-label β > 0, 通过 exp 参数化, 上限 clamp 防止过大."""
        return torch.exp(self.log_beta).clamp(max=2.0)

    def forward(self, model_logits: torch.Tensor, retrieval_prior: torch.Tensor) -> torch.Tensor:
        """
        融合模型预测和检索投票.

        Args:
            model_logits: [B, num_labels] 模型原始 logits
            retrieval_prior: [B, num_labels] kNN 投票 (0~1 之间)

        Returns:
            final_logits: [B, num_labels]
        """
        # Clamp retrieval_prior 到 [eps, 1-eps] 防止极端 logit
        eps = 0.01
        retrieval_clamped = retrieval_prior.clamp(min=eps, max=1.0 - eps)

        # 转为 logit 空间
        retrieval_logits = torch.log(retrieval_clamped) - torch.log(1.0 - retrieval_clamped)

        # Clamp logit 幅度, 防止近邻投票压过主模型
        retrieval_logits = retrieval_logits.clamp(-self.clamp_range, self.clamp_range)

        # Per-label 加权融合
        final_logits = model_logits + self.beta.unsqueeze(0) * retrieval_logits + self.retrieval_bias

        return final_logits

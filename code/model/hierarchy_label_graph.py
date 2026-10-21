"""
hierarchy_label_graph.py — Hierarchical Label Graph GCN

核心创新: 将 L1 (11) 和 L2 (146) 标签构建为一个统一的层级图 (157 节点).
三种边:
  1. L2-L2 共现边: 从训练集标签共现矩阵构建
  2. L1-L2 层级边: parent-child 双向边
  3. L1-L1 共现边 (可选): L1 超类间的关系

GCN 在层级图上传播后:
  - L2 尾部标签可以从其 L1 父类"借力"
  - 同一 L1 下的 L2 标签通过共享父节点传播信息
  - 最终输出 W_label_L2 [146, d] 作为分类权重
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 图构建工具
# ===========================================================================

def build_l2_cooccurrence_graph(
    label_matrix: np.ndarray,
    threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从训练集 L2 标签矩阵构建共现图.

    Args:
        label_matrix: [N, 146] binary
        threshold: 条件概率阈值

    Returns:
        adj: [146, 146] float
        edge_index: [2, E] int
    """
    N, L = label_matrix.shape
    counts = label_matrix.sum(axis=0).astype(np.float64) + 1e-8
    cooccur = label_matrix.T @ label_matrix
    cond_prob = cooccur / counts[:, None]
    adj = (cond_prob + cond_prob.T) / 2.0
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)

    # symmetric normalization
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj = D_inv_sqrt @ adj @ D_inv_sqrt

    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst], axis=0)
    return adj.astype(np.float32), edge_index.astype(np.int64)


def build_hierarchy_graph(
    l2_label_matrix: np.ndarray,
    l2_to_l1_map: Dict[int, int],
    num_l1: int = 11,
    num_l2: int = 146,
    l2_threshold: float = 0.02,
    hierarchy_weight: float = 0.5,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建完整的层级标签图 (157 节点).

    节点编排:
      [0, 11)    → L1 节点
      [11, 157)  → L2 节点

    边来源:
      1. L2-L2 共现边 (offset by num_l1)
      2. L1-L2 层级边 (parent-child 双向)
      3. 自环 (所有节点)

    Args:
        l2_label_matrix: [N, 146] 训练集 L2 标签矩阵
        l2_to_l1_map: {l2_index: l1_index}
        hierarchy_weight: 层级边的权重

    Returns:
        edge_index: [2, E] LongTensor
        edge_weight: [E] FloatTensor
    """
    total_nodes = num_l1 + num_l2

    # --- L2-L2 共现边 ---
    l2_adj, l2_edge_idx = build_l2_cooccurrence_graph(l2_label_matrix, threshold=l2_threshold)

    all_src = []
    all_dst = []
    all_weight = []

    # L2-L2 (offset by num_l1)
    if l2_edge_idx.shape[1] > 0:
        l2_src = l2_edge_idx[0] + num_l1
        l2_dst = l2_edge_idx[1] + num_l1
        l2_weights = l2_adj[l2_edge_idx[0], l2_edge_idx[1]]
        all_src.append(l2_src)
        all_dst.append(l2_dst)
        all_weight.append(l2_weights)

    # --- L1-L2 层级边 (双向) ---
    for l2_idx, l1_idx in l2_to_l1_map.items():
        l2_node = int(l2_idx) + num_l1
        l1_node = int(l1_idx)
        # L1 → L2 (parent to child)
        all_src.append(np.array([l1_node]))
        all_dst.append(np.array([l2_node]))
        all_weight.append(np.array([hierarchy_weight]))
        # L2 → L1 (child to parent)
        all_src.append(np.array([l2_node]))
        all_dst.append(np.array([l1_node]))
        all_weight.append(np.array([hierarchy_weight]))

    # --- L1-L1 共现边 (从 L2 矩阵推导) ---
    # 如果两个 L1 下的 L2 标签经常共现, 则 L1 之间加边
    l1_cooccur = np.zeros((num_l1, num_l1), dtype=np.float32)
    for l2_i, l1_i in l2_to_l1_map.items():
        for l2_j, l1_j in l2_to_l1_map.items():
            if l1_i != l1_j and l2_i < num_l2 and l2_j < num_l2:
                l1_cooccur[int(l1_i), int(l1_j)] += l2_adj[int(l2_i), int(l2_j)]

    # normalize and threshold
    l1_max = l1_cooccur.max() + 1e-8
    l1_cooccur = l1_cooccur / l1_max
    l1_cooccur[l1_cooccur < 0.1] = 0.0

    l1_src, l1_dst = np.nonzero(l1_cooccur)
    if len(l1_src) > 0:
        all_src.append(l1_src)
        all_dst.append(l1_dst)
        all_weight.append(l1_cooccur[l1_src, l1_dst])

    # --- 自环 ---
    if add_self_loops:
        self_nodes = np.arange(total_nodes)
        all_src.append(self_nodes)
        all_dst.append(self_nodes)
        all_weight.append(np.ones(total_nodes, dtype=np.float32))

    # 拼接
    all_src = np.concatenate(all_src).astype(np.int64)
    all_dst = np.concatenate(all_dst).astype(np.int64)
    all_weight = np.concatenate(all_weight).astype(np.float32)

    # --- 整图对称归一化 (D^{-1/2} A D^{-1/2}) ---
    # 先构建邻接矩阵, 然后统一归一化, 确保不同来源的边数值语义一致
    adj_full = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    for i in range(len(all_src)):
        adj_full[all_src[i], all_dst[i]] += all_weight[i]

    # 对称归一化
    deg_full = adj_full.sum(axis=1)
    deg_inv_sqrt_full = np.where(deg_full > 0, 1.0 / np.sqrt(deg_full), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt_full)
    adj_norm = D_inv_sqrt @ adj_full @ D_inv_sqrt

    # 重新提取 edge_index 和 edge_weight (只保留非零边)
    norm_src, norm_dst = np.nonzero(adj_norm)
    norm_weight = adj_norm[norm_src, norm_dst].astype(np.float32)

    edge_index = torch.tensor(np.stack([norm_src, norm_dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(norm_weight, dtype=torch.float32)

    return edge_index, edge_weight


# ===========================================================================
# Hierarchical Label GCN
# ===========================================================================

class HierarchyLabelGCNLayer(nn.Module):
    """单层 GCN (带边权): X' = σ(A @ X @ W)"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, edge_weight):
        h = self.linear(x)
        row, col = edge_index
        L = x.size(0)
        msg = h[col] * edge_weight.unsqueeze(-1)
        out = torch.zeros(L, h.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return out


class HierarchicalLabelGCN(nn.Module):
    """
    层级标签图 GCN: 在 157 节点 (11 L1 + 146 L2) 图上做 GCN.

    输入:
        label_embeddings: [157, d_in] — L1 和 L2 的初始嵌入 (BioBERT)
        edge_index + edge_weight: 层级标签图

    输出:
        W_label_L1: [11, d_out]  — L1 标签嵌入 (用于 hierarchy gating)
        W_label_L2: [146, d_out] — L2 标签嵌入 (用作分类权重)
    """

    def __init__(self,
                 num_l1: int = 11,
                 num_l2: int = 146,
                 in_dim: int = 256,
                 hidden_dim: int = 256,
                 out_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 init_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.total_nodes = num_l1 + num_l2
        self.out_dim = out_dim

        # 标签初始嵌入 [157, in_dim]
        if init_embeddings is not None:
            assert init_embeddings.shape == (self.total_nodes, in_dim)
            self.label_emb = nn.Parameter(init_embeddings.clone())
        else:
            self.label_emb = nn.Parameter(torch.empty(self.total_nodes, in_dim))
            nn.init.xavier_uniform_(self.label_emb)

        # GCN layers
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(HierarchyLabelGCNLayer(dims[i], dims[i + 1]))
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.num_layers = num_layers

        # LayerNorm
        self.ln = nn.LayerNorm(out_dim)

        # 存储图结构
        self.register_buffer("_edge_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_edge_weight", torch.zeros(0, dtype=torch.float))

    def set_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """设置层级图结构 (训练前调一次)."""
        device = self._edge_index.device
        self._edge_index = edge_index.to(device)
        self._edge_weight = edge_weight.to(device)

    def forward(self, edge_index=None, edge_weight=None):
        """
        Returns:
            W_label_L1: [11, out_dim]
            W_label_L2: [146, out_dim]
        """
        if edge_index is None:
            edge_index = self._edge_index
            edge_weight = self._edge_weight

        x = self.label_emb  # [157, in_dim]

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x, negative_slope=0.2)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.ln(x)  # [157, out_dim]

        W_label_L1 = x[:self.num_l1]          # [11, out_dim]
        W_label_L2 = x[self.num_l1:]          # [146, out_dim]

        return W_label_L1, W_label_L2

    def get_all_embeddings(self):
        """获取所有标签嵌入 (用于 retrieval augmentation)."""
        return self.forward()


# ===========================================================================
# 工具: 从 JSON 加载层级映射
# ===========================================================================

def load_l2_to_l1_mapping(l2_json_path: str) -> Tuple[Dict[int, int], int, List[str]]:
    """
    从 label_space_L2.json 加载 L2→L1 映射.

    Returns:
        l2_to_l1: {l2_index(int): l1_index(int)}
        num_l2: L2 标签数
        l2_names: L2 标签名列表
    """
    with open(l2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_l2 = data["n_classes"]
    map_to_l1 = data.get("map_to_l1_index", {})
    index_to_map = data.get("index_to_map", {})
    index_to_name = data.get("index_to_name", {})

    l2_to_l1 = {}
    for l2_idx_str, map_id in index_to_map.items():
        l2_idx = int(l2_idx_str)
        l1_idx = map_to_l1.get(map_id)
        if l1_idx is not None:
            l2_to_l1[l2_idx] = int(l1_idx)

    l2_names = [index_to_name.get(str(i), f"pathway_{i}") for i in range(num_l2)]

    return l2_to_l1, num_l2, l2_names


def build_parent_map_tensor(l2_to_l1: Dict[int, int], num_l2: int = 146) -> torch.LongTensor:
    """
    构建 L2→L1 的映射张量 (用于 hierarchy gating).

    Returns:
        parent_map: [146] LongTensor, parent_map[l2_idx] = l1_idx
    """
    parent_map = torch.zeros(num_l2, dtype=torch.long)
    for l2_idx, l1_idx in l2_to_l1.items():
        parent_map[int(l2_idx)] = int(l1_idx)
    return parent_map

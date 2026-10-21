"""
ra_glgn_model.py — RA-GLGN: Retrieval-Augmented Graph Label Network

简化版 Stage 2 模型, 基于消融实验结论:
  1. Encoder (from Stage 1, slow-lr fine-tune)
  2. Hierarchical Label GCN (157 节点, L1+L2) — 核心组件
  3. kNN Retrieval Voting — 核心组件
  4. Label-Anchored Contrastive Loss — 唯一辅助损失

移除 (消融实验表明无正向贡献或贡献有限):
  - Hierarchy L1 Gating (soft mask)
  - Pathway Contrastive / Batch Prototype Loss
  - Hierarchy Consistency Loss
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MolecularEncoder
from hierarchy_label_graph import HierarchicalLabelGCN
from retrieval_module import KNNVotingModule


class RAGLGN(nn.Module):
    """
    RA-GLGN: Stage 2 简化模型.

    Forward pass:
      1. Encoder → h_c [B, 256]
      2. Hierarchy LabelGCN → W_label_L1 [11, 256], W_label_L2 [146, 256]
         (若 use_label_gcn=False, 则用 learnable embedding 替代)
      3. logits = h_c @ W_label_L2^T + bias
      4. (optional) kNN Voting: final = logits + β * retrieval_prior
    """

    def __init__(self,
                 # Encoder params
                 node_in_dim: int,
                 edge_in_dim: int,
                 fp_dim: int,
                 graph_hidden_dim: int = 128,
                 graph_out_dim: int = 256,
                 fp_hidden_dim: int = 256,
                 fusion_hidden_dim: int = 256,
                 final_dim: int = 256,
                 pool_ratio: float = 0.5,
                 gate_scale_mode: str = "straight_through",
                 # Label Graph params
                 num_l1: int = 11,
                 num_l2: int = 146,
                 label_emb_dim: int = 256,
                 label_gcn_hidden: int = 256,
                 label_gcn_layers: int = 2,
                 label_gcn_dropout: float = 0.1,
                 # Retrieval
                 use_retrieval: bool = True,
                 retrieval_beta: float = 0.3,
                 # Label GCN toggle
                 use_label_gcn: bool = True,
                 # Label embeddings init
                 init_label_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.final_dim = final_dim
        self.use_retrieval = use_retrieval
        self.use_label_gcn = use_label_gcn

        # === 1. Molecular Encoder ===
        self.encoder = MolecularEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            fp_dim=fp_dim,
            graph_hidden_dim=graph_hidden_dim,
            graph_out_dim=graph_out_dim,
            fp_hidden_dim=fp_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            final_dim=final_dim,
            pool_ratio=pool_ratio,
            gate_scale_mode=gate_scale_mode,
        )

        # === 2. Label Embeddings ===
        if use_label_gcn:
            # Hierarchical Label GCN
            self.hierarchy_label_gcn = HierarchicalLabelGCN(
                num_l1=num_l1,
                num_l2=num_l2,
                in_dim=label_emb_dim,
                hidden_dim=label_gcn_hidden,
                out_dim=final_dim,
                num_layers=label_gcn_layers,
                dropout=label_gcn_dropout,
                init_embeddings=init_label_embeddings,
            )
        else:
            # Learnable label embeddings (no GCN propagation)
            total_labels = num_l1 + num_l2
            if init_label_embeddings is not None:
                self.label_embeddings = nn.Parameter(init_label_embeddings.clone())
            else:
                self.label_embeddings = nn.Parameter(torch.empty(total_labels, label_emb_dim))
                nn.init.xavier_uniform_(self.label_embeddings)
            # Project to final_dim if needed
            if label_emb_dim != final_dim:
                self.label_proj = nn.Linear(label_emb_dim, final_dim)
            else:
                self.label_proj = nn.Identity()
            self.label_ln = nn.LayerNorm(final_dim)

        # === 3. Classification bias ===
        self.bias_l2 = nn.Parameter(torch.zeros(num_l2))

        # === 4. kNN Voting Module ===
        if use_retrieval:
            self.knn_voter = KNNVotingModule(num_labels=num_l2, init_beta=retrieval_beta)
            self.retrieval_index = None  # 外部设置

    def set_hierarchy_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """设置层级标签图 (仅在 use_label_gcn=True 时有效)."""
        if self.use_label_gcn:
            self.hierarchy_label_gcn.set_graph(edge_index, edge_weight)

    def load_encoder_from_stage1(self, stage1_ckpt_path: str, strict: bool = False):
        """
        从 Stage 1 checkpoint 加载 Encoder 权重.

        Stage1Model 中 encoder 的 key 形如: encoder.graph_encoder.conv1.xxx
        本模型 (RAGLGN) 中 encoder 的 key 形如: encoder.graph_encoder.conv1.xxx
        因此需要从 Stage1 state_dict 中提取 "encoder." 前缀的 key,
        去掉 "encoder." 前缀后加载到 self.encoder 中。
        """
        ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model", ckpt)

        encoder_state = {}
        s1_prefix = "encoder."

        for key, value in state_dict.items():
            if key.startswith(s1_prefix):
                new_key = key[len(s1_prefix):]
                encoder_state[new_key] = value

        if len(encoder_state) == 0:
            # Fallback: 兼容旧版格式
            for key, value in state_dict.items():
                if (key.startswith("graph_encoder.") or
                    key.startswith("fp_encoder.") or
                    key.startswith("fusion.")):
                    encoder_state[key] = value

        if len(encoder_state) == 0:
            print(f"[Load Encoder] WARNING: No encoder keys found in Stage 1 checkpoint!")
            print(f"[Load Encoder] Available keys (first 10): {list(state_dict.keys())[:10]}")
            return

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[Load Encoder] Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    {k}")
        if unexpected:
            print(f"[Load Encoder] Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    {k}")

        loaded_count = len(encoder_state) - len(unexpected)
        print(f"[Load Encoder] Successfully loaded {loaded_count}/{len(encoder_state)} "
              f"parameters from Stage 1")

    def forward(
        self,
        batch_graph,
        fps,
        retrieval_prior: Optional[torch.Tensor] = None,
        return_emb: bool = False,
        precomputed_h_c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            batch_graph: PyG Batch
            fps: [B, fp_dim]
            retrieval_prior: [B, 146] kNN 投票结果 (推断时外部计算后传入)
            return_emb: 是否返回分子表征
            precomputed_h_c: [B, final_dim] 预计算的分子嵌入 (避免 encoder 重复 forward)

        Returns:
            如果 return_emb=False:
                logits_l2: [B, 146]
            如果 return_emb=True:
                (h_c, logits_l2, W_label_L1, W_label_L2)
        """
        # === 1. Encode ===
        if precomputed_h_c is not None:
            h_c = precomputed_h_c
        else:
            h_c = self.encoder(batch_graph, fps)  # [B, final_dim]

        # === 2. Label Embeddings (GCN or learnable) ===
        if self.use_label_gcn:
            W_label_L1, W_label_L2 = self.hierarchy_label_gcn()  # [11, d], [146, d]
        else:
            emb = self.label_ln(self.label_proj(self.label_embeddings))
            W_label_L1 = emb[:self.num_l1]   # [11, d]
            W_label_L2 = emb[self.num_l1:]   # [146, d]

        # === 3. L2 logits (直接点积, 无 gating) ===
        raw_logits = h_c @ W_label_L2.t() + self.bias_l2  # [B, 146]

        # === 4. kNN Retrieval Voting ===
        if self.use_retrieval and retrieval_prior is not None:
            final_logits = self.knn_voter(raw_logits, retrieval_prior)
        else:
            final_logits = raw_logits

        if return_emb:
            return h_c, final_logits, W_label_L1, W_label_L2
        return final_logits

    def get_label_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前标签嵌入."""
        if self.use_label_gcn:
            return self.hierarchy_label_gcn()
        else:
            emb = self.label_ln(self.label_proj(self.label_embeddings))
            return emb[:self.num_l1], emb[self.num_l1:]


# ===========================================================================
# Stage 1 Model (消融版: 只用 BCE + LabelAnchor, 无 SupCon / BatchProto)
# ===========================================================================

class Stage1Model(nn.Module):
    """
    Stage 1: L1 预训练模型 (简化版).
    与 RA_HLGN 的 Stage1Model 架构相同, 但训练时只使用:
      - BCE Loss
      - Label-Anchored Contrastive Loss

    支持 use_label_gcn=False 模式 (用 learnable embedding 替代 GCN).
    """

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 fp_dim: int,
                 num_l1: int = 11,
                 graph_hidden_dim: int = 128,
                 graph_out_dim: int = 256,
                 fp_hidden_dim: int = 256,
                 fusion_hidden_dim: int = 256,
                 final_dim: int = 256,
                 pool_ratio: float = 0.5,
                 gate_scale_mode: str = "straight_through",
                 # Label GCN
                 label_emb_dim: int = 256,
                 label_gcn_hidden: int = 256,
                 label_gcn_layers: int = 2,
                 label_gcn_dropout: float = 0.1,
                 use_label_gcn: bool = True,
                 init_label_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_l1 = num_l1
        self.use_label_gcn = use_label_gcn

        # Encoder (与 Stage 2 共享架构)
        self.encoder = MolecularEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            fp_dim=fp_dim,
            graph_hidden_dim=graph_hidden_dim,
            graph_out_dim=graph_out_dim,
            fp_hidden_dim=fp_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            final_dim=final_dim,
            pool_ratio=pool_ratio,
            gate_scale_mode=gate_scale_mode,
        )

        # Label embeddings
        self.label_emb = nn.Parameter(
            init_label_embeddings if init_label_embeddings is not None
            else torch.empty(num_l1, label_emb_dim)
        )
        if init_label_embeddings is None:
            nn.init.xavier_uniform_(self.label_emb)

        if use_label_gcn:
            # L1 Label GCN (11 节点)
            from hierarchy_label_graph import HierarchyLabelGCNLayer
            self.gcn1 = HierarchyLabelGCNLayer(label_emb_dim, label_gcn_hidden)
            self.gcn2 = HierarchyLabelGCNLayer(label_gcn_hidden, final_dim)
            self.label_ln = nn.LayerNorm(final_dim)
            self.label_gcn_dropout = label_gcn_dropout
        else:
            # 直接投影 (no GCN propagation)
            if label_emb_dim != final_dim:
                self.label_proj = nn.Linear(label_emb_dim, final_dim)
            else:
                self.label_proj = nn.Identity()
            self.label_ln = nn.LayerNorm(final_dim)

        # Bias
        self.bias = nn.Parameter(torch.zeros(num_l1))

        # 存储 L1 图 (仅 GCN 模式使用)
        self.register_buffer("_l1_edge_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_l1_edge_weight", torch.zeros(0, dtype=torch.float))

    def set_label_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """设置 L1 标签图 (仅 GCN 模式使用)."""
        if self.use_label_gcn:
            device = self._l1_edge_index.device
            self._l1_edge_index = edge_index.to(device)
            self._l1_edge_weight = edge_weight.to(device)

    def get_label_embeddings(self) -> torch.Tensor:
        """获取 L1 标签嵌入 (用于 Label-Anchored Loss)."""
        if self.use_label_gcn:
            x = self.label_emb
            x = self.gcn1(x, self._l1_edge_index, self._l1_edge_weight)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.label_gcn_dropout, training=self.training)
            x = self.gcn2(x, self._l1_edge_index, self._l1_edge_weight)
            x = self.label_ln(x)
            return x
        else:
            return self.label_ln(self.label_proj(self.label_emb))

    def forward(self, batch_graph, fps, return_emb: bool = False):
        h_c = self.encoder(batch_graph, fps)  # [B, final_dim]

        # L1 Label GCN → logits
        W_label = self.get_label_embeddings()  # [num_l1, final_dim]
        logits = h_c @ W_label.t() + self.bias  # [B, num_l1]

        if return_emb:
            return h_c, logits
        return logits

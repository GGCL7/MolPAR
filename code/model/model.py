"""
model.py — RA-HLGN Shared Encoder

Encoder 模块 (Stage 1 & Stage 2 共用):
- GINGraphEncoder: 3层 GINE → MVPool → 1层精炼 + JK readout
- FingerprintEncoder: 3层 MLP
- BiGateFusion: LayerNorm + 低秩双线性 + 残差

与 model_fixed/model.py 的 Encoder 部分完全一致,
确保 Stage 1 预训练权重可以无缝加载到 Stage 2。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_add(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='add')
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import degree, subgraph


# ===========================================================================
# MVPool: Multi-View Pooling Scorer
# ===========================================================================
class MVPool(nn.Module):
    """
    Multi-View Pooling scorer.
    三视图 (结构度/特征MLP/扩散一致性) 融合打分 → (0,1) 重要性分数。
    """

    def __init__(self, in_channels: int, ratio: float, edge_attr_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.feat_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )

        self.view_weight = nn.Parameter(torch.ones(3))

    @staticmethod
    def _standardize(v: torch.Tensor) -> torch.Tensor:
        mean = v.mean()
        std = v.std()
        return (v - mean) / (std + 1e-6)

    def _diffusion_step(self, x, edge_index, edge_attr):
        row, col = edge_index
        N = x.size(0)

        if edge_attr is not None:
            w_raw = self.edge_mlp(edge_attr).squeeze(-1)
            edge_weight = F.softplus(w_raw) + 1e-6
        else:
            edge_weight = x.new_ones(row.size(0))

        deg = scatter_add(edge_weight, row, dim=0, dim_size=N)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        x_diff = scatter_add(
            x[col] * norm.unsqueeze(-1),
            row, dim=0, dim_size=N
        )
        return x_diff

    def compute_scores(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        row, _ = edge_index
        N = x.size(0)

        # view 1: structural (degree)
        deg_vec = degree(row, num_nodes=N, dtype=x.dtype)
        struct_raw = torch.log(deg_vec + 1.0)
        struct_score = self._standardize(struct_raw)

        # view 2: feature (MLP)
        feat_raw = self.feat_mlp(x).squeeze(-1)
        feat_score = self._standardize(feat_raw)

        # view 3: diffusion consistency
        x_diff = self._diffusion_step(x, edge_index, edge_attr)
        dot = (x * x_diff).sum(dim=-1)
        norm_x = x.norm(dim=-1) + 1e-6
        norm_xd = x_diff.norm(dim=-1) + 1e-6
        diff_raw = dot / (norm_x * norm_xd)
        diff_score = self._standardize(diff_raw)

        score_mat = torch.stack([struct_score, feat_score, diff_score], dim=-1)
        raw = score_mat @ self.view_weight
        score = torch.sigmoid(raw)
        return score


def topk_per_graph(score, batch_ids, ratio):
    """Per-graph Top-K node selection."""
    device = score.device
    num_graphs = int(batch_ids.max().item()) + 1

    idx_list = []
    for g in range(num_graphs):
        mask = (batch_ids == g)
        n_g = int(mask.sum().item())
        if n_g == 0:
            continue
        k_g = max(1, int(math.ceil(ratio * n_g)))
        s_g = score[mask]
        _, topk_loc = torch.topk(s_g.detach(), k_g, sorted=False)
        global_idx = mask.nonzero(as_tuple=False).view(-1)[topk_loc]
        idx_list.append(global_idx)

    if len(idx_list) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.sort(torch.cat(idx_list, dim=0))[0]


# ===========================================================================
# GINGraphEncoder
# ===========================================================================
class GINGraphEncoder(nn.Module):
    """
    3层 GINE (3-hop) → MVPool → 1层 GINE (子图精炼) + JK readout
    """

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 256,
                 pool_ratio: float = 0.5,
                 dropout: float = 0.2,
                 gate_scale_mode: str = "straight_through"):
        super().__init__()
        assert gate_scale_mode in ("straight_through", "soft_mask")
        self.gate_scale_mode = gate_scale_mode
        self.pool_ratio = pool_ratio
        self.hidden_dim = hidden_dim

        # 3 层 GINE 编码
        nn1 = nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim=edge_in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn2, edge_dim=edge_in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(nn3, edge_dim=edge_in_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # MVPool
        self.pool = MVPool(in_channels=hidden_dim, ratio=pool_ratio, edge_attr_dim=edge_in_dim)

        # 池化后精炼层
        nn4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv4 = GINEConv(nn4, edge_dim=edge_in_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # JK readout
        self.jk_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def _apply_score_gate(self, h_sub, score_sub):
        if self.gate_scale_mode == "straight_through":
            st = score_sub - score_sub.detach() + 1.0
            return h_sub * st.unsqueeze(-1)
        else:
            return h_sub * score_sub.unsqueeze(-1)

    def forward(self, data, return_all: bool = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index, edge_attr))))
        h = self.dropout(F.relu(self.bn2(self.conv2(h, edge_index, edge_attr))))
        h = self.dropout(F.relu(self.bn3(self.conv3(h, edge_index, edge_attr))))

        g_pre = global_mean_pool(h, batch)

        scores = self.pool.compute_scores(h, edge_index, edge_attr, batch)
        idx_keep = topk_per_graph(scores, batch, self.pool_ratio)

        sub_edge_index, sub_edge_attr = subgraph(
            idx_keep, edge_index, edge_attr, relabel_nodes=True, num_nodes=h.size(0)
        )

        h_sub = self._apply_score_gate(h[idx_keep], scores[idx_keep])
        sub_batch = batch[idx_keep]

        h_sub = self.dropout(F.relu(self.bn4(self.conv4(h_sub, sub_edge_index, sub_edge_attr))))

        g_post = global_mean_pool(h_sub, sub_batch)

        g = torch.cat([g_pre, g_post], dim=-1)
        g = self.jk_proj(g)

        if return_all:
            return g, h, scores, batch, idx_keep
        return g


# ===========================================================================
# FingerprintEncoder
# ===========================================================================
class FingerprintEncoder(nn.Module):
    """3层指纹编码器"""

    def __init__(self, fp_dim: int, hidden_dim: int = 256, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, fp):
        return self.mlp(fp)


# ===========================================================================
# BiGateFusion
# ===========================================================================
class BiGateFusion(nn.Module):
    """LayerNorm + 门控加权 + 低秩双线性交互 + 残差"""

    def __init__(self,
                 in_dim_graph: int,
                 in_dim_fp: int,
                 hidden_dim: int = 256,
                 out_dim: int = 256,
                 bilinear_rank: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.proj_g = nn.Linear(in_dim_graph, hidden_dim)
        self.proj_f = nn.Linear(in_dim_fp, hidden_dim)
        self.ln_g = nn.LayerNorm(hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.bilinear_U = nn.Linear(hidden_dim, bilinear_rank, bias=False)
        self.bilinear_V = nn.Linear(hidden_dim, bilinear_rank, bias=False)
        self.bilinear_out = nn.Linear(bilinear_rank, hidden_dim)

        self.out_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        if hidden_dim != out_dim:
            self.residual_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, h_g, h_fp):
        g = self.ln_g(self.proj_g(h_g))
        f = self.ln_f(self.proj_f(h_fp))

        concat = torch.cat([g, f], dim=-1)
        gate = torch.sigmoid(self.gate_net(concat))
        h_mix = gate * g + (1.0 - gate) * f

        u = self.bilinear_U(g)
        v = self.bilinear_V(f)
        h_bi = self.bilinear_out(u * v)

        fused = torch.cat([h_mix, h_bi], dim=-1)
        out = self.out_net(fused)

        residual = self.residual_proj((g + f) / 2.0)
        out = out + residual
        return out


# ===========================================================================
# MolecularEncoder: 完整的分子编码器 (GIN + FP + Fusion)
# ===========================================================================
class MolecularEncoder(nn.Module):
    """
    完整的分子编码器, 输出 h_c [B, final_dim].
    Stage 1 和 Stage 2 共享此 Encoder.
    """

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 fp_dim: int,
                 graph_hidden_dim: int = 128,
                 graph_out_dim: int = 256,
                 fp_hidden_dim: int = 256,
                 fusion_hidden_dim: int = 256,
                 final_dim: int = 256,
                 pool_ratio: float = 0.5,
                 gate_scale_mode: str = "straight_through"):
        super().__init__()

        self.graph_encoder = GINGraphEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            pool_ratio=pool_ratio,
            gate_scale_mode=gate_scale_mode,
        )

        self.fp_encoder = FingerprintEncoder(
            fp_dim=fp_dim,
            hidden_dim=fp_hidden_dim,
            out_dim=graph_out_dim,
        )

        self.fusion = BiGateFusion(
            in_dim_graph=graph_out_dim,
            in_dim_fp=graph_out_dim,
            hidden_dim=fusion_hidden_dim,
            out_dim=final_dim,
        )

    def forward(self, batch_graph, fps):
        h_g = self.graph_encoder(batch_graph)
        h_fp = self.fp_encoder(fps)
        h_c = self.fusion(h_g, h_fp)
        return h_c

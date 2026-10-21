"""
dataset.py — 数据集 (支持 L1/L2 双标签)

与 model_fixed/dataset.py 相比:
  - 支持同时返回 L1 和 L2 标签 (用于 hierarchy consistency loss)
  - 支持分层训练 (只返回 L1 或只返回 L2)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch as GeomBatch

from get_feature import smiles_to_data, extract_fingerprints


class HierarchicalMoleculeDataset(Dataset):
    """
    分层多标签分子数据集.
    同时提供 L1 (11) 和 L2 (146) 标签.
    """

    def __init__(
        self,
        df_or_path,
        num_l1: int = 11,
        num_l2: int = 146,
        smiles_col: str = "smiles",
        label_l1_col: str = "labels_L1",
        label_l2_col: str = "labels_L2",
        mode: str = "l2",  # "l1" | "l2" | "both"
        cache: bool = True,
    ):
        """
        Args:
            mode: "l1" → 只返回 L1 标签 (Stage 1)
                  "l2" → 只返回 L2 标签
                  "both" → 返回 L1 + L2 (Stage 2 with hierarchy loss)
        """
        super().__init__()

        if isinstance(df_or_path, str):
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path

        self.df = df.reset_index(drop=True)
        self.num_l1 = num_l1
        self.num_l2 = num_l2
        self.smiles_col = smiles_col
        self.label_l1_col = label_l1_col
        self.label_l2_col = label_l2_col
        self.mode = mode
        self.cache = cache

        self.smiles_list = self.df[self.smiles_col].tolist()

        if mode in ("l1", "both"):
            self.labels_l1_str = self.df[self.label_l1_col].tolist()
        if mode in ("l2", "both"):
            self.labels_l2_str = self.df[self.label_l2_col].tolist()

        self.graph_cache = {} if cache else None
        self.fp_cache = {} if cache else None

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_label(label_str, num_classes):
        """解析逗号分隔的标签索引为 one-hot."""
        if isinstance(label_str, str) and label_str.strip():
            indices = [int(i) for i in label_str.split(",") if i.strip()]
            if indices:
                onehot = F.one_hot(
                    torch.tensor(indices, dtype=torch.long),
                    num_classes=num_classes
                ).sum(dim=0).float()
                return onehot
        return torch.zeros(num_classes, dtype=torch.float32)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]

        # Graph
        if self.cache and self.graph_cache is not None and idx in self.graph_cache:
            data_graph = self.graph_cache[idx]
        else:
            data_graph = smiles_to_data(smiles)
            if self.cache and self.graph_cache is not None:
                self.graph_cache[idx] = data_graph

        # Fingerprint
        if self.cache and self.fp_cache is not None and idx in self.fp_cache:
            fp = self.fp_cache[idx]
        else:
            fp = extract_fingerprints(smiles)
            if self.cache and self.fp_cache is not None:
                self.fp_cache[idx] = fp

        # Labels
        if self.mode == "l1":
            label = self._parse_label(self.labels_l1_str[idx], self.num_l1)
            return data_graph, fp, label
        elif self.mode == "l2":
            label = self._parse_label(self.labels_l2_str[idx], self.num_l2)
            return data_graph, fp, label
        else:  # "both"
            label_l1 = self._parse_label(self.labels_l1_str[idx], self.num_l1)
            label_l2 = self._parse_label(self.labels_l2_str[idx], self.num_l2)
            return data_graph, fp, label_l1, label_l2


def collate_fn_l1(batch):
    """Stage 1 collate: (graph, fp, label_l1)"""
    graphs, fps, labels = zip(*batch)
    batch_graph = GeomBatch.from_data_list(graphs)
    fps = torch.stack(fps, dim=0)
    labels = torch.stack(labels, dim=0)
    return batch_graph, fps, labels


def collate_fn_l2(batch):
    """Stage 2 (L2 only) collate: (graph, fp, label_l2)"""
    graphs, fps, labels = zip(*batch)
    batch_graph = GeomBatch.from_data_list(graphs)
    fps = torch.stack(fps, dim=0)
    labels = torch.stack(labels, dim=0)
    return batch_graph, fps, labels


def collate_fn_both(batch):
    """Stage 2 (both L1+L2) collate: (graph, fp, label_l1, label_l2)"""
    graphs, fps, labels_l1, labels_l2 = zip(*batch)
    batch_graph = GeomBatch.from_data_list(graphs)
    fps = torch.stack(fps, dim=0)
    labels_l1 = torch.stack(labels_l1, dim=0)
    labels_l2 = torch.stack(labels_l2, dim=0)
    return batch_graph, fps, labels_l1, labels_l2


def get_collate_fn(mode: str):
    """根据 mode 返回对应的 collate_fn."""
    if mode == "l1":
        return collate_fn_l1
    elif mode == "l2":
        return collate_fn_l2
    else:
        return collate_fn_both

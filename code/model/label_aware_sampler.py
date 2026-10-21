"""
label_aware_sampler.py — Label-Aware Batch Sampler for Contrastive Learning

策略:
  每个 batch 由若干 "label groups" 组成, 每组围绕一个 "焦点标签" 采样:
  1. 随机选 num_anchors 个焦点标签
  2. 对每个焦点标签, 从持有该标签的样本中随机采 samples_per_anchor 个
  3. 这样保证 batch 内有大量共享标签的正对
  4. 如果填不满 batch_size, 用随机样本补齐

这确保:
  - 即使标签稀疏, batch 内也有足够的正对让 Contrastive Loss 起作用
  - 每个 epoch 所有样本仍然被大致均匀地遍历
"""

import random
from typing import List, Dict, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler, Subset


class LabelAwareBatchSampler(Sampler[List[int]]):
    """
    生成 batch indices, 保证每个 batch 内有多组共享标签的样本.
    
    Args:
        labels: [N, num_labels] 的 numpy array 或 tensor, 二值矩阵
        batch_size: 每个 batch 的总样本数
        num_anchors: 每个 batch 选多少个焦点标签 (默认 8)
        samples_per_anchor: 每个焦点标签采多少个正样本 (默认 4)
        drop_last: 是否丢弃最后一个不满的 batch
        seed: 随机种子
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 128,
        num_anchors: int = 8,
        samples_per_anchor: int = 4,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.samples_per_anchor = samples_per_anchor
        self.drop_last = drop_last
        self.seed = seed

        # labels: [N, C] binary matrix
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels = (labels > 0.5).astype(np.float32)
        self.num_samples = len(labels)
        self.num_labels = labels.shape[1]

        # 预计算: 每个标签对应哪些样本 (label -> list of sample indices)
        self.label_to_samples: Dict[int, List[int]] = {}
        for label_idx in range(self.num_labels):
            indices = np.where(self.labels[:, label_idx] > 0)[0].tolist()
            if len(indices) >= 2:  # 至少2个样本才有意义做对比
                self.label_to_samples[label_idx] = indices

        # 有效标签 (至少有2个样本)
        self.valid_labels = list(self.label_to_samples.keys())
        
        # 计算每个 epoch 的 batch 数
        self.num_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches += 1

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        self.seed += 1  # 每个 epoch 不同的随机序列

        # 全量索引, 用于补齐
        all_indices = list(range(self.num_samples))
        rng.shuffle(all_indices)
        fill_ptr = 0  # 指向 all_indices 的指针

        for _ in range(self.num_batches):
            batch_set = set()

            # Step 1: 选焦点标签并采正样本
            if self.valid_labels:
                # 随机选 num_anchors 个标签 (可重复如果标签不够多)
                anchor_labels = rng.choices(self.valid_labels, k=self.num_anchors)

                for lbl in anchor_labels:
                    candidates = self.label_to_samples[lbl]
                    # 采 samples_per_anchor 个 (如果候选不够就全采)
                    k = min(self.samples_per_anchor, len(candidates))
                    selected = rng.sample(candidates, k)
                    batch_set.update(selected)

                    # 如果 batch 已满, 提前停止
                    if len(batch_set) >= self.batch_size:
                        break

            # Step 2: 用随机样本补齐到 batch_size
            while len(batch_set) < self.batch_size and fill_ptr < len(all_indices):
                batch_set.add(all_indices[fill_ptr])
                fill_ptr += 1
                # 如果全量用完, 重新 shuffle
                if fill_ptr >= len(all_indices):
                    rng.shuffle(all_indices)
                    fill_ptr = 0

            batch_list = list(batch_set)[:self.batch_size]
            rng.shuffle(batch_list)  # 打乱 batch 内顺序
            yield batch_list

    def __len__(self) -> int:
        return self.num_batches


def build_label_aware_sampler(
    dataset,
    subset_indices: list = None,
    num_l2: int = 146,
    label_l2_col: str = "labels_L2",
    batch_size: int = 128,
    num_anchors: int = 8,
    samples_per_anchor: int = 4,
    seed: int = 42,
) -> LabelAwareBatchSampler:
    """
    从 dataset (或 Subset) 构建 LabelAwareBatchSampler.
    
    Args:
        dataset: HierarchicalMoleculeDataset 或 Subset
        subset_indices: 如果是 Subset, 传入 subset.indices
        num_l2: L2 标签数
        batch_size: batch 大小
        num_anchors: 每 batch 焦点标签数
        samples_per_anchor: 每焦点标签采样数
    """
    # 获取底层 dataset 和 indices
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = subset_indices or list(range(len(dataset)))

    # 构建 label matrix [N_subset, num_l2]
    label_matrix = np.zeros((len(indices), num_l2), dtype=np.float32)
    for local_idx, global_idx in enumerate(indices):
        label_str = base_dataset.df.iloc[global_idx][label_l2_col]
        if isinstance(label_str, str) and label_str.strip():
            label_indices = [int(i) for i in label_str.split(",") if i.strip()]
            for li in label_indices:
                if 0 <= li < num_l2:
                    label_matrix[local_idx, li] = 1.0

    return LabelAwareBatchSampler(
        labels=label_matrix,
        batch_size=batch_size,
        num_anchors=num_anchors,
        samples_per_anchor=samples_per_anchor,
        seed=seed,
    )

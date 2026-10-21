#!/usr/bin/env python3
"""
train_stage2.py — Stage 2: RA-GLGN 完整训练 (简化版)

目标: 在 Stage 1 Encoder 基础上, 训练 L2 (146 标签) 预测

核心组件 (保留):
  - Encoder 使用 slow-lr (0.1× 主学习率)
  - Hierarchical Label GCN (157 节点)
  - kNN Retrieval Voting (每 N epoch 更新索引)
  - Label-Anchored Contrastive Loss
  - Weighted BCE Loss
  - Label-Aware Sampling

移除 (基于消融实验):
  - Hierarchy L1 Gating
  - Pathway Contrastive Loss
  - Batch Prototype Loss
  - Hierarchy Consistency Loss
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 fork 后 tokenizers 死锁警告
import ast
import json
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from label_aware_sampler import LabelAwareBatchSampler, build_label_aware_sampler

from dataset import HierarchicalMoleculeDataset, collate_fn_both, collate_fn_l2
from ra_glgn_model import RAGLGN
from losses import (
    compute_class_weights,
    RAGLGNLoss,
)
from retrieval_module import RetrievalIndex, GPURetrievalIndex
from hierarchy_label_graph import (
    build_hierarchy_graph,
    load_l2_to_l1_mapping,
    build_parent_map_tensor,
)
from label_semantic_init import generate_or_load_embeddings
from utils import set_seed, multilabel_metrics


# ===========================================================================
# Split I/O
# ===========================================================================
def load_split_indices(index_path: str) -> Tuple[list, list, list]:
    data_index = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_index.append(ast.literal_eval(line))
    if len(data_index) != 3:
        raise ValueError("index file must contain exactly 3 non-empty lines")
    return data_index[0], data_index[1], data_index[2]


# ===========================================================================
# Build Training Label Matrix
# ===========================================================================
def build_train_label_matrix(train_indices, df, label_col, num_labels):
    """从训练集构建标签矩阵 [N_train, L]."""
    label_matrix = np.zeros((len(train_indices), num_labels), dtype=np.float32)
    for i, tidx in enumerate(train_indices):
        label_str = df.iloc[tidx][label_col]
        if isinstance(label_str, str) and label_str.strip():
            for idx_s in label_str.split(","):
                idx_s = idx_s.strip()
                if idx_s:
                    label_idx = int(idx_s)
                    if 0 <= label_idx < num_labels:
                        label_matrix[i, label_idx] = 1.0
    return label_matrix


# ===========================================================================
# Build Retrieval Index (GPU-native, no CPU-GPU transfer per batch)
# ===========================================================================
@torch.no_grad()
def build_retrieval_index(model, train_loader, device, num_l2):
    """用当前 Encoder 构建 GPU 检索索引 (避免每 batch CPU-GPU 搬运)."""
    model.eval()
    all_embs = []
    all_labels = []

    for batch in train_loader:
        if len(batch) == 4:
            batch_graph, fps, _, labels_l2 = batch
        else:
            batch_graph, fps, labels_l2 = batch

        batch_graph = batch_graph.to(device)
        fps = fps.to(device)

        h_c = model.encoder(batch_graph, fps)
        all_embs.append(h_c)  # keep on GPU
        all_labels.append(labels_l2)

    all_embs = torch.cat(all_embs, dim=0)       # [N_train, dim] on GPU
    all_labels = torch.cat(all_labels, dim=0)   # [N_train, num_labels] on CPU

    retrieval_index = GPURetrievalIndex(device=device)
    retrieval_index.build(all_embs, all_labels.to(device))
    return retrieval_index


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, loader, device, retrieval_index=None,
             use_retrieval=False, k=32, threshold=0.5):
    model.eval()
    total_loss = 0.0
    preds_all, trues_all = [], []

    for batch in loader:
        if len(batch) == 4:
            batch_graph, fps, labels_l1, labels_l2 = batch
        else:
            batch_graph, fps, labels_l2 = batch

        batch_graph = batch_graph.to(device)
        fps = fps.to(device)
        labels_l2 = labels_l2.to(device)

        # encoder forward
        h_c_eval = model.encoder(batch_graph, fps)

        # kNN retrieval (GPU 原生)
        retrieval_prior = None
        if use_retrieval and retrieval_index is not None and retrieval_index.is_built:
            retrieval_prior = retrieval_index.knn_vote(h_c_eval, k=k)

        h_c, logits_l2, W_l1, W_l2 = model(
            batch_graph, fps,
            retrieval_prior=retrieval_prior,
            return_emb=True,
            precomputed_h_c=h_c_eval,
        )

        # 评估时只用 BCE loss
        loss = F.binary_cross_entropy_with_logits(logits_l2, labels_l2)
        total_loss += loss.item() * labels_l2.size(0)

        preds_all.append(torch.sigmoid(logits_l2))
        trues_all.append(labels_l2)

    total_loss /= len(loader.dataset)
    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    metrics = multilabel_metrics(trues_all, preds_all, threshold=threshold)
    return total_loss, metrics


# ===========================================================================
# Per-bucket evaluation (Head / Mid / Tail)
# ===========================================================================
def evaluate_per_bucket(preds_all, trues_all, label_counts, threshold=0.5):
    """按标签频率桶 (Head/Mid/Tail) 评估."""
    preds_np = preds_all.cpu().numpy()
    trues_np = trues_all.cpu().numpy()
    preds_bin = (preds_np >= threshold).astype(int)

    head_mask = label_counts >= 500
    mid_mask = (label_counts >= 50) & (label_counts < 500)
    tail_mask = label_counts < 50

    results = {}
    for name, mask in [("head", head_mask), ("mid", mid_mask), ("tail", tail_mask)]:
        if mask.sum() == 0:
            results[name] = {"f1": 0.0, "n_labels": 0}
            continue

        from sklearn.metrics import f1_score
        f1 = f1_score(
            trues_np[:, mask], preds_bin[:, mask],
            average="macro", zero_division=0
        )
        results[name] = {"f1": float(f1), "n_labels": int(mask.sum())}

    return results


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="RA-GLGN Stage 2: L2 Training (Simplified)")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Path to kegg2026_hier.csv")
    p.add_argument("--index_path", type=str, required=True,
                   help="Split file (3 lines: train/val/test)")
    p.add_argument("--label_l2_json", type=str, required=True,
                   help="Path to label_space_L2.json")
    p.add_argument("--label_l1_json", type=str, default=None,
                   help="Path to label_space_L1.json (for semantic init)")

    # Stage 1 checkpoint
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Stage 1 best checkpoint path")

    # Labels
    p.add_argument("--num_l1", type=int, default=11)
    p.add_argument("--num_l2", type=int, default=146)
    p.add_argument("--smiles_col", type=str, default="smiles")
    p.add_argument("--label_l1_col", type=str, default="labels_L1")
    p.add_argument("--label_l2_col", type=str, default="labels_L2")

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    # Label-aware sampling
    p.add_argument("--label_aware_sampling", action="store_true", default=True,
                   help="Use label-aware batch sampler")
    p.add_argument("--no_label_aware_sampling", dest="label_aware_sampling", action="store_false")
    p.add_argument("--num_anchors", type=int, default=10,
                   help="Number of anchor labels per batch for label-aware sampling")
    p.add_argument("--samples_per_anchor", type=int, default=6,
                   help="Samples to draw per anchor label")
    p.add_argument("--encoder_lr_ratio", type=float, default=0.1,
                   help="Encoder lr = lr * ratio (slow fine-tuning)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)

    # Loss weights
    p.add_argument("--lambda_bce", type=float, default=1.0)
    p.add_argument("--lambda_anchor", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weight_method", type=str, default="sqrt_inv",
                   choices=["sqrt_inv", "inv", "effective"])

    # Model dims
    p.add_argument("--graph_hidden_dim", type=int, default=128)
    p.add_argument("--graph_out_dim", type=int, default=256)
    p.add_argument("--fp_hidden_dim", type=int, default=256)
    p.add_argument("--fusion_hidden_dim", type=int, default=256)
    p.add_argument("--final_dim", type=int, default=256)
    p.add_argument("--pool_ratio", type=float, default=0.5)
    p.add_argument("--label_emb_dim", type=int, default=256)
    p.add_argument("--label_gcn_layers", type=int, default=2)
    p.add_argument("--label_graph_threshold", type=float, default=0.02)
    p.add_argument("--no_label_gcn", action="store_true",
                   help="Disable Label GCN, use learnable embedding instead")

    # Retrieval
    p.add_argument("--use_retrieval", action="store_true", default=True,
                   help="Enable kNN retrieval (default: True)")
    p.add_argument("--no_retrieval", dest="use_retrieval", action="store_false",
                   help="Disable kNN retrieval")
    p.add_argument("--retrieval_k", type=int, default=32)
    p.add_argument("--retrieval_beta", type=float, default=0.3)
    p.add_argument("--retrieval_update_freq", type=int, default=5,
                   help="Update FAISS index every N epochs")

    # Semantic init
    p.add_argument("--semantic_init", action="store_true")
    p.add_argument("--biobert_model", type=str, default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--semantic_emb_cache", type=str, default=None)

    # Output
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--cache", action="store_true")

    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2] Device: {device}")
    print(f"[Stage 2] Model: RA-GLGN (Retrieval + Label GCN + Label-Anchored Loss)")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "stage2_best.pth")

    # --- Data ---
    df = pd.read_csv(args.csv_path)
    print(f"[Stage 2] Total samples: {len(df)}")

    train_idx, val_idx, test_idx = load_split_indices(args.index_path)
    print(f"[Stage 2] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # L2→L1 映射 (用于构建 Label Graph)
    l2_to_l1, num_l2_actual, l2_names = load_l2_to_l1_mapping(args.label_l2_json)
    print(f"[Stage 2] L2→L1 mapping loaded: {len(l2_to_l1)} entries")

    # Dataset (both mode 以兼容 collate_fn_both)
    dataset = HierarchicalMoleculeDataset(
        df_or_path=df,
        num_l1=args.num_l1,
        num_l2=args.num_l2,
        smiles_col=args.smiles_col,
        label_l1_col=args.label_l1_col,
        label_l2_col=args.label_l2_col,
        mode="both",
        cache=bool(args.cache),
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # 训练集: 使用 label-aware sampler 或普通 shuffle
    NUM_WORKERS = 4
    if args.label_aware_sampling:
        print(f"[Stage 2] Using Label-Aware Sampling: "
              f"num_anchors={args.num_anchors}, samples_per_anchor={args.samples_per_anchor}")
        train_batch_sampler = build_label_aware_sampler(
            dataset=train_set,
            num_l2=args.num_l2,
            label_l2_col=args.label_l2_col,
            batch_size=args.batch_size,
            num_anchors=args.num_anchors,
            samples_per_anchor=args.samples_per_anchor,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_set, batch_sampler=train_batch_sampler, collate_fn=collate_fn_both,
            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_both,
            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True,
        )

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_both, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn_both, num_workers=NUM_WORKERS, pin_memory=True)

    # 获取数据维度
    sample_graph, sample_fp, _, _ = dataset[0]
    node_in_dim = sample_graph.x.size(1)
    edge_in_dim = sample_graph.edge_attr.size(1)
    fp_dim = sample_fp.size(0)
    print(f"[Stage 2] Dims: node={node_in_dim}, edge={edge_in_dim}, fp={fp_dim}")

    # --- Class Weights ---
    train_label_matrix = build_train_label_matrix(
        train_idx, df, args.label_l2_col, args.num_l2
    )
    class_weights = compute_class_weights(train_label_matrix, method=args.weight_method)
    label_counts = train_label_matrix.sum(axis=0)
    print(f"[Stage 2] Label counts: min={label_counts.min():.0f}, "
          f"max={label_counts.max():.0f}, median={np.median(label_counts):.0f}")
    print(f"[Stage 2] Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    # --- Semantic Init (L1+L2 = 157 labels) ---
    init_label_emb = None
    if args.semantic_init:
        l1_json = args.label_l1_json
        l2_json = args.label_l2_json

        print(f"[Stage 2] Loading L1 semantic embeddings...")
        emb_l1 = generate_or_load_embeddings(
            label_json_path=l1_json,
            cache_path=args.semantic_emb_cache.replace(".pt", "_L1.pt") if args.semantic_emb_cache else None,
            model_name=args.biobert_model,
            target_dim=args.label_emb_dim,
            use_descriptions=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"[Stage 2] Loading L2 semantic embeddings...")
        emb_l2 = generate_or_load_embeddings(
            label_json_path=l2_json,
            cache_path=args.semantic_emb_cache.replace(".pt", "_L2.pt") if args.semantic_emb_cache else None,
            model_name=args.biobert_model,
            target_dim=args.label_emb_dim,
            use_descriptions=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        init_label_emb = torch.cat([emb_l1, emb_l2], dim=0)
        print(f"[Stage 2] Combined label embeddings: {init_label_emb.shape}")

    # --- Model ---
    use_label_gcn = not args.no_label_gcn
    model = RAGLGN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        fp_dim=fp_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_out_dim=args.graph_out_dim,
        fp_hidden_dim=args.fp_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        final_dim=args.final_dim,
        pool_ratio=args.pool_ratio,
        num_l1=args.num_l1,
        num_l2=args.num_l2,
        label_emb_dim=args.label_emb_dim,
        label_gcn_layers=args.label_gcn_layers,
        use_retrieval=args.use_retrieval,
        retrieval_beta=args.retrieval_beta,
        use_label_gcn=use_label_gcn,
        init_label_embeddings=init_label_emb,
    ).to(device)
    print(f"[Stage 2] Label GCN: {'Enabled' if use_label_gcn else 'Disabled (learnable emb)'}")

    # Load Stage 1 encoder weights
    print(f"[Stage 2] Loading encoder from Stage 1: {args.stage1_ckpt}")
    model.load_encoder_from_stage1(args.stage1_ckpt)

    # --- Hierarchy Label Graph ---
    edge_index, edge_weight = build_hierarchy_graph(
        l2_label_matrix=train_label_matrix,
        l2_to_l1_map=l2_to_l1,
        num_l1=args.num_l1,
        num_l2=args.num_l2,
        l2_threshold=args.label_graph_threshold,
    )
    model.set_hierarchy_graph(edge_index.to(device), edge_weight.to(device))
    print(f"[Stage 2] Hierarchy Label Graph: {args.num_l1 + args.num_l2} nodes, "
          f"{edge_index.size(1)} edges")

    # --- Optimizer (differential lr) ---
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("encoder.") and p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_ratio},
        {"params": other_params, "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # --- Loss ---
    criterion = RAGLGNLoss(
        class_weights=class_weights.to(device),
        lambda_bce=args.lambda_bce,
        lambda_anchor=args.lambda_anchor,
        anchor_temperature=args.temperature,
    )

    # --- Retrieval Setup ---
    retrieval_index = None
    if args.use_retrieval:
        print("[Stage 2] Building initial retrieval index...")
        retrieval_loader = DataLoader(
            train_set, batch_size=args.batch_size * 2,
            shuffle=False, collate_fn=collate_fn_both,
        )
        retrieval_index = build_retrieval_index(model, retrieval_loader, device, args.num_l2)
        print(f"[Stage 2] Initial index built: {retrieval_index.embeddings.shape[0]} vectors")

    # --- Training ---
    best_val_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loss_components = {"bce": 0., "anchor": 0.}

        for batch_graph, fps, labels_l1, labels_l2 in tqdm(
            train_loader, desc=f"[S2] Epoch {epoch}"
        ):
            batch_graph = batch_graph.to(device)
            fps = fps.to(device)
            labels_l2 = labels_l2.to(device)

            optimizer.zero_grad()

            # === Encoder forward (带梯度) ===
            h_c_grad = model.encoder(batch_graph, fps)

            # kNN retrieval prior: GPU 原生检索 (无 CPU-GPU 搬运)
            retrieval_prior = None
            if args.use_retrieval and retrieval_index is not None and retrieval_index.is_built:
                retrieval_prior = retrieval_index.knn_vote(h_c_grad.detach(), k=args.retrieval_k)

            # 模型 forward (precomputed 避免重复 encode)
            h_c, logits_l2, W_l1, W_l2 = model(
                batch_graph, fps,
                retrieval_prior=retrieval_prior,
                return_emb=True,
                precomputed_h_c=h_c_grad,
            )

            # Loss (简化: 只有 BCE + Anchor)
            losses = criterion(
                logits_l2=logits_l2,
                labels_l2=labels_l2,
                h_c=h_c,
                W_label_L2=W_l2,
            )

            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels_l2.size(0)
            for k in loss_components:
                loss_components[k] += losses[k].item() * labels_l2.size(0)

        scheduler.step()
        n_train = len(train_set)
        total_loss /= n_train
        for k in loss_components:
            loss_components[k] /= n_train

        # --- Update Retrieval Index ---
        if (args.use_retrieval and
            epoch % args.retrieval_update_freq == 0 and
            epoch < args.epochs):
            print(f"  [Retrieval] Rebuilding GPU index at epoch {epoch}...")
            retrieval_loader = DataLoader(
                train_set, batch_size=args.batch_size * 2,
                shuffle=False, collate_fn=collate_fn_both,
            )
            retrieval_index = build_retrieval_index(model, retrieval_loader, device, args.num_l2)

        # --- Validation ---
        val_loss, val_metrics = evaluate(
            model, val_loader, device,
            retrieval_index=retrieval_index,
            use_retrieval=args.use_retrieval,
            k=args.retrieval_k,
            threshold=args.threshold,
        )

        print(
            f"[S2 Epoch {epoch}] "
            f"Loss={total_loss:.4f} (BCE={loss_components['bce']:.4f} "
            f"Anchor={loss_components['anchor']:.4f}) | "
            f"Val: Loss={val_loss:.4f} F1={val_metrics['f1']:.4f} "
            f"MacroF1={val_metrics['macro_f1']:.4f} "
            f"AUPRC={val_metrics['auprc']:.4f} "
            f"Prec={val_metrics['prec']:.4f} Rec={val_metrics['rec']:.4f}"
        )

        # 使用 macro-F1 选择 best checkpoint
        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_metrics['f1'],
                "val_macro_f1": best_val_macro_f1,
                "val_auprc": val_metrics['auprc'],
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → Saved best (MacroF1={best_val_macro_f1:.4f}, "
                  f"SampleF1={val_metrics['f1']:.4f}, AUPRC={val_metrics['auprc']:.4f})")

    # --- Test ---
    print(f"\n[Stage 2] Best epoch: {best_epoch} (Val MacroF1={best_val_macro_f1:.4f})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    # Final retrieval index for test
    if args.use_retrieval:
        retrieval_loader = DataLoader(
            train_set, batch_size=args.batch_size * 2,
            shuffle=False, collate_fn=collate_fn_both,
        )
        retrieval_index = build_retrieval_index(model, retrieval_loader, device, args.num_l2)

    test_loss, test_metrics = evaluate(
        model, test_loader, device,
        retrieval_index=retrieval_index,
        use_retrieval=args.use_retrieval,
        k=args.retrieval_k,
        threshold=args.threshold,
    )
    print(f"[Stage 2] Test: Loss={test_loss:.4f} F1={test_metrics['f1']:.4f} "
          f"MacroF1={test_metrics['macro_f1']:.4f} AUPRC={test_metrics['auprc']:.4f} "
          f"Prec={test_metrics['prec']:.4f} Rec={test_metrics['rec']:.4f}")

    # --- Per-bucket evaluation ---
    print("\n[Stage 2] Per-bucket evaluation:")
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                bg, fp, _, l2 = batch
            else:
                bg, fp, l2 = batch
            bg = bg.to(device)
            fp = fp.to(device)
            l2 = l2.to(device)

            retrieval_prior = None
            if args.use_retrieval and retrieval_index is not None:
                h_eval = model.encoder(bg, fp)
                retrieval_prior = retrieval_index.knn_vote(h_eval, k=args.retrieval_k)

            logits = model(bg, fp, retrieval_prior=retrieval_prior)
            preds_all.append(torch.sigmoid(logits))
            trues_all.append(l2)

    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)

    bucket_results = evaluate_per_bucket(preds_all, trues_all, label_counts, args.threshold)
    for bucket, res in bucket_results.items():
        print(f"  {bucket:>4s}: F1={res['f1']:.4f} (n_labels={res['n_labels']})")

    # --- Save Summary ---
    summary = {
        "stage": 2,
        "model": "RA-GLGN",
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "bucket_results": bucket_results,
        "ckpt_path": ckpt_path,
        "stage1_ckpt": args.stage1_ckpt,
        "config": {
            "use_retrieval": args.use_retrieval,
            "label_gcn_layers": args.label_gcn_layers,
            "lambda_bce": args.lambda_bce,
            "lambda_anchor": args.lambda_anchor,
            "label_aware_sampling": args.label_aware_sampling,
        },
    }
    summary_path = os.path.join(args.ckpt_dir, "stage2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Stage 2] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

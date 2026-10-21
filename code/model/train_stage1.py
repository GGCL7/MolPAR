#!/usr/bin/env python3
"""
train_stage1.py — Stage 1: L1 预训练

目标: 训练高质量的 MolecularEncoder (GIN + FP + Fusion)
     使用 L1 (11 大类) 数据充足的标签进行训练
     训练完成后 Encoder 权重迁移到 Stage 2

训练策略:
  - Label GCN (11 节点) + BioBERT 语义初始化
  - SupCon + BatchProto + LabelAnchored 对比学习
  - BCE Loss (无加权, 因为 L1 数据相对均衡)
"""

import os
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

from dataset import HierarchicalMoleculeDataset, collate_fn_l1
from ra_glgn_model import Stage1Model
from losses import label_anchored_contrastive_loss
from utils import set_seed, multilabel_metrics
from label_semantic_init import generate_or_load_embeddings


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


def build_l1_label_graph(train_indices, df, label_col, num_labels, threshold=0.02):
    """从训练集构建 L1 标签共现图."""
    from hierarchy_label_graph import build_l2_cooccurrence_graph

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

    adj, edge_idx = build_l2_cooccurrence_graph(label_matrix, threshold=threshold)

    # 加自环
    L = adj.shape[0]
    adj = adj + np.eye(L, dtype=np.float32)

    # 对称归一化 D^{-1/2} A D^{-1/2} (与 Stage2 build_hierarchy_graph 保持一致)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj = D_inv_sqrt @ adj @ D_inv_sqrt

    src, dst = np.nonzero(adj)
    weights = adj[src, dst].astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight, label_matrix


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    preds_all, trues_all = [], []

    for batch_graph, fps, labels in loader:
        batch_graph = batch_graph.to(device)
        fps = fps.to(device)
        labels = labels.to(device)

        logits = model(batch_graph, fps)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds_all.append(torch.sigmoid(logits))
        trues_all.append(labels)

    total_loss /= len(loader.dataset)
    preds_all = torch.cat(preds_all, dim=0)
    trues_all = torch.cat(trues_all, dim=0)
    metrics = multilabel_metrics(trues_all, preds_all, threshold=threshold)
    return total_loss, metrics


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="RA-HLGN Stage 1: L1 Pretraining")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Path to kegg2026_hier.csv")
    p.add_argument("--index_path", type=str, required=True,
                   help="Split file (3 lines: train/val/test)")
    p.add_argument("--label_json_path", type=str, default=None,
                   help="Path to label_space_L1.json (for semantic init)")

    p.add_argument("--num_l1", type=int, default=11)
    p.add_argument("--smiles_col", type=str, default="smiles")
    p.add_argument("--label_col", type=str, default="labels_L1")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--lambda_anchor", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.1)

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
    print(f"[Stage 1] Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "stage1_best.pth")

    # --- Data ---
    df = pd.read_csv(args.csv_path)
    print(f"[Stage 1] Total samples: {len(df)}")

    train_idx, val_idx, test_idx = load_split_indices(args.index_path)
    print(f"[Stage 1] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    dataset = HierarchicalMoleculeDataset(
        df_or_path=df,
        num_l1=args.num_l1,
        smiles_col=args.smiles_col,
        label_l1_col=args.label_col,
        mode="l1",
        cache=bool(args.cache),
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_l1)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_l1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_l1)

    # 获取数据维度
    sample_graph, sample_fp, _ = dataset[0]
    node_in_dim = sample_graph.x.size(1)
    edge_in_dim = sample_graph.edge_attr.size(1)
    fp_dim = sample_fp.size(0)
    print(f"[Stage 1] Dims: node={node_in_dim}, edge={edge_in_dim}, fp={fp_dim}")

    # --- Semantic Init ---
    init_label_emb = None
    if args.semantic_init:
        if args.label_json_path is None:
            csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
            candidate = os.path.join(csv_dir, "label_space_L1.json")
            if not os.path.exists(candidate):
                candidate = os.path.join(os.path.dirname(csv_dir), "processed", "label_space_L1.json")
            args.label_json_path = candidate

        print(f"[Stage 1] Loading BioBERT semantic embeddings...")
        init_label_emb = generate_or_load_embeddings(
            label_json_path=args.label_json_path,
            cache_path=args.semantic_emb_cache,
            model_name=args.biobert_model,
            target_dim=args.label_emb_dim,
            use_descriptions=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # --- Model ---
    use_label_gcn = not args.no_label_gcn
    model = Stage1Model(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        fp_dim=fp_dim,
        num_l1=args.num_l1,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_out_dim=args.graph_out_dim,
        fp_hidden_dim=args.fp_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        final_dim=args.final_dim,
        pool_ratio=args.pool_ratio,
        label_emb_dim=args.label_emb_dim,
        label_gcn_layers=args.label_gcn_layers,
        use_label_gcn=use_label_gcn,
        init_label_embeddings=init_label_emb,
    ).to(device)
    print(f"[Stage 1] Label GCN: {'Enabled' if use_label_gcn else 'Disabled (learnable emb)'}")

    # --- Label Graph ---
    edge_index, edge_weight, _ = build_l1_label_graph(
        train_idx, df, args.label_col, args.num_l1, threshold=args.label_graph_threshold
    )
    model.set_label_graph(edge_index.to(device), edge_weight.to(device))
    print(f"[Stage 1] L1 Label Graph: {args.num_l1} nodes, {edge_index.size(1)} edges")

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_graph, fps, labels in tqdm(train_loader, desc=f"[S1] Epoch {epoch}"):
            batch_graph = batch_graph.to(device)
            fps = fps.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            h_c, logits = model(batch_graph, fps, return_emb=True)

            loss_bce = criterion(logits, labels)

            # Label-Anchored Contrastive Loss (唯一辅助损失)
            W_label = model.get_label_embeddings()
            loss_anchor = label_anchored_contrastive_loss(
                h_c, labels, W_label, args.temperature
            )

            loss = loss_bce + args.lambda_anchor * loss_anchor

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        scheduler.step()
        total_loss /= len(train_set)

        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, args.threshold)

        print(
            f"[S1 Epoch {epoch}] Loss={total_loss:.4f} | "
            f"Val: Loss={val_loss:.4f} F1={val_metrics['f1']:.4f} "
            f"MacroF1={val_metrics['macro_f1']:.4f} AUPRC={val_metrics['auprc']:.4f} "
            f"Prec={val_metrics['prec']:.4f} Rec={val_metrics['rec']:.4f}"
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → Saved best (F1={best_val_f1:.4f})")

    # --- Test ---
    print(f"\n[Stage 1] Best epoch: {best_epoch} (Val F1={best_val_f1:.4f})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    test_loss, test_metrics = evaluate(model, test_loader, device, criterion, args.threshold)
    print(f"[Stage 1] Test: Loss={test_loss:.4f} F1={test_metrics['f1']:.4f} "
          f"MacroF1={test_metrics['macro_f1']:.4f} AUPRC={test_metrics['auprc']:.4f} "
          f"Prec={test_metrics['prec']:.4f} Rec={test_metrics['rec']:.4f}")

    # Save summary
    summary = {
        "stage": 1,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "ckpt_path": ckpt_path,
    }
    summary_path = os.path.join(args.ckpt_dir, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Stage 1] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

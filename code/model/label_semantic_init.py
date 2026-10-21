"""
label_semantic_init.py — BioBERT 语义初始化 (L1+L2 共 157 标签)

用 BioBERT 为 KEGG 代谢通路标签生成语义嵌入, 作为 Hierarchy Label GCN 的初始嵌入。

与 model_fixed/label_semantic_init.py 完全兼容,
额外支持 L2 (146 标签) 的语义嵌入生成。
"""

import os
import json
import argparse
from typing import Dict, List, Optional

import torch
import numpy as np


# KEGG L1 代谢通路的增强描述
KEGG_L1_DESCRIPTIONS: Dict[int, str] = {
    0: "Carbohydrate metabolism: glycolysis, gluconeogenesis, pentose phosphate pathway, "
       "TCA cycle, starch and sucrose metabolism, fructose and mannose metabolism",
    1: "Energy metabolism: oxidative phosphorylation, photosynthesis, carbon fixation, "
       "nitrogen metabolism, sulfur metabolism, methane metabolism",
    2: "Lipid metabolism: fatty acid biosynthesis and degradation, glycerolipid metabolism, "
       "sphingolipid metabolism, steroid biosynthesis, arachidonic acid metabolism",
    3: "Nucleotide metabolism: purine metabolism, pyrimidine metabolism, nucleotide "
       "biosynthesis and salvage pathways",
    4: "Amino acid metabolism: alanine, aspartate, glutamate, glycine, serine, threonine, "
       "cysteine, methionine, valine, leucine, isoleucine, lysine, arginine, proline, "
       "histidine, phenylalanine, tyrosine, tryptophan metabolism",
    5: "Metabolism of other amino acids: beta-alanine, taurine, D-amino acid, glutathione, "
       "selenocompound metabolism, cyanoamino acid metabolism",
    6: "Glycan biosynthesis and metabolism: N-glycan, O-glycan, glycosaminoglycan, "
       "glycosphingolipid biosynthesis, lipopolysaccharide biosynthesis",
    7: "Metabolism of cofactors and vitamins: thiamine, riboflavin, vitamin B6, "
       "nicotinate, pantothenate, biotin, folate, retinol, porphyrin metabolism, "
       "ubiquinone and coenzyme Q biosynthesis",
    8: "Metabolism of terpenoids and polyketides: terpenoid backbone biosynthesis, "
       "monoterpenoid, diterpenoid, carotenoid, sesquiterpenoid biosynthesis, "
       "type I and II polyketide biosynthesis",
    9: "Biosynthesis of other secondary metabolites: phenylpropanoid, flavonoid, "
       "isoflavonoid, stilbenoid, alkaloid, indole alkaloid, isoquinoline alkaloid "
       "biosynthesis, penicillin and cephalosporin biosynthesis",
    10: "Xenobiotics biodegradation and metabolism: benzoate, aminobenzoate, fluorobenzoate "
        "degradation, chloroalkane, chlorocyclohexane, nitrotoluene, styrene, "
        "drug metabolism by cytochrome P450 and other enzymes",
}


def get_label_texts(
    label_json_path: str,
    use_descriptions: bool = True,
) -> List[str]:
    """
    从 label_space JSON 获取标签文本列表。

    Args:
        label_json_path: label_space_L1.json 或 label_space_L2.json 的路径
        use_descriptions: 是否使用增强描述 (仅 L1 支持)

    Returns:
        texts: [L] 每个标签对应的文本
    """
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_classes = data["n_classes"]
    index_to_name = data.get("index_to_name", {})

    texts = []
    for i in range(n_classes):
        name = index_to_name.get(str(i), f"pathway_{i}")

        if use_descriptions and i in KEGG_L1_DESCRIPTIONS and n_classes <= 11:
            texts.append(KEGG_L1_DESCRIPTIONS[i])
        else:
            # L2 或无描述时: 使用 "KEGG metabolic pathway: <name>"
            texts.append(f"KEGG metabolic pathway: {name}")

    return texts


def generate_biobert_embeddings(
    texts: List[str],
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    target_dim: Optional[int] = 256,
    pooling: str = "cls",
    device: str = "cpu",
) -> torch.Tensor:
    """
    用 BioBERT 生成文本的语义嵌入。

    Args:
        texts:      [L] 标签文本列表
        model_name: HuggingFace 模型名
        target_dim: 目标维度 (如果不为 None, 通过 PCA 降维)
        pooling:    "cls" | "mean"
        device:     "cpu" | "cuda"

    Returns:
        embeddings: [L, target_dim] FloatTensor
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"[Semantic Init] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            ).to(device)

            outputs = model(**inputs)

            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

            embeddings.append(emb.squeeze(0).cpu())

    embeddings = torch.stack(embeddings, dim=0)  # [L, 768]
    print(f"[Semantic Init] Raw embeddings shape: {embeddings.shape}")

    # 降维到 target_dim
    if target_dim is not None and embeddings.size(1) != target_dim:
        embeddings = _pca_project(embeddings, target_dim)
        print(f"[Semantic Init] Projected to: {embeddings.shape}")

    # L2 normalize
    embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)

    return embeddings


def _pca_project(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """用 PCA 或随机正交映射降维."""
    L, D = x.shape

    if L >= target_dim:
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(x_centered, full_matrices=False)
        return x_centered @ V[:target_dim, :].t()
    else:
        # L < target_dim: 用随机正交映射
        proj = torch.randn(D, target_dim)
        q, _ = torch.linalg.qr(proj)
        projected = x @ q
        return projected


def generate_or_load_embeddings(
    label_json_path: str,
    cache_path: Optional[str] = None,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    target_dim: int = 256,
    use_descriptions: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    生成或从缓存加载语义嵌入的便捷接口。
    """
    # 尝试加载缓存
    if cache_path and os.path.exists(cache_path):
        print(f"[Semantic Init] Loading cached embeddings from {cache_path}")
        emb = torch.load(cache_path, map_location="cpu")
        if emb.size(1) == target_dim:
            return emb
        else:
            print(f"[Semantic Init] Cache dim mismatch ({emb.size(1)} vs {target_dim}), regenerating...")

    # 生成
    texts = get_label_texts(label_json_path, use_descriptions=use_descriptions)
    print(f"[Semantic Init] Generating embeddings for {len(texts)} labels:")
    for i, t in enumerate(texts[:5]):
        print(f"  [{i}] {t[:80]}{'...' if len(t) > 80 else ''}")
    if len(texts) > 5:
        print(f"  ... ({len(texts) - 5} more)")

    embeddings = generate_biobert_embeddings(
        texts,
        model_name=model_name,
        target_dim=target_dim,
        device=device,
    )

    # 缓存
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"[Semantic Init] Saved embeddings to {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BioBERT semantic embeddings for KEGG pathway labels"
    )
    parser.add_argument("--label_json", type=str, required=True,
                        help="Path to label_space_L1.json or L2.json")
    parser.add_argument("--output", type=str, default="label_emb_biobert.pt",
                        help="Output .pt file")
    parser.add_argument("--model_name", type=str,
                        default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument("--target_dim", type=int, default=256)
    parser.add_argument("--no_descriptions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])

    args = parser.parse_args()

    embeddings = generate_or_load_embeddings(
        label_json_path=args.label_json,
        cache_path=args.output,
        model_name=args.model_name,
        target_dim=args.target_dim,
        use_descriptions=not args.no_descriptions,
        device=args.device,
    )

    print(f"\n✓ Final embeddings: {embeddings.shape}")
    print(f"  Norm range: [{embeddings.norm(dim=-1).min():.4f}, {embeddings.norm(dim=-1).max():.4f}]")

    sim = embeddings @ embeddings.t()
    print(f"\n  Cosine similarity stats:")
    print(f"    Diagonal mean: {sim.diag().mean():.4f}")
    off_diag = sim - torch.eye(sim.size(0)) * sim.diag()
    print(f"    Off-diagonal mean: {off_diag.sum() / (sim.numel() - sim.size(0)):.4f}")

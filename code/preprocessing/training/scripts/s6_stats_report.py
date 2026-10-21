"""s6_stats_report.py — 生成数据集统计报告 (Markdown + 可选图)。

产物: reports/dataset_stats.md, reports/figures/*.png
用法: python scripts/s6_stats_report.py
"""
from __future__ import annotations
import sys
from collections import Counter

import numpy as np
import pandas as pd

from common import load_config, rp, load_json


def parse_labels(s):
    s = str(s)
    if not s or s == "nan":
        return []
    return [int(x) for x in s.split(",") if x != ""]


def main():
    cfg = load_config()
    processed = rp(cfg, "processed")
    reports = rp(cfg, "reports")
    figdir = reports / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fp = processed / "kegg2026_hier.csv"
    if not fp.exists():
        sys.exit("缺少 kegg2026_hier.csv, 请先运行 s4_build_labels.py")
    df = pd.read_csv(fp)
    l1_space = load_json(processed / "label_space_L1.json")
    l2_space = load_json(processed / "label_space_L2.json")

    n = len(df)
    l1 = df["labels_L1"].apply(parse_labels)
    l2 = df["labels_L2"].apply(parse_labels)
    l1_card = l1.apply(len)
    l2_card = l2.apply(len)

    l1_counter = Counter()
    for xs in l1:
        l1_counter.update(xs)
    l2_counter = Counter()
    for xs in l2:
        l2_counter.update(xs)

    lines = []
    lines.append("# KEGG 2026 分层数据集统计报告\n")
    lines.append(f"- 化合物总数: **{n}**")
    lines.append(f"- L1 超类数: **{l1_space['n_classes']}**")
    lines.append(f"- L2 通路数: **{l2_space['n_classes']}** (min_count={l2_space['min_count']})")
    lines.append(f"- L1 平均标签数(cardinality): {l1_card.mean():.2f} (max {l1_card.max()})")
    lines.append(f"- L2 平均标签数(cardinality): {l2_card.mean():.2f} (max {l2_card.max()})")
    lines.append(f"- 重原子数: mean {df['num_heavy'].mean():.1f}, median {df['num_heavy'].median():.0f}, max {df['num_heavy'].max()}\n")

    lines.append("## L1 超类分布\n")
    lines.append("| idx | 超类 | 化合物数 | 占比 |")
    lines.append("|----|------|--------|------|")
    idx2name = l1_space["index_to_name"]
    for i in range(l1_space["n_classes"]):
        c = l1_counter.get(i, 0)
        lines.append(f"| {i} | {idx2name[str(i)]} | {c} | {c/n*100:.1f}% |")

    lines.append("\n## L2 通路 Top-20 (按化合物数)\n")
    lines.append("| map | 名称 | 数量 |")
    lines.append("|-----|------|------|")
    idx2map = l2_space["index_to_map"]
    m2name = l2_space["map_to_name"]
    top = sorted(l2_counter.items(), key=lambda x: -x[1])[:20]
    for li, c in top:
        m = idx2map[str(li)]
        lines.append(f"| {m} | {m2name.get(m, '')} | {c} |")

    (reports / "dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] 报告 -> {reports / 'dataset_stats.md'}")

    # 图 (可选)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        xs = list(range(l1_space["n_classes"]))
        ys = [l1_counter.get(i, 0) for i in xs]
        ax.bar(xs, ys, color="#4C72B0")
        ax.set_xticks(xs)
        ax.set_xlabel("L1 super-class index")
        ax.set_ylabel("#compounds")
        ax.set_title("L1 label distribution")
        fig.tight_layout()
        fig.savefig(figdir / "l1_distribution.png", dpi=150)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(l2_card, bins=range(0, int(l2_card.max()) + 2), color="#55A868")
        ax.set_xlabel("#L2 labels per compound")
        ax.set_ylabel("#compounds")
        ax.set_title("L2 label cardinality")
        fig.tight_layout()
        fig.savefig(figdir / "l2_cardinality.png", dpi=150)
        print(f"[ok] 图 -> {figdir}")
    except Exception as e:
        print(f"[warn] 绘图跳过: {e}")


if __name__ == "__main__":
    main()

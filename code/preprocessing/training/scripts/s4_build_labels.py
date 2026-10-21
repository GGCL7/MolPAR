"""s4_build_labels.py — 构建分层多标签。

L1: 11 个超类(索引 0-10, 与旧版严格对齐, 向后兼容历史结果)
L2: 超类下的具体 KEGG map (按 l2_min_count 过滤稀有类)
父子一致性: 任一 L2 标签必蕴含其 L1 父标签。

产物:
  processed/kegg2026_hier.csv   smiles,kegg_id,inchikey,num_heavy,labels_L1,labels_L2
  processed/label_space_L1.json
  processed/label_space_L2.json
用法: python scripts/s4_build_labels.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict, Counter

import pandas as pd

from common import load_config, rp, save_json


def build_map_hierarchy(brite: dict, cfg: dict):
    """返回 map_id(mapXXXXX) -> (l1_index, l1_name, map_name)。"""
    l1_map = cfg["labels"]["l1_kegg_superclasses"]          # {idx: name}
    name2idx = {v: int(k) for k, v in l1_map.items()}
    exclude = set(cfg["labels"]["exclude_superclasses"])

    def find_metabolism(node):
        for c in node.get("children", []):
            if c.get("name") == "Metabolism":
                return c
        for c in node.get("children", []):
            r = find_metabolism(c)
            if r:
                return r
        return None

    meta = find_metabolism(brite)
    result = {}
    for sc in meta.get("children", []):
        sc_name = sc.get("name", "")
        if sc_name in exclude or sc_name not in name2idx:
            continue
        idx = name2idx[sc_name]
        for leaf in sc.get("children", []):
            parts = leaf.get("name", "").split(None, 1)
            code = parts[0]
            mname = parts[1] if len(parts) > 1 else code
            if code.isdigit():
                result[f"map{code}"] = (idx, sc_name, mname)
    return result


def main():
    cfg = load_config()
    raw = rp(cfg, "raw")
    interim = rp(cfg, "interim")
    processed = rp(cfg, "processed")

    cleaned = interim / "cleaned_compounds.csv"
    if not cleaned.exists():
        sys.exit("缺少 cleaned_compounds.csv, 请先运行 s3_clean.py")
    df = pd.read_csv(cleaned)

    brite = json.loads((raw / "brite_br08901.json").read_text(encoding="utf-8"))
    map_hier = build_map_hierarchy(brite, cfg)   # map -> (l1idx, l1name, mapname)

    # 读 compound -> maps
    cpd2maps = defaultdict(set)
    for line in (raw / "compound_pathway.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cpd, path = line.split("\t")
        cid = cpd.split(":")[-1]
        mid = path.split(":")[-1]
        if mid in map_hier:
            cpd2maps[cid].add(mid)

    # 展开合并的 kegg_id(去重时用 '|' 连接), 收集每个化合物的 maps
    def maps_for_row(kegg_id_field):
        ms = set()
        for cid in str(kegg_id_field).split("|"):
            ms |= cpd2maps.get(cid, set())
        return ms

    df["_maps"] = df["kegg_id"].apply(maps_for_row)
    df = df[df["_maps"].apply(len) > 0].reset_index(drop=True)

    # L2 频次过滤
    map_counter = Counter()
    for ms in df["_maps"]:
        map_counter.update(ms)
    min_c = cfg["labels"]["l2_min_count"]
    kept_maps = sorted([m for m, c in map_counter.items() if c >= min_c])
    l2_index = {m: i for i, m in enumerate(kept_maps)}
    print(f"[info] L2 通路: 总 {len(map_counter)} 个, 过滤(>= {min_c}) 后保留 {len(kept_maps)} 个")

    # 生成标签
    l1_rows, l2_rows = [], []
    for ms in df["_maps"]:
        ms_kept = [m for m in ms if m in l2_index]
        l2_ids = sorted(l2_index[m] for m in ms_kept)
        l1_ids = sorted({map_hier[m][0] for m in ms_kept})  # 父类一致性: 由保留的L2回溯
        l1_rows.append(",".join(map(str, l1_ids)))
        l2_rows.append(",".join(map(str, l2_ids)))

    df["labels_L1"] = l1_rows
    df["labels_L2"] = l2_rows
    # 过滤掉因 L2 频次过滤后变空的化合物
    out = df[df["labels_L1"].str.len() > 0].copy()

    final = out[["canonical_smiles", "kegg_id", "inchikey", "num_heavy",
                 "labels_L1", "labels_L2"]].rename(columns={"canonical_smiles": "smiles"})
    fp = processed / "kegg2026_hier.csv"
    final.to_csv(fp, index=False)

    # label spaces
    l1_space = {str(k): v for k, v in cfg["labels"]["l1_kegg_superclasses"].items()}
    save_json({"n_classes": len(l1_space), "index_to_name": l1_space},
              processed / "label_space_L1.json")
    l2_space = {
        "n_classes": len(kept_maps),
        "min_count": min_c,
        "index_to_map": {str(i): m for m, i in l2_index.items()},
        "map_to_name": {m: map_hier[m][2] for m in kept_maps},
        "map_to_l1_index": {m: map_hier[m][0] for m in kept_maps},
    }
    save_json(l2_space, processed / "label_space_L2.json")

    print(f"[done] 最终化合物 {len(final)} -> {fp}")
    print(f"       L1={len(l1_space)} 类, L2={len(kept_maps)} 类")


if __name__ == "__main__":
    main()

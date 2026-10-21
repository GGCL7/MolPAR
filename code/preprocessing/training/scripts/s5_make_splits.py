"""s5_make_splits.py — 生成三类数据划分 (替代旧的固定 data_index.txt)。

1. random x5 seeds     : 80/10/10 随机划分
2. scaffold            : Bemis-Murcko 骨架分组, 骨架不跨集(顶刊标准)
3. temporal            :
     T-A (主): 旧 4192(历史快照)=train/val, 新增化合物=test -> 模拟"预测未标注新分子"
     T-B (附): 以 KEGG C-number 作时间代理, 新化合物做 test (稳健性检验)

产物: processed/splits/{random_seed*.json, scaffold.json, temporal.json}
用法: python scripts/s5_make_splits.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

from common import load_config, rp, save_json

RDLogger.DisableLog("rdApp.*")


def random_split(n, seed, tr, va):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n * tr)
    n_va = int(n * va)
    return {"train": sorted(idx[:n_tr]),
            "val": sorted(idx[n_tr:n_tr + n_va]),
            "test": sorted(idx[n_tr + n_va:])}


def scaffold_split(smiles_list, tr, va, seed=0):
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        try:
            scf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
        except Exception:
            scf = ""
        groups[scf or f"_singleton_{i}"].append(i)
    # 大骨架优先放入 train, 保证 test 由较少见骨架组成
    buckets = sorted(groups.values(), key=lambda x: (-len(x), x[0]))
    n = len(smiles_list)
    n_tr, n_va = int(n * tr), int(n * va)
    train, val, test = [], [], []
    for b in buckets:
        if len(train) + len(b) <= n_tr:
            train += b
        elif len(val) + len(b) <= n_va:
            val += b
        else:
            test += b
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def temporal_A(df, legacy_csv: Path, tr_of_old=0.9):
    """旧4192=train/val, 其余=test。返回 split 或 None。"""
    if not legacy_csv.exists():
        sys.stderr.write(f"[warn] 未找到旧数据 {legacy_csv}, 跳过 T-A\n")
        return None
    old = pd.read_csv(legacy_csv)
    old_keys = set()
    for smi in old["smiles"].astype(str):
        m = Chem.MolFromSmiles(smi)
        if m:
            try:
                old_keys.add(inchi.MolToInchiKey(m))
            except Exception:
                pass
    is_old = df["inchikey"].isin(old_keys).values
    old_idx = [i for i, v in enumerate(is_old) if v]
    new_idx = [i for i, v in enumerate(is_old) if not v]
    random.Random(0).shuffle(old_idx)
    n_tr = int(len(old_idx) * tr_of_old)
    return {"train": sorted(old_idx[:n_tr]),
            "val": sorted(old_idx[n_tr:]),
            "test": sorted(new_idx),
            "_note": f"T-A: old(train+val)={len(old_idx)}, new(test)={len(new_idx)}"}


def temporal_B(df, test_frac):
    """C-number 时间代理: 编号大(较新)的做 test。"""
    def min_cnum(field):
        nums = []
        for cid in str(field).split("|"):
            d = "".join(ch for ch in cid if ch.isdigit())
            if d:
                nums.append(int(d))
        return min(nums) if nums else 10**9
    order = sorted(range(len(df)), key=lambda i: min_cnum(df.iloc[i]["kegg_id"]))
    n = len(order)
    n_test = int(n * test_frac)
    train_pool = order[:n - n_test]
    test = order[n - n_test:]
    random.Random(0).shuffle(train_pool)
    n_va = int(len(train_pool) * 0.111)  # ~10% of total
    return {"train": sorted(train_pool[n_va:]),
            "val": sorted(train_pool[:n_va]),
            "test": sorted(test),
            "_note": f"T-B: C-number 代理, test=较新 {len(test)}"}


def main():
    cfg = load_config()
    processed = rp(cfg, "processed")
    fp = processed / "kegg2026_hier.csv"
    if not fp.exists():
        sys.exit("缺少 kegg2026_hier.csv, 请先运行 s4_build_labels.py")
    df = pd.read_csv(fp)
    n = len(df)
    sp = cfg["splits"]
    outdir = processed / "splits"
    outdir.mkdir(parents=True, exist_ok=True)

    # random x5
    for seed in sp["random_seeds"]:
        s = random_split(n, seed, sp["train_frac"], sp["val_frac"])
        save_json(s, outdir / f"random_seed{seed}.json")
    print(f"[ok] random x{len(sp['random_seeds'])} 划分")

    # scaffold
    s = scaffold_split(df["smiles"].tolist(), sp["train_frac"], sp["val_frac"])
    save_json(s, outdir / "scaffold.json")
    print(f"[ok] scaffold 划分 (train/val/test = {len(s['train'])}/{len(s['val'])}/{len(s['test'])})")

    # temporal
    legacy = (processed.parent / sp["legacy_old_csv"]).resolve()
    tA = temporal_A(df, legacy)
    tB = temporal_B(df, sp["cnumber_test_frac"])
    save_json({"T_A": tA, "T_B": tB}, outdir / "temporal.json")
    if tA:
        print(f"[ok] T-A: {tA['_note']}")
    print(f"[ok] T-B: {tB['_note']}")
    print(f"[done] 划分文件 -> {outdir}")


if __name__ == "__main__":
    main()

"""s2_build_structures.py — 解析 raw/mol/*.mol 为 canonical SMILES + InChIKey。

产物: interim/raw_compounds.csv
  列: kegg_id, canonical_smiles, inchikey, num_heavy, elements
用法: python scripts/s2_build_structures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

from common import load_config, rp

RDLogger.DisableLog("rdApp.*")


def main():
    cfg = load_config()
    raw = rp(cfg, "raw")
    interim = rp(cfg, "interim")
    mol_dir = raw / "mol"

    files = sorted(mol_dir.glob("*.mol"))
    if not files:
        sys.exit("未找到 mol 文件，请先运行 s1_fetch_kegg.py")
    print(f"[info] 解析 {len(files)} 个 mol 文件 ...")

    rows = []
    n_fail = 0
    for i, f in enumerate(files, 1):
        cid = f.stem
        try:
            mol = Chem.MolFromMolBlock(f.read_text(encoding="utf-8"), sanitize=True)
        except Exception:
            mol = None
        if mol is None:
            n_fail += 1
            continue
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
            ikey = inchi.MolToInchiKey(mol)
            heavy = mol.GetNumHeavyAtoms()
            elems = "".join(sorted({a.GetSymbol() for a in mol.GetAtoms()}))
        except Exception:
            n_fail += 1
            continue
        if not smi or not ikey:
            n_fail += 1
            continue
        rows.append({"kegg_id": cid, "canonical_smiles": smi,
                     "inchikey": ikey, "num_heavy": heavy, "elements": elems})
        if i % 2000 == 0:
            print(f"  ... {i}/{len(files)}")

    df = pd.DataFrame(rows)
    out = interim / "raw_compounds.csv"
    df.to_csv(out, index=False)
    print(f"[done] 解析成功 {len(df)}, 失败 {n_fail} -> {out}")


if __name__ == "__main__":
    main()

"""s3_clean.py — 清洗与去重。

清洗规则(每步统计淘汰数, 写入报告):
  1. 去盐 + (可选)电荷中和
  2. 去含通配符 R/*/[*] 的 generic/markush 结构
  3. 去混合物(去盐后仍多组分)
  4. 原子数下/上限过滤
  5. (可选)剔除金属/无机: 元素白名单 + 必须含碳
  6. 按 InChIKey 去重(同结构多 CPD 合并, kegg_id 用 '|' 连接)

产物: interim/cleaned_compounds.csv
用法: python scripts/s3_clean.py
"""
from __future__ import annotations
import sys
from collections import OrderedDict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.SaltRemover import SaltRemover

from common import load_config, rp

RDLogger.DisableLog("rdApp.*")

_saltremover = SaltRemover()


def neutralize(mol):
    """简单电荷中和(加/减 H)。"""
    patt = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(patt)
    if not at_matches:
        return mol
    for (idx,) in at_matches:
        atom = mol.GetAtomWithIdx(idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(max(0, hcount - chg))
        atom.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def main():
    cfg = load_config()
    interim = rp(cfg, "interim")
    cc = cfg["clean"]

    src = interim / "raw_compounds.csv"
    if not src.exists():
        sys.exit("缺少 raw_compounds.csv, 请先运行 s2_build_structures.py")
    df = pd.read_csv(src)
    stats = OrderedDict()
    stats["输入"] = len(df)

    allow = set(cc["organic_elements"])
    kept = []
    removed_wild = removed_mix = removed_size = removed_metal = removed_parse = 0

    for _, r in df.iterrows():
        smi = str(r["canonical_smiles"])
        cid = r["kegg_id"]

        # 通配符
        if cc["remove_generic_wildcards"] and ("*" in smi or "R" in smi.replace("Br", "").replace("Ra", "")):
            # 更稳妥: 用 mol 检查是否含 dummy atom
            pass
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            removed_parse += 1
            continue
        # dummy/通配原子
        if cc["remove_generic_wildcards"] and any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            removed_wild += 1
            continue

        # 去盐
        try:
            mol = _saltremover.StripMol(mol, dontRemoveEverything=True)
        except Exception:
            pass
        # 混合物
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if cc["remove_mixtures"] and len(frags) > 1:
            removed_mix += 1
            continue

        if cc["neutralize"]:
            mol = neutralize(mol)

        # 原子数
        nheavy = mol.GetNumHeavyAtoms()
        if nheavy < cc["min_heavy_atoms"] or nheavy > cc["max_heavy_atoms"]:
            removed_size += 1
            continue

        # 金属/无机
        if cc["remove_metals_inorganic"]:
            syms = {a.GetSymbol() for a in mol.GetAtoms()}
            if not syms.issubset(allow):
                removed_metal += 1
                continue
            if cc.get("require_carbon", True) and "C" not in syms:
                removed_metal += 1
                continue

        try:
            can = Chem.MolToSmiles(mol, canonical=True)
            ikey = inchi.MolToInchiKey(mol)
        except Exception:
            removed_parse += 1
            continue
        if not can or not ikey:
            removed_parse += 1
            continue
        kept.append({"kegg_id": cid, "canonical_smiles": can, "inchikey": ikey,
                     "num_heavy": mol.GetNumHeavyAtoms(),
                     "elements": "".join(sorted(syms if cc["remove_metals_inorganic"] else {a.GetSymbol() for a in mol.GetAtoms()}))})

    stats["解析失败"] = removed_parse
    stats["含通配符去除"] = removed_wild
    stats["混合物去除"] = removed_mix
    stats["原子数越界去除"] = removed_size
    stats["金属/无机去除"] = removed_metal

    kdf = pd.DataFrame(kept)
    # 按 InChIKey 去重, 合并 kegg_id
    merged = (kdf.groupby("inchikey")
                 .agg(kegg_id=("kegg_id", lambda s: "|".join(sorted(set(s)))),
                      canonical_smiles=("canonical_smiles", "first"),
                      num_heavy=("num_heavy", "first"),
                      elements=("elements", "first"))
                 .reset_index())
    stats["去重前"] = len(kdf)
    stats["去重后(最终)"] = len(merged)

    out = interim / "cleaned_compounds.csv"
    merged.to_csv(out, index=False)

    print("===== 清洗统计 =====")
    for k, v in stats.items():
        print(f"  {k:16s}: {v}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()

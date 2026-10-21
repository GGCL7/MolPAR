"""s1_fetch_kegg.py — 从 KEGG REST API 拉取原始数据 (带缓存/断点续跑)。

产物 (raw/):
  compound_pathway.tsv   化合物<->pathway map 映射
  pathway_list.tsv       所有 reference pathway (map) 的名称
  brite_br08901.json     KEGG pathway 分类层级树
  target_compounds.txt   落在 11 个 Metabolism 超类下的目标化合物列表
  mol/<Cid>.mol          每个目标化合物的 MOL 结构缓存
  mol_failed.txt         无结构/拉取失败的化合物

用法: python scripts/s1_fetch_kegg.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from common import load_config, rp, KeggClient, save_json


def parse_metabolism_maps(brite: dict, cfg: dict) -> set[str]:
    """从 br08901 层级树中提取 11 个超类下的所有 map id (形如 map00010)。"""
    l1_names = set(cfg["labels"]["l1_kegg_superclasses"].values())
    exclude = set(cfg["labels"]["exclude_superclasses"])
    maps: set[str] = set()

    def find_metabolism(node):
        for child in node.get("children", []):
            if child.get("name") == "Metabolism":
                return child
        # 兜底: 深搜
        for child in node.get("children", []):
            r = find_metabolism(child)
            if r:
                return r
        return None

    meta = find_metabolism(brite)
    if meta is None:
        raise RuntimeError("br08901 中未找到 'Metabolism' 分支")

    for superclass in meta.get("children", []):
        sc_name = superclass.get("name", "")
        if sc_name in exclude:
            continue
        if sc_name not in l1_names:
            # 记录但不纳入 (以防 KEGG 改名)
            sys.stderr.write(f"[note] Metabolism 下未匹配到配置的超类: {sc_name}\n")
            continue
        for leaf in superclass.get("children", []):
            code = leaf.get("name", "").split(None, 1)[0]  # '00010'
            if code.isdigit():
                maps.add(f"map{code}")
    return maps


def main():
    cfg = load_config()
    raw = rp(cfg, "raw")
    client = KeggClient(cfg)

    # 1) 通路列表
    f_plist = raw / "pathway_list.tsv"
    if not f_plist.exists():
        txt = client.get("/list/pathway")
        if not txt:
            sys.exit("拉取 /list/pathway 失败")
        f_plist.write_text(txt, encoding="utf-8")
        print(f"[ok] pathway_list.tsv ({txt.count(chr(10))} 行)")
    else:
        print("[skip] pathway_list.tsv 已存在")

    # 2) 化合物<->pathway 映射
    f_cp = raw / "compound_pathway.tsv"
    if not f_cp.exists():
        txt = client.get("/link/pathway/compound")
        if not txt:
            sys.exit("拉取 /link/pathway/compound 失败")
        f_cp.write_text(txt, encoding="utf-8")
        print(f"[ok] compound_pathway.tsv ({txt.count(chr(10))} 行)")
    else:
        print("[skip] compound_pathway.tsv 已存在")

    # 3) BRITE 层级
    f_brite = raw / "brite_br08901.json"
    if not f_brite.exists():
        txt = client.get("/get/br:br08901/json")
        if not txt:
            sys.exit("拉取 br:br08901 失败")
        f_brite.write_text(txt, encoding="utf-8")
        print("[ok] brite_br08901.json")
    else:
        print("[skip] brite_br08901.json 已存在")

    import json
    brite = json.loads(f_brite.read_text(encoding="utf-8"))
    meta_maps = parse_metabolism_maps(brite, cfg)
    print(f"[info] Metabolism 11 超类下共 {len(meta_maps)} 个 map")

    # 4) 目标化合物 = 链接到任一 metabolism map 的化合物
    targets: set[str] = set()
    for line in f_cp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cpd, path = line.split("\t")
        cid = cpd.split(":")[-1]           # C00031
        mid = path.split(":")[-1]          # map00010
        if mid in meta_maps:
            targets.add(cid)
    targets = sorted(targets)
    (raw / "target_compounds.txt").write_text("\n".join(targets), encoding="utf-8")
    print(f"[info] 目标化合物 (落在代谢通路下): {len(targets)}")

    # 5) 逐个下载 mol (断点续跑)
    mol_dir = raw / "mol"
    mol_dir.mkdir(exist_ok=True)
    failed_path = raw / "mol_failed.txt"
    known_failed = set()
    if failed_path.exists():
        known_failed = set(failed_path.read_text(encoding="utf-8").split())

    todo = [c for c in targets
            if not (mol_dir / f"{c}.mol").exists() and c not in known_failed]
    print(f"[info] 待下载 mol: {len(todo)} (已缓存 {len(targets) - len(todo) - len(known_failed)}, 已知失败 {len(known_failed)})")

    new_failed = []
    for i, cid in enumerate(todo, 1):
        txt = client.get(f"/get/cpd:{cid}/mol")
        if txt and "V2000" in txt or (txt and "$$$$" in txt):
            (mol_dir / f"{cid}.mol").write_text(txt, encoding="utf-8")
        elif txt and txt.strip():
            (mol_dir / f"{cid}.mol").write_text(txt, encoding="utf-8")
        else:
            new_failed.append(cid)
        if i % 200 == 0:
            print(f"  ... {i}/{len(todo)}")
            # 增量写入失败记录
            if new_failed:
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(new_failed) + "\n")
                new_failed = []
    if new_failed:
        with open(failed_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_failed) + "\n")

    n_ok = len(list(mol_dir.glob("*.mol")))
    print(f"[done] mol 缓存完成: {n_ok} 个结构")


if __name__ == "__main__":
    main()

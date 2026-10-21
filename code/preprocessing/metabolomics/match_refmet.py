#!/usr/bin/env python3
"""Resolve repository-supplied RefMet names to explicit structures.

Sum-composition lipids commonly have a RefMet class/formula but no SMILES or
InChIKey. They remain excluded from structure-based model inference.
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def lookup(name):
    url = "https://www.metabolomicsworkbench.org/rest/refmet/name/" + quote(name, safe="") + "/all"
    try:
        with urlopen(Request(url, headers={"User-Agent":"CRC-metabolomics-audit/1.0"}), timeout=45) as h:
            x = json.loads(h.read().decode())
        return x if isinstance(x, dict) else {}
    except Exception as e:
        return {"_error": type(e).__name__}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--studies", nargs="+", default=["ST003798","ST002787","ST002468"]); ap.add_argument("--workers", type=int, default=4); args = ap.parse_args()
    path = HERE / "metabolite_structure_map.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    cache_path = HERE / "refmet_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    names = sorted({r.get("refmet_name", "") for r in rows if r["study_id"] in args.studies and r.get("refmet_name") and r.get("refmet_name") not in cache})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(lookup, n):n for n in names}
        for i, fut in enumerate(as_completed(futures), 1):
            cache[futures[fut]] = fut.result()
            if i % 25 == 0: cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
    for r in rows:
        if r["study_id"] not in args.studies or r.get("smiles") or not r.get("refmet_name"): continue
        x = cache.get(r["refmet_name"], {})
        if x.get("smiles") and x.get("inchi_key"):
            r.update(structure_id=(f"CID:{x['pubchem_cid']}" if x.get("pubchem_cid") else f"RefMet:{x.get('refmet_id','')}"), smiles=x["smiles"], inchi_key=x["inchi_key"], identifier_type="RefMet", identification_level="repository RefMet structure", match_basis="Metabolomics Workbench RefMet REST", ambiguity="none reported by RefMet", include_primary="review", notes="Structure supplied for repository RefMet name; stereochemistry still requires study-level identification review")
        elif x and not x.get("_error"):
            r["ambiguity"] = "class or sum-composition; no unique RefMet structure"
            r["notes"] = "RefMet record lacks both SMILES and InChIKey; excluded from structure inference"
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({sid:{"records":sum(r["study_id"]==sid for r in rows),"structures":sum(r["study_id"]==sid and bool(r.get("smiles")) for r in rows)} for sid in args.studies}, indent=2))


if __name__ == "__main__": main()

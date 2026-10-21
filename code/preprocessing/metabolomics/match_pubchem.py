"""Polite first-pass PubChem structure matching for repository names.

Only exact-name resolver results are recorded. A failed or ambiguous resolver
response remains unverified and therefore cannot qualify a cohort.
"""
import argparse, csv, json, time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent

def resolve(name):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + quote(name, safe="") + "/property/CanonicalSMILES,InChIKey/JSON"
    req = Request(url, headers={"User-Agent": "CRC-metabolomics-audit/1.0 (mailto:research@example.org)"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as h:
                obj = json.loads(h.read().decode("utf-8"))
            props = obj.get("PropertyTable", {}).get("Properties", [])
            if len(props) != 1: return None, "ambiguous_or_missing"
            p = props[0]
            if not p.get("ConnectivitySMILES") or not p.get("InChIKey"): return None, "missing_structure"
            return p, "exact_name_pubchem"
        except Exception:
            if attempt == 2: return None, "resolver_error"
            time.sleep(2 ** attempt)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--studies", nargs="+", default=["ST000284"]); ap.add_argument("--delay", type=float, default=1.0); ap.add_argument("--max-new", type=int, default=100000); args = ap.parse_args()
    path = ROOT / "metabolite_structure_map.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    cache_path = ROOT / "pubchem_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    new_count = 0
    for row in rows:
        if row["study_id"] not in args.studies or not row.get("original_name"): continue
        name = row["original_name"]
        if name not in cache:
            if new_count >= args.max_new: continue
            p, status = resolve(name); cache[name] = {"property": p, "status": status}; cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n"); time.sleep(args.delay)
            new_count += 1
        c = cache[name]
        if c["property"]:
            p = c["property"]; row.update(structure_id=f"CID:{p['CID']}", smiles=p["ConnectivitySMILES"], inchi_key=p["InChIKey"], identification_level="PubChem exact-name", match_basis="PubChem PUG REST", ambiguity="unresolved stereochemistry possible", include_primary="review")
        else:
            row["notes"] = f"PubChem status: {c['status']}"
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"rows": len(rows), "cached_names": len(cache), "matched_rows": sum(bool(r.get("smiles")) for r in rows if r["study_id"] in args.studies)}, ensure_ascii=False))

if __name__ == "__main__": raise SystemExit(main())

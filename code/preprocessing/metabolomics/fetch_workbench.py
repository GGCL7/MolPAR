"""Fetch public Metabolomics Workbench REST records for the audit.

The files under raw/ are source snapshots, not analysis-ready matrices. The
parser intentionally leaves subject identity and chemical structure verification
explicitly unresolved when the repository does not provide those fields.
"""
import csv, json, re, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
STUDIES = ["ST000284", "ST003798", "ST002787", "ST002468"]

def get_json(study, output):
    url = f"https://www.metabolomicsworkbench.org/rest/study/study_id/{study}/{output}"
    req = Request(url, headers={"User-Agent": "CRC-metabolomics-audit/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=90) as h:
                return json.loads(h.read().decode("utf-8"))
        except Exception:
            if attempt == 2: raise
            time.sleep(2 ** attempt)

def parse_factors(text):
    out = {}
    for item in str(text).split("|"):
        if ":" not in item: continue
        key, value = item.split(":", 1)
        out[key.strip().lower().replace(" ", "_")] = value.strip()
    return out

def fetch():
    RAW.mkdir(exist_ok=True)
    for study in STUDIES:
        for output in ("summary", "allfactors", "data", "metabolites"):
            path = RAW / f"{study}_{output}.json"
            if not path.exists():
                path.write_text(json.dumps(get_json(study, output), ensure_ascii=False, indent=2) + "\n")
                time.sleep(1)

def parse():
    sample_rows, structure_rows = [], []
    existing_path = ROOT / "metabolite_structure_map.csv"
    existing_rows = list(csv.DictReader(existing_path.open(encoding="utf-8", newline=""))) if existing_path.exists() else []
    existing = {(r.get("study_id", ""), r.get("original_id", ""), r.get("original_name", "")): r for r in existing_rows}
    manifest = list(csv.DictReader((ROOT / "study_manifest.csv").open(encoding="utf-8", newline="")))
    by_id = {r["study_id"]: r for r in manifest}
    for study in STUDIES:
        factors = json.loads((RAW / f"{study}_allfactors.json").read_text(encoding="utf-8"))
        data = json.loads((RAW / f"{study}_data.json").read_text(encoding="utf-8"))
        sample_ids = set()
        for row in factors.values():
            sid = row.get("local_sample_id", "")
            sample_ids.add(sid)
            f = parse_factors(row.get("factors", ""))
            group = f.get("patient_group") or f.get("group") or f.get("group_type") or ""
            group_l = group.lower()
            if "crc" in group_l or "cancer" in group_l: disease = "CRC"
            elif "healthy" in group_l or "control" in group_l or group_l in {"ctrl", "ctl", "hc"}: disease = "healthy"
            elif "adenoma" in group_l or group_l == "aa" or group_l == "ad": disease = "adenoma"
            else: disease = group
            age = f.get("age_at_consent") or f.get("age(years)") or f.get("age") or ""
            sex = f.get("gender") or f.get("sex") or ""
            batch = f.get("batch", "")
            sample_rows.append({"study_id": study, "sample_id": sid, "subject_id": "", "disease_group": disease, "age": age, "sex": sex, "bmi": f.get("bmi", ""), "medication": "", "treatment": "", "batch": batch, "source_file": f.get("raw_file_name", "")})
        # Each data record is one metabolite with a DATA mapping of local sample IDs.
        for rec in data.values():
            name = rec.get("metabolite_name", "")
            original_id = rec.get("metabolite_id", "")
            old = existing.get((study, original_id, name), {})
            structure_rows.append({"study_id": study, "original_id": original_id, "original_name": name, "refmet_name": rec.get("refmet_name", ""), "structure_id": old.get("structure_id", ""), "smiles": old.get("smiles", ""), "inchi_key": old.get("inchi_key", ""), "identifier_type": old.get("identifier_type", "Workbench metabolite ID"), "identification_level": old.get("identification_level", "repository name only"), "match_basis": old.get("match_basis", "Metabolomics Workbench REST"), "ambiguity": old.get("ambiguity", "unverified"), "existing_pathway_annotation": old.get("existing_pathway_annotation", ""), "include_primary": old.get("include_primary", ""), "notes": old.get("notes", "Unique chemical structure and stereochemistry require external verification")})
        m = by_id[study]
        m["matrix_status"] = "downloaded" if sample_ids and data else "unavailable"
        m["structure_status"] = f"{len(data)} repository records; structures unverified"
        m["independent_subjects_status"] = "unverified: local sample IDs only"
        m["processing_status"] = "metadata retrieved; processing audit pending"
        m["batch_status"] = "reported" if any(r.get("batch") for r in sample_rows if r["study_id"] == study) else "not reported"
    with (ROOT / "sample_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["study_id","sample_id","subject_id","disease_group","age","sex","bmi","medication","treatment","batch","source_file"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sample_rows)
    with (ROOT / "metabolite_structure_map.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["study_id","original_id","original_name","refmet_name","structure_id","smiles","inchi_key","identifier_type","identification_level","match_basis","ambiguity","existing_pathway_annotation","include_primary","notes"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(structure_rows)
    with (ROOT / "study_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(manifest[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(manifest)

if __name__ == "__main__":
    fetch(); parse()
    print(f"Fetched {len(STUDIES)} studies into {RAW}")

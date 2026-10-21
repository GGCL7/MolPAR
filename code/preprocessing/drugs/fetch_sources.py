#!/usr/bin/env python3
"""Fetch versioned ChEMBL and KEGG source snapshots for drug mechanism analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
KEGG = "https://rest.kegg.jp/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def request(session: requests.Session, url: str, attempts: int = 5) -> requests.Response:
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=90)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def paginated(session: requests.Session, endpoint: str, params: dict, key: str) -> dict:
    url = urljoin(CHEMBL, endpoint) + "?" + urlencode(params)
    rows, first_meta = [], None
    while url:
        payload = request(session, url).json()
        rows.extend(payload[key])
        first_meta = first_meta or payload.get("page_meta", {})
        nxt = payload.get("page_meta", {}).get("next")
        url = urljoin("https://www.ebi.ac.uk", nxt) if nxt else None
        print(f"{key}: {len(rows)}", flush=True)
    return {key: rows, "page_meta": first_meta}


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    SOURCES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MVP-702-drug-mechanism-recovery/1.0"})

    status_path = SOURCES / "chembl_status.json"
    molecules_path = SOURCES / "chembl_approved_small_molecules.json"
    mechanisms_path = SOURCES / "chembl_approved_mechanisms.json"
    targets_path = SOURCES / "chembl_targets.json"

    if args.refresh or not status_path.exists():
        write_json(status_path, request(session, urljoin(CHEMBL, "status.json")).json())
    if args.refresh or not molecules_path.exists():
        payload = paginated(
            session,
            "molecule.json",
            {"max_phase": 4, "molecule_type": "Small molecule", "limit": 1000},
            "molecules",
        )
        write_json(molecules_path, payload)
    if args.refresh or not mechanisms_path.exists():
        payload = paginated(
            session,
            "mechanism.json",
            {"max_phase": 4, "limit": 1000},
            "mechanisms",
        )
        write_json(mechanisms_path, payload)

    mechanisms = json.loads(mechanisms_path.read_text())["mechanisms"]
    target_ids = sorted({m.get("target_chembl_id") for m in mechanisms if m.get("target_chembl_id")})
    if args.refresh or not targets_path.exists():
        targets = []
        for start in range(0, len(target_ids), 50):
            batch = target_ids[start : start + 50]
            payload = paginated(
                session,
                "target.json",
                {"target_chembl_id__in": ",".join(batch), "limit": 1000},
                "targets",
            )
            targets.extend(payload["targets"])
            print(f"target batches: {min(start + 50, len(target_ids))}/{len(target_ids)}", flush=True)
        write_json(targets_path, {"targets": targets, "requested_target_ids": target_ids})

    text_sources = {
        "kegg_info.txt": "info/kegg",
        "kegg_uniprot_to_hsa.tsv": "conv/hsa/uniprot",
        "kegg_hsa_to_pathway.tsv": "link/pathway/hsa",
    }
    for filename, endpoint in text_sources.items():
        path = SOURCES / filename
        if args.refresh or not path.exists():
            path.write_text(request(session, urljoin(KEGG, endpoint)).text)
            time.sleep(0.4)

    files = sorted(p for p in SOURCES.iterdir() if p.is_file() and p.name != "manifest.json")
    status = json.loads(status_path.read_text())
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_version": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "selection_query": "max_phase=4, molecule_type=Small molecule; local therapeutic/withdrawn/structure filters",
        "kegg_info": (SOURCES / "kegg_info.txt").read_text().strip(),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }
    write_json(SOURCES / "manifest.json", manifest)
    print(json.dumps({"chembl_version": manifest["chembl_version"], "targets": len(target_ids), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a clean pool of KEGG compounds without metabolic pathway maps."""
from __future__ import annotations

import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.SaltRemover import SaltRemover

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RDLogger.DisableLog("rdApp.*")


class Kegg:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MVP-702-pathway-nomination"
        self.last_request = 0.0

    def get(self, endpoint: str, timeout: int = 120) -> str:
        path = CACHE / endpoint.replace("/", "_").replace("+", "-")
        if path.exists():
            return path.read_text()
        url = "https://rest.kegg.jp/" + endpoint
        error = None
        for attempt in range(5):
            wait = 0.34 - (time.time() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, timeout=timeout)
                self.last_request = time.time()
                if response.status_code == 200:
                    path.write_text(response.text)
                    return response.text
                if response.status_code == 404:
                    path.write_text("")
                    return ""
                error = RuntimeError(f"HTTP {response.status_code}")
            except requests.RequestException as exc:
                error = exc
                self.session.close()
                self.session = requests.Session()
            time.sleep(2**attempt)
        raise RuntimeError(f"Failed to fetch {url}: {error}")


def find_metabolism_maps(brite: dict, l1_names: set[str]) -> set[str]:
    def find(node):
        if node.get("name") == "Metabolism":
            return node
        for child in node.get("children", []):
            found = find(child)
            if found:
                return found
        return None

    metabolism = find(brite)
    if metabolism is None:
        raise RuntimeError("Metabolism branch missing from KEGG BRITE")
    maps = set()
    for superclass in metabolism.get("children", []):
        if superclass.get("name") not in l1_names:
            continue
        for leaf in superclass.get("children", []):
            code = leaf.get("name", "").split(None, 1)[0]
            if code.isdigit():
                maps.add("map" + code)
    return maps


def parse_links(text: str, target_prefix: str) -> dict[str, set[str]]:
    links: dict[str, set[str]] = defaultdict(set)
    for line in text.splitlines():
        if not line.strip():
            continue
        left, right = line.split("\t")
        cid = left.split(":")[-1]
        target = right.split(":")[-1]
        if target.startswith(target_prefix):
            links[cid].add(target)
    return links


def neutralize(mol: Chem.Mol) -> Chem.Mol:
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    for (idx,) in mol.GetSubstructMatches(pattern):
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - charge))
        atom.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def clean_molecule(mol: Chem.Mol, config: dict) -> tuple[dict | None, str]:
    if mol is None:
        return None, "no_or_invalid_structure"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None, "generic_or_wildcard"
    try:
        mol = SaltRemover().StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass
    if config["remove_mixtures"] and len(Chem.GetMolFrags(mol)) > 1:
        return None, "mixture"
    if config["neutralize"]:
        mol = neutralize(mol)
    heavy = mol.GetNumHeavyAtoms()
    if heavy < config["min_heavy_atoms"] or heavy > config["max_heavy_atoms"]:
        return None, "size_outside_training_domain"
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    allowed = set(config["organic_elements"])
    if not elements.issubset(allowed) or (config.get("require_carbon", True) and "C" not in elements):
        return None, "metal_or_inorganic"
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if Chem.MolFromSmiles(smiles) is None:
            return None, "model_input_parse_failed"
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        return None, "standardization_failed"
    if not smiles or not inchikey:
        return None, "standardization_failed"
    return {
        "smiles": smiles,
        "inchikey": inchikey,
        "connectivity_key": inchikey.split("-")[0],
        "num_heavy": heavy,
        "elements": ";".join(sorted(elements)),
    }, "kept"


def parse_sdf(text: str) -> list[Chem.Mol | None]:
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(text.encode()), sanitize=True, removeHs=False)
    return list(supplier)


def fetch_structures(client: Kegg, compound_ids: list[str]) -> dict[str, Chem.Mol | None]:
    structures = {}
    batches = [compound_ids[i : i + 10] for i in range(0, len(compound_ids), 10)]
    for number, batch in enumerate(batches, 1):
        endpoint = "get/" + "+".join("cpd:" + cid for cid in batch) + "/mol"
        try:
            mols = parse_sdf(client.get(endpoint))
        except RuntimeError:
            mols = []
        structures.update({cid: None for cid in batch})
        for mol in mols:
            if mol is None or not mol.HasProp("ENTRY"):
                continue
            cid = mol.GetProp("ENTRY").split(":")[-1]
            if cid in structures:
                structures[cid] = mol
        if number % 100 == 0:
            print(f"Structures: {min(number * 10, len(compound_ids))}/{len(compound_ids)}", flush=True)
    return structures


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    client = Kegg()
    info = client.get("info/kegg")
    compounds_text = client.get("list/compound")
    pathway_text = client.get("link/pathway/compound", timeout=240)
    reaction_text = client.get("link/reaction/compound", timeout=240)
    brite = json.loads(client.get("get/br:br08901/json"))

    compounds = {}
    for line in compounds_text.splitlines():
        cid, name = line.split("\t", 1)
        compounds[cid.split(":")[-1]] = name
    pathway_links = parse_links(pathway_text, "map")
    reaction_links = parse_links(reaction_text, "R")
    l1_names = set(config["labels"]["l1_kegg_superclasses"].values())
    metabolism_maps = find_metabolism_maps(brite, l1_names)
    model_maps = set(json.loads((ROOT / "MVP_model_0706/data/label_space_L2.json").read_text())["index_to_map"].values())

    no_any_pathway = {cid for cid in compounds if not pathway_links.get(cid)}
    no_metabolism_pathway = {
        cid for cid in compounds if not (pathway_links.get(cid, set()) & metabolism_maps)
    }
    no_model_pathway = {cid for cid in compounds if not (pathway_links.get(cid, set()) & model_maps)}
    candidate_ids = sorted(no_metabolism_pathway)
    print(
        f"KEGG compounds={len(compounds)}, no_any_pathway={len(no_any_pathway)}, "
        f"no_metabolism_pathway={len(no_metabolism_pathway)}, no_model_pathway={len(no_model_pathway)}"
    )

    structures = fetch_structures(client, candidate_ids)
    training = pd.read_csv(ROOT / "MVP_model_0706/data/kegg2026_hier.csv")
    training_inchikey = set(training.inchikey.astype(str))
    training_connectivity = {key.split("-")[0] for key in training_inchikey}
    rows, exclusions = [], Counter()
    for cid in candidate_ids:
        cleaned, reason = clean_molecule(structures.get(cid), config["clean"])
        if cleaned is None:
            exclusions[reason] += 1
            continue
        exact_training = cleaned["inchikey"] in training_inchikey
        connectivity_training = cleaned["connectivity_key"] in training_connectivity
        if connectivity_training:
            exclusions["training_structure_duplicate"] += 1
        rows.append(
            {
                "kegg_id": cid,
                "name": compounds[cid],
                **cleaned,
                "no_any_pathway": cid in no_any_pathway,
                "has_kegg_reaction": bool(reaction_links.get(cid)),
                "kegg_reactions": ";".join(sorted(reaction_links.get(cid, set()))),
                "exact_training_duplicate": exact_training,
                "connectivity_training_duplicate": connectivity_training,
            }
        )

    cleaned = pd.DataFrame(rows)
    before_dedup = len(cleaned)
    cleaned = (
        cleaned.sort_values("kegg_id")
        .groupby("inchikey", as_index=False)
        .agg(
            kegg_id=("kegg_id", "|".join),
            name=("name", "first"),
            smiles=("smiles", "first"),
            connectivity_key=("connectivity_key", "first"),
            num_heavy=("num_heavy", "first"),
            elements=("elements", "first"),
            no_any_pathway=("no_any_pathway", "all"),
            has_kegg_reaction=("has_kegg_reaction", "any"),
            kegg_reactions=("kegg_reactions", lambda x: ";".join(sorted({v for s in x for v in s.split(";") if v}))),
            exact_training_duplicate=("exact_training_duplicate", "any"),
            connectivity_training_duplicate=("connectivity_training_duplicate", "any"),
        )
    )
    cleaned["eligible_for_prediction"] = ~cleaned.connectivity_training_duplicate
    cleaned.to_csv(OUT / "candidate_pool.csv", index=False)

    eligible = cleaned[cleaned.eligible_for_prediction]
    funnel = {
        "kegg_state": info.splitlines()[1].strip() if len(info.splitlines()) > 1 else "current",
        "all_kegg_compounds": len(compounds),
        "linked_to_any_pathway": len(compounds) - len(no_any_pathway),
        "without_any_pathway": len(no_any_pathway),
        "without_metabolism_pathway": len(no_metabolism_pathway),
        "without_model_146_pathway": len(no_model_pathway),
        "structure_cleaning_exclusions": dict(exclusions),
        "clean_structures_before_deduplication": before_dedup,
        "clean_unique_structures": len(cleaned),
        "eligible_nontraining_structures": len(eligible),
        "eligible_without_any_pathway": int(eligible.no_any_pathway.sum()),
        "eligible_with_kegg_reaction": int(eligible.has_kegg_reaction.sum()),
        "eligible_without_any_pathway_and_with_reaction": int((eligible.no_any_pathway & eligible.has_kegg_reaction).sum()),
    }
    (OUT / "funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")
    print(json.dumps(funnel, indent=2))
    print(f"Candidate table: {OUT / 'candidate_pool.csv'}")


if __name__ == "__main__":
    main()

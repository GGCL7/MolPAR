#!/usr/bin/env python3
"""Standardize approved drugs and construct strict ChEMBL-target-KEGG evidence."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.SaltRemover import SaltRemover

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = HERE / "sources"
ALLOWED = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Se", "Si", "As"}
RDLogger.DisableLog("rdApp.*")
REMOVER = SaltRemover()


def joined(values) -> str:
    return "|".join(sorted({str(v) for v in values if v is not None and str(v).strip()}))


def neutralize(mol):
    mol = Chem.Mol(mol)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    for (idx,) in mol.GetSubstructMatches(pattern):
        atom = mol.GetAtomWithIdx(idx)
        charge, hydrogens = atom.GetFormalCharge(), atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(max(0, hydrogens - charge))
        atom.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def standardize(smiles: str) -> tuple[dict | None, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, "invalid_smiles"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None, "wildcard"
    try:
        mol = REMOVER.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "mixture_after_desalting"
    mol = neutralize(mol)
    if mol is None:
        return None, "neutralization_failure"
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 2 or heavy > 150:
        return None, "heavy_atom_range"
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in elements or not elements.issubset(ALLOWED):
        return None, "element_filter"
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        return None, "standardization_failure"
    if not canonical or not inchikey:
        return None, "standardization_failure"
    return {
        "smiles": canonical,
        "inchikey": inchikey,
        "connectivity_key": inchikey.split("-")[0],
        "num_heavy": heavy,
        "elements": ";".join(sorted(elements)),
    }, "included"


def parse_kegg() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    uniprot_to_genes: dict[str, set[str]] = defaultdict(set)
    for line in (SOURCES / "kegg_uniprot_to_hsa.tsv").read_text().splitlines():
        if not line.strip():
            continue
        accession, gene = line.split("\t")[:2]
        uniprot_to_genes[accession.removeprefix("up:")].add(gene.removeprefix("hsa:"))
    gene_to_maps: dict[str, set[str]] = defaultdict(set)
    for line in (SOURCES / "kegg_hsa_to_pathway.tsv").read_text().splitlines():
        if not line.strip():
            continue
        gene, pathway = line.split("\t")[:2]
        gene = gene.removeprefix("hsa:")
        pathway = pathway.removeprefix("path:hsa")
        if len(pathway) == 5 and pathway.isdigit():
            gene_to_maps[gene].add("map" + pathway)
    return uniprot_to_genes, gene_to_maps


def main() -> None:
    molecules = json.loads((SOURCES / "chembl_approved_small_molecules.json").read_text())["molecules"]
    mechanisms = json.loads((SOURCES / "chembl_approved_mechanisms.json").read_text())["mechanisms"]
    targets = json.loads((SOURCES / "chembl_targets.json").read_text())["targets"]
    molecule_by_id = {m["molecule_chembl_id"]: m for m in molecules}
    target_by_id = {t["target_chembl_id"]: t for t in targets}

    audit, standardized = [], []
    for molecule in molecules:
        mid = molecule["molecule_chembl_id"]
        reason = "candidate"
        if molecule.get("therapeutic_flag") is not True:
            reason = "not_therapeutic"
        elif molecule.get("withdrawn_flag") is True:
            reason = "withdrawn"
        elif not (molecule.get("molecule_structures") or {}).get("canonical_smiles"):
            reason = "no_structure"
        parent_id = (molecule.get("molecule_hierarchy") or {}).get("parent_chembl_id") or mid
        structure_record = molecule_by_id.get(parent_id, molecule)
        source_smiles = (structure_record.get("molecule_structures") or {}).get("canonical_smiles")
        clean, clean_reason = (standardize(source_smiles) if source_smiles else (None, "no_structure"))
        if reason == "candidate":
            reason = clean_reason
        audit.append({
            "molecule_chembl_id": mid,
            "parent_chembl_id": parent_id,
            "pref_name": molecule.get("pref_name") or "",
            "therapeutic_flag": molecule.get("therapeutic_flag"),
            "withdrawn_flag": molecule.get("withdrawn_flag"),
            "first_approval": molecule.get("first_approval"),
            "filter_status": reason,
        })
        if reason == "included" and clean:
            standardized.append({
                **clean,
                "molecule_chembl_id": mid,
                "parent_chembl_id": parent_id,
                "pref_name": molecule.get("pref_name") or structure_record.get("pref_name") or mid,
                "first_approval": molecule.get("first_approval"),
                "natural_product": molecule.get("natural_product"),
                "prodrug": molecule.get("prodrug"),
                "atc_classifications": joined(molecule.get("atc_classifications") or []),
            })
    pd.DataFrame(audit).to_csv(HERE / "drug_filtering_audit.csv", index=False)

    groups = []
    for inchikey, rows in pd.DataFrame(standardized).groupby("inchikey", sort=True):
        first = rows.iloc[0]
        groups.append({
            "drug_id": sorted(rows.molecule_chembl_id)[0],
            "drug_name": next((x for x in rows.pref_name if x and not x.startswith("CHEMBL")), first.pref_name),
            "chembl_ids": joined(rows.molecule_chembl_id),
            "parent_chembl_ids": joined(rows.parent_chembl_id),
            "smiles": first.smiles,
            "inchikey": inchikey,
            "connectivity_key": first.connectivity_key,
            "num_heavy": int(first.num_heavy),
            "elements": first.elements,
            "first_approval": pd.to_numeric(rows.first_approval, errors="coerce").min(),
            "natural_product": bool(pd.to_numeric(rows.natural_product, errors="coerce").fillna(0).max()),
            "prodrug": bool(pd.to_numeric(rows.prodrug, errors="coerce").fillna(0).max()),
            "atc_classifications": joined("|".join(rows.atc_classifications).split("|")),
            "approval_status": "ChEMBL max_phase=4, therapeutic, not withdrawn",
        })
    drugs = pd.DataFrame(groups)
    id_to_drug = {}
    for _, row in drugs.iterrows():
        for value in (row.chembl_ids + "|" + row.parent_chembl_ids).split("|"):
            id_to_drug[value] = row.drug_id

    label = json.loads((ROOT / "MVP_model_0706/data/label_space_L2.json").read_text())
    model_maps = set(label["index_to_map"].values())
    map_to_index = {v: int(k) for k, v in label["index_to_map"].items()}
    map_to_name = label["map_to_name"]
    map_to_l1 = label["map_to_l1_index"]
    uniprot_to_genes, gene_to_maps = parse_kegg()

    evidence_rows, pathway_rows = [], []
    drug_paths: dict[str, set[str]] = defaultdict(set)
    drug_targets: dict[str, set[str]] = defaultdict(set)
    strict_targets: dict[str, set[str]] = defaultdict(set)
    for mechanism in mechanisms:
        drug_id = id_to_drug.get(mechanism.get("molecule_chembl_id")) or id_to_drug.get(mechanism.get("parent_molecule_chembl_id"))
        if not drug_id:
            continue
        target_id = mechanism.get("target_chembl_id") or ""
        target = target_by_id.get(target_id, {})
        components = target.get("target_components") or []
        organism = target.get("organism") or ""
        tax_id = target.get("tax_id")
        strict = bool(
            mechanism.get("direct_interaction") == 1
            and mechanism.get("molecular_mechanism") == 1
            and mechanism.get("disease_efficacy") == 1
            and organism == "Homo sapiens"
            and target.get("target_type") == "SINGLE PROTEIN"
            and len(components) == 1
        )
        if target_id:
            drug_targets[drug_id].add(target_id)
            if strict:
                strict_targets[drug_id].add(target_id)
        if not components:
            components = [{}]
        for component in components:
            accession = component.get("accession") or ""
            genes = sorted(uniprot_to_genes.get(accession, set()))
            maps = sorted({m for g in genes for m in gene_to_maps.get(g, set()) if m in model_maps})
            refs = mechanism.get("mechanism_refs") or []
            evidence_rows.append({
                "drug_id": drug_id,
                "molecule_chembl_id": mechanism.get("molecule_chembl_id") or "",
                "parent_molecule_chembl_id": mechanism.get("parent_molecule_chembl_id") or "",
                "mechanism_of_action": mechanism.get("mechanism_of_action") or "",
                "action_type": mechanism.get("action_type") or "",
                "direct_interaction": mechanism.get("direct_interaction"),
                "molecular_mechanism": mechanism.get("molecular_mechanism"),
                "disease_efficacy": mechanism.get("disease_efficacy"),
                "target_chembl_id": target_id,
                "target_name": target.get("pref_name") or "",
                "target_type": target.get("target_type") or "",
                "target_organism": organism,
                "target_tax_id": tax_id,
                "uniprot_accession": accession,
                "kegg_hsa_genes": joined(genes),
                "model_l2_maps": joined(maps),
                "strict_primary_mechanism": strict,
                "reference_ids": joined(r.get("ref_id") for r in refs),
                "reference_types": joined(r.get("ref_type") for r in refs),
            })
            if strict:
                for model_map in maps:
                    drug_paths[drug_id].add(model_map)
                    pathway_rows.append({
                        "drug_id": drug_id,
                        "target_chembl_id": target_id,
                        "target_name": target.get("pref_name") or "",
                        "uniprot_accession": accession,
                        "kegg_hsa_genes": joined(genes),
                        "l2_index": map_to_index[model_map],
                        "l2_map": model_map,
                        "l2_name": map_to_name[model_map],
                        "l1_index": map_to_l1[model_map],
                        "mapping_evidence": "ChEMBL direct human single-protein MoA; UniProt-KEGG conversion; human gene-pathway link",
                    })

    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(HERE / "drug_target_mechanism_evidence.csv", index=False)
    pathways = pd.DataFrame(pathway_rows).drop_duplicates() if pathway_rows else pd.DataFrame()
    pathways.to_csv(HERE / "drug_target_pathway_evidence.csv", index=False)

    drugs["known_targets"] = [joined(drug_targets.get(d, set())) for d in drugs.drug_id]
    drugs["strict_known_targets"] = [joined(strict_targets.get(d, set())) for d in drugs.drug_id]
    drugs["known_target_kegg_L2"] = [joined(drug_paths.get(d, set())) for d in drugs.drug_id]
    drugs["n_known_target_kegg_L2"] = [len(drug_paths.get(d, set())) for d in drugs.drug_id]
    drugs["mechanism_evaluable"] = drugs.n_known_target_kegg_L2 > 0
    drugs["labels_L1"] = ""
    drugs["labels_L2"] = ""
    drugs.to_csv(HERE / "drug_catalogue.csv", index=False)

    summary = {
        "chembl_max_phase4_small_molecule_records": len(molecules),
        "included_records_after_local_filters": len(standardized),
        "unique_standardized_drug_structures": len(drugs),
        "drugs_with_any_mechanism_target": int((drugs.known_targets != "").sum()),
        "drugs_with_strict_human_direct_target": int((drugs.strict_known_targets != "").sum()),
        "mechanism_evaluable_drugs": int(drugs.mechanism_evaluable.sum()),
        "strict_drug_target_pathway_edges": len(pathways),
        "covered_model_l2": int(pathways.l2_map.nunique()) if len(pathways) else 0,
        "strict_definition": "direct_interaction=1, molecular_mechanism=1, disease_efficacy=1, Homo sapiens, SINGLE PROTEIN, one target component",
    }
    (HERE / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

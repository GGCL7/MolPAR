#!/usr/bin/env python3
"""Summarize frozen RA-GLGN drug mechanism recovery and novelty analyses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE


def boot(values, function, seed=20260921, n=2000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None}
    estimate = float(function(values))
    samples = np.empty(n, dtype=float)
    for i in range(n):
        samples[i] = function(values[rng.integers(0, len(values), len(values))])
    return {"estimate": estimate, "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975))}


def main() -> None:
    drugs = pd.read_csv(INPUT / "drug_catalogue.csv").fillna("")
    training = pd.read_csv(ROOT / "MVP_model_0706/data/kegg2026_hier.csv")
    evidence = pd.read_csv(INPUT / "drug_target_pathway_evidence.csv").fillna("")
    scores = np.load(INPUT / "predictions/ensemble_drug_scores.npz", allow_pickle=True)
    l1_by_seed = scores["l1_by_seed"]
    l2_by_seed = scores["l2_by_seed"]
    nearest_by_seed = scores["nearest_embedding_similarity_by_seed"]
    if list(scores["drug_id"].astype(str)) != drugs.drug_id.astype(str).tolist():
        raise RuntimeError("Drug ordering differs between catalogue and prediction cache")

    labels = json.loads((ROOT / "MVP_model_0706/data/label_space_L2.json").read_text())
    l1_labels = json.loads((ROOT / "MVP_model_0706/data/label_space_L1.json").read_text())
    l1_names = {int(k): v for k, v in l1_labels["index_to_name"].items()}
    maps = {int(k): v for k, v in labels["index_to_map"].items()}
    names = labels["map_to_name"]
    parent = labels["map_to_l1_index"]
    known = evidence.groupby("drug_id").apply(lambda frame: set(frame.l2_map.astype(str))).to_dict()
    target_names = evidence.groupby("drug_id").apply(lambda frame: "|".join(sorted(set(frame.target_name.astype(str))))).to_dict()
    target_ids = evidence.groupby("drug_id").apply(lambda frame: "|".join(sorted(set(frame.target_chembl_id.astype(str))))).to_dict()

    train_mols = [Chem.MolFromSmiles(s) for s in training.smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols]
    train_ikey = set(training.inchikey.astype(str))
    train_scaffolds = set()
    for mol in train_mols:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold:
            train_scaffolds.add(scaffold)
    max_sim, exact_duplicate, scaffold_status = [], [], []
    for smiles, inchikey in zip(drugs.smiles, drugs.inchikey):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        max_sim.append(float(sim))
        exact_duplicate.append(str(inchikey) in train_ikey)
        scaffold_status.append("acyclic" if not scaffold else ("seen" if scaffold in train_scaffolds else "unseen"))
    drugs["max_train_ecfp4_tanimoto"] = max_sim
    drugs["exact_training_structure_duplicate"] = exact_duplicate
    drugs["scaffold_status"] = scaffold_status
    drugs["known_target_l2_maps"] = ["|".join(sorted(known.get(d, set()))) for d in drugs.drug_id]
    drugs["known_target_names"] = [target_names.get(d, "") for d in drugs.drug_id]
    drugs["known_target_chembl_ids"] = [target_ids.get(d, "") for d in drugs.drug_id]
    drugs["mechanism_evaluable"] = drugs.drug_id.isin(set(known))

    l2_mean = l2_by_seed.mean(axis=0)
    l1_mean = l1_by_seed.mean(axis=0)
    top_order = np.argsort(-l2_mean, axis=1, kind="stable")
    l1_order = np.argsort(-l1_mean, axis=1, kind="stable")
    top1_seed = np.argmax(l2_by_seed, axis=2)
    top3_seed = np.argsort(-l2_by_seed, axis=2)[:, :, :3]
    drugs["predicted_l1_index"] = l1_order[:, 0]
    drugs["predicted_l1_name"] = [l1_names.get(int(i), "") for i in l1_order[:, 0]]
    drugs["l1_top1_probability"] = l1_mean[np.arange(len(drugs)), l1_order[:, 0]]
    drugs["predicted_l2_index"] = top_order[:, 0]
    drugs["predicted_l2_map"] = [maps[i] for i in top_order[:, 0]]
    drugs["predicted_l2_name"] = [names[maps[i]] for i in top_order[:, 0]]
    drugs["l2_top1_probability"] = l2_mean[np.arange(len(drugs)), top_order[:, 0]]
    drugs["l2_seed_top1_consensus"] = [len(set(top1_seed[:, i])) == 1 for i in range(len(drugs))]
    drugs["l2_top3_mean_jaccard"] = [
        np.mean([len(set(top3_seed[a, i]) & set(top3_seed[b, i])) / len(set(top3_seed[a, i]) | set(top3_seed[b, i])) for a, b in ((0, 1), (0, 2), (1, 2))])
        for i in range(len(drugs))
    ]
    drugs["l2_score_seed_sd_top1"] = [np.std([l2_by_seed[s, i, top_order[i, 0]] for s in range(l2_by_seed.shape[0])], ddof=1) for i in range(len(drugs))]
    drugs["nearest_embedding_similarity_mean"] = nearest_by_seed.mean(axis=0)
    for rank in range(5):
        index = top_order[:, rank]
        drugs[f"l2_top{rank + 1}_map"] = [maps[i] for i in index]
        drugs[f"l2_top{rank + 1}_name"] = [names[maps[i]] for i in index]
        drugs[f"l2_top{rank + 1}_probability"] = l2_mean[np.arange(len(drugs)), index]

    def top_maps(row, k):
        return set(row[f"l2_top{i}_map"] for i in range(1, k + 1))
    for k in (1, 3, 5):
        drugs[f"mechanism_hit_at_{k}"] = [bool(top_maps(row, k) & known.get(row.drug_id, set())) for _, row in drugs.iterrows()]
        drugs[f"mechanism_recall_at_{k}"] = [
            (len(top_maps(row, k) & known.get(row.drug_id, set())) / len(known[row.drug_id])) if row.drug_id in known else np.nan
            for _, row in drugs.iterrows()
        ]
    drugs["mechanism_first_hit_rank"] = [
        next((rank for rank in range(1, 147) if maps[top_order[i, rank - 1]] in known.get(row.drug_id, set())), np.nan)
        for i, (_, row) in enumerate(drugs.iterrows())
    ]
    drugs["mechanism_mrr"] = drugs.mechanism_first_hit_rank.map(lambda x: 1.0 / x if pd.notna(x) else np.nan)
    drugs["primary_evaluable"] = drugs.mechanism_evaluable & ~drugs.exact_training_structure_duplicate
    drugs.to_csv(HERE / "drug_mechanism_recovery.csv", index=False)

    subgroup_specs = {
        "all_mechanism_evaluable": drugs.mechanism_evaluable,
        "primary_no_exact_training_duplicate": drugs.primary_evaluable,
        "primary_tanimoto_below_0_8": drugs.primary_evaluable & (drugs.max_train_ecfp4_tanimoto < .8),
        "primary_tanimoto_below_0_6": drugs.primary_evaluable & (drugs.max_train_ecfp4_tanimoto < .6),
        "primary_unseen_scaffold": drugs.primary_evaluable & (drugs.scaffold_status == "unseen"),
        "primary_acyclic": drugs.primary_evaluable & (drugs.scaffold_status == "acyclic"),
    }
    summaries = []
    for subgroup, mask in subgroup_specs.items():
        selected = drugs[mask].copy()
        row = {"subgroup": subgroup, "n": len(selected), "mean_tanimoto": float(selected.max_train_ecfp4_tanimoto.mean()) if len(selected) else None}
        for k in (1, 3, 5):
            row[f"hit_at_{k}"] = boot(selected[f"mechanism_hit_at_{k}"].astype(float), np.mean)
            row[f"recall_at_{k}"] = boot(selected[f"mechanism_recall_at_{k}"].dropna().astype(float), np.mean)
        row["mrr"] = boot(selected.mechanism_mrr.dropna().astype(float), np.mean)
        row["seed_top1_consensus_rate"] = float(selected.l2_seed_top1_consensus.mean()) if len(selected) else None
        row["mean_top3_jaccard"] = float(selected.l2_top3_mean_jaccard.mean()) if len(selected) else None
        summaries.append(row)
    summary_df = pd.json_normalize(summaries, sep=".")
    summary_df.to_csv(HERE / "mechanism_recovery_summary.csv", index=False)

    subgroup_rows = []
    for category in sorted(set(evidence.target_name) - {""}):
        ids = set(evidence.loc[evidence.target_name == category, "drug_id"])
        selected = drugs[drugs.primary_evaluable & drugs.drug_id.isin(ids)]
        if len(selected) >= 3:
            subgroup_rows.append({"target_name": category, "n": len(selected), "hit_at_3": float(selected.mechanism_hit_at_3.mean()), "recall_at_5": float(selected.mechanism_recall_at_5.mean()), "seed_top1_consensus_rate": float(selected.l2_seed_top1_consensus.mean())})
    pd.DataFrame(subgroup_rows).sort_values(["n", "hit_at_3"], ascending=[False, False]).to_csv(HERE / "target_subgroup_analysis.csv", index=False)

    eligible = drugs[drugs.primary_evaluable].copy()
    cases = []
    def add_case(series, case_type):
        record = series.to_dict()
        record["case_type"] = case_type
        cases.append(record)
    hit = eligible[eligible.mechanism_hit_at_3]
    if len(hit):
        add_case(hit.sort_values(["l2_seed_top1_consensus", "l2_top1_probability"], ascending=False).iloc[0], "stable_hit")
        add_case(hit.sort_values("max_train_ecfp4_tanimoto").iloc[0], "low_similarity_hit")
        unseen = hit[hit.scaffold_status == "unseen"]
        if len(unseen): add_case(unseen.sort_values("max_train_ecfp4_tanimoto").iloc[0], "unseen_scaffold_hit")
    miss = eligible[~eligible.mechanism_hit_at_5]
    if len(miss): add_case(miss.sort_values("l2_top1_probability", ascending=False).iloc[0], "stable_boundary_miss")
    multi = eligible[eligible.known_target_l2_maps.str.contains("\\|")]
    if len(multi): add_case(multi.sort_values("n_known_target_kegg_L2", ascending=False).iloc[0], "multi_pathway_drug")
    case_df = pd.DataFrame(cases).drop_duplicates(subset=["case_type"]) if cases else pd.DataFrame()
    if len(case_df):
        case_df.to_csv(HERE / "case_studies.csv", index=False)

    source_summary = json.loads((INPUT / "dataset_summary.json").read_text())
    final = {
        "dataset": source_summary,
        "prediction_rows": len(drugs),
        "primary_evaluable_n": int(drugs.primary_evaluable.sum()),
        "primary_hits_at_3": int(drugs.loc[drugs.primary_evaluable, "mechanism_hit_at_3"].sum()),
        "primary_hits_at_5": int(drugs.loc[drugs.primary_evaluable, "mechanism_hit_at_5"].sum()),
        "top_level_interpretation": "Frozen RA-GLGN pathway associations were evaluated against independently curated direct human ChEMBL MoA target to KEGG L2 mappings; non-mappable drugs are coverage exclusions, not prediction errors.",
        "no_other_model_comparisons": True,
    }
    (HERE / "analysis_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(final, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    import subprocess, sys
    from pathlib import Path
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / 'run_downstream.py'), '--analysis', 'drugs', *sys.argv[1:]]))

#!/usr/bin/env python3
"""Full pathway-driven Stage B using frozen RA-GLGN predictions.

This implementation does not use a drug's known targets in scoring. It maps
each model L2 pathway to all available human KEGG genes, aggregates Open
Targets genetic evidence over those genes with pathway-size normalisation,
and weights the resulting pathway-disease matrix by frozen RA-GLGN scores.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
GENETIC_SOURCES = [
    "gwas_credible_sets", "gene_burden", "eva", "eva_somatic",
    "orphanet", "uniprot_variants", "cancer_gene_census",
    "gene2phenotype", "genomics_england", "clingen", "uniprot_literature",
]
TARGET_QUERY = """
query($ids:[String!]!,$ds:[DatasourceSettingsInput!]!,$page:Int!){
 targets(ensemblIds:$ids){ id approvedSymbol
  associatedDiseases(enableIndirect:false,datasources:$ds,page:{index:$page,size:100}){
   count rows{ disease{id name} datasourceScores{id score} datatypeScores{id score} }
  }
 }
}
"""


def post(query: str, variables: dict, retries: int = 5) -> dict:
    last = None
    for attempt in range(retries):
        try:
            response = requests.post(OT_API, json={"query": query, "variables": variables}, timeout=180)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload["data"]
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Open Targets request failed: {last}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def map_entrez(entrez: list[str]) -> pd.DataFrame:
    cache = HERE / "stage_b_pathway_gene_mapping.csv"
    if cache.exists():
        frame = pd.read_csv(cache, dtype={"entrez_id": str}).fillna("")
        frame["entrez_id"] = frame.entrez_id.astype(str)
        return frame
    rows = []
    for start in range(0, len(entrez), 500):
        block = entrez[start:start + 500]
        response = requests.post(
            "https://mygene.info/v3/query",
            data={"q": ",".join(block), "scopes": "entrezgene", "species": "human", "fields": "entrezgene,ensembl.gene,symbol"},
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json():
            ens = item.get("ensembl", {})
            if isinstance(ens, list):
                genes = sorted({str(x.get("gene", "")) for x in ens if x.get("gene")})
            elif isinstance(ens, dict):
                genes = [str(ens.get("gene"))] if ens.get("gene") else []
            else:
                genes = []
            rows.append({"entrez_id": str(item.get("query", "")), "ensembl_ids": "|".join(genes), "symbol": item.get("symbol", ""), "notfound": bool(item.get("notfound", False))})
    frame = pd.DataFrame(rows).drop_duplicates("entrez_id")
    frame.to_csv(cache, index=False)
    return frame


def fetch_genetic_associations(ensembl_ids: list[str]) -> pd.DataFrame:
    cache = HERE / "stage_b_pathway_gene_disease_genetic.csv"
    if cache.exists():
        return pd.read_csv(cache).fillna("")
    batches = [ensembl_ids[i:i + 40] for i in range(0, len(ensembl_ids), 40)]
    jobs = [(source, batch) for source in GENETIC_SOURCES for batch in batches]

    def fetch(job):
        source, ids = job
        settings = [{"id": source, "weight": 1.0, "propagate": False, "required": True}]
        out = []
        page = 0
        while True:
            data = post(TARGET_QUERY, {"ids": ids, "ds": settings, "page": page})
            more = False
            for target in data["targets"]:
                association = target["associatedDiseases"]
                if (page + 1) * 100 < int(association["count"]):
                    more = True
                for row in association["rows"]:
                    ds = {x["id"]: float(x.get("score") or 0) for x in row.get("datasourceScores", [])}
                    source_score = ds.get(source, 0.0)
                    if source_score <= 0:
                        continue
                    dt = {x["id"]: float(x.get("score") or 0) for x in row.get("datatypeScores", [])}
                    score = max(dt.get("genetic_association", 0.0), dt.get("genetic_literature", 0.0))
                    if score > 0:
                        out.append({"ensembl_id": target["id"], "symbol": target["approvedSymbol"], "disease_id": row["disease"]["id"], "disease_name": row["disease"]["name"], "genetic_score": score, "trigger_source": source, "trigger_source_score": source_score})
            if not more:
                break
            page += 1
        return out

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for block in pool.map(fetch, jobs):
            records.extend(block)
    raw = pd.DataFrame(records)
    if raw.empty:
        raise RuntimeError("No pathway gene-disease genetic associations were retrieved")
    agg = raw.groupby(["ensembl_id", "symbol", "disease_id", "disease_name"], as_index=False).agg(
        genetic_score=("genetic_score", "max"),
        trigger_sources=("trigger_source", lambda x: "|".join(sorted(set(x)))),
        max_trigger_source_score=("trigger_source_score", "max"),
    )
    agg.to_csv(cache, index=False)
    return agg


def dcg_at(ranks: list[int], k: int) -> float:
    return sum(1.0 / np.log2(r + 1) for r in ranks if r <= k)


def main() -> None:
    labels = json.loads((ROOT / "MVP_model_0706/data/label_space_L2.json").read_text())
    index_to_map = {int(k): v for k, v in labels["index_to_map"].items()}
    model_maps = set(index_to_map.values())
    links = pd.read_csv(HERE / "sources/kegg_hsa_to_pathway.tsv", sep="\t", header=None, names=["gene", "pathway"])
    links["entrez_id"] = links.gene.str.replace("hsa:", "", regex=False)
    links["l2_map"] = links.pathway.str.replace("path:hsa", "map", regex=False)
    links = links[links.l2_map.isin(model_maps)].copy()
    mapping = map_entrez(sorted(links.entrez_id.unique()))
    links = links.merge(mapping, on="entrez_id", how="left")
    links = links.assign(ensembl_id=links.ensembl_ids.fillna("").str.split("|")).explode("ensembl_id")
    links = links[links.ensembl_id.astype(str).str.startswith("ENSG")].drop_duplicates(["l2_map", "ensembl_id"])
    links[["l2_map", "entrez_id", "ensembl_id", "symbol"]].to_csv(HERE / "stage_b_pathway_gene_membership.csv", index=False)
    ensembl_ids = sorted(links.ensembl_id.unique())
    assoc = fetch_genetic_associations(ensembl_ids)

    # Pathway-disease score: mean genetic score over every mapped pathway gene;
    # genes without evidence for a disease contribute zero to the denominator.
    sizes = links.groupby("l2_map").ensembl_id.nunique().to_dict()
    edges = links[["l2_map", "ensembl_id"]].merge(assoc, on="ensembl_id", how="inner")
    pd_score = edges.groupby(["l2_map", "disease_id", "disease_name"], as_index=False).agg(
        genetic_score_sum=("genetic_score", "sum"), evidence_gene_n=("ensembl_id", "nunique")
    )
    pd_score["pathway_gene_n"] = pd_score.l2_map.map(sizes)
    pd_score["pathway_disease_score"] = pd_score.genetic_score_sum / pd_score.pathway_gene_n
    pd_score.to_csv(HERE / "stage_b_pathway_disease_matrix_long.csv", index=False)

    diseases = pd_score[["disease_id", "disease_name"]].drop_duplicates("disease_id").sort_values("disease_id").reset_index(drop=True)
    disease_index = {x: i for i, x in enumerate(diseases.disease_id)}
    matrix = np.zeros((146, len(diseases)), dtype=np.float32)
    map_to_index = {v: k for k, v in index_to_map.items()}
    for row in pd_score.itertuples():
        matrix[map_to_index[row.l2_map], disease_index[row.disease_id]] = row.pathway_disease_score
    np.savez_compressed(HERE / "stage_b_pathway_disease_matrix.npz", matrix=matrix, disease_id=diseases.disease_id.to_numpy(), disease_name=diseases.disease_name.to_numpy())

    drugs = pd.read_csv(HERE / "drug_mechanism_recovery.csv").fillna("")
    frozen = np.load(HERE / "predictions/ensemble_drug_scores.npz", allow_pickle=True)
    l2_by_seed = frozen["l2_by_seed"]
    l2_mean = l2_by_seed.mean(axis=0)
    top5 = np.argsort(-l2_mean, axis=1)[:, :5]
    mapped_indices = set(map_to_index[m] for m in sizes)
    weighted = np.zeros_like(l2_mean)
    for i in range(len(drugs)):
        weighted[i, top5[i]] = l2_mean[i, top5[i]]
    scores = weighted @ matrix
    mapped_top5_n = np.asarray([sum(int(j in mapped_indices) for j in top5[i]) for i in range(len(drugs))])
    total_top5_weight = np.take_along_axis(l2_mean, top5, axis=1).sum(1)
    mapped_top5_weight = np.asarray([sum(l2_mean[i, j] for j in top5[i] if j in mapped_indices) for i in range(len(drugs))])

    indications = pd.read_csv(HERE / "stage_b_approved_indications.csv").fillna("")
    truth = indications.groupby("drug_id").apply(lambda x: set(x.disease_id.astype(str))).to_dict()
    primary = ~drugs.exact_training_structure_duplicate.astype(str).str.lower().eq("true")
    metric_rows, top_rows = [], []
    for i, drug in drugs.iterrows():
        positive = np.flatnonzero(scores[i] > 0)
        order = positive[np.argsort(-scores[i, positive], kind="stable")]
        ranked_ids = diseases.disease_id.to_numpy()[order]
        for rank, pos in enumerate(order[:50], 1):
            top_rows.append({"drug_id": drug.drug_id, "rank": rank, "disease_id": diseases.iloc[pos].disease_id, "disease_name": diseases.iloc[pos].disease_name, "ra_pathway_score": float(scores[i, pos])})
        if drug.drug_id not in truth:
            continue
        known = truth[drug.drug_id]
        ranks = [int(np.where(ranked_ids == x)[0][0]) + 1 for x in known if x in disease_index and x in set(ranked_ids)]
        row = {"drug_id": drug.drug_id, "primary_no_exact_duplicate": bool(primary.iloc[i]), "n_approved_indications": len(known), "n_indications_in_candidate_space": len(ranks), "candidate_coverage": len(ranks) / len(known), "first_hit_rank": min(ranks) if ranks else np.nan, "mrr": 1.0 / min(ranks) if ranks else 0.0, "mapped_top5_n": int(mapped_top5_n[i]), "mapped_top5_weight_fraction": float(mapped_top5_weight[i] / total_top5_weight[i]) if total_top5_weight[i] else 0.0}
        for k in (10, 20, 50):
            hits = sum(r <= k for r in ranks)
            row[f"hit_at_{k}"] = bool(hits)
            row[f"recall_at_{k}"] = hits / len(known)
            ideal = dcg_at(list(range(1, min(len(known), k) + 1)), k)
            row[f"ndcg_at_{k}"] = dcg_at(ranks, k) / ideal if ideal else 0.0
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(HERE / "stage_b_pathway_indication_recovery.csv", index=False)
    pd.DataFrame(top_rows).to_csv(HERE / "stage_b_pathway_top50_diseases.csv", index=False)

    # Seed sensitivity uses the same fixed pathway-disease matrix and Top-5 rule.
    seed_rows = []
    eval_mask = metrics.primary_no_exact_duplicate.astype(bool)
    eval_ids = set(metrics.loc[eval_mask, "drug_id"])
    for seed in range(l2_by_seed.shape[0]):
        seed_l2 = l2_by_seed[seed]
        seed_top = np.argsort(-seed_l2, axis=1)[:, :5]
        seed_weighted = np.zeros_like(seed_l2)
        for i in range(len(drugs)):
            seed_weighted[i, seed_top[i]] = seed_l2[i, seed_top[i]]
        seed_scores = seed_weighted @ matrix
        per_drug = []
        for i, drug in drugs.iterrows():
            if drug.drug_id not in eval_ids:
                continue
            positive = np.flatnonzero(seed_scores[i] > 0)
            order = positive[np.argsort(-seed_scores[i, positive], kind="stable")]
            ranked = diseases.disease_id.to_numpy()[order]
            known = truth[drug.drug_id]
            ranks = [int(np.where(ranked == x)[0][0]) + 1 for x in known if x in disease_index and x in set(ranked)]
            per_drug.append({"hit20": any(r <= 20 for r in ranks), "recall20": sum(r <= 20 for r in ranks) / len(known), "mrr": 1/min(ranks) if ranks else 0})
        frame = pd.DataFrame(per_drug)
        seed_rows.append({"seed": seed, "n": len(frame), "hit_at_20": frame.hit20.mean(), "recall_at_20": frame.recall20.mean(), "mrr": frame.mrr.mean()})
    pd.DataFrame(seed_rows).to_csv(HERE / "stage_b_pathway_seed_sensitivity.csv", index=False)

    selected = metrics[metrics.primary_no_exact_duplicate.astype(bool)]
    summary = {
        "method": "frozen RA Top-5 L2 scores times pathway-size-normalized human KEGG gene/Open Targets genetic disease scores",
        "known_drug_targets_used_in_scoring": False,
        "model_l2_n": 146,
        "human_gene_mappable_l2_n": len(sizes),
        "unique_kegg_human_entrez_n": int(links.entrez_id.nunique()),
        "unique_ensembl_gene_n": len(ensembl_ids),
        "disease_candidate_n": len(diseases),
        "evaluation_drug_n": len(selected),
        "approved_indication_n": int(selected.n_approved_indications.sum()),
        "indication_candidate_coverage": float(selected.n_indications_in_candidate_space.sum() / selected.n_approved_indications.sum()),
        "hit_at_10": float(selected.hit_at_10.mean()), "hit_at_20": float(selected.hit_at_20.mean()), "hit_at_50": float(selected.hit_at_50.mean()),
        "recall_at_10": float(selected.recall_at_10.mean()), "recall_at_20": float(selected.recall_at_20.mean()), "recall_at_50": float(selected.recall_at_50.mean()),
        "ndcg_at_20": float(selected.ndcg_at_20.mean()), "mrr": float(selected.mrr.mean()),
        "mean_mapped_top5_n": float(selected.mapped_top5_n.mean()), "mean_mapped_top5_weight_fraction": float(selected.mapped_top5_weight_fraction.mean()),
        "genetic_source_allowlist": GENETIC_SOURCES,
        "input_hashes": {"l2_labels": sha256(ROOT / "MVP_model_0706/data/label_space_L2.json"), "kegg_hsa_links": sha256(HERE / "sources/kegg_hsa_to_pathway.tsv"), "frozen_scores": sha256(HERE / "predictions/ensemble_drug_scores.npz")},
    }
    (HERE / "stage_b_pathway_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

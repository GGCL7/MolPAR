#!/usr/bin/env python3
"""Evaluate nominations in a side-aware 1-3 step Rhea reaction network.

Rhea master equations use an equals sign and should not be interpreted as
thermodynamically directed. Here, "side-aware" means that a graph step may only
connect participants on opposite sides of an unambiguously aligned equation.
Same-side co-participants are never treated as reaction neighbours.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from review_candidate_evidence import load_two_column_links, parse_rhea_sides, split_values


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "MVP_model_0706"
NOMINATION = ROOT / "downstream/pathway_nomination"
SOURCE = NOMINATION / "external_source"
OUT = Path(__file__).resolve().parent


def build_network():
    kegg_to_chebi: dict[str, set[str]] = defaultdict(set)
    chebi_to_kegg: dict[str, set[str]] = defaultdict(set)
    with (SOURCE / "kegg_compound_to_chebi.tsv").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            kegg, chebi = line.rstrip("\n").split("\t")
            kid = kegg.split(":")[-1]
            cid = "CHEBI:" + chebi.split(":")[-1]
            kegg_to_chebi[kid].add(cid)
            chebi_to_kegg[cid].add(kid)

    rhea = pd.read_csv(SOURCE / "rhea_reactions.tsv", sep="\t", dtype=str).fillna("")
    rhea.columns = ["rhea_id", "equation", "chebi_ids", "ec_numbers", "enzyme_count", "pubmed_ids"]
    participant_degree: Counter[str] = Counter()
    parsed_rows = []
    for row in rhea.itertuples(index=False):
        parsed = parse_rhea_sides(row.equation, row.chebi_ids)
        parsed_rows.append((row.rhea_id, parsed))
        for participant in set(split_values(row.chebi_ids)):
            participant_degree[participant] += 1
    currency_cutoff = max(50, int(np.quantile(list(participant_degree.values()), 0.99)))

    adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    aligned_reactions = 0
    retained_reactions = 0
    for rhea_id, parsed in parsed_rows:
        if not parsed["aligned"]:
            continue
        aligned_reactions += 1
        left = {cid for cid in parsed["left"] if participant_degree[cid] < currency_cutoff}
        right = {cid for cid in parsed["right"] if participant_degree[cid] < currency_cutoff}
        if not left or not right:
            continue
        retained_reactions += 1
        for source in left:
            for target in right:
                if source != target:
                    adjacency[source][target].add(rhea_id)
                    adjacency[target][source].add(rhea_id)
    return rhea, kegg_to_chebi, chebi_to_kegg, adjacency, participant_degree, currency_cutoff, aligned_reactions, retained_reactions


def shortest_pathway_distances(
    starts: set[str],
    adjacency: dict[str, dict[str, set[str]]],
    chebi_pathways: dict[str, set[str]],
    max_steps: int = 3,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]], int]:
    distance: dict[str, int] = {}
    example_path: dict[str, tuple[str, ...]] = {}
    queue = deque((node, 0, (node,)) for node in starts if node in adjacency)
    visited = {node: 0 for node in starts}
    reached_nodes = set(starts)
    while queue:
        node, depth, path = queue.popleft()
        if depth >= max_steps:
            continue
        for neighbour in adjacency.get(node, {}):
            next_depth = depth + 1
            if neighbour not in visited or next_depth < visited[neighbour]:
                visited[neighbour] = next_depth
                next_path = path + (neighbour,)
                queue.append((neighbour, next_depth, next_path))
                reached_nodes.add(neighbour)
                for pathway in chebi_pathways.get(neighbour, set()):
                    if pathway not in distance or next_depth < distance[pathway]:
                        distance[pathway] = next_depth
                        example_path[pathway] = next_path
    return distance, example_path, len(reached_nodes - starts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(NOMINATION / "candidate_catalogue_with_support.csv")
    high = candidates.loc[candidates.high_confidence_nomination].copy()
    label_space = json.loads((MODEL_ROOT / "data/label_space_L2.json").read_text())
    model_maps = set(label_space["index_to_map"].values())
    map_to_name = label_space["map_to_name"]
    map_to_l1 = {pathway: int(value) for pathway, value in label_space["map_to_l1_index"].items()}
    maps_by_l1: dict[int, list[str]] = defaultdict(list)
    for pathway in sorted(model_maps):
        maps_by_l1[map_to_l1[pathway]].append(pathway)

    compound_pathways = load_two_column_links(NOMINATION / "cache/link_pathway_compound")
    (
        rhea,
        kegg_to_chebi,
        chebi_to_kegg,
        adjacency,
        participant_degree,
        currency_cutoff,
        aligned_reactions,
        retained_reactions,
    ) = build_network()

    chebi_pathways: dict[str, set[str]] = defaultdict(set)
    pathway_targets: dict[str, set[str]] = defaultdict(set)
    for kegg_id, pathways in compound_pathways.items():
        covered = pathways & model_maps
        for chebi in kegg_to_chebi.get(kegg_id, set()):
            chebi_pathways[chebi].update(covered)
            for pathway in covered:
                pathway_targets[pathway].add(chebi)
    pathway_sizes = {pathway: len(pathway_targets[pathway]) for pathway in model_maps}

    controls = {}
    if 'CONTROL_PATH' in globals():
        fixed = pd.read_csv(CONTROL_PATH).fillna('')
        controls = fixed.set_index('kegg_id').size_matched_same_l1_maps.to_dict()
    rows = []
    for candidate in high.itertuples(index=False):
        kegg_ids = set(str(candidate.kegg_id).split("|"))
        starts = set().union(*(kegg_to_chebi.get(kid, set()) for kid in kegg_ids))
        distances, paths, reached_count = shortest_pathway_distances(
            starts, adjacency, chebi_pathways, max_steps=3
        )
        predicted = candidate.predicted_l2_map
        predicted_distance = distances.get(predicted)
        same_l1 = maps_by_l1[int(candidate.predicted_l1_index)]
        ordered_same_l1 = sorted(
            (pathway for pathway in same_l1 if pathway in distances),
            key=lambda pathway: (distances[pathway], -pathway_sizes[pathway], pathway),
        )
        predicted_rank = (
            1 + sum(distances[pathway] < distances[predicted] for pathway in same_l1 if pathway in distances)
            if predicted in distances
            else np.nan
        )
        alternatives = [p for p in same_l1 if p != predicted]
        matched = sorted(
            alternatives,
            key=lambda pathway: abs(
                np.log1p(pathway_sizes[pathway]) - np.log1p(pathway_sizes[predicted])
            ),
        )[:5]
        if candidate.kegg_id in controls:
            matched = split_values(controls[candidate.kegg_id])
            if any(p not in alternatives for p in matched):
                raise ValueError('Invalid within-class reference control')
        path = paths.get(predicted, ())
        path_kegg = []
        for chebi in path:
            mapped = sorted(chebi_to_kegg.get(chebi, set()))
            path_kegg.append(mapped[0] if mapped else chebi)
        rows.append(
            {
                "kegg_id": candidate.kegg_id,
                "name": candidate.name,
                "predicted_l1_name": candidate.predicted_l1_name,
                "predicted_l2_map": predicted,
                "predicted_l2_name": candidate.predicted_l2_name,
                "l2_ensemble_probability": candidate.l2_ensemble_probability,
                "max_train_ecfp4_tanimoto": candidate.max_train_ecfp4_tanimoto,
                "scaffold_status": candidate.scaffold_status,
                "start_chebi_ids": ";".join(sorted(starts)),
                "network_start_available": any(start in adjacency for start in starts),
                "reached_noncurrency_nodes_within_3_steps": reached_count,
                "predicted_l2_min_steps": predicted_distance if predicted_distance is not None else np.nan,
                "predicted_l2_within_1_steps": predicted_distance is not None and predicted_distance <= 1,
                "predicted_l2_within_2_steps": predicted_distance is not None and predicted_distance <= 2,
                "predicted_l2_within_3_steps": predicted_distance is not None and predicted_distance <= 3,
                "predicted_l2_same_l1_network_rank": predicted_rank,
                "predicted_l2_example_chebi_path": ";".join(path),
                "predicted_l2_example_kegg_path": ";".join(path_kegg),
                "size_matched_same_l1_maps": ";".join(matched),
                "matched_within_1_fraction": float(np.mean([distances.get(p, 999) <= 1 for p in matched])) if matched else np.nan,
                "matched_within_2_fraction": float(np.mean([distances.get(p, 999) <= 2 for p in matched])) if matched else np.nan,
                "matched_within_3_fraction": float(np.mean([distances.get(p, 999) <= 3 for p in matched])) if matched else np.nan,
                "reachable_same_l1_pathways": ";".join(ordered_same_l1),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "multihop_network_support.csv", index=False)
    evaluable = result.loc[result.network_start_available].copy()

    rng = np.random.default_rng(20260908)
    summary_by_step = []
    for step in (1, 2, 3):
        observed = evaluable[f"predicted_l2_within_{step}_steps"].to_numpy(dtype=bool)
        matched_prob = evaluable[f"matched_within_{step}_fraction"].fillna(0).to_numpy(float)
        observed_count = int(observed.sum())
        null_counts = np.asarray([
            int((rng.random(len(evaluable)) < matched_prob).sum()) for _ in range(10000)
        ])
        null_rate = float(matched_prob.mean())
        observed_rate = float(observed.mean())
        summary_by_step.append(
            {
                "max_reaction_steps": step,
                "n_evaluable": len(evaluable),
                "predicted_pathway_supported": observed_count,
                "predicted_pathway_support_rate": observed_rate,
                "matched_same_l1_support_rate": null_rate,
                "fold_enrichment": observed_rate / null_rate if null_rate else None,
                "permutation_p_value": float((1 + (null_counts >= observed_count).sum()) / 10001),
                "same_l1_network_rank_top1_among_reachable": float(
                    (evaluable.loc[evaluable.predicted_l2_min_steps.notna(), "predicted_l2_same_l1_network_rank"] == 1).mean()
                ),
                "same_l1_network_rank_top3_among_reachable": float(
                    (evaluable.loc[evaluable.predicted_l2_min_steps.notna(), "predicted_l2_same_l1_network_rank"] <= 3).mean()
                ),
            }
        )
    pd.DataFrame(summary_by_step).to_csv(OUT / "multihop_network_summary.csv", index=False)

    summary = {
        "high_confidence_nominations": len(high),
        "rhea_rows": len(rhea),
        "side_aligned_rhea_reactions": aligned_reactions,
        "noncurrency_side_usable_reactions": retained_reactions,
        "currency_degree_cutoff": currency_cutoff,
        "network_start_evaluable_nominations": len(evaluable),
        "by_max_steps": summary_by_step,
        "network_semantics": "Rhea master equations are traversed across opposite sides in both directions; same-side participants are excluded.",
    }
    (OUT / "multihop_network_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import subprocess, sys
    from pathlib import Path
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / 'run_downstream.py'), '--analysis', 'network', *sys.argv[1:]]))

#!/usr/bin/env python3
"""Reassess pathway nominations using direction-aware Rhea evidence.

The original nomination analysis treated every non-currency participant in a
Rhea reaction as a neighbour.  This script uses the ordered Rhea equation and
ChEBI participant fields to distinguish opposite-side biochemical products or
substrates from same-side co-participants.  Reactions whose fields cannot be
aligned unambiguously are retained as indirect evidence only.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOMINATION = ROOT / "downstream/pathway_nomination"
OUT = Path(__file__).resolve().parent
SOURCE = NOMINATION / "external_source"
MODEL_ROOT = ROOT / "MVP_model_0706"


def split_values(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def load_two_column_links(path: Path) -> dict[str, set[str]]:
    links: dict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = line.rstrip("\n").split("\t")
            links[left.split(":")[-1]].add(right.split(":")[-1])
    return links


def parse_rhea_sides(equation: str, chebi_field: str) -> dict[str, object]:
    """Align ordered ChEBI identifiers with equation sides when unambiguous."""
    sides = re.split(r"\s=\s", equation, maxsplit=1)
    chebi = split_values(chebi_field)
    if len(sides) != 2:
        return {"aligned": False, "left": [], "right": [], "left_names": [], "right_names": []}
    names = [
        [token.strip() for token in re.split(r"\s\+\s", side) if token.strip()]
        for side in sides
    ]
    if len(names[0]) + len(names[1]) != len(chebi):
        return {
            "aligned": False,
            "left": [],
            "right": [],
            "left_names": names[0],
            "right_names": names[1],
        }
    cut = len(names[0])
    return {
        "aligned": True,
        "left": chebi[:cut],
        "right": chebi[cut:],
        "left_names": names[0],
        "right_names": names[1],
    }


def join(values: set[str] | list[str]) -> str:
    return ";".join(sorted(set(values)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(NOMINATION / "candidate_catalogue_with_support.csv")
    supported_ids = set(pd.read_csv(NOMINATION / "supported_nominations.csv").kegg_id)

    label_space = json.loads((MODEL_ROOT / "data/label_space_L2.json").read_text())
    model_maps = set(label_space["index_to_map"].values())
    compound_pathways = load_two_column_links(NOMINATION / "cache/link_pathway_compound")
    reaction_pathways = load_two_column_links(SOURCE / "kegg_reaction_to_pathway.tsv")
    enzyme_pathways = load_two_column_links(SOURCE / "kegg_enzyme_to_pathway.tsv")

    kegg_to_chebi: dict[str, set[str]] = defaultdict(set)
    chebi_to_kegg: dict[str, set[str]] = defaultdict(set)
    with (SOURCE / "kegg_compound_to_chebi.tsv").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            kegg, chebi = line.rstrip("\n").split("\t")
            kegg_id = kegg.split(":")[-1]
            chebi_id = "CHEBI:" + chebi.split(":")[-1]
            kegg_to_chebi[kegg_id].add(chebi_id)
            chebi_to_kegg[chebi_id].add(kegg_id)

    rhea = pd.read_csv(SOURCE / "rhea_reactions.tsv", sep="\t", dtype=str).fillna("")
    rhea.columns = ["rhea_id", "equation", "chebi_ids", "ec_numbers", "enzyme_count", "pubmed_ids"]
    reaction_rows: dict[str, dict[str, object]] = {}
    chebi_reactions: dict[str, set[str]] = defaultdict(set)
    participant_degree: Counter[str] = Counter()
    for row in rhea.itertuples(index=False):
        participant_ids = set(split_values(row.chebi_ids))
        parsed = parse_rhea_sides(row.equation, row.chebi_ids)
        parsed.update(
            equation=row.equation,
            participants=participant_ids,
            ecs={x.removeprefix("EC:") for x in split_values(row.ec_numbers)},
            pmids=set(split_values(row.pubmed_ids)),
        )
        reaction_rows[row.rhea_id] = parsed
        for chebi in participant_ids:
            chebi_reactions[chebi].add(row.rhea_id)
            participant_degree[chebi] += 1

    currency_cutoff = max(50, int(np.quantile(list(participant_degree.values()), 0.99)))
    id_to_name = dict(zip(candidates.kegg_id.astype(str), candidates.name.fillna("").astype(str)))
    training = pd.read_csv(MODEL_ROOT / "data/kegg2026_hier.csv")
    if "kegg_id" in training.columns:
        id_to_name.update(dict(zip(training.kegg_id.astype(str), training.get("name", ""))))

    records = []
    for candidate in candidates.loc[candidates.kegg_id.isin(supported_ids)].itertuples(index=False):
        kegg_ids = set(str(candidate.kegg_id).split("|"))
        candidate_chebi = set().union(*(kegg_to_chebi.get(cid, set()) for cid in kegg_ids))
        reaction_ids = set().union(*(chebi_reactions.get(cid, set()) for cid in candidate_chebi))
        predicted_map = candidate.predicted_l2_map

        aligned_reactions: set[str] = set()
        ambiguous_reactions: set[str] = set()
        left_reactions: set[str] = set()
        right_reactions: set[str] = set()
        opposite_chebi: set[str] = set()
        same_side_chebi: set[str] = set()
        direct_support_reactions: set[str] = set()
        direct_support_opposite_kegg: set[str] = set()
        direct_support_pmids: set[str] = set()
        direct_support_ecs: set[str] = set()
        all_ecs: set[str] = set()
        all_pmids: set[str] = set()

        for reaction_id in reaction_ids:
            reaction = reaction_rows[reaction_id]
            all_ecs.update(reaction["ecs"])
            all_pmids.update(reaction["pmids"])
            if not reaction["aligned"]:
                ambiguous_reactions.add(reaction_id)
                continue
            left = set(reaction["left"])
            right = set(reaction["right"])
            on_left = bool(candidate_chebi & left)
            on_right = bool(candidate_chebi & right)
            if on_left == on_right:
                ambiguous_reactions.add(reaction_id)
                continue
            aligned_reactions.add(reaction_id)
            if on_left:
                left_reactions.add(reaction_id)
                opposite = right
                same_side = left - candidate_chebi
            else:
                right_reactions.add(reaction_id)
                opposite = left
                same_side = right - candidate_chebi
            opposite = {
                cid for cid in opposite
                if cid not in candidate_chebi and participant_degree[cid] < currency_cutoff
            }
            same_side = {
                cid for cid in same_side
                if cid not in candidate_chebi and participant_degree[cid] < currency_cutoff
            }
            opposite_chebi.update(opposite)
            same_side_chebi.update(same_side)

            opposite_kegg = set().union(*(chebi_to_kegg.get(cid, set()) for cid in opposite))
            pathway_kegg = {
                cid for cid in opposite_kegg if predicted_map in compound_pathways.get(cid, set())
            }
            if pathway_kegg:
                direct_support_reactions.add(reaction_id)
                direct_support_opposite_kegg.update(pathway_kegg)
                direct_support_pmids.update(reaction["pmids"])
                direct_support_ecs.update(reaction["ecs"])

        opposite_kegg_all = set().union(*(chebi_to_kegg.get(cid, set()) for cid in opposite_chebi))
        same_side_kegg_all = set().union(*(chebi_to_kegg.get(cid, set()) for cid in same_side_chebi))
        opposite_maps: set[str] = set()
        for cid in opposite_kegg_all:
            opposite_maps.update(compound_pathways.get(cid, set()) & model_maps)
        same_side_predicted = {
            cid for cid in same_side_kegg_all if predicted_map in compound_pathways.get(cid, set())
        }

        kegg_reactions = set(split_values(candidate.kegg_reactions))
        kegg_support = {rid for rid in kegg_reactions if predicted_map in reaction_pathways.get(rid, set())}
        ec_support = {ec for ec in all_ecs if predicted_map in enzyme_pathways.get(ec, set())}
        has_direct = bool(direct_support_reactions)
        has_mechanistic = bool(ec_support or kegg_support)
        if has_direct and has_mechanistic:
            tier = "Strong"
        elif has_direct:
            tier = "Direct"
        elif has_mechanistic or same_side_predicted or bool(candidate.predicted_l2_network_supported):
            tier = "Indirect"
        else:
            tier = "Unresolved"

        if direct_support_reactions:
            roles = []
            if left_reactions & direct_support_reactions:
                roles.append("substrate/left")
            if right_reactions & direct_support_reactions:
                roles.append("product/right")
            reaction_role = ";".join(roles)
        else:
            reaction_role = "unresolved" if ambiguous_reactions else "no direct pathway reaction"

        if has_direct:
            n_maps = len(opposite_maps)
            pathway_specificity = "high" if n_maps <= 2 else "moderate" if n_maps <= 5 else "broad"
        else:
            pathway_specificity = "not directly supported"

        opposite_names = [
            f"{cid}:{id_to_name.get(cid, '')}".rstrip(":") for cid in sorted(direct_support_opposite_kegg)
        ]
        direct_equations = [reaction_rows[rid]["equation"] for rid in sorted(direct_support_reactions)]
        limitations = []
        if ambiguous_reactions:
            limitations.append(f"{len(ambiguous_reactions)} Rhea reaction(s) could not be side-aligned")
        if len(candidate_chebi) > 1:
            limitations.append("multiple KEGG-ChEBI cross-references")
        if not has_direct:
            limitations.append("no unambiguous opposite-side compound in nominated L2")
        if candidate.scaffold_status == "seen" and candidate.max_train_ecfp4_tanimoto >= 0.8:
            limitations.append("close to a training compound")

        records.append(
            {
                "kegg_id": candidate.kegg_id,
                "name": candidate.name,
                "smiles": candidate.smiles,
                "predicted_l1_name": candidate.predicted_l1_name,
                "predicted_l2_map": predicted_map,
                "predicted_l2_name": candidate.predicted_l2_name,
                "l2_ensemble_probability": candidate.l2_ensemble_probability,
                "max_train_ecfp4_tanimoto": candidate.max_train_ecfp4_tanimoto,
                "scaffold_status": candidate.scaffold_status,
                "nearest_train_kegg_id": candidate.nearest_train_kegg_id,
                "candidate_chebi_ids": join(candidate_chebi),
                "identity_mapping_status": "unique" if len(candidate_chebi) == 1 else "multiple" if candidate_chebi else "unmapped",
                "rhea_reaction_count": len(reaction_ids),
                "side_aligned_reaction_count": len(aligned_reactions),
                "side_ambiguous_reaction_count": len(ambiguous_reactions),
                "candidate_reaction_role": reaction_role,
                "direct_support": has_direct,
                "direct_supporting_rhea_reactions": join(direct_support_reactions),
                "direct_supporting_equations": " | ".join(direct_equations),
                "opposite_side_pathway_compounds": join(direct_support_opposite_kegg),
                "opposite_side_pathway_compound_names": " | ".join(opposite_names),
                "direct_supporting_ec_numbers": join(direct_support_ecs),
                "direct_supporting_pubmed_ids": join(direct_support_pmids),
                "all_rhea_ec_numbers": join(all_ecs),
                "all_rhea_pubmed_ids": join(all_pmids),
                "ec_to_nominated_pathway_support": bool(ec_support),
                "supporting_ec_numbers": join(ec_support),
                "kegg_reaction_to_nominated_pathway_support": bool(kegg_support),
                "supporting_kegg_reactions": join(kegg_support),
                "same_side_only_pathway_compounds": join(same_side_predicted - direct_support_opposite_kegg),
                "opposite_side_supported_model_pathway_count": len(opposite_maps),
                "pathway_specificity": pathway_specificity,
                "evidence_tier": tier,
                "limitations": "; ".join(limitations),
            }
        )

    review = pd.DataFrame(records)
    tier_order = pd.CategoricalDtype(["Strong", "Direct", "Indirect", "Unresolved"], ordered=True)
    review["evidence_tier"] = review.evidence_tier.astype(tier_order)
    review = review.sort_values(
        ["evidence_tier", "max_train_ecfp4_tanimoto", "l2_ensemble_probability"],
        ascending=[True, True, False],
    )
    review["evidence_tier"] = review.evidence_tier.astype(str)
    review.to_csv(OUT / "candidate_evidence_review.csv", index=False)

    summary = {
        "input_supported_nominations": len(review),
        "rhea_equation_rows": len(rhea),
        "rhea_equations_side_aligned": int(sum(parse_rhea_sides(r.equation, r.chebi_ids)["aligned"] for r in rhea.itertuples(index=False))),
        "currency_participant_degree_cutoff": currency_cutoff,
        "tier_counts": review.evidence_tier.value_counts().to_dict(),
        "unambiguous_direct_support": int(review.direct_support.sum()),
        "direct_support_with_pubmed": int((review.direct_support & review.direct_supporting_pubmed_ids.ne("")).sum()),
        "direct_support_max_tanimoto_below_0_8": int((review.direct_support & review.max_train_ecfp4_tanimoto.lt(0.8)).sum()),
        "direct_support_unseen_scaffold": int((review.direct_support & review.scaffold_status.eq("unseen")).sum()),
    }
    (OUT / "evidence_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

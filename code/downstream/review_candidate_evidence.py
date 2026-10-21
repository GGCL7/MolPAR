"""Parsers from the study reaction-evidence analysis."""
from pathlib import Path
from collections import defaultdict
import re

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

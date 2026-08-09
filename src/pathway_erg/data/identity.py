"""Identity resolution.

Produces canonical global IDs and PERG repeat-link components.

- LEOP: participant ID defines the canonical person; recording IDs are
  collision-safe (never relying on wave_id alone).
- PERG: `rep_record` references other records (`Id:XXXX - Id:YYYY ...`); all
  visit records connected through these references form one canonical subject
  so that every visit of a repeat-connected subject stays in one fold.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .schemas import SplitAssignment, VisitRecord, WaveformRecord

PERG_ID_PATTERN = re.compile(r"(?i)id[:#]?\s*([a-z0-9]{1,8})")


def parse_repeat_edges_with_source(rep_record: str | None) -> list[str]:
    """Extract referenced IDs from one rep_record cell."""
    if not rep_record:
        return []
    return [m.group(1) for m in PERG_ID_PATTERN.finditer(rep_record)]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            key, self.parent[key] = self.parent[key], root
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for key in self.parent:
            groups[self.find(key)].append(key)
        return dict(sorted(groups.items()))


def connected_subject_components(visit_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    """Map every visit id to a canonical component id.

    `visit_ids` are already-resolved record ids (e.g. "Id:0003");
    `edges` are (a, b) pairs of ids that are the same person.
    """
    uf = UnionFind()
    for vid in visit_ids:
        uf.find(vid)
    for a, b in edges:
        uf.union(a, b)
    comp = {}
    for key in visit_ids:
        comp[key] = uf.find(key)
    return comp


def canonical_perg_subject_id(component_id: str) -> str:
    """Stable canonical subject id from a component root id."""
    return f"PERG_SUBJ_{component_id.removeprefix('Id:')}"


def assign_perg_identity_edges(
    visits: list[VisitRecord], rep_records: dict[str, str | None]
) -> dict[str, str]:
    """Map global_visit_id -> canonical global_subject_id using rep_record."""
    record_to_visit = {v.source_record_id: v.global_visit_id for v in visits}
    edges: list[tuple[str, str]] = []
    for record_id, rep in rep_records.items():
        for ref in parse_repeat_edges_with_source(rep):
            ref_norm = f"{int(ref):04d}"
            if ref_norm in record_to_visit:
                edges.append((record_id, ref_norm))
    ids = list(record_to_visit)
    comp = connected_subject_components(ids, edges)
    mapping = {}
    for record_id, root in comp.items():
        mapping[record_to_visit[record_id]] = canonical_perg_subject_id(root)
    return mapping


def assert_partition_disjointness(assignments: list[SplitAssignment]) -> None:
    """Assert no unit appears in two different outer partitions."""
    seen: dict[tuple[str, str], str] = {}
    for a in assignments:
        key = (a.dataset.value, a.unit_id)
        if key in seen and seen[key] != f"{a.outer_fold}:{a.partition}":
            raise ValueError(
                f"unit {key} assigned to both {seen[key]} and {a.outer_fold}:{a.partition}"
            )
        seen[key] = f"{a.outer_fold}:{a.partition}"


def assert_unique_recording_ids(records: list[WaveformRecord]) -> None:
    ids = [r.global_recording_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global_recording_id values detected")

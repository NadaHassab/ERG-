"""Versioned label construction.

Every endpoint is locked in a versioned, reviewable mapping table.  There is
no fallback or default class: every raw label must be mapped explicitly, and
null labels make a unit ineligible for that endpoint.

LEOP primary endpoint: Control (0) versus ASD (1); ASD+ADHD is excluded from
primary training (explored separately).  PERG primary endpoint:
Normal (0) versus any non-empty non-normal diagnosis1 (1); null diagnosis1 is
ineligible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..provenance import sha256_text

ENDPOINT_LEOPS_PRIMARY = "leops_control_vs_asd"
ENDPOINT_PERG_PRIMARY = "perg_normal_vs_abnormal"

LEOPS_MAPPING_VERSION = "leops_labels_v1"
PERG_MAPPING_VERSION = "perg_labels_v1"


@dataclass
class DiagnosisMapping:
    endpoint: str
    version: str
    table: pd.DataFrame  # columns: raw_label, mapped_value, rationale
    reviewer: str
    review_date: str

    @property
    def mapping_hash(self) -> str:
        canonical = self.table.sort_values("raw_label").to_csv(index=False)
        return sha256_text(f"{self.version}\n{canonical}")

    def map(self, raw_label: str | None) -> int | None:
        if raw_label is None or str(raw_label).strip() == "":
            return None
        norm = str(raw_label).strip()
        row = self.table[self.table["raw_label"] == norm]
        if row.empty:
            raise ValueError(f"unmapped label {raw_label!r} for endpoint {self.endpoint}")
        value = row.iloc[0]["mapped_value"]
        if pd.isna(value):
            return None
        return int(value)


def make_leops_target(group_raw: str | None, endpoint: str) -> int | None:
    """LEOP endpoint mapping.  None for ineligible units."""
    if endpoint != ENDPOINT_LEOPS_PRIMARY:
        raise ValueError(f"unknown LEOP endpoint {endpoint}")
    if group_raw is None:
        return None
    group = str(group_raw).strip()
    if group == "Control":
        return 0
    if group == "ASD":
        return 1
    return None  # ASD+ADHD and any unknown group are excluded from primary


def build_perg_mapping(observed: pd.Series, endpoint: str) -> DiagnosisMapping:
    """Construct the explicit PERG diagnosis mapping for an endpoint.

    Requires clinician review before locking; this function only produces the
    candidate table with counts.
    """
    if endpoint != ENDPOINT_PERG_PRIMARY:
        raise ValueError(f"unknown PERG endpoint {endpoint}")
    counts = observed.dropna().astype(str).str.strip().value_counts().sort_index()
    rows = []
    for label, count in counts.items():
        if label == "Normal":
            mapped, rationale = 0, "Reference clinical normal group"
        else:
            mapped, rationale = 1, "Any non-normal diagnosis (heterogeneous endpoint)"
        rows.append({"raw_label": label, "mapped_value": mapped, "count": int(count), "rationale": rationale})
    table = pd.DataFrame(rows)
    return DiagnosisMapping(
        endpoint=endpoint,
        version=PERG_MAPPING_VERSION,
        table=table,
        reviewer="PENDING_CLINICAL_REVIEW",
        review_date="",
    )


def make_perg_target(diagnosis_raw: str | None, mapping: DiagnosisMapping) -> int | None:
    return mapping.map(diagnosis_raw)


def audit_label_coverage(observed: pd.Series, mapping: DiagnosisMapping) -> dict:
    """Report coverage: every observed label must be in the mapping."""
    observed = observed.dropna().astype(str).str.strip()
    mapped = set(mapping.table["raw_label"])
    missing = sorted(set(observed) - mapped)
    return {
        "observed_labels": len(set(observed)),
        "mapped_labels": len(mapped),
        "unmapped": missing,
        "complete": not missing,
    }


def write_mapping_csv(mapping: DiagnosisMapping, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "endpoint": mapping.endpoint,
        "version": mapping.version,
        "reviewer": mapping.reviewer,
        "review_date": mapping.review_date,
        "mapping_hash": mapping.mapping_hash,
    }
    mapping.table.to_csv(path, index=False)
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path

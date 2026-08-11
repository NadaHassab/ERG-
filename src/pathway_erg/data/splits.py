"""Grouped nested fold construction and leakage prevention.

One locked nested grouped evaluation is used by every model:

- LEOP: outer/inner folds group by participant; constraints balance the
  primary endpoint, site, sex, and age bins; all curves/eyes/OP stay with the
  person.
- PERG: folds group by canonical (repeat-link resolved) subject; constraints
  balance normal/abnormal, sex, age bins, and visit counts; all visits and
  sessions stay with the subject.

Deterministic stratified greedy assignment minimizes deviations of each
constraint dimension from its target proportion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import INNER_FOLDS_TEMPLATE, OUTER_FOLDS_TEMPLATE
from ..provenance import sha256_text
from .schemas import Dataset, Partition, SplitAssignment


@dataclass(frozen=True)
class FoldConstraint:
    """One categorical constraint whose proportions should be balanced."""

    column: str
    weight: float


@dataclass(frozen=True)
class FoldConfig:
    n_outer: int
    n_inner: int
    outer_seed: int
    inner_seed: int
    version: str
    age_bins: tuple[float, ...]
    constraints: tuple[FoldConstraint, ...]


def _age_bin(age: float | None, bins: tuple[float, ...]) -> str:
    """Assign an age to one of len(bins)+1 intervals defined by the cutpoints."""
    if age is None or pd.isna(age):
        return "unknown"
    idx = sum(1 for b in bins if age >= b)
    if idx == 0:
        return f"<{bins[0]:g}"
    if idx == len(bins):
        return f"{bins[-1]:g}+"
    return f"{bins[idx - 1]:g}-{bins[idx]:g}"


def _unit_frame(
    subjects_df: pd.DataFrame, visits_df: pd.DataFrame, dataset: str, age_bins: tuple[float, ...]
) -> pd.DataFrame:
    """One row per split unit with aggregated constraint columns."""
    if dataset == "LEOP":
        units = subjects_df[subjects_df["dataset"] == "LEOP"].copy()
        units["unit_id"] = units["global_subject_id"]
        units["class"] = units["group_raw"].map({"Control": 0, "ASD": 1}).astype("Float64")
        units["n_visits"] = 1
    else:
        units = subjects_df[subjects_df["dataset"] == "PERG"].copy()
        units["unit_id"] = units["global_subject_id"]
        v = visits_df[visits_df["dataset"] == "PERG"]
        target = v.groupby("global_subject_id")["target_binary"].first()
        n_visits = v.groupby("global_subject_id").size()
        units["class"] = units["unit_id"].map(target)
        units["n_visits"] = units["unit_id"].map(n_visits).fillna(0).astype(int)
    units["age_bin"] = units["age_years"].map(lambda a: _age_bin(a, age_bins))
    units["sex"] = units["sex_standardized"].fillna("unknown")
    return units.reset_index(drop=True)


def _deviation(
    units: pd.DataFrame, folds: dict[str, int], n_folds: int, constraints: tuple[FoldConstraint, ...]
) -> float:
    """Weighted deviation of observed fold proportions from targets.

    Weights come from the fold config constraints (class, site, sex, age-bin
    balance) plus visit-count balance.
    """
    total = len(units)
    if total == 0:
        return 0.0
    target = 1.0 / n_folds
    cost = 0.0
    counts = pd.DataFrame(
        {"unit_id": units["unit_id"], "fold": units["unit_id"].map(folds)}
    )
    for constraint in constraints:
        col, weight = constraint.column, constraint.weight
        if col not in units:
            continue
        for value in units[col].dropna().unique():
            n = int((units[col] == value).sum())
            if n == 0:
                continue
            observed = np.array(
                [int(((units[col] == value) & (counts["fold"] == f)).sum()) for f in range(n_folds)]
            )
            prop = observed / n
            cost += weight * float(np.sum(np.abs(prop - target)))
    visit_weights = units["n_visits"].astype(float)
    total_visits = float(visit_weights.sum())
    if total_visits > 0:
        for f in range(n_folds):
            got = float(visit_weights[counts["fold"] == f].sum())
            cost += abs(got / total_visits - target)
    return cost


def _greedy_assign(
    units: pd.DataFrame, n_folds: int, seed: int, constraints: tuple[FoldConstraint, ...]
) -> dict[str, int]:
    """Deterministic greedy assignment minimizing constraint deviation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    folds: dict[str, int] = {}
    for idx in order:
        unit = units.iloc[idx]
        best_fold, best_cost = 0, float("inf")
        for f in range(n_folds):
            folds[unit["unit_id"]] = f
            cost = _deviation(units, folds, n_folds, constraints)
            if cost < best_cost:
                best_cost, best_fold = cost, f
        folds[unit["unit_id"]] = best_fold
    return folds


def make_outer_folds(
    subjects_df: pd.DataFrame, visits_df: pd.DataFrame, config: FoldConfig
) -> pd.DataFrame:
    """Outer grouped folds.  Returns rows unit_id, dataset, outer_fold."""
    frames = []
    for dataset in ("LEOP", "PERG"):
        units = _unit_frame(subjects_df, visits_df, dataset, config.age_bins)
        assignments = _greedy_assign(units, config.n_outer, config.outer_seed, config.constraints)
        out = units[["unit_id"]].copy()
        out["dataset"] = dataset
        out["outer_fold"] = out["unit_id"].map(assignments)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def make_inner_folds(
    outer_assignments: pd.DataFrame,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    outer_fold: int,
    config: FoldConfig,
) -> pd.DataFrame:
    """Inner grouped folds within one outer training partition."""
    keep = outer_assignments["outer_fold"] != outer_fold
    outer_train = outer_assignments[keep]
    frames = []
    for dataset in ("LEOP", "PERG"):
        units = _unit_frame(subjects_df, visits_df, dataset, config.age_bins)
        units = units[units["unit_id"].isin(set(outer_train[outer_train["dataset"] == dataset]["unit_id"]))]
        if units.empty:
            continue
        assignments = _greedy_assign(units, config.n_inner, config.inner_seed + outer_fold, config.constraints)
        out = units[["unit_id"]].copy()
        out["dataset"] = dataset
        out["outer_fold"] = outer_fold
        out["inner_fold"] = out["unit_id"].map(assignments)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def assignment_records(
    outer: pd.DataFrame, inner: pd.DataFrame | None = None
) -> list[SplitAssignment]:
    """Expand fold tables into per-partition SplitAssignment records."""
    records: list[SplitAssignment] = []
    inner_map: dict[tuple[str, str], int] = {}
    if inner is not None:
        inner_map = {
            (r.dataset, r.unit_id): int(r.inner_fold) for r in inner.itertuples(index=False)
        }
    for r in outer.itertuples(index=False):
        unit, ds, fold = r.unit_id, r.dataset, int(r.outer_fold)
        key = (ds, unit)
        inner_fold = inner_map.get(key)
        records.append(
            SplitAssignment(
                unit_id=unit,
                dataset=Dataset(ds),
                outer_fold=fold,
                partition=Partition.OUTER_TRAIN if inner_fold is not None else Partition.OUTER_TEST,
                inner_fold=inner_fold,
            )
        )
    return records


def summarize_folds(
    assignments: pd.DataFrame,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    age_bins: tuple[float, ...],
) -> dict[str, Any]:
    """Fold balance report per dataset and fold."""
    report: dict[str, Any] = {}
    for dataset in ("LEOP", "PERG"):
        sub = assignments[assignments["dataset"] == dataset]
        report[dataset] = {}
        for fold in sorted(sub["outer_fold"].unique()):
            unit_ids = set(sub[sub["outer_fold"] == fold]["unit_id"])
            units = subjects_df[subjects_df["global_subject_id"].isin(unit_ids)]
            v = visits_df[visits_df["global_subject_id"].isin(unit_ids)]
            report[dataset][int(fold)] = {
                "n_units": len(unit_ids),
                "n_visits": int(v["global_visit_id"].nunique()),
                "class_counts": (
                    units["group_raw"].value_counts().to_dict()
                    if dataset == "LEOP"
                    else v["target_binary"].fillna(-1).value_counts().to_dict()
                ),
                "site_counts": units["site"].value_counts().to_dict() if dataset == "LEOP" else {},
                "sex_counts": units["sex_standardized"].fillna("unknown").value_counts().to_dict(),
                "age_bin_counts": units["age_years"].map(lambda a: _age_bin(a, age_bins)).value_counts().to_dict(),
            }
    return report


def assert_no_leakage(
    assignments: pd.DataFrame,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
) -> None:
    """Every leakage assertion from plan Section 13.5 that can be checked statically.

Dataset-agnostic: every dataset present in ``assignments`` (LEOP/PERG or
external URFU/FLINDERS) undergoes the same unit-in-one-fold, visit, and
recording coverage checks; external assignments are subject-keyed exactly
like the LEOP/PERG ones.
"""

    def _fails(msg: str) -> None:
        raise AssertionError(f"leakage assertion failed: {msg}")

    for dataset in sorted(assignments["dataset"].unique()):
        sub = assignments[assignments["dataset"] == dataset]
        for unit, group in sub.groupby("unit_id"):
            if group["outer_fold"].nunique() > 1:
                _fails(f"{dataset} unit {unit} in multiple outer folds")
    assigned_datasets = set(assignments["dataset"].unique())
    visits = visits_df[visits_df["dataset"].isin(assigned_datasets)].merge(
        assignments[["unit_id", "outer_fold"]].rename(columns={"unit_id": "global_subject_id"}),
        on="global_subject_id",
        how="left",
    )
    if visits["outer_fold"].isna().any():
        _fails("some visits lack an outer-fold assignment")
    rec = recordings_df[recordings_df["dataset"].isin(assigned_datasets)].merge(
        visits[["global_visit_id", "outer_fold"]], on="global_visit_id", how="left"
    )
    if rec["outer_fold"].isna().any():
        _fails("some recordings lack an outer-fold assignment")
    dupes = rec.groupby(["dataset", "source_wave_id"])["outer_fold"].nunique()
    if (dupes > 1).any():
        _fails("duplicate source waveform across partitions")
    pair_folds = rec.groupby(["dataset", "erg_pair_id"])["outer_fold"].nunique().dropna()
    if (pair_folds > 1).any():
        _fails("ERG/OP pair split across partitions")


def write_splits(
    outer: pd.DataFrame,
    inner_by_fold: dict[int, pd.DataFrame],
    report: dict[str, Any],
    artifact_root: Path,
    split_version: str,
) -> dict[str, Path]:
    """Persist split manifests plus a summary and split hash."""
    split_dir = Path(artifact_root) / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    outer_path = split_dir / OUTER_FOLDS_TEMPLATE.format(version=split_version)
    inner_path = split_dir / INNER_FOLDS_TEMPLATE.format(version=split_version)
    outer.to_parquet(outer_path, index=False)
    inner_all = pd.concat(
        [df.assign(outer_fold_sel=k) for k, df in inner_by_fold.items()], ignore_index=True
    )
    inner_all.to_parquet(inner_path, index=False)
    split_hash = sha256_text(
        outer.sort_values(["dataset", "unit_id", "outer_fold"]).to_csv(index=False)
    )
    summary = {
        "version": split_version,
        "split_hash": split_hash,
        "report": report,
    }
    summary_path = split_dir / f"split_summary_{split_version}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {
        "outer": outer_path,
        "inner": inner_path,
        "summary": summary_path,
    }

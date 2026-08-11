"""External (URFU/FLINDERS) grouped fold construction (plan integration §11.2.3).

Fold units are subjects for both external datasets so a subject's visits and
sessions always stay inside one outer fold (subject-level leakage
discipline).  URFU *training* bags are visits and FLINDERS bags are subjects,
but every folds row is keyed by subject — the same pattern PERG uses (fold by
canonical subject, bag by visit).  The locked external table is written to
its own ``outer_folds_external_<version>.parquet``; the frozen LEOP/PERG
tables are never rewritten.

Deterministic stratified greedy assignment mirrors ``splits.py``; constraints
only reference columns that exist for the external datasets (age bin, sex,
visit count; class only where labels exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .splits import (
    FoldConfig,
    _age_bin,
    _greedy_assign,
    assert_no_leakage,
    write_splits,
)

EXTERNAL_DATASETS = ("URFU", "FLINDERS")
EXTERNAL_FOLD_VERSION = "external_v1"


@dataclass(frozen=True)
class ExternalSplitResult:
    outer: pd.DataFrame
    inner_by_fold: dict[int, pd.DataFrame]
    report: dict[str, Any]
    version: str
    paths: dict[str, Path]


def external_unit_frame(
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    dataset: str,
    age_bins: tuple[float, ...],
) -> pd.DataFrame:
    """One row per external split unit (subject) with constraint columns.

    URFU subjects may carry a target (first visit diagnosis, possibly
    missing); FLINDERS is healthy-only so its class is always 0.
    """
    if dataset not in EXTERNAL_DATASETS:
        raise ValueError(f"external datasets are {EXTERNAL_DATASETS}, got {dataset!r}")
    units = subjects_df[subjects_df["dataset"] == dataset].copy()
    units["unit_id"] = units["global_subject_id"]
    v = visits_df[visits_df["dataset"] == dataset]
    if dataset == "URFU":
        target = v.groupby("global_subject_id")["target_binary"].first()
        units["class"] = units["unit_id"].map(target)
    else:
        units["class"] = 0
    n_visits = v.groupby("global_subject_id").size()
    units["n_visits"] = units["unit_id"].map(n_visits).fillna(0).astype(int)
    units["age_bin"] = units["age_years"].map(lambda a: _age_bin(a, age_bins))
    units["sex"] = units["sex_standardized"].fillna("unknown")
    return units.reset_index(drop=True)


def make_external_outer_folds(
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    config: FoldConfig,
    datasets: tuple[str, ...] = EXTERNAL_DATASETS,
) -> pd.DataFrame:
    """Outer grouped folds keyed by subject for the external datasets."""
    frames = []
    for dataset in datasets:
        units = external_unit_frame(subjects_df, visits_df, dataset, config.age_bins)
        if units.empty:
            continue
        assignments = _greedy_assign(units, config.n_outer, config.outer_seed, config.constraints)
        out = units[["unit_id"]].copy()
        out["dataset"] = dataset
        out["outer_fold"] = out["unit_id"].map(assignments)
        frames.append(out)
    if not frames:
        raise ValueError("no external subjects to fold")
    return pd.concat(frames, ignore_index=True)


def make_external_inner_folds(
    outer_assignments: pd.DataFrame,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    outer_fold: int,
    config: FoldConfig,
    datasets: tuple[str, ...] = EXTERNAL_DATASETS,
) -> pd.DataFrame:
    """Inner grouped folds within one outer external training partition."""
    keep = outer_assignments["outer_fold"] != outer_fold
    outer_train = outer_assignments[keep]
    frames = []
    for dataset in datasets:
        units = external_unit_frame(subjects_df, visits_df, dataset, config.age_bins)
        units = units[units["unit_id"].isin(set(outer_train.loc[outer_train["dataset"] == dataset, "unit_id"]))]
        if units.empty:
            continue
        assignments = _greedy_assign(units, config.n_inner, config.inner_seed + outer_fold, config.constraints)
        out = units[["unit_id"]].copy()
        out["dataset"] = dataset
        out["outer_fold"] = outer_fold
        out["inner_fold"] = out["unit_id"].map(assignments)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def summarize_external_folds(
    assignments: pd.DataFrame,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    age_bins: tuple[float, ...],
) -> dict[str, Any]:
    """Fold balance report per external dataset and fold."""
    report: dict[str, Any] = {}
    for dataset in sorted(assignments["dataset"].unique()):
        sub = assignments[assignments["dataset"] == dataset]
        report[dataset] = {}
        for fold in sorted(sub["outer_fold"].unique()):
            unit_ids = set(sub[sub["outer_fold"] == fold]["unit_id"])
            units = subjects_df[subjects_df["global_subject_id"].isin(unit_ids)]
            v = visits_df[visits_df["global_subject_id"].isin(unit_ids)]
            classes = None
            if dataset == "URFU":
                classes = v["target_binary"].fillna(-1).value_counts().to_dict()
            else:
                classes = {"healthy": 0}
            report[dataset][int(fold)] = {
                "n_units": len(unit_ids),
                "n_visits": int(v["global_visit_id"].nunique()),
                "class_counts": classes,
                "sex_counts": units["sex_standardized"].fillna("unknown").value_counts().to_dict(),
                "age_bin_counts": units["age_years"].map(lambda a: _age_bin(a, age_bins)).value_counts().to_dict(),
            }
    return report


def build_external_splits(
    artifact_root: str | Path,
    subjects_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
    config: FoldConfig,
    datasets: tuple[str, ...] = EXTERNAL_DATASETS,
    version: str = EXTERNAL_FOLD_VERSION,
) -> ExternalSplitResult:
    """Build, validate, and persist the locked external split tables."""
    subj = subjects_df[subjects_df["dataset"].isin(datasets)]
    vis = visits_df[visits_df["dataset"].isin(datasets)]
    rec = recordings_df[recordings_df["dataset"].isin(datasets)]
    outer = make_external_outer_folds(subj, vis, config, datasets)
    assert_no_leakage(outer, subj, vis, rec)
    inner_by_fold: dict[int, pd.DataFrame] = {}
    for fold in sorted(outer["outer_fold"].unique()):
        inner = make_external_inner_folds(outer, subj, vis, fold, config, datasets)
        if not inner.empty:
            inner_by_fold[int(fold)] = inner
    report = summarize_external_folds(outer, subj, vis, config.age_bins)
    paths = write_splits(outer, inner_by_fold, report, artifact_root, version)
    return ExternalSplitResult(
        outer=outer,
        inner_by_fold=inner_by_fold,
        report=report,
        version=version,
        paths=paths,
    )
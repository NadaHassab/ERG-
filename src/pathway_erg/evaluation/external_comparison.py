"""Authoritative paired comparison for the four-domain SSL-init arm."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import BOOTSTRAP_SEED, DEFAULT_CONFIDENCE, DEFAULT_N_BOOTSTRAP_REPS
from .comparisons import paired_compare


REQUIRED_COLUMNS = {
    "task",
    "outer_fold",
    "unit_id",
    "subject_id",
    "target",
    "calibrated_probability",
}


def _task_frame(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    out = predictions.loc[
        predictions["task"].eq(task),
        ["outer_fold", "unit_id", "subject_id", "target", "calibrated_probability"],
    ].copy()
    if out.empty:
        raise ValueError(f"prediction table has no {task} rows")
    if set(out["outer_fold"].astype(int)) != set(range(5)):
        raise ValueError(f"{task}: incomplete outer-fold coverage")
    if out["unit_id"].duplicated().any():
        raise ValueError(f"{task}: duplicate unit_id rows")
    if out.groupby("subject_id")["target"].nunique().gt(1).any():
        raise ValueError(f"{task}: inconsistent labels within a subject")
    probabilities = out["calibrated_probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{task}: probabilities must be finite and in [0, 1]")
    return out.sort_values(["outer_fold", "unit_id"], kind="stable").reset_index(drop=True)


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    n_tests = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (n_tests - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def compare_external_sslinit(
    external_path: str | Path,
    internal_path: str | Path,
    output_path: str | Path,
    *,
    tasks: tuple[str, ...] = ("LEOP", "PERG"),
    metric: str = "roc_auc",
    seed: int = BOOTSTRAP_SEED,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    n_perm: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, object]:
    """Compare external-minus-internal predictions on exactly paired OOF units."""
    external_path = Path(external_path)
    internal_path = Path(internal_path)
    output_path = Path(output_path)
    external = pd.read_parquet(external_path)
    internal = pd.read_parquet(internal_path)

    reports: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    for task in tasks:
        model_a = _task_frame(external, task)
        model_b = _task_frame(internal, task)
        for column in ("outer_fold", "unit_id", "subject_id", "target"):
            if not np.array_equal(model_a[column].to_numpy(), model_b[column].to_numpy()):
                raise ValueError(f"{task}: models differ on paired column {column!r}")
        adapt = {
            "target": "y_true",
            "calibrated_probability": "y_prob",
        }
        comparison = paired_compare(
            model_a.rename(columns=adapt),
            model_b.rename(columns=adapt),
            cluster_col="subject_id",
            metric=metric,
            seed=seed,
            n_reps=n_reps,
            n_perm=n_perm,
            confidence=confidence,
        )
        reports[task] = asdict(comparison)
        raw_p_values[task] = comparison.p_value

    adjusted = _holm_adjust(raw_p_values)
    for task, p_value in adjusted.items():
        reports[task]["p_value_holm"] = p_value

    result: dict[str, object] = {
        "difference": "four_domain_external_sslinit - leop_perg_only_sslinit",
        "metric": metric,
        "external_predictions": str(external_path),
        "internal_predictions": str(internal_path),
        "seed": seed,
        "n_bootstrap_reps": n_reps,
        "n_permutations": n_perm,
        "confidence": confidence,
        "tasks": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result

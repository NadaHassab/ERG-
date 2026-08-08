"""Fold-safe robust amplitude scaling (plan Step 12.13).

Median + 1.4826*MAD scales are fitted on outer-training people only, per
(dataset, protocol, waveform_kind, component) stratum, and reused on
validation/test.  No per-curve standard-deviation division ever.

Degenerate strata (empty, or zero finite spread) raise: a silent identity
scaler would masquerade a broken stratum as a fitted one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import MAD_SCALE
from ..provenance import RunManifest, sha256_text


@dataclass(frozen=True)
class RobustScaler:
    median: float
    scale: float
    fit_count: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.median) / self.scale

    def to_dict(self) -> dict:
        return {
            "median": self.median,
            "scale": self.scale,
            "fit_count": self.fit_count,
        }


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Median / (1.4826*MAD); raises on empty or zero-spread input."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot fit a scaler on an empty stratum")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = MAD_SCALE * mad
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError(f"degenerate stratum: zero or non-finite spread (MAD={mad})")
    return median, scale


def stratum_key(recording: pd.Series) -> tuple:
    kind = recording["waveform_kind"]
    return (recording["dataset"], recording["protocol"], kind)


def fit_fold_safe_scalers(
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    folds: pd.DataFrame,
    canonical_arrays: np.ndarray,
    outer_fold: int,
    cache_dir: Path,
) -> dict[tuple, RobustScaler]:
    """Fit scalers for one outer fold on outer-training people only.

    components/recordings must carry global_subject_id; folds maps unit_id to
    (outer_fold, is_train).  Scalers are keyed by (dataset, protocol, kind,
    component_id) and saved with a hash for provenance.
    """
    # people in this fold's outer train set (outer_fold != f), test = outer_fold == f
    train_units = set(folds[(folds["outer_fold"] != outer_fold)]["unit_id"])
    test_units = set(folds[(folds["outer_fold"] == outer_fold)]["unit_id"])
    overlap = train_units & test_units
    if overlap:
        raise RuntimeError(f"leakage: {len(overlap)} units in both train and test")
    rec = recordings.merge(
        folds[["unit_id", "outer_fold"]],
        left_on="global_subject_id",
        right_on="unit_id",
        how="inner",
    )
    # also verify every recording's subject is present in the fold table
    missing = recordings[~recordings["global_subject_id"].isin(folds["unit_id"])]["global_subject_id"].nunique()
    if missing:
        raise RuntimeError(f"{missing} subjects missing from fold table")

    # align canonical arrays to component rows
    comp = components.copy()
    comp = comp.merge(
        rec[["global_recording_id", "global_subject_id", "dataset", "protocol", "waveform_kind"]],
        on="global_recording_id",
        how="left",
    )
    comp = comp[comp["global_subject_id"].notna()]
    # per-component row index into the canonical array (components.parquet row order == array order)
    comp["array_row"] = np.arange(len(components))[comp.index]

    scalers: dict[tuple, RobustScaler] = {}
    for key, group in comp.groupby(["dataset", "protocol", "waveform_kind", "component_id"], dropna=False):
        train_mask = group["global_subject_id"].isin(train_units)
        rows = group.loc[train_mask, "array_row"].values
        if len(rows) == 0:
            raise ValueError(f"stratum {key}: no training components in this fold")
        vals = canonical_arrays[rows]
        if vals.size == 0:
            raise ValueError(f"stratum {key}: empty canonical values")
        median, scale = _robust_scale(vals)
        scalers[key] = RobustScaler(median, scale, int(len(rows)))

    # persist
    payload = {json.dumps(list(k)): v.to_dict() for k, v in scalers.items()}
    scaler_hash = sha256_text(json.dumps(payload, sort_keys=True))
    out = cache_dir / f"scalers_fold{outer_fold}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    manifest = RunManifest(kind="fold_scaler", name=f"fold{outer_fold}")
    manifest.extra["scaler_hash"] = scaler_hash
    manifest.extra["n_strata"] = len(scalers)
    manifest.write_atomic(cache_dir / f"scalers_fold{outer_fold}_manifest.json")
    return scalers


def apply_fold_scaler(canonical: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(canonical)

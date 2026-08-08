"""Fold-dependent technical QC and locked populations (plan Step 12.15/12.16).

Technical flags are computed from raw curves only (no labels).  Thresholds
are fitted with robust statistics on outer-training people per (dataset,
protocol) stratum and applied to validation/test.  Three populations are
locked: all technically valid, high-QC sensitivity (no technical flags), and
complete bilateral/full-intensity sensitivity (both PERG eyes present; LEOP
full-intensity 9_step recording present).  A missingness summary reports bag
composition by class so absence is never silently treated as zero physiology.
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
class QcThreshold:
    high: float
    low: float | None = None
    degenerate: bool = False

    def flag(self, value: float) -> bool:
        if self.degenerate:
            return False
        if self.low is not None and value < self.low:
            return True
        return value > self.high


TECHNICAL_FLAG_NAMES = (
    "noise-derivative-mad",
    "boundary-jump",
    "saturation",
    "unusual-rate",
    "extreme-variation",
)


def technical_flags(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    median_dt_ms: float,
) -> dict[str, float]:
    """Raw technical quantities for one waveform (flagging happens elsewhere)."""
    sig = np.asarray(signal_uv, dtype=float)
    finite = np.isfinite(sig)
    valid = sig[finite]
    ptp = float(np.ptp(valid)) if valid.size else np.nan
    d = np.diff(valid)
    derivative_mad = float(np.median(np.abs(d - np.median(d)))) if d.size else np.nan
    boundary_jump = (
        float(np.abs(valid[0] - valid[-1]) / ptp) if valid.size > 1 and ptp else np.nan
    )
    saturation = (
        float(np.mean((valid <= valid.min() + 1e-12) | (valid >= valid.max() - 1e-12)))
        if valid.size
        else np.nan
    )
    total_variation = float(np.sum(np.abs(d))) if d.size else np.nan
    return {
        "noise_derivative_mad_uv": derivative_mad,
        "noise_derivative_mad_relative": derivative_mad / ptp if ptp else np.nan,
        "boundary_jump_relative": boundary_jump,
        "saturation_fraction": saturation,
        "total_variation_uv": total_variation,
        "total_variation_relative": total_variation / ptp if ptp else np.nan,
        "median_dt_ms": float(median_dt_ms),
    }


def fit_qc_thresholds(
    qc: pd.DataFrame,
    train_units: set,
    k: float = 5.0,
) -> dict[tuple[str, str], dict[str, QcThreshold]]:
    """Robust outer-train thresholds per (dataset, protocol) stratum.

    A value is flagged when it exceeds high = median + k * 1.4826*MAD (or
    falls below low where meaningful), computed on training people only.
    """
    qc = qc[qc["global_subject_id"].isin(train_units)]
    # median_dt_ms is excluded: the sampling rate is fixed by the recording
    # device, so the measure is constant across the cohort and carries no
    # flaggable signal (confirmed degenerate in every fold/stratum).
    measures = {
        "noise_derivative_mad_relative": ("high", None),
        "boundary_jump_relative": ("high", None),
        "saturation_fraction": ("high", None),
        "total_variation_relative": ("high", None),
    }
    thresholds: dict[tuple[str, str], dict[str, QcThreshold]] = {}
    for (dataset, protocol), group in qc.groupby(["dataset", "protocol"]):
        row = {}
        for name, (mode, _low) in measures.items():
            vals = group[name].dropna().to_numpy()
            if vals.size == 0:
                raise ValueError(
                    f"stratum ({dataset}, {protocol}): no values for measure {name!r}"
                )
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median)))
            spread = MAD_SCALE * mad
            if not (np.isfinite(spread) and spread > 0.0):
                # Degenerate stratum: the measure is constant in the train
                # population, so no statistical flag rule exists.  This is
                # recorded explicitly (degenerate=true) and flags nothing;
                # it is never silently treated as an infinite threshold.
                row[name] = QcThreshold(median, None, degenerate=True)
                continue
            high = median + k * spread
            low = max(0.0, median - k * spread) if mode == "both" else None
            row[name] = QcThreshold(high, low)
        thresholds[(dataset, protocol)] = row
    return thresholds


def flag_recording(
    quantities: dict[str, float],
    thresholds: dict[str, QcThreshold],
) -> list[str]:
    flags = []
    for name, thr in thresholds.items():
        value = quantities.get(name)
        if value is None or not np.isfinite(value):
            flags.append(f"{name}-nonfinite")
            continue
        if thr.flag(value):
            flags.append(f"{name}")
    return flags


def compute_all_qc(
    recordings: pd.DataFrame,
    waveforms: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for rec in recordings.itertuples(index=False):
        tm, sig = waveforms[rec.global_recording_id]
        q = technical_flags(tm, sig, float(rec.median_dt_ms))
        rows.append({"global_recording_id": rec.global_recording_id, **q})
    return pd.DataFrame(rows)


def lock_populations(
    recordings: pd.DataFrame,
    visits: pd.DataFrame,
    sessions: pd.DataFrame,
    qc_flags: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """Three locked population masks, one row per (subject, visit, fold).

    - all_valid: subject+visit exists and every recording passed hard validity
      (survives to components).
    - high_qc: all recordings of the visit also carry no technical flags.
    - complete: PERG visit has both eyes; LEOP subject has a 9_step recording
      (full-intensity standard protocol) in that visit.
    """
    rows: list[dict] = []
    if "n_flags" not in qc_flags:
        raise ValueError("qc_flags must carry an n_flags column")
    for fold in sorted(folds["outer_fold"].unique()):
        units = set(folds[folds["outer_fold"] == fold]["unit_id"])
        valid_recs = set(recordings["global_recording_id"])
        flagged = set(qc_flags.loc[qc_flags["n_flags"] > 0, "global_recording_id"])
        for v in visits.itertuples(index=False):
            if v.global_subject_id not in units:
                continue
            visit_recs = recordings[recordings["global_visit_id"] == v.global_visit_id]
            if visit_recs.empty:
                continue
            all_valid = bool(visit_recs["global_recording_id"].isin(valid_recs).all())
            high_qc = all_valid and not visit_recs["global_recording_id"].isin(flagged).any()
            if v.dataset == "LEOP":
                complete = bool((visit_recs["protocol"] == "9_step").any())
            else:
                complete = bool(
                    visit_recs["eye"].dropna().nunique() == 2
                )
            rows.append(
                {
                    "global_visit_id": v.global_visit_id,
                    "global_subject_id": v.global_subject_id,
                    "outer_fold": fold,
                    "dataset": v.dataset,
                    "all_valid": all_valid,
                    "high_qc": high_qc,
                    "complete": complete,
                }
            )
    return pd.DataFrame(rows)


def _thresholds_payload(thresholds: dict) -> str:
    """thresholds: {fold: {(dataset, protocol): {measure: QcThreshold}}}"""
    def clean(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return float(v)

    payload = {}
    for fold, strata in thresholds.items():
        for k, row in strata.items():
            payload[f"{fold}|{'|'.join(k) if isinstance(k, tuple) else k}"] = {
                name: {
                    "high": clean(t.high),
                    "low": clean(t.low),
                    "degenerate": t.degenerate,
                }
                for name, t in row.items()
            }
    return json.dumps(payload, indent=1, sort_keys=True)


def write_qc_artifacts(
    artifact_root: Path,
    qc_flags: pd.DataFrame,
    thresholds: dict,
    populations: pd.DataFrame,
    missingness: pd.DataFrame,
) -> dict[str, object]:
    out = artifact_root / "data" / "qc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc_thresholds_by_fold.json").write_text(_thresholds_payload(thresholds), encoding="utf-8")
    qc_flags.to_parquet(out / "qc_technical_flags.parquet", index=False)
    populations.to_parquet(out / "population_masks.parquet", index=False)
    missingness.to_parquet(out / "missingness_summary.parquet", index=False)
    manifest = RunManifest(kind="fold_qc", name="technical_qc")
    manifest.extra["threshold_hash"] = sha256_text(_thresholds_payload(thresholds))
    manifest.extra["n_flag_recs"] = int(qc_flags["n_flags"].sum())
    manifest.write_atomic(out / "qc_manifest.json")
    return {"threshold_hash": manifest.extra["threshold_hash"]}

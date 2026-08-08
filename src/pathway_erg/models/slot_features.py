"""Fixed slot feature representation (plan Section 16.6, Phase 2).

A *slot* is the fixed grid ``component_type x eye x intensity x protocol``.
For LEOP: component_type in {L_EARLY_A, L_A_TO_B, L_LATE, L_OP} (with
P_EARLY/P_LATE for PERG), eye in {RE, LE}, intensity = stimulus_value
quantized to one decimal (113.04 -> 113.0), protocol in {9_step, 2_step,
LA3, PERG}.

Rules enforced here:

- every feature is computed strictly within one slot (median/MAD of the
  component-level physical features inside that slot), so components, eyes,
  intensities and protocols are never averaged across each other;
- a unit missing an entire slot keeps NaN for it — imputation happens inside
  the CV loops only (see ``_fit_transform_features``), never here;
- the slot grid is fixed from the full recordings table before any split, so
  slot definitions cannot leak label information.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .baselines import FeatureSet, _flagged, _reindex_to_units, _unit_mapping

SLOT_PHYSICAL_FEATURES: tuple[str, ...] = (
    "area_above_ref_uv_ms",
    "peak_to_peak_uv",
    "max_rising_slope_uv_per_ms",
    "max_falling_slope_uv_per_ms",
    "mass_pos",
    "mass_neg",
    "log_mass_pos",
    "log_mass_neg",
    "max_uv",
    "min_uv",
    "duration_ms",
    "max_latency_ms",
    "min_latency_ms",
)


def slot_stimulus(value: object) -> str:
    """Intensity axis for the slot grid: one-decimal quantization, NA-safe."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.1f}"


def slot_key(row: pd.Series) -> tuple[str, str, str, str]:
    """(protocol, eye, intensity, component_type) for one component row."""
    protocol = str(row.get("protocol")) if pd.notna(row.get("protocol")) else "NA"
    eye = str(row.get("eye")) if pd.notna(row.get("eye")) else "NA"
    return (protocol, eye, slot_stimulus(row.get("stimulus_value")), str(row.get("component_id")))


def _mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a - med)))


def _slot_component_table(
    components: pd.DataFrame, recordings: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    """Component rows joined with slot dimensions, unit, and parsed features."""
    rec = recordings[["global_recording_id", "protocol", "eye", "stimulus_value"]].copy()
    if "dataset" in recordings:
        rec = rec.loc[recordings["dataset"] == dataset]
    rec_ids = set(rec["global_recording_id"])
    long = components[components["global_recording_id"].isin(rec_ids)].merge(
        rec, on="global_recording_id", how="left"
    )
    unit_map = _unit_mapping(recordings, dataset)
    long["unit"] = long["global_recording_id"].map(unit_map)
    long["flagged"] = _flagged(long)
    parsed = long["physical_features_json"].map(
        lambda s: json.loads(s) if isinstance(s, str) and s else {}
    )
    for col in SLOT_PHYSICAL_FEATURES:
        long[col] = parsed.map(lambda d, c=col: d.get(c, np.nan))
    long["slot"] = long.apply(slot_key, axis=1)
    return long


def e4_slot_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
) -> FeatureSet:
    """Per-slot median + MAD of component physical features, per unit.

    Returns one wide feature block per slot (sorted by the fixed grid), with
    ``..._median``, ``..._mad``, ``..._n`` and ``..._flagged_rate`` columns.
    Missing slots stay NaN; ``per_unit_n`` counts every used component.
    """
    long = _slot_component_table(components, recordings, dataset)
    grid = sorted(long["slot"].dropna().unique())
    if not grid:
        return FeatureSet(
            unit_id=units["unit_id"].reset_index(drop=True),
            X=np.empty((len(units), 0)),
            names=[],
            per_unit_n=np.zeros(len(units), dtype=int),
            notes={"n_slots": 0, "note": "no components available"},
        )
    feat_cols = list(SLOT_PHYSICAL_FEATURES)
    grouped = long.groupby(["unit", "slot"])
    med = grouped[feat_cols].median()
    dev = long.copy()
    dev[feat_cols] = (long[feat_cols] - grouped[feat_cols].transform("median")).abs()
    mad = dev.groupby(["unit", "slot"])[feat_cols].median()
    n_comp = grouped.size()
    flagged_rate = grouped["flagged"].mean()

    blocks: list[pd.DataFrame] = []
    names: list[str] = []
    for slot in grid:
        proto, eye, stim, ctype = slot
        prefix = f"slot_{proto}_{eye}_{stim}_{ctype}"
        sel = med.index.get_level_values("slot") == slot
        block = med.loc[sel].droplevel("slot")
        block.columns = [f"{prefix}_{c}_median" for c in block.columns]
        names += list(block.columns)
        mad_block = mad.loc[sel].droplevel("slot")
        mad_block.columns = [f"{prefix}_{c}_mad" for c in mad_block.columns]
        names += list(mad_block.columns)
        n_block = n_comp.loc[sel].to_frame(f"{prefix}_n")
        n_block.index = n_block.index.get_level_values("unit")
        names += [f"{prefix}_n"]
        fr_block = flagged_rate.loc[sel].to_frame(f"{prefix}_flagged_rate")
        fr_block.index = fr_block.index.get_level_values("unit")
        names += [f"{prefix}_flagged_rate"]
        blocks += [block, mad_block, n_block, fr_block]

    table = pd.concat(blocks, axis=1)
    table = _reindex_to_units(table, units)
    n_cols = [c for c in names if c.endswith("_n")]
    per_unit_n = np.nansum(table[n_cols].to_numpy(float), axis=1)
    missing_rate = 1.0 - table[n_cols].notna().to_numpy(float).sum(axis=1) / len(grid)
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=table[names].to_numpy(float),
        names=names,
        per_unit_n=per_unit_n.astype(int),
        notes={
            "n_slots": len(grid),
            "grid": [list(s) for s in grid],
            "mean_slots_missing_per_unit": float(np.nanmean(missing_rate)),
            "physical_features": list(feat_cols),
        },
    )

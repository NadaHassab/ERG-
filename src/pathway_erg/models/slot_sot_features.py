"""Per-slot signed-OT features (plan Section 16.6, v2 Phase 6).

Why this exists: the unit-mean ``scdt`` baseline averages canonical curves
across component types and time domains and then takes quantiles of the
average — an invalid operation (recorded AUROC 0.41-0.49, i.e. below
chance). ``e4_derot_features`` is a valid descriptor but still unit-averages
across slots, diluting slot-specific morphology.

Here the per-component signed derivative-OT descriptor (plan Section 7) is
aggregated **strictly within each fixed slot** (``component_type x eye x
intensity x protocol``), mirroring ``slot_features``: elementwise median and
MAD of the descriptor vector inside a slot, plus per-slot ``n`` and
``flagged_rate``. Nothing is averaged across components, eyes, intensities
or protocols; a unit missing a slot keeps NaN (imputation happens inside the
CV loops only, never here).

Descriptor semantics (declared, per v2 Phase 6):

- each sign's variation measure is normalized to a probability measure and
  represented by its quantile map on a fixed grid — i.e. the 1D SCDT against
  a **uniform reference measure** on the probability grid;
- masses are **not** discarded: retained separately as log-masses plus
  total/net variation, so amplitude information survives normalization;
- +/− channels are kept separate throughout (sign information is never
  collapsed);
- property behaviour (translation -> quantile shift, amplitude -> log-mass
  shift, sign flip -> channel swap) is verified end-to-end at the feature
  level in ``tests/models/test_slot_sot_features.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import FeatureSet, _flagged, _reindex_to_units, _unit_mapping
from .qc_flags import is_hard_invalid
from .slot_features import slot_key

# Fixed tail of the signed-OT vector after the two quantile blocks
# (see SignedOTResult.to_vector; v2 layout: 7 trailing scalars).
SOT_TAIL_FEATURES: tuple[str, ...] = (
    "log_mass_pos",
    "log_mass_neg",
    "mass_pos_frac",
    "total_variation",
    "net_variation",
    "valid_pos",
    "valid_neg",
)


def sot_vector_names(n_quantiles: int) -> list[str]:
    """Column names for one signed-OT descriptor vector."""
    return (
        [f"qpos_{i}" for i in range(n_quantiles)]
        + [f"qneg_{i}" for i in range(n_quantiles)]
        + list(SOT_TAIL_FEATURES)
    )


def infer_n_quantiles(vector_dim: int) -> int:
    """Descriptor layout is 2*n_quantiles + len(SOT_TAIL_FEATURES)."""
    n, r = divmod(vector_dim - len(SOT_TAIL_FEATURES), 2)
    if r != 0 or n <= 0:
        raise ValueError(
            f"cannot infer quantile count from signed-OT vector dim {vector_dim}"
        )
    return n


def e4_slot_sot_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    sot: np.ndarray,
) -> FeatureSet:
    """Per-slot elementwise median + MAD of signed-OT descriptor vectors.

    ``sot`` is positionally aligned with ``components`` rows (the cache
    layout). Returns one wide block per slot with ``..._median``, ``..._mad``,
    ``..._n`` and ``..._flagged_rate`` columns, following ``slot_features``.
    """
    n_q = infer_n_quantiles(sot.shape[1])
    feat_names = sot_vector_names(n_q)

    rec = recordings[["global_recording_id", "protocol", "eye", "stimulus_value"]].copy()
    if "dataset" in recordings:
        rec = rec.loc[recordings["dataset"] == dataset]
    rec_ids = set(rec["global_recording_id"])
    comp = components.copy()
    comp["_row"] = np.arange(len(comp))
    long = comp[comp["global_recording_id"].isin(rec_ids)].merge(
        rec, on="global_recording_id", how="left"
    )
    unit_map = _unit_mapping(recordings, dataset)
    long["unit"] = long["global_recording_id"].map(unit_map)
    long["flagged"] = _flagged(long)
    long["slot"] = long.apply(slot_key, axis=1)
    # Phase 4: hard-invalid components are excluded from every feature family,
    # including the transport-derived ones.
    keep = ~is_hard_invalid(long).to_numpy()
    n_excluded = int((~keep).sum())
    long = long[keep].reset_index(drop=True)
    vecs = sot[long["_row"].to_numpy()]

    grid = sorted(long["slot"].dropna().unique())
    if not grid:
        return FeatureSet(
            unit_id=units["unit_id"].reset_index(drop=True),
            X=np.empty((len(units), 0)),
            names=[],
            per_unit_n=np.zeros(len(units), dtype=int),
            notes={"n_slots": 0, "note": "no components available"},
        )

    vec_df = pd.DataFrame(vecs, columns=feat_names, index=long.index)
    grouped = long.groupby(["unit", "slot"])
    med = vec_df.groupby([long["unit"], long["slot"]]).median()
    dev = vec_df.copy()
    dev[feat_names] = (
        vec_df[feat_names] - vec_df.groupby([long["unit"], long["slot"]]).transform("median")
    ).abs()
    mad = dev.groupby([long["unit"], long["slot"]])[feat_names].median()
    n_comp = grouped.size()
    flagged_rate = grouped["flagged"].mean()

    blocks: list[pd.DataFrame] = []
    names: list[str] = []
    for slot in grid:
        proto, eye, stim, ctype = slot
        prefix = f"slotsot_{proto}_{eye}_{stim}_{ctype}"
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
            "n_components_excluded_qc": n_excluded,
            "sot_features": feat_names,
            "reference": "uniform (1D SCDT on each normalized sign measure)",
        },
    )

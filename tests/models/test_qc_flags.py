"""Phase 4: categorized QC flag semantics.

Replaces "any flag = drop" with four categories applied alike to the
clinical, curve, spectral and transport feature families:

- hard_invalid                -> component excluded from every feature
- low_confidence              -> kept (measurement exists, lower trust)
- truncated_or_limited_support -> kept (values computed on observed support)
- informational_qc            -> kept, only counted in QC-rate features

The dominant real-world case is ``truncated-low``/``truncated-high`` on
relative-phase L_LATE segments (5290 of 5309 L_LATE rows in the v2 cache):
the old predicate silently dropped those from clinical features.  Unknown
future flags must never be dropped by default.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pathway_erg.models.baselines import (
    LEOP_AMP_COL,
    LEOP_LAT_COL,
    _clean_pairs,
    e4_curve_features,
    e4_derot_features,
)

PIPELINE_FLAG_VOCABULARY = {
    "no-samples-in-window",
    "no-prominence-peak",
    "boundary-extreme",
    "late-support-too-short",
    "fallback-window",
    "late-landmark-invalid",
    "truncated-low",
    "truncated-high",
    "supplied-only",
    "disagrees-with-supplied",
}


def _phys(**kwargs) -> str:
    base = {
        "area_above_ref_uv_ms": 1.0,
        "peak_to_peak_uv": 2.0,
        "max_rising_slope_uv_per_ms": 0.1,
        "max_falling_slope_uv_per_ms": -0.1,
        "max_uv": 2.0,
        "min_uv": -2.0,
        "duration_ms": 10.0,
        "max_latency_ms": 20.0,
        "min_latency_ms": 5.0,
    }
    base.update(kwargs)
    return json.dumps(base)


def _tables():
    """Two LEOP units; one flagged component of each kind."""
    recordings = pd.DataFrame(
        [
            {"global_recording_id": "R0", "global_subject_id": "U1", "dataset": "LEOP", "eye": "RE", "protocol": "9_step", "stimulus_value": 113.04},
            {"global_recording_id": "R1", "global_subject_id": "U2", "dataset": "LEOP", "eye": "RE", "protocol": "9_step", "stimulus_value": 113.04},
        ]
    )
    components = pd.DataFrame(
        [
            {"global_component_id": "C0", "global_recording_id": "R0", "component_id": "L_LATE", "component_qc_flags": "truncated-low", "physical_features_json": _phys(min_uv=-5.0, min_latency_ms=55.0)},
            {"global_component_id": "C1", "global_recording_id": "R0", "component_id": "L_A_TO_B", "component_qc_flags": "fallback-window", "physical_features_json": _phys(max_uv=9.0, max_latency_ms=28.0)},
            {"global_component_id": "C2", "global_recording_id": "R1", "component_id": "L_LATE", "component_qc_flags": "no-prominence-peak", "physical_features_json": _phys(min_uv=-6.0, min_latency_ms=58.0)},
            {"global_component_id": "C3", "global_recording_id": "R1", "component_id": "L_A_TO_B", "component_qc_flags": "disagrees-with-supplied", "physical_features_json": _phys(max_uv=10.0, max_latency_ms=29.0)},
        ]
    )
    units = pd.DataFrame({"unit_id": ["U1", "U2"], "dataset": "LEOP"})
    return units, components, recordings


# ---------------------------------------------------------------------------
# Category table
# ---------------------------------------------------------------------------


def test_flag_vocabulary_fully_categorized():
    from pathway_erg.models.qc_flags import FLAG_CATEGORIES

    mapped = {f for cat in FLAG_CATEGORIES.values() for f in cat}
    assert PIPELINE_FLAG_VOCABULARY <= mapped
    assert mapped == PIPELINE_FLAG_VOCABULARY  # no typo'd flags in the table


def test_category_membership():
    from pathway_erg.models.qc_flags import FLAG_CATEGORIES

    assert set(FLAG_CATEGORIES["hard_invalid"]) == {
        "no-samples-in-window",
        "late-support-too-short",
        "fallback-window",
        "late-landmark-invalid",
    }
    assert set(FLAG_CATEGORIES["low_confidence"]) == {"no-prominence-peak", "supplied-only"}
    assert set(FLAG_CATEGORIES["truncated_or_limited_support"]) == {"truncated-low", "truncated-high", "boundary-extreme"}
    assert set(FLAG_CATEGORIES["informational_qc"]) == {"disagrees-with-supplied"}


def test_unknown_flag_never_dropped():
    from pathway_erg.models.qc_flags import flag_categories, is_hard_invalid

    cats = flag_categories("brand-new-flag")
    assert cats == ("informational_qc",)
    comp = pd.DataFrame({"component_qc_flags": ["brand-new-flag", "truncated-low"]})
    assert not is_hard_invalid(comp).any()


def test_truncated_flags_are_kept():
    from pathway_erg.models.qc_flags import flag_categories, is_hard_invalid

    for flags in ("truncated-low", "truncated-high", "truncated-low|truncated-high", "boundary-extreme"):
        assert "hard_invalid" not in flag_categories(flags)
    comp = pd.DataFrame({"component_qc_flags": ["truncated-low", "no-prominence-peak", "disagrees-with-supplied", ""]})
    assert not is_hard_invalid(comp).any()


def test_real_cache_no_late_truncation_dropped():
    """The 5290 truncated L_LATE rows in the v2 cache must all be kept."""
    from pathway_erg.models.qc_flags import is_hard_invalid

    components = pd.read_parquet("artifacts/data/interim/components_v2.parquet")
    hard = is_hard_invalid(components)
    flagged = components["component_qc_flags"].fillna("").astype(str).ne("")
    assert flagged.sum() > 5000
    assert not hard[flagged].any()
    n_hard = int(hard.sum())
    assert n_hard == int(
        components.loc[flagged, "component_qc_flags"].astype(str)
        .apply(lambda s: any(f in s for f in ("fallback-window", "late-landmark-invalid", "no-samples-in-window", "late-support-too-short")))
        .sum()
    )


# ---------------------------------------------------------------------------
# Feature-family behavior
# ---------------------------------------------------------------------------


def test_clean_pairs_keeps_truncated_and_drops_hard_invalid():
    units, components, recordings = _tables()
    long = _clean_pairs(components, recordings, "LEOP", LEOP_AMP_COL, LEOP_LAT_COL)
    got = long.set_index(["unit", "component_id"])["valid"]
    # U1 L_LATE truncated-low: kept
    assert got.loc[("U1", "L_LATE")]
    # U1 L_A_TO_B fallback-window: dropped (hard_invalid)
    assert not got.loc[("U1", "L_A_TO_B")]
    # U2 rows: low_confidence / informational kept
    assert got.loc[("U2", "L_LATE")]
    assert got.loc[("U2", "L_A_TO_B")]
    # the kept L_LATE value is the flagged one, not a dropped-NaN
    assert long.loc[(long["unit"] == "U1") & (long["component_id"] == "L_LATE"), "amp"].iloc[0] == pytest.approx(-5.0)


def test_curve_features_exclude_hard_invalid():
    units, components, recordings = _tables()
    n = len(components)
    curves = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n), np.zeros(n)])
    valid_mask = np.ones((n, 4), dtype=bool)
    fs = e4_curve_features(units, components, recordings, "LEOP", curves, valid_mask, "raw_rbf")
    assert fs.notes["n_components_excluded_qc"] == 1  # only the fallback-window row
    u1 = fs.unit_id == "U1"
    assert fs.X[u1, 0] == pytest.approx(0.0)  # U1 mean of kept rows only (C1 dropped)
    assert fs.per_unit_n[u1] == 1
    assert fs.per_unit_n[fs.unit_id == "U2"] == 2


def test_curve_features_still_exclude_nan_rows():
    units, components, recordings = _tables()
    n = len(components)
    curves = np.zeros((n, 3))
    valid_mask = np.ones((n, 3), dtype=bool)
    valid_mask[0, :] = False  # C0 has NaN curve points
    fs = e4_curve_features(units, components, recordings, "LEOP", curves, valid_mask, "pca_fpca")
    assert fs.notes["n_components_excluded_qc"] == 2  # NaN row + hard_invalid row


def test_derot_features_exclude_hard_invalid():
    units, components, recordings = _tables()
    n = len(components)
    sot = np.arange(n * 3, dtype=float).reshape(n, 3)
    fs = e4_derot_features(units, components, recordings, "LEOP", sot)
    u1 = fs.unit_id == "U1"
    assert fs.per_unit_n[u1] == 1  # C1 (hard_invalid) excluded
    assert fs.X[u1, 0] == pytest.approx(0.0)  # mean of C0 row only (C1 row dropped)
    assert fs.per_unit_n[fs.unit_id == "U2"] == 2

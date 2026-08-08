"""Phase 2: fixed slot feature representation tests (plan Section 16.6).

Slots are the fixed grid ``component_type x eye x intensity x protocol``.
Every feature is defined strictly within one slot (median/MAD of component
features in that slot); components/eyes/intensities/protocols are never
averaged across each other, and missing slots are NaN (imputed inside CV
only, never filled by the feature builder).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pathway_erg.config import BaselinesConfig


def _phys_row(**kwargs) -> str:
    base = {
        "area_above_ref_uv_ms": 1.0,
        "peak_to_peak_uv": 2.0,
        "max_rising_slope_uv_per_ms": 0.1,
        "max_falling_slope_uv_per_ms": -0.1,
        "mass_pos": 3.0,
        "mass_neg": 3.0,
        "log_mass_pos": 1.0,
        "log_mass_neg": 1.0,
        "max_uv": 2.0,
        "min_uv": -2.0,
        "duration_ms": 10.0,
        "max_latency_ms": 20.0,
        "min_latency_ms": 5.0,
    }
    base.update(kwargs)
    return json.dumps(base)


def _synthetic():
    """Minimal LEOP-like tables: 4 units, 9-step only, two eyes.

    Slot grid: RE@113.0, RE@21.0, LE@113.0 (x L_A_TO_B / L_EARLY_A). U4 is
    missing RE@113.0 entirely (missing-slot -> NaN case).
    """
    slots = [
        ("U1", "RE", 113.04, 1.0),
        ("U1", "RE", 21.0, 100.0),
        ("U1", "LE", 113.04, 100.0),
        ("U2", "RE", 113.0, 2.0),
        ("U2", "RE", 21.0, 50.0),
        ("U2", "LE", 113.0, 50.0),
        ("U3", "RE", 113.0, 3.0),
        ("U3", "RE", 21.0, 30.0),
        ("U3", "LE", 113.0, 30.0),
        ("U4", "RE", 21.0, 4.0),
        ("U4", "LE", 113.0, 40.0),
    ]
    rec_rows, comp_rows = [], []
    for i, (unit, eye, stim, val) in enumerate(slots):
        rec_rows.append(
            {
                "global_recording_id": f"R{i}",
                "global_subject_id": unit,
                "dataset": "LEOP",
                "protocol": "9_step",
                "eye": eye,
                "stimulus_value": stim,
                "waveform_kind": "ERG",
            }
        )
        for j, cid in enumerate(("L_A_TO_B", "L_EARLY_A")):
            comp_rows.append(
                {
                    "global_component_id": f"C{i}_{j}",
                    "global_recording_id": f"R{i}",
                    "component_id": cid,
                    "component_qc_flags": "",
                    "physical_features_json": _phys_row(
                        area_above_ref_uv_ms=val + j,
                        peak_to_peak_uv=val + j,
                        mass_pos=val + j,
                    ),
                }
            )
    recordings = pd.DataFrame(rec_rows)
    components = pd.DataFrame(comp_rows)
    units = pd.DataFrame(
        {
            "unit_id": ["U1", "U2", "U3", "U4"],
            "dataset": "LEOP",
            "target_binary": [1.0, 0.0, 1.0, 0.0],
            "outer_fold": [0, 0, 1, 1],
        }
    )
    return units, components, recordings


def test_slot_key_quantizes_intensity():
    from pathway_erg.models.slot_features import slot_stimulus

    assert slot_stimulus(113.04) == "113.0"
    assert slot_stimulus(113.0) == "113.0"
    assert slot_stimulus(21.48) == "21.5"
    assert slot_stimulus(np.nan) == "NA"


def test_no_cross_slot_averaging():
    """U1 has value 1 in RE@113 and 100 elsewhere: each slot keeps its own."""
    from pathway_erg.models.slot_features import e4_slot_features

    units, components, recordings = _synthetic()
    fs = e4_slot_features(units, components, recordings, "LEOP")
    wide = pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id)
    area_cols = [
        c
        for c in fs.names
        if c.endswith("area_above_ref_uv_ms_median") and "L_A_TO_B" in c
    ]
    re113 = [c for c in area_cols if "RE" in c and "113.0" in c]
    le113 = [c for c in area_cols if "LE" in c and "113.0" in c]
    re21 = [c for c in area_cols if "RE" in c and "21.0" in c]
    assert len(re113) == 1 and len(le113) == 1 and len(re21) == 1
    assert wide.loc["U1", re113[0]] == pytest.approx(1.0)
    assert wide.loc["U1", le113[0]] == pytest.approx(100.0)
    assert wide.loc["U1", re21[0]] == pytest.approx(100.0)
    # No pooling: the within-slot median never equals a cross-slot mean.
    assert wide.loc["U1", re113[0]] != pytest.approx((1.0 + 100.0) / 2)


def test_median_and_mad_are_exact():
    from pathway_erg.models.slot_features import _mad

    assert np.median([1.0, 2.0, 3.0, 100.0]) == 2.5
    assert _mad(np.asarray([1.0, 2.0, 3.0, 100.0])) == pytest.approx(1.0)


def test_missing_slots_are_nan_not_zero():
    """U4 never has a RE recording at 21.0 -> that slot must be NaN."""
    from pathway_erg.models.slot_features import e4_slot_features

    units, components, recordings = _synthetic()
    fs = e4_slot_features(units, components, recordings, "LEOP")
    wide = pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id)
    area_cols = [
        c
        for c in fs.names
        if c.endswith("area_above_ref_uv_ms_median") and "L_A_TO_B" in c
    ]
    u4 = wide.loc["U4", area_cols].dropna()
    assert len(u4) == 2  # RE@21 and LE@113 present; RE@113 missing -> NaN
    n_slot_col = [
        c
        for c in fs.names
        if c.endswith("_n") and "RE" in c and "113.0" in c and "L_A_TO_B" in c
    ][0]
    assert np.isnan(wide.loc["U4", n_slot_col])  # absent slot stays NaN (imputed in CV)


def test_slot_grid_matches_recordings_domain():
    from pathway_erg.models.slot_features import e4_slot_features

    units, components, recordings = _synthetic()
    fs = e4_slot_features(units, components, recordings, "LEOP")
    n_slot_names = [c for c in fs.names if c.endswith("_n")]
    # 6 slots: 3 recording-slots (RE@113, RE@21, LE@113) x 2 component types
    assert len(n_slot_names) == 6
    assert fs.notes["n_slots"] == 6


def test_slot_run_end_to_end(tmp_path):
    """Slot method runs inside the v2 pipeline (LEOP primary cohort)."""
    from pathway_erg.config import DataConfig, load_config
    from pathway_erg.models.baselines import run_baselines

    dc = load_config(DataConfig, "configs/data/local.yaml")
    cfg = BaselinesConfig(
        name="phase2_test",
        datasets=("LEOP",),
        leop_cohort="primary_nine_step",
        e0_methods=("prevalence",),
        e4_methods=("slot",),
        outer_folds=(0,),
        models=("logreg",),
        output_subdir=str(tmp_path / "phase2"),
    )
    results = run_baselines(cfg, dc)
    key = "LEOP_primary_nine_step/slot_logreg"
    assert key in results.metrics
    assert results.metrics[key]["roc_auc"] is not None
    assert results.predictions["cohort"].eq("primary_nine_step").all()


def test_perg_slots_are_eye_and_component_only(tmp_path):
    from pathway_erg.config import DataConfig, load_config
    from pathway_erg.models.baselines import run_baselines

    dc = load_config(DataConfig, "configs/data/local.yaml")
    cfg = BaselinesConfig(
        name="phase2_test",
        datasets=("PERG",),
        e0_methods=("prevalence",),
        e4_methods=("slot",),
        outer_folds=(0,),
        models=("logreg",),
        output_subdir=str(tmp_path / "phase2"),
    )
    results = run_baselines(cfg, dc)
    notes = results.notes["PERG_feature_notes"]["slot"]
    assert notes["n_slots"] == 4  # P_EARLY/P_LATE x RE/LE

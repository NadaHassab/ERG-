"""Phase 6: per-slot signed-OT feature tests (plan Section 16.6, v2 Phase 6).

The unit-mean ``scdt`` baseline is invalid (quantiles of a curve averaged
across component types/domains are meaningless; recorded AUROC 0.41-0.49,
i.e. below chance). The valid replacement aggregates the *per-component*
signed derivative-OT descriptor strictly within each fixed slot
(``component_type x eye x intensity x protocol``) — never across slots —
using elementwise median/MAD on the descriptor vector. The descriptor's
declared reference measure is the uniform distribution on the probability
grid (1D SCDT with uniform reference applied to each normalized sign
measure); masses are retained separately as log-masses.

Property tests run the real ``signed_derivative_ot`` transform on synthetic
traces and then the builder, so translation / amplitude / sign behaviour is
verified end-to-end at the feature level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.signal.signed_ot import signed_derivative_ot
from pathway_erg.signal.smoothing import SmoothingConfig

N_Q = 64


def _tables(vals_by_unit):
    """One component per recording; two slots per unit (RE@113.0, RE@21.0)."""
    rec_rows, comp_rows, unit_rows = [], [], []
    idx = 0
    for u, _vals in vals_by_unit.items():
        unit_rows.append(
            {
                "unit_id": u,
                "dataset": "LEOP",
                "target_binary": 1.0 if u in ("U1", "U3") else 0.0,
                "outer_fold": 0 if u in ("U1", "U2") else 1,
            }
        )
        for stim in (113.04, 21.0):
            rec_rows.append(
                {
                    "global_recording_id": f"R{idx}",
                    "global_subject_id": u,
                    "dataset": "LEOP",
                    "protocol": "9_step",
                    "eye": "RE",
                    "stimulus_value": stim,
                    "waveform_kind": "ERG",
                }
            )
            comp_rows.append(
                {
                    "global_component_id": f"C{idx}",
                    "global_recording_id": f"R{idx}",
                    "component_id": "L_A_TO_B",
                    "component_qc_flags": "",
                    "physical_features_json": "{}",
                }
            )
            idx += 1
    return (
        pd.DataFrame(unit_rows),
        pd.DataFrame(comp_rows),
        pd.DataFrame(rec_rows),
    )


def _synthetic():
    units, components, recordings = _tables(
        {"U1": (1.0, 10.0), "U2": (2.0, 20.0), "U3": (3.0, 30.0), "U4": (4.0, 40.0)}
    )
    sot = np.zeros((len(components), 2 * N_Q + 7))
    vals = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
    for i, v in enumerate(vals):
        sot[i, :] = v
        sot[i, :N_Q] = np.arange(N_Q) + v  # qpos ramp per component
    return units, components, recordings, sot


def _col(fs, stim_token, feature, stat="median"):
    hits = [
        c
        for c in fs.names
        if c.endswith(f"{feature}_{stat}") and f"_{stim_token}_" in c
    ]
    assert len(hits) == 1, f"expected 1 column for {stim_token}/{feature}, got {hits}"
    return hits[0]


def test_no_cross_slot_averaging_sot():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    units, components, recordings, sot = _synthetic()
    fs = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    wide = pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id)
    c113 = _col(fs, "113.0", "log_mass_pos")
    c21 = _col(fs, "21.0", "log_mass_pos")
    assert wide.loc["U1", c113] == pytest.approx(1.0)
    assert wide.loc["U1", c21] == pytest.approx(10.0)
    # never a cross-slot pooled value
    assert wide.loc["U1", c113] != pytest.approx((1.0 + 10.0) / 2)


def test_elementwise_median_exact_sot():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    units, components, recordings, sot = _synthetic()
    fs = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    wide = pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id)
    q0 = _col(fs, "113.0", "qpos_0")
    q63 = _col(fs, "113.0", "qpos_63")
    assert wide.loc["U1", q0] == pytest.approx(1.0)
    assert wide.loc["U1", q63] == pytest.approx(63.0 + 1.0)
    assert wide.loc["U2", q63] == pytest.approx(63.0 + 2.0)


def test_missing_slots_stay_nan_sot():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    # U4 has no 21.0 recording at all -> NaN row for that slot
    units, components, recordings = _tables({"U1": (1.0, 10.0)})
    units = units.iloc[:1]
    sot = np.ones((len(components), 2 * N_Q + 7))
    fs = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    wide = pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id)
    present = _col(fs, "113.0", "log_mass_pos")
    assert np.isfinite(wide.loc["U1", present])
    # grid was built from available slots only; a unit lacking a slot is NaN
    absent_units = pd.DataFrame(
        {
            "unit_id": ["U1", "U9"],
            "dataset": ["LEOP", "LEOP"],
            "target_binary": [1.0, 0.0],
            "outer_fold": [0, 1],
        }
    )
    fs2 = e4_slot_sot_features(absent_units, components, recordings, "LEOP", sot)
    wide2 = pd.DataFrame(fs2.X, columns=fs2.names, index=fs2.unit_id)
    assert np.isnan(wide2.loc["U9", present])


def test_grid_matches_slot_features():
    from pathway_erg.models.slot_features import e4_slot_features
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    units, components, recordings, sot = _synthetic()
    fs_sot = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    fs_phys = e4_slot_features(units, components, recordings, "LEOP")
    assert fs_sot.notes["grid"] == fs_phys.notes["grid"]


def _real_sot(t, sig):
    return signed_derivative_ot(
        np.asarray(t, float),
        np.asarray(sig, float),
        median_dt_ms=float(np.median(np.diff(t))),
        smoothing=SmoothingConfig(method="none"),
        n_quantiles=N_Q,
        mass_tolerance=1e-8,
    ).to_vector()


def _bump(t, center, sign=1.0, scale=1.0):
    return sign * scale * 12.0 * np.exp(-0.5 * ((t - center) / 3.0) ** 2)


def _one_unit_tables(vec):
    units = pd.DataFrame(
        {"unit_id": ["U1"], "dataset": ["LEOP"], "target_binary": [1.0], "outer_fold": [0]}
    )
    recordings = pd.DataFrame(
        {
            "global_recording_id": ["R0"],
            "global_subject_id": ["U1"],
            "dataset": ["LEOP"],
            "protocol": ["9_step"],
            "eye": ["RE"],
            "stimulus_value": [113.0],
            "waveform_kind": ["ERG"],
        }
    )
    components = pd.DataFrame(
        {
            "global_component_id": ["C0"],
            "global_recording_id": ["R0"],
            "component_id": ["L_A_TO_B"],
            "component_qc_flags": [""],
            "physical_features_json": ["{}"],
        }
    )
    return units, components, recordings, vec.reshape(1, -1)


def _feature_row(fs):
    return pd.DataFrame(fs.X, columns=fs.names, index=fs.unit_id).loc["U1"]


def test_translation_shifts_quantiles_not_masses():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    t = np.arange(0.0, 120.0, 0.5)
    v0 = _real_sot(t, _bump(t, 40.0))
    v1 = _real_sot(t, _bump(t, 45.0))  # +5 ms shift
    row0 = _feature_row(e4_slot_sot_features(*_one_unit_tables(v0)[:3], "LEOP", _one_unit_tables(v0)[3]))
    row1 = _feature_row(e4_slot_sot_features(*_one_unit_tables(v1)[:3], "LEOP", _one_unit_tables(v1)[3]))
    qpos_cols = [c for c in row0.index if "_qpos_" in c and c.endswith("_median")]
    q0 = row0[qpos_cols].to_numpy(float)
    q1 = row1[qpos_cols].to_numpy(float)
    assert np.allclose(q1 - q0, 5.0, atol=0.6)
    assert row1[_col_name(row0, "log_mass_pos")] == pytest.approx(
        row0[_col_name(row0, "log_mass_pos")], abs=1e-6
    )


def _col_name(row, feature):
    hits = [c for c in row.index if f"_{feature}_" in c and c.endswith("_median")]
    assert len(hits) == 1
    return hits[0]


def test_amplitude_scaling_moves_only_log_masses():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    t = np.arange(0.0, 120.0, 0.5)
    v0 = _real_sot(t, _bump(t, 40.0))
    v3 = _real_sot(t, _bump(t, 40.0, scale=3.0))
    row0 = _feature_row(e4_slot_sot_features(*_one_unit_tables(v0)[:3], "LEOP", _one_unit_tables(v0)[3]))
    row3 = _feature_row(e4_slot_sot_features(*_one_unit_tables(v3)[:3], "LEOP", _one_unit_tables(v3)[3]))
    lm = _col_name(row0, "log_mass_pos")
    assert row3[lm] - row0[lm] == pytest.approx(np.log(3.0), abs=1e-3)
    qpos_cols = [c for c in row0.index if "_qpos_" in c and c.endswith("_median")]
    assert np.allclose(
        row0[qpos_cols].to_numpy(float), row3[qpos_cols].to_numpy(float), atol=1e-6
    )


def test_sign_flip_swaps_channels():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    t = np.arange(0.0, 120.0, 0.5)
    vp = _real_sot(t, _bump(t, 40.0, sign=1.0))
    vn = _real_sot(t, _bump(t, 40.0, sign=-1.0))
    rowp = _feature_row(e4_slot_sot_features(*_one_unit_tables(vp)[:3], "LEOP", _one_unit_tables(vp)[3]))
    rown = _feature_row(e4_slot_sot_features(*_one_unit_tables(vn)[:3], "LEOP", _one_unit_tables(vn)[3]))
    qpos = [c for c in rowp.index if "_qpos_" in c and c.endswith("_median")]
    qneg = [c for c in rowp.index if "_qneg_" in c and c.endswith("_median")]
    assert np.allclose(
        rowp[qpos].to_numpy(float), rown[qneg].to_numpy(float), atol=1e-6
    )
    assert np.allclose(
        rowp[qneg].to_numpy(float), rown[qpos].to_numpy(float), atol=1e-6
    )


def test_deterministic_sot_builder():
    from pathway_erg.models.slot_sot_features import e4_slot_sot_features

    units, components, recordings, sot = _synthetic()
    a = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    b = e4_slot_sot_features(units, components, recordings, "LEOP", sot)
    assert a.names == b.names
    assert np.allclose(a.X, b.X, equal_nan=True)

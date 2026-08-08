"""Tests for validity, baseline, smoothing, landmarks, segments, resample."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

from pathway_erg.config import (
    LandmarkDetectionConfig,
    LeopsSegmentationConfig,
    PergSegmentationConfig,
    SegmentationConfig,
)
from pathway_erg.signal.baseline import apply_offset
from pathway_erg.signal.landmarks import (
    detect_leops_landmarks,
    detect_perg_landmarks,
)
from pathway_erg.signal.resample import canonicalize_relative_phase, resample_pchip
from pathway_erg.signal.segments import make_leops_segments, make_perg_segments
from pathway_erg.signal.smoothing import SmoothingConfig, odd_window_samples, smooth_for_analysis
from pathway_erg.signal.validity import check_hard_validity, interpolate_isolated_nan

HARD_FRACTION = 0.95
MAX_NAN_GAPS = 1
T0_MS = 0.0
MAD_MULTIPLE = 3.0

LM_CFG = LandmarkDetectionConfig(
    leops_a_range=(5.0, 25.0),
    leops_b_range=(15.0, 55.0),
    leops_late_range=(55.0, 110.0),
    leops_min_prominence_frac=0.05,
    leops_max_disagreement_ms=10.0,
    leops_min_separation_ms=5.0,
    leops_late_separation_ms=10.0,
    perg_n35_range=(20.0, 45.0),
    perg_p50_range=(35.0, 70.0),
    perg_n95_range=(65.0, 130.0),
    perg_min_prominence_frac=0.05,
    perg_min_separation_ms=5.0,
    perg_late_separation_ms=10.0,
)

SEG_CFG = SegmentationConfig(
    relative_phase_range=(-0.2, 1.2),
    late_end_ms=110.0,
    op_default_confidence=1.0,
    leops=LeopsSegmentationConfig(
        early_a_pad_ms=5.0,
        early_a_bound_ms=25.0,
        early_a_fallback_ms=(0.0, 25.0),
        a_to_b_pad_ms=(-5.0, 10.0),
        a_to_b_fallback_ms=(0.0, 50.0),
        late_pad_ms=10.0,
        late_fallback_lo_ms=50.0,
    ),
    perg=PergSegmentationConfig(
        early_pad_ms=(-10.0, 10.0),
        early_fallback_ms=(20.0, 70.0),
        late_pad_ms=(-10.0, 20.0),
        late_fallback_ms=(50.0, 130.0),
    ),
)

N_POINTS = 128


def _t(fs=1953.125, n=235):
    return np.arange(n) * 1000.0 / fs


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


def test_validity_rejects_length_mismatch():
    r = check_hard_validity(np.zeros(10), np.zeros(11), HARD_FRACTION, MAX_NAN_GAPS)
    assert not r.valid
    assert "length-mismatch" in r.reasons


def test_validity_rejects_low_finite_fraction():
    t = _t()
    sig = np.zeros_like(t)
    sig[100:] = np.nan
    r = check_hard_validity(t, sig, HARD_FRACTION, MAX_NAN_GAPS)
    assert not r.valid
    assert "finite-fraction" in r.reasons


def test_validity_accepts_one_isolated_nan():
    t = _t()
    sig = np.zeros_like(t)
    sig[50] = np.nan
    r = check_hard_validity(t, sig, HARD_FRACTION, MAX_NAN_GAPS)
    assert r.valid
    assert r.fixed_mask is not None and r.fixed_mask[50]


def test_validity_rejects_two_nan_runs():
    t = _t()
    sig = np.zeros_like(t)
    sig[50] = np.nan
    sig[150] = np.nan
    r = check_hard_validity(t, sig, HARD_FRACTION, MAX_NAN_GAPS)
    assert not r.valid
    assert "too-many-nan-gaps" in r.reasons


def test_validity_rejects_nonmonotonic_time():
    t = _t()
    t[100], t[101] = t[101], t[100]
    r = check_hard_validity(t, np.zeros_like(t), HARD_FRACTION, MAX_NAN_GAPS)
    assert not r.valid
    assert "non-monotonic-time" in r.reasons


def test_interpolate_isolated_nan():
    t = _t()
    sig = np.zeros_like(t)
    sig[50] = np.nan
    r = check_hard_validity(t, sig, HARD_FRACTION, MAX_NAN_GAPS)
    out = interpolate_isolated_nan(t, sig, r.fixed_mask)
    assert np.isclose(out[50], 0.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# baseline / offset
# ---------------------------------------------------------------------------


def test_prestimulus_median_subtraction():
    t = np.arange(-50, 200) * 0.512
    sig = np.full_like(t, 3.0)
    sig[t > 0] += 10.0
    out, stats = apply_offset(t, sig, T0_MS, "prestimulus_median", MAD_MULTIPLE)
    assert np.isclose(stats.prestimulus_median_uv, 3.0)
    assert np.isclose(out[t > 0].mean(), 10.0, atol=1e-9)
    assert np.isclose(out[t <= 0].mean(), 0.0, atol=1e-9)


def test_offset_none_preserves_level():
    t = _t()
    sig = np.linspace(0, 5, t.size)
    out, stats = apply_offset(t, sig, T0_MS, "none", MAD_MULTIPLE)
    assert np.array_equal(out, sig)


def test_peak_to_peak_invariant_to_constant_centering():
    t = _t()
    sig = 5.0 + 2.0 * np.sin(t / 10.0)
    a, _ = apply_offset(t, sig, T0_MS, "none", MAD_MULTIPLE)
    b, _ = apply_offset(t, sig, T0_MS, "whole_trace_median", MAD_MULTIPLE)
    assert np.ptp(a) == pytest.approx(np.ptp(b))


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def test_odd_window_samples():
    assert odd_window_samples(3.0, 0.512) % 2 == 1
    assert odd_window_samples(3.0, 0.512) >= 3
    with pytest.raises(ValueError):
        odd_window_samples(3.0, 0.0)


def test_smoothing_reduces_high_frequency_noise():
    t = _t()
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * t / 50.0) + rng.normal(0, 1, t.size)
    out = smooth_for_analysis(t, sig, 0.512, SmoothingConfig())
    assert np.std(out) < np.std(sig)


def test_smoothing_matches_scipy_reference():
    t = _t()
    sig = np.cos(t / 30.0)
    out = smooth_for_analysis(t, sig, 0.512, SmoothingConfig(window_ms=3.0, polyorder=3))
    ref = savgol_filter(sig, 7, 3)
    assert np.allclose(out, ref)


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------


def test_leops_landmark_recovery_on_synthetic():
    t = _t()
    a_center, b_center = 12.0, 28.0
    sig = -3.0 * np.exp(-0.5 * ((t - a_center) / 2.0) ** 2) + 12.0 * np.exp(
        -0.5 * ((t - b_center) / 3.0) ** 2
    )
    lm = detect_leops_landmarks(t, sig, LM_CFG)
    assert lm["a_trough"].time_ms is not None
    assert np.isclose(lm["a_trough"].time_ms, a_center, atol=2.0)
    assert np.isclose(lm["b_peak"].time_ms, b_center, atol=2.0)
    assert lm["b_peak"].amplitude_uv > lm["a_trough"].amplitude_uv


def test_leops_fallback_without_peaks():
    t = _t()
    sig = np.linspace(0, 1, t.size)
    lm = detect_leops_landmarks(t, sig, LM_CFG)
    assert lm["late_trough"].source == "fallback"
    assert lm["late_trough"].confidence == 0.0


def test_leops_supplied_disagreement_flags():
    t = _t()
    sig = -3.0 * np.exp(-0.5 * ((t - 12.0) / 2.0) ** 2) + 12.0 * np.exp(-0.5 * ((t - 28.0) / 3.0) ** 2)
    lm = detect_leops_landmarks(t, sig, LM_CFG, supplied={"a_time_ms": 30.0, "b_time_ms": 60.0})
    assert any("disagrees-with-supplied" in f for f in lm["a_trough"].flags)


def test_perg_landmark_recovery_on_synthetic():
    t = np.arange(255) * 1000.0 / 1700.0
    sig = (
        -1.5 * np.exp(-0.5 * ((t - 30.0) / 3.0) ** 2)
        + 3.0 * np.exp(-0.5 * ((t - 55.0) / 4.0) ** 2)
        - 2.0 * np.exp(-0.5 * ((t - 95.0) / 5.0) ** 2)
    )
    lm = detect_perg_landmarks(t, sig, LM_CFG)
    assert np.isclose(lm["p50"].time_ms, 55.0, atol=3.0)
    assert np.isclose(lm["n95"].time_ms, 95.0, atol=5.0)
    assert lm["n35"].time_ms is not None


# ---------------------------------------------------------------------------
# resample and segments
# ---------------------------------------------------------------------------


def test_resample_pchip_no_extrapolation():
    t = np.linspace(0, 100, 235)
    sig = np.sin(t / 10.0)
    grid, out = resample_pchip(t, sig, 128)
    assert grid.size == 128
    assert grid[0] == pytest.approx(t[0]) and grid[-1] == pytest.approx(t[-1])
    assert np.all(np.isfinite(out))


def test_canonicalize_relative_phase_domain():
    t = np.linspace(20, 110, 235)
    sig = np.cos(t / 30.0)
    ctime, csig, ctype, flags = canonicalize_relative_phase(t, sig, 40.0, 95.0, SEG_CFG.relative_phase_range, N_POINTS)
    assert ctype == "relative_phase"
    assert ctime.size == 128
    assert np.all(ctime <= 1.2 + 1e-9)
    assert ctime[0] == pytest.approx(-0.2, abs=1e-6)


def test_make_leops_segments_all_present():
    t = _t()
    sig = (
        -3.0 * np.exp(-0.5 * ((t - 12.0) / 2.0) ** 2)
        + 12.0 * np.exp(-0.5 * ((t - 28.0) / 3.0) ** 2)
        - 4.0 * np.exp(-0.5 * ((t - 60.0) / 6.0) ** 2)
    )
    lm = detect_leops_landmarks(t, sig, LM_CFG)
    segs = make_leops_segments(t, sig, lm, SEG_CFG, N_POINTS)
    assert [s.component_id.value for s in segs] == ["L_EARLY_A", "L_A_TO_B", "L_LATE"]
    for s in segs:
        assert s.time_ms.size >= 2
        assert s.time_ms[0] >= t[0] and s.time_ms[-1] <= t[-1]


def test_make_perg_segments_all_present():
    t = np.arange(255) * 1000.0 / 1700.0
    sig = (
        -1.5 * np.exp(-0.5 * ((t - 30.0) / 3.0) ** 2)
        + 3.0 * np.exp(-0.5 * ((t - 55.0) / 4.0) ** 2)
        - 2.0 * np.exp(-0.5 * ((t - 95.0) / 5.0) ** 2)
    )
    lm = detect_perg_landmarks(t, sig, LM_CFG)
    segs = make_perg_segments(t, sig, lm, SEG_CFG, N_POINTS)
    assert [s.component_id.value for s in segs] == ["P_EARLY", "P_LATE"]
    for s in segs:
        assert s.time_ms[0] >= t[0] and s.time_ms[-1] <= t[-1]


def test_no_segment_exceeds_observed_support():
    t = _t()
    sig = 10.0 * np.exp(-0.5 * ((t - 30.0) / 5.0) ** 2)
    lm = detect_leops_landmarks(t, sig, LM_CFG)
    segs = make_leops_segments(t, sig, lm, SEG_CFG, N_POINTS)
    for s in segs:
        assert s.time_ms.min() >= t.min() and s.time_ms.max() <= t.max()

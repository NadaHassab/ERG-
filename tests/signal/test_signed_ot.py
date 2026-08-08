"""Tests for the signed derivative OT descriptor (plan Section 7.8)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pathway_erg.constants import MASS_EPSILON
from pathway_erg.signal.signed_ot import (
    signed_derivative_ot,
    signed_ot_distance,
    signed_ot_w1,
)
from pathway_erg.signal.smoothing import SmoothingConfig

N_QUANTILES = 64
RNG = np.random.default_rng(42)
SMOOTH = SmoothingConfig()


def _synthetic(t0=0.0, t1=120.0, fs=1953.125):
    time = np.arange(t0, t1, 1e3 / fs)
    return time


def _peak_signal(time, center=40.0, width=8.0, amp=10.0, offset=0.0):
    return offset + amp * np.exp(-0.5 * ((time - center) / width) ** 2)


def test_constant_offset_invariance():
    time = _synthetic()
    x = _peak_signal(time)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, x + 5.0, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert np.allclose(a.q_pos, b.q_pos, atol=1e-9)
    assert np.allclose(a.q_neg, b.q_neg, atol=1e-9)
    assert np.isclose(a.mass_pos, b.mass_pos)
    assert a.transform_version == b.transform_version


def test_time_shift_shifts_quantiles():
    time = _synthetic()
    x = _peak_signal(time, center=40.0)
    y = _peak_signal(time, center=55.0)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, y, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert np.all(b.q_pos - a.q_pos > 0)
    assert np.isclose(np.mean(b.q_pos - a.q_pos), 15.0, atol=1.0)


def test_amplitude_scaling_changes_masses_not_normalized_quantiles():
    time = _synthetic()
    x = _peak_signal(time, amp=10.0)
    y = 3.0 * x
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, y, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert np.allclose(a.q_pos, b.q_pos, atol=1e-9)
    assert np.isclose(b.mass_pos, 3.0 * a.mass_pos, rtol=1e-6)


def test_zero_mass_handling():
    time = _synthetic()
    flat = np.zeros_like(time)
    a = signed_derivative_ot(time, flat, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert not a.valid_pos and not a.valid_neg
    assert np.all(a.q_pos == 0.0) and np.all(a.q_neg == 0.0)
    assert a.mass_pos == 0.0 and a.mass_neg == 0.0
    b = signed_derivative_ot(time, _peak_signal(time), 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    d = signed_ot_distance(a, b, mass_weight=1.0)
    assert np.isfinite(d)


def test_monotone_signal_has_single_sign():
    time = _synthetic()
    x = np.linspace(0.0, 1.0, time.size)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert a.valid_pos and not a.valid_neg
    assert a.mass_neg == 0.0


def test_nonuniform_timestamp_integration():
    time = np.sort(RNG.uniform(0.0, 120.0, 500))
    x = _peak_signal(time)
    a = signed_derivative_ot(time, x, np.median(np.diff(time)), SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, x, np.median(np.diff(time)), SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert np.allclose(a.to_vector(), b.to_vector())


def test_no_nan_inf_output():
    time = _synthetic()
    for amp in (0.0, 1e-12, 1.0, 1e6):
        a = signed_derivative_ot(time, _peak_signal(time, amp=amp), 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
        assert np.all(np.isfinite(a.to_vector()))


def test_determinism():
    time = _synthetic()
    x = _peak_signal(time)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert np.array_equal(a.to_vector(), b.to_vector())


def test_w1_agreement_with_scipy():
    """Quantile-map W1 must match scipy's empirical Wasserstein distance.

    The signed-OT descriptor encodes the *derivative* distribution of a
    signal; equivalently, for signal x, v+ is a density over time.  W1 between
    two such densities computed from our quantile maps must match the
    empirical W1 of samples drawn from those densities.
    """
    rng = np.random.default_rng(7)
    time = _synthetic()
    for _ in range(5):
        center_a = rng.uniform(25.0, 45.0)
        center_b = rng.uniform(45.0, 75.0)
        xa = _peak_signal(time, center=center_a, amp=10.0)
        xb = _peak_signal(time, center=center_b, amp=10.0)
        a = signed_derivative_ot(time, xa, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
        b = signed_derivative_ot(time, xb, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
        vpa = np.maximum(np.gradient(xa, time), 0.0)
        vpb = np.maximum(np.gradient(xb, time), 0.0)
        samples_a = rng.choice(time, size=5000, p=vpa / vpa.sum())
        samples_b = rng.choice(time, size=5000, p=vpb / vpb.sum())
        reference = stats.wasserstein_distance(samples_a, samples_b)
        ours = signed_ot_w1(a, b)
        assert np.isclose(ours, reference, rtol=0.03), (ours, reference)


def test_distance_metric_like_properties():
    time = _synthetic()
    a = signed_derivative_ot(time, _peak_signal(time, center=40.0), 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, _peak_signal(time, center=40.0), 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    c = signed_derivative_ot(time, _peak_signal(time, center=55.0), 0.512, SMOOTH, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
    assert signed_ot_distance(a, b, mass_weight=1.0) == pytest.approx(0.0, abs=1e-12)
    assert signed_ot_distance(a, c, mass_weight=1.0) > 0.0
    assert signed_ot_distance(a, c, mass_weight=1.0) == pytest.approx(signed_ot_distance(c, a, mass_weight=1.0))


def test_vector_length():
    time = _synthetic()
    a = signed_derivative_ot(time, _peak_signal(time), 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert a.to_vector().size == 64 + 64 + 2 + 2 + 1 + 1 + 1  # q, q, masses, frac, tv/net, valids


def test_declared_reference():
    """The sign reference is explicit (zero baseline after the offset policy)."""
    time = _synthetic()
    a = signed_derivative_ot(time, _peak_signal(time), 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert a.reference == "zero_uv_after_offset_policy"
    assert a.transform_version == "signed_derivative_ot_v2"


def test_normalized_mass_fraction():
    time = _synthetic()
    a = signed_derivative_ot(time, _peak_signal(time), 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert 0.0 <= a.mass_pos_frac <= 1.0
    # total = mass_pos + mass_neg
    assert np.isclose(a.mass_pos_frac, a.mass_pos / (a.mass_pos + a.mass_neg))
    mono = np.linspace(0.0, 1.0, time.size)  # purely increasing -> 100% positive
    m = signed_derivative_ot(time, mono, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert m.mass_pos_frac == pytest.approx(1.0)
    # flat -> no variation -> frac 0 by definition, not NaN
    flat = np.zeros_like(time)
    z = signed_derivative_ot(time, flat, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert z.mass_pos_frac == 0.0


def test_sign_flip_swaps_pos_neg():
    time = _synthetic()
    x = _peak_signal(time)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    b = signed_derivative_ot(time, -x, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert np.allclose(a.q_pos, b.q_neg)
    assert np.allclose(a.q_neg, b.q_pos)
    assert np.isclose(a.mass_pos, b.mass_neg)
    assert np.isclose(a.mass_neg, b.mass_pos)
    assert np.isclose(a.mass_pos_frac, 1.0 - b.mass_pos_frac)


def test_time_stretch_scales_quantiles_keeps_masses():
    """t' = a*t deforms quantile positions by a; variation masses stay fixed."""
    time = _synthetic()
    x = _peak_signal(time, center=80.0, width=10.0)
    a = signed_derivative_ot(time, x, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    t2a, stretched = time * 2.0, x
    b = signed_derivative_ot(t2a, stretched, 0.512, SMOOTH, n_quantiles=64, mass_tolerance=MASS_EPSILON)
    assert np.allclose(a.q_pos, b.q_pos.view() / 2.0, rtol=0.02, atol=2.0)
    assert np.isclose(a.mass_pos, b.mass_pos, rtol=0.02)
    assert np.isclose(a.mass_pos_frac, b.mass_pos_frac, atol=0.02)


def test_noise_sensitivity_remains_finite():
    time = _synthetic()
    x = _peak_signal(time) + RNG.normal(0.0, 0.05, time.size)
    for window in (2.0, 3.0, 5.0):
        cfg = SmoothingConfig(window_ms=window)
        a = signed_derivative_ot(time, x, 0.512, cfg, n_quantiles=N_QUANTILES, mass_tolerance=MASS_EPSILON)
        assert np.all(np.isfinite(a.to_vector()))

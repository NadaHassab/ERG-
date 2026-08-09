"""Synthetic VMD tests (plan Section 15.4).

Covers: hertz conversion (calibration), mode sorting by physical frequency,
reconstruction error, padding artifacts (crop correctness), stability under
neighbouring hyperparameters, NaN safety, and determinism, at both LEOP
(0.512 ms) and PERG (0.600 ms) sampling rates with 10/25/60/120 Hz components.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathway_erg.signal.vmd import (
    VMDConfig,
    calibrate_vmd_frequency,
    decompose_vmd,
    extract_vmd_features,
    vmd_feature_names,
)

FS_LEOP = 1000.0 / 0.512
FS_PERG = 1000.0 / 0.600
TONES = (10.0, 25.0, 60.0, 120.0)

CONVENTION = calibrate_vmd_frequency()
CFG = VMDConfig(K=5, alpha=2000.0, tol=1e-7, mirror_pad_ms=25.0)


def _signal(
    fs: float,
    n: int = 1024,
    tones: tuple[float, ...] = (25.0, 120.0),
    amps: tuple[float, ...] = (1.0, 0.5),
    noise_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / fs
    x = np.zeros(n)
    for fh, a in zip(tones, amps, strict=True):
        x += a * np.sin(2.0 * np.pi * fh * t + 0.3 * fh)
    if noise_sd > 0.0:
        rng = np.random.default_rng(7)
        x += rng.normal(0.0, noise_sd, n)
    return t * 1000.0, x


def _dominant_hz(result) -> np.ndarray:
    idx = int(np.argmax(result.mode_energy))
    return result.center_freqs_hz[idx]


@pytest.mark.parametrize("fs", [FS_LEOP, FS_PERG])
@pytest.mark.parametrize("tone", TONES)
def test_hz_conversion_calibrated(fs, tone):
    """Physical Hz from the calibration convention matches the tone (15.4)."""
    t_ms, x = _signal(fs, tones=(tone,), amps=(1.0,))
    res = decompose_vmd(t_ms, x, VMDConfig(K=2, alpha=2000.0, tol=1e-7), CONVENTION)
    dom = _dominant_hz(res)
    assert abs(dom - tone) / tone < 0.10, f"{tone} Hz -> {dom:.2f}"


@pytest.mark.parametrize("fs", [FS_LEOP, FS_PERG])
def test_modes_sorted_by_center_frequency(fs):
    t_ms, x = _signal(fs)
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    assert res.sorted
    assert np.all(np.isfinite(res.center_freqs_hz))


@pytest.mark.parametrize("fs", [FS_LEOP, FS_PERG])
def test_reconstruction_error_small(fs):
    t_ms, x = _signal(fs)
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    assert res.recon_rms_rel < 0.05
    assert res.residual_energy_rel < 1e-2


@pytest.mark.parametrize("fs", [FS_LEOP, FS_PERG])
def test_padding_does_not_distort_center_frequencies(fs):
    """Mirror padding must not shift recovered frequencies (15.2 step 4).

    The padded decomposition must recover the true tones (boundary artifacts
    are what mirror padding removes); the unpadded short-window result may be
    distorted, which is the failure mode padding exists to prevent.
    """
    t_ms, x = _signal(fs)
    res_pad = decompose_vmd(t_ms, x, VMDConfig(K=3, mirror_pad_ms=50.0), CONVENTION)
    for m in np.argsort(res_pad.mode_energy)[-2:]:
        fh = res_pad.center_freqs_hz[m]
        assert min(abs(fh - 25.0) / 25.0, abs(fh - 120.0) / 120.0) < 0.05


def test_transient_envelope_recovered():
    """Transient envelope (plan 15.4) still recovers the tone frequency."""
    fs = FS_PERG
    n = 1024
    t = np.arange(n) / fs
    env = np.exp(-((t - 0.05) ** 2) / (2 * 0.03**2))
    x = env * np.sin(2.0 * np.pi * 60.0 * t)
    res = decompose_vmd(t * 1000.0, x, VMDConfig(K=3), CONVENTION)
    dom = _dominant_hz(res)
    assert abs(dom - 60.0) / 60.0 < 0.10


def test_noise_robust():
    fs = FS_LEOP
    t_ms, x = _signal(fs, noise_sd=0.05)
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    dom = _dominant_hz(res)
    assert abs(dom - 120.0) / 120.0 < 0.10 or abs(dom - 25.0) / 25.0 < 0.10


def test_deterministic():
    t_ms, x = _signal(FS_LEOP)
    a = decompose_vmd(t_ms, x, CFG, CONVENTION, seed=42)
    b = decompose_vmd(t_ms, x, CFG, CONVENTION, seed=42)
    assert np.array_equal(a.modes, b.modes)
    assert np.array_equal(a.center_freqs_hz, b.center_freqs_hz)


def test_nan_signal_nan_features():
    t_ms, x = _signal(FS_PERG, n=256)
    x[10] = np.nan
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    assert np.isnan(res.center_freqs_hz).all()


def test_feature_vector_shape_and_names():
    t_ms, x = _signal(FS_PERG)
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    neigh = [
        decompose_vmd(t_ms, x, VMDConfig(K=k, mirror_pad_ms=25.0), CONVENTION)
        for k in CFG.stability_neighbors
    ]
    vec = extract_vmd_features(res, CFG, 1000.0 / 0.6, neighbor_results=neigh)
    names = vmd_feature_names(CFG)
    assert vec.size == len(names)
    assert names[0] == "mode0_center_freq_hz"
    assert names[-1] == "n_iterations"
    assert np.all(np.isfinite(vec))


def test_stability_score_present():
    t_ms, x = _signal(FS_LEOP)
    res = decompose_vmd(t_ms, x, CFG, CONVENTION)
    vec_none = extract_vmd_features(res, CFG, 1000.0 / 0.512)
    assert vec_none[14] == 1.0  # mode0 stability without neighbours

"""Phase 5: real multiscale spectral features.

The spectral feature set is computed on the *physical* component window
(uniform sampling at the recording's median dt — the canonical curves are not
uniform in time for relative-phase segments, so FFT on them would be
meaningless).  Features: per-band energy (log), per-band relative energy,
normalized spectral entropy, and dominant frequency.  The OP band
(80-300 Hz) is explicit.  Known-frequency synthetic tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.config import PreprocessingConfig, SpectralConfig, load_config
from pathway_erg.signal.spectral import (
    spectral_feature_names,
    spectral_features,
)

LEOP_FS = 1953.125  # Hz (median_dt_ms = 0.512)


def _pre_cfg() -> PreprocessingConfig:
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def _sine(freq_hz: float, seconds: float = 0.25, fs: float = LEOP_FS) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    return t, np.sin(2 * np.pi * freq_hz * t)


def _names(pre: PreprocessingConfig) -> list[str]:
    return spectral_feature_names(pre.spectral.bands)


# ---------------------------------------------------------------------------
# Known-frequency behavior
# ---------------------------------------------------------------------------


def test_pure_sine_dominant_frequency():
    pre = _pre_cfg()
    t, sig = _sine(150.0)
    feats = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    names = _names(pre)
    idx = names.index("dominant_freq_hz")
    assert abs(feats[idx] - 150.0) < 6.0  # DFT bin resolution ~4 Hz


def test_op_band_dominates_150hz():
    """150 Hz is inside the explicit OP band (80-300 Hz)."""
    pre = _pre_cfg()
    t, sig = _sine(150.0)
    feats = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    names = _names(pre)
    op_rel = feats[names.index("op_rel_energy")]
    slow_rel = feats[names.index("slow_rel_energy")]
    assert op_rel > 0.5
    assert slow_rel < 0.1


def test_slow_sine_energy_in_slow_band():
    pre = _pre_cfg()
    t, sig = _sine(5.0)
    feats = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    names = _names(pre)
    assert feats[names.index("slow_rel_energy")] > 0.7
    assert feats[names.index("op_rel_energy")] < 0.05


def test_entropy_sine_low_noise_high():
    pre = _pre_cfg()
    rng = np.random.default_rng(7)
    names = _names(pre)
    e_idx = names.index("spectral_entropy")
    _, sine = _sine(150.0)
    es = spectral_features(_sine(150.0)[0], sine, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    noise = rng.standard_normal(489)
    en = spectral_features(_sine(150.0)[0], noise, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    assert es[e_idx] < 0.4
    assert en[e_idx] > 0.85


def test_relative_energies_bounded():
    pre = _pre_cfg()
    rng = np.random.default_rng(3)
    names = _names(pre)
    rel = [names.index(f"{b}_rel_energy") for b, _, _ in pre.spectral.bands]
    for gen in (lambda: _sine(50.0), lambda: (np.arange(600) / LEOP_FS, rng.standard_normal(600))):
        t, sig = gen()
        feats = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
        vals = feats[rel]
        assert ((vals >= 0.0) & (vals <= 1.0)).all()
        assert vals.sum() <= 1.0 + 1e-9


def test_nan_input_all_nan():
    pre = _pre_cfg()
    t = np.arange(256) / LEOP_FS
    sig = np.full(256, np.nan)
    feats = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    assert np.isnan(feats).all()


def test_deterministic():
    pre = _pre_cfg()
    t, sig = _sine(80.0)
    a = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    b = spectral_features(t, sig, LEOP_FS, pre.spectral.bands, pre.spectral.dominant_range)
    assert np.array_equal(a, b)


def test_feature_names_match_bands():
    pre = _pre_cfg()
    assert {b for b, _, _ in pre.spectral.bands} == {"slow", "mid", "op", "fast"}
    names = _names(pre)
    expected = [f"{b}_{s}" for b, _, _ in pre.spectral.bands for s in ("logenergy", "rel_energy")]
    expected += ["spectral_entropy", "dominant_freq_hz"]
    assert names == expected
    assert len(names) == 2 * len(pre.spectral.bands) + 2


def test_op_band_is_explicit_in_default_config():
    default = SpectralConfig()
    assert ("op", 80.0, 300.0) in default.bands


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_process_recording_emits_spectral_vectors():
    from pathway_erg.signal.component_cache import process_recording

    t = np.arange(235) * 1000.0 / LEOP_FS
    sig = (
        -3.0 * np.exp(-0.5 * ((t - 12.0) / 2.0) ** 2)
        + 12.0 * np.exp(-0.5 * ((t - 28.0) / 3.0) ** 2)
        - 4.0 * np.exp(-0.5 * ((t - 60.0) / 6.0) ** 2)
    )
    record = pd.Series(
        {
            "dataset": "LEOP",
            "waveform_kind": "ERG",
            "median_dt_ms": 0.512,
            "global_recording_id": "SYNTH_A",
            "array_key": "SYNTH_A",
            "supplied_features_json": "",
        }
    )
    out = process_recording(record, t, sig, _pre_cfg())
    assert out["valid"]
    names = _names(_pre_cfg())
    for row in out["rows"]:
        vec = out["arrays"][row["canonical_array_key"]]["spectral_vector"]
        assert vec.shape == (len(names),)
        assert np.isfinite(vec).all()


def test_cache_spectral_alignment_with_components():
    """spectral zarr rows align 1:1 with components.parquet rows."""
    import zarr

    from pathway_erg.signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths

    root = "artifacts"
    cache = cache_paths(root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    spec = np.asarray(
        zarr.open_group(str(cache["spectral_zarr"]), mode="r")["components"]["spectral_vector"][:]
    )
    assert spec.shape[0] == len(components)
    names = _names(_pre_cfg())
    assert spec.shape[1] == len(names)


def test_spectral_builder_excludes_hard_invalid():
    from pathway_erg.models.baselines import e4_spectral_features

    recordings = pd.DataFrame(
        [
            {"global_recording_id": "R0", "global_subject_id": "U1", "dataset": "LEOP", "eye": "RE"},
            {"global_recording_id": "R1", "global_subject_id": "U2", "dataset": "LEOP", "eye": "RE"},
        ]
    )
    components = pd.DataFrame(
        [
            {"global_component_id": "C0", "global_recording_id": "R0", "component_id": "L_LATE", "component_qc_flags": "truncated-low"},
            {"global_component_id": "C1", "global_recording_id": "R0", "component_id": "L_A_TO_B", "component_qc_flags": "fallback-window"},
            {"global_component_id": "C2", "global_recording_id": "R1", "component_id": "L_LATE", "component_qc_flags": ""},
        ]
    )
    units = pd.DataFrame({"unit_id": ["U1", "U2"], "dataset": "LEOP"})
    n_feat = len(_names(_pre_cfg()))
    spectral = np.tile(np.arange(n_feat, dtype=float), (3, 1))
    fs = e4_spectral_features(units, components, recordings, "LEOP", spectral, _names(_pre_cfg()))
    u1 = fs.unit_id == "U1"
    assert fs.per_unit_n[u1] == 1  # hard_invalid C1 excluded
    assert fs.per_unit_n[fs.unit_id == "U2"] == 1
    assert fs.X[u1, 0] == pytest.approx(0.0)  # only C0 contributes

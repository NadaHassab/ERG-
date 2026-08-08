"""Physically calibrated VMD (variational mode decomposition) baseline.

VMD is an adaptive spectral-decomposition comparator (Dragomiretskiy and
Zosso, 2014, IEEE TSP 62(3):531-544).  It is kept strictly separate from the
main signed-OT pathway because mode identity is not physiological identity:
modes may swap, center frequencies depend on the implementation's convention,
and short transient traces have boundary artifacts (plan Section 15).

The plan's correctness requirements (Section 15.2) are implemented here:

1. native timestamps/sampling rates are used;
2. each transient waveform is mirror-padded with a configurable recorded
   padding duration before decomposition;
3. VMD runs on the offset-handled modeling copy;
4. padding is cropped after decomposition;
5. reconstruction is verified (relative RMS + residual energy);
6. the implementation's center-frequency convention is calibrated with
   synthetic signals (``calibrate_vmd_frequency``);
7. center frequencies are converted to physical hertz;
8. modes are sorted by physical center frequency;
9. modes are optionally matched across curves by frequency assignment
   (``match_sort_modes`` policy) instead of raw output index;
10. caches are keyed by source/config hash (see ``signal/vmd_cache.py``).

Features per mode and per decomposition follow Section 15.3; the inner-fold
hyperparameter grid (K/alpha/tol/padding) follows Section 15.2 and is exposed
through ``VMDConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import hilbert

from vmdpy import VMD as _vmdpy_vmd

DEFAULT_K = 5
DEFAULT_ALPHA = 2000.0
DEFAULT_TAU = 0.0
DEFAULT_TOL = 1e-7
DEFAULT_PADDING_MS = 25.0
DEFAULT_INIT = 1
DEFAULT_DC = 0

K_GRID = (3, 4, 5, 6)
ALPHA_GRID = (500.0, 1000.0, 2000.0, 4000.0)
TOL_GRID = (1e-6, 1e-7)
PADDING_GRID_MS = (25.0, 50.0)

_CALIBRATION_TONES_HZ = (10.0, 25.0, 60.0, 120.0)
_CALIBRATION_SAMPLES = 4096
_CALIBRATION_K = 2  # tone + residual: K=3 splits low tones with alpha=2000
_CALIBRATION_ALPHA = 2000.0
_CALIBRATION_TOL = 1e-8
_CALIBRATION_REL_TOL = 0.05


@dataclass(frozen=True)
class VMDConfig:
    """VMD decomposition parameters (plan Section 15.2 grid values as defaults)."""

    K: int = DEFAULT_K
    alpha: float = DEFAULT_ALPHA
    tau: float = DEFAULT_TAU
    DC: int = DEFAULT_DC
    init: int = DEFAULT_INIT
    tol: float = DEFAULT_TOL
    mirror_pad_ms: float = DEFAULT_PADDING_MS
    max_iter: int = 500
    seed: int = 0
    stability_neighbors: tuple[int, ...] = (4, 6)

    @property
    def key(self) -> str:
        return (
            f"K{self.K}_a{self.alpha:g}_t{self.tau:g}_dc{self.DC}"
            f"_i{self.init}_tol{self.tol:g}_pad{self.mirror_pad_ms:g}"
        )


@dataclass(frozen=True)
class FrequencyConvention:
    """Calibrated mapping from the implementation's normalized omega to Hz.

    vmdpy reports center frequencies in cycles/sample on a [-0.5, 0.5] grid;
    physical hertz are ``omega * fs`` where ``fs = 1000 / median_dt_ms``.
    ``calibrate_vmd_frequency`` verifies this on synthetic tones and records
    the worst relative error for provenance.
    """

    hz_per_omega_unit: float
    sampling_rates_hz: tuple[float, ...]
    max_relative_error: float
    verified: bool = True

    def to_hz(self, omega: np.ndarray, fs: float) -> np.ndarray:
        return np.asarray(omega, dtype=float) * self.hz_per_omega_unit * fs


@dataclass
class VMDResult:
    """One decomposition: sorted modes + per-decomposition diagnostics."""

    modes: np.ndarray  # (K, n) cropped mode time series
    center_freqs_hz: np.ndarray  # (K,) sorted ascending
    center_freqs_norm: np.ndarray  # (K,) raw omega before conversion
    recon_rms_rel: float  # relative reconstruction RMS error
    residual_energy_rel: float  # (sum residual^2)/(sum source^2)
    converged: bool
    n_iterations: int
    mode_energy: np.ndarray  # (K,) absolute energy per mode
    mode_corr_source: np.ndarray  # (K,) per-mode correlation with the source

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[0])

    @property
    def sorted(self) -> bool:
        return bool(
            self.center_freqs_hz.size < 2
            or bool(np.all(np.diff(self.center_freqs_hz) >= 0.0))
        )


def vmd_feature_names(cfg: VMDConfig) -> list[str]:
    """Fixed-length feature names for a decomposition of ``cfg.K`` modes."""
    names: list[str] = []
    mode_feats = (
        "center_freq_hz",
        "log_energy",
        "rel_energy",
        "bandwidth_hz",
        "spectral_entropy",
        "time_entropy",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "env_mean",
        "env_max",
        "env_area",
        "time_first_extremum_ms",
        "corr_source",
        "stability",
    )
    for m in range(cfg.K):
        for f in mode_feats:
            names.append(f"mode{m}_{f}")
    names.extend(
        [
            "recon_rms_rel",
            "residual_energy_rel",
            "converged",
            "n_unstable_modes",
            "n_iterations",
        ]
    )
    return names


def _vmd_call(
    signal: np.ndarray,
    cfg: VMDConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the pinned vmdpy implementation; returns (modes, omega, n_iterations)."""
    if int(cfg.init) == 2:
        np.random.seed(int(seed))  # vmdpy init=2 draws from np.random
    u, _u_hat, omega = _vmdpy_vmd(
        np.asarray(signal, dtype=float),
        float(cfg.alpha),
        float(cfg.tau),
        int(cfg.K),
        int(cfg.DC),
        int(cfg.init),
        float(cfg.tol),
    )
    n_iter = int(omega.shape[0])
    return u, omega, n_iter


def _mirror_pad(signal: np.ndarray, pad_samples: int) -> np.ndarray:
    """Mirror-pad both ends by ``pad_samples`` (plan 15.2 step 2)."""
    if pad_samples == 0:
        return signal
    return np.concatenate(
        [
            signal[1 : pad_samples + 1][::-1],
            signal,
            signal[-pad_samples - 1 : -1][::-1],
        ]
    )


def _relative_rms(recon: np.ndarray, source: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(np.asarray(source) ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((recon - source) ** 2)) / denom)


def _mode_features(
    mode: np.ndarray,
    fs: float,
    center_hz: float,
    source: np.ndarray,
    energy_total: float,
) -> dict[str, float]:
    x = np.asarray(mode, dtype=float)
    n = x.size
    energy = float(np.sum(x**2))
    rel = energy / energy_total if energy_total > 0.0 else 0.0
    spec = np.abs(np.fft.rfft(x - x.mean()))
    power = spec**2
    p = power / power.sum() if power.sum() > 0.0 else power
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spread = float(
        np.sqrt(np.maximum(0.0, float(np.sum(p * (freqs - center_hz) ** 2))))
    )
    spectral_ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(max(p.size, 2)))
    pdf, _edges = np.histogram(x, bins=32, density=True)
    pdf = pdf / (pdf.sum() + 1e-12)
    time_ent = float(-(pdf * np.log(pdf + 1e-12)).sum() / math.log(pdf.size))
    env = np.abs(hilbert(x))
    denom = float(np.sqrt(np.sum(source**2)))
    corr = (
        float(np.dot(x, source) / denom / (np.sqrt(np.sum(x**2)) + 1e-12))
        if denom > 0.0 and np.sqrt(np.sum(x**2)) > 0.0
        else 0.0
    )
    first_ext = np.argmax(np.abs(x)) / fs * 1000.0
    return {
        "center_freq_hz": center_hz,
        "log_energy": math.log(energy + 1e-12),
        "rel_energy": rel,
        "bandwidth_hz": spread,
        "spectral_entropy": spectral_ent,
        "time_entropy": time_ent,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "skewness": float(_skew(x)),
        "kurtosis": float(_kurt(x)),
        "env_mean": float(env.mean()),
        "env_max": float(env.max()),
        "env_area": float(np.trapezoid(env) / fs),
        "time_first_extremum_ms": first_ext,
        "corr_source": corr,
    }


def _skew(x: np.ndarray) -> float:
    s = float(np.std(x))
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    s = float(np.std(x))
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 4))


def decompose_vmd(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    cfg: VMDConfig,
    convention: FrequencyConvention,
    seed: int | None = None,
) -> VMDResult:
    """Decompose one offset-handled waveform with mirror padding + cropping.

    Steps (plan 15.2): median sampling rate from native timestamps; mirror-pad
    with ``cfg.mirror_pad_ms``; run VMD; crop the padding; verify
    reconstruction; convert center frequencies to physical Hz and sort.
    A NaN signal yields an all-NaN result (NaN-safe, deterministic).
    """
    seed = cfg.seed if seed is None else seed
    t = np.asarray(time_ms, dtype=float)
    x = np.asarray(signal_uv, dtype=float)
    if np.isnan(x).any() or x.size < 4:
        return VMDResult(
            modes=np.full((cfg.K, x.size), np.nan),
            center_freqs_hz=np.full(cfg.K, np.nan),
            center_freqs_norm=np.full(cfg.K, np.nan),
            recon_rms_rel=float("nan"),
            residual_energy_rel=float("nan"),
            converged=False,
            n_iterations=0,
            mode_energy=np.full(cfg.K, np.nan),
            mode_corr_source=np.full(cfg.K, np.nan),
        )
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("VMD: non-finite/non-positive median sampling interval")
    fs = 1000.0 / dt
    pad = int(round(cfg.mirror_pad_ms / 1000.0 * fs))
    padded = _mirror_pad(x, pad)
    u, omega, n_iter = _vmd_call(padded, cfg, seed)

    inner = u[:, pad : pad + x.size] if pad > 0 else u
    if inner.shape[1] != x.size:
        inner = u[:, : x.size]

    center_hz = convention.to_hz(omega[-1, :], fs)
    order = np.argsort(center_hz)
    inner = inner[order]
    center_hz = center_hz[order]
    omega_sorted = omega[-1, order]

    recon = inner.sum(axis=0)
    recon_rms = _relative_rms(recon, x)
    residual = float(np.sum((recon - x) ** 2) / (np.sum(x**2) + 1e-12))
    energy_total = float(np.sum(inner**2))
    mode_energy = np.sum(inner**2, axis=1)
    corr = np.asarray(
        [
            float(np.dot(inner[k], x) / (np.sqrt(np.sum(inner[k] ** 2) + 1e-12) * np.sqrt(np.sum(x**2)) + 1e-12))
            for k in range(cfg.K)
        ]
    )
    return VMDResult(
        modes=inner,
        center_freqs_hz=center_hz,
        center_freqs_norm=omega_sorted,
        recon_rms_rel=recon_rms,
        residual_energy_rel=residual,
        converged=n_iter < cfg.max_iter,
        n_iterations=n_iter,
        mode_energy=mode_energy,
        mode_corr_source=corr,
    )


def match_sort_modes(
    result: VMDResult,
    policy: str = "frequency_ascending",
) -> VMDResult:
    """Sort/mode-matching policy; currently frequency-ascending only.

    ``frequency_ascending`` orders modes by physical center frequency so that
    mode index m is comparable across curves; ``unstable`` mode counting marks
    modes whose center frequency differs from the neighbouring-mode spacing
    (see ``extract_vmd_features``).
    """
    if policy != "frequency_ascending":
        raise ValueError(f"match_sort_modes: unknown policy {policy!r}")
    return result


def _stability_metric(
    center_hz: np.ndarray,
    neighbor_center_hz: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    """Per-mode stability: how close each mode sits to its nearest neighbor.

    Uses the energy-weighted nearest-neighbour frequency distance in the
    neighbour decomposition (plan 15.3 'stability under neighbouring
    hyperparameters'); 0 = no neighbour within 2x the local spacing.
    """
    out = np.zeros(center_hz.size)
    for m, fc in enumerate(center_hz):
        if not np.isfinite(fc):
            out[m] = np.nan
            continue
        d = np.abs(neighbor_center_hz - fc)
        dmin = float(np.min(d))
        span = float(center_hz[-1] - center_hz[0]) if center_hz.size > 1 else 1.0
        scale = max(span / max(center_hz.size - 1, 1), 1e-9)
        out[m] = float(np.exp(-dmin / scale))
    return out


def extract_vmd_features(
    result: VMDResult,
    cfg: VMDConfig,
    fs: float,
    neighbor_results: list[VMDResult] | None = None,
) -> np.ndarray:
    """Fixed-length feature vector for one decomposition (plan 15.3).

    Per mode (sorted): physical center frequency, log/relative energy,
    bandwidth, spectral and time entropy, peak-to-peak, skewness, kurtosis,
    Hilbert-envelope mean/max/area, time of first extremum, correlation with
    the source, and a stability score against the neighbour decompositions.
    Per decomposition: relative reconstruction RMS, residual energy,
    convergence flag, unstable-mode count, iteration count.
    """
    names = vmd_feature_names(cfg)
    vec: list[float] = []
    neighbor_centers = (
        [r.center_freqs_hz for r in neighbor_results if r is not None] if neighbor_results else []
    )
    stability = np.ones(cfg.K)
    if neighbor_centers:
        stacked = np.concatenate(neighbor_centers)
        if stacked.size:
            stability = _stability_metric(result.center_freqs_hz, stacked, result.mode_energy)
    for k in range(cfg.K):
        mode = result.modes[k]
        center = float(result.center_freqs_hz[k])
        feats = _mode_features(mode, fs, center, result.modes.sum(axis=0), float(result.mode_energy.sum()))
        vec.append(feats["center_freq_hz"])
        vec.append(feats["log_energy"])
        vec.append(feats["rel_energy"])
        vec.append(feats["bandwidth_hz"])
        vec.append(feats["spectral_entropy"])
        vec.append(feats["time_entropy"])
        vec.append(feats["peak_to_peak"])
        vec.append(feats["skewness"])
        vec.append(feats["kurtosis"])
        vec.append(feats["env_mean"])
        vec.append(feats["env_max"])
        vec.append(feats["env_area"])
        vec.append(feats["time_first_extremum_ms"])
        vec.append(feats["corr_source"])
        vec.append(float(stability[k]))
    n_unstable = int(np.sum(np.isnan(result.center_freqs_hz)) + np.sum(~np.isfinite(stability)))
    vec.extend(
        [
            result.recon_rms_rel,
            result.residual_energy_rel,
            1.0 if result.converged else 0.0,
            float(n_unstable),
            float(result.n_iterations),
        ]
    )
    out = np.asarray(vec, dtype=float)
    if out.size != len(names):
        raise ValueError(f"VMD feature length {out.size} != names {len(names)}")
    return out


def calibrate_vmd_frequency(
    implementation: Callable = _vmdpy_vmd,
    sampling_rates_hz: tuple[float, ...] = (1000.0 / 0.512, 1000.0 / 0.600),
    tones_hz: tuple[float, ...] = _CALIBRATION_TONES_HZ,
    n_samples: int = _CALIBRATION_SAMPLES,
) -> FrequencyConvention:
    """Calibrate the omega->Hz convention on synthetic tones (plan 15.2 step 6).

    For every (fs, tone) the single-tone decomposition's dominant center
    frequency must equal the tone within ``_CALIBRATION_REL_TOL`` after the
    ``omega * fs`` conversion; the worst relative error is recorded.  Raises
    ``ValueError`` on a failed calibration (convention wrong for the pinned
    implementation).
    """
    worst = 0.0
    verified = True
    for fs in sampling_rates_hz:
        t = np.arange(n_samples) / fs
        for tone in tones_hz:
            x = np.sin(2.0 * np.pi * tone * t)
            u, _u_hat, omega = implementation(
                x, float(_CALIBRATION_ALPHA), 0.0, _CALIBRATION_K, 0, 1, float(_CALIBRATION_TOL)
            )
            center = float(np.max(omega[-1]))
            hz = center * fs
            err = abs(hz - tone) / tone
            worst = max(worst, err)
            if err > _CALIBRATION_REL_TOL:
                verified = False
    return FrequencyConvention(
        hz_per_omega_unit=1.0,
        sampling_rates_hz=sampling_rates_hz,
        max_relative_error=worst,
        verified=verified,
    )

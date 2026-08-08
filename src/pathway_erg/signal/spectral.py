"""Multiscale spectral features on physical component windows.

Spectral analysis requires uniform time sampling, so these features are
computed on the physical ``seg.time_ms/seg.signal_uv`` window (recording
sampling rate from ``median_dt_ms``) — never on the canonical arrays, which
are not uniform in time for relative-phase segments.

Per component: per-band log energy and relative energy for the configured
bands, normalized spectral entropy, and the dominant frequency within the
configured search range.  The OP band (80-300 Hz) is explicit in the default
``SpectralConfig``.  All values are deterministic and NaN-safe: a component
with non-finite samples yields an all-NaN vector.
"""

from __future__ import annotations

import numpy as np


def spectral_feature_names(bands: tuple[tuple[str, float, float], ...]) -> list[str]:
    names: list[str] = []
    for name, _lo, _hi in bands:
        names.append(f"{name}_logenergy")
        names.append(f"{name}_rel_energy")
    names.append("spectral_entropy")
    names.append("dominant_freq_hz")
    return names


def periodogram(
    time_ms: np.ndarray, signal_uv: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram; returns (freqs_hz, power)."""
    x = np.asarray(signal_uv, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("spectral features require at least 2 samples")
    window = np.hanning(n)
    xw = (x - x.mean()) * window
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, np.abs(spec) ** 2


def _band_energy(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    return float(power[mask].sum() * df)


def spectral_features(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    fs: float,
    bands: tuple[tuple[str, float, float], ...],
    dominant_range: tuple[float, float],
) -> np.ndarray:
    """Per-component spectral feature vector (see module docstring)."""
    if np.isnan(signal_uv).any():
        return np.full(2 * len(bands) + 2, np.nan)
    freqs, power = periodogram(time_ms, signal_uv, fs)
    total = _band_energy(freqs, power, 0.0, float(freqs[-1]))
    out: list[float] = []
    for _name, lo, hi in bands:
        energy = _band_energy(freqs, power, lo, hi)
        out.append(float(np.log1p(energy)))
        out.append(energy / total if total > 0.0 else 0.0)
    p = power / power.sum()
    out.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size)))
    lo, hi = dominant_range
    mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0.0)
    out.append(float(freqs[mask][int(np.argmax(power[mask]))]))
    return np.asarray(out, dtype=float)

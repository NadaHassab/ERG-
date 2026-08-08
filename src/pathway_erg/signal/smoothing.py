"""Smoothing copies for derivatives and landmarks.

The raw signal is never filtered; a separate smoothed copy supports stable
derivatives and landmark detection (plan Section 12.10).  Reference copy uses
Savitzky-Golay order 3 with approximately 3 ms windows; windows are expressed
in milliseconds and converted to valid odd sample counts per dataset rate.

Smoothing parameters come from `config.SmoothingConfig`; signals too short to
smooth raise instead of silently returning the raw trace.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..config import SmoothingConfig
from ..constants import MIN_SMOOTH_POINTS

__all__ = ["SmoothingConfig", "odd_window_samples", "smooth_for_analysis", "derivative_spacing_aware"]


def odd_window_samples(window_ms: float, median_dt_ms: float) -> int:
    """Convert a millisecond window to the nearest odd sample count (>= 3)."""
    if not median_dt_ms or median_dt_ms <= 0:
        raise ValueError(f"invalid median dt: {median_dt_ms}")
    n = int(round(window_ms / median_dt_ms))
    if n % 2 == 0:
        n += 1
    return max(3, n)


def smooth_for_analysis(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    median_dt_ms: float,
    config: SmoothingConfig,
) -> np.ndarray:
    """Smooth the modeling copy for derivative/landmark use."""
    signal_uv = np.asarray(signal_uv, dtype=float)
    if not np.all(np.isfinite(signal_uv)):
        raise ValueError("smoothing requires finite input")
    if signal_uv.size < MIN_SMOOTH_POINTS:
        raise ValueError(f"signal too short to smooth ({signal_uv.size} points)")
    if config.method == "none":
        return signal_uv.astype(float, copy=True)
    if config.method == "savitzky_golay":
        window = odd_window_samples(config.window_ms, median_dt_ms)
        if window > signal_uv.size:
            window = signal_uv.size if signal_uv.size % 2 == 1 else signal_uv.size - 1
        if window < MIN_SMOOTH_POINTS:
            raise ValueError(f"window {window} too small for savitzky_golay")
        poly = min(config.polyorder, window - 1)
        return savgol_filter(signal_uv, window, poly)
    raise ValueError(f"unsupported smoothing method {config.method!r}")


def derivative_spacing_aware(
    time_ms: np.ndarray, signal_uv: np.ndarray
) -> np.ndarray:
    """Spacing-aware derivative on the smoothed copy (plan Section 7.3)."""
    if time_ms.size < 2:
        raise ValueError("derivative requires at least two points")
    dt = np.diff(time_ms)
    if np.any(dt <= 0):
        raise ValueError("derivative requires increasing timestamps")
    slope = np.diff(signal_uv) / dt
    mid = 0.5 * (time_ms[:-1] + time_ms[1:])
    out = np.interp(time_ms, mid, slope)
    out[0] = out[1]
    out[-1] = out[-2]
    return out

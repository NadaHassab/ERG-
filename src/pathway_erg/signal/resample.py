"""Component resampling and relative-phase canonicalization.

Fixed-point local inputs without extrapolation.  PCHIP is the primary
interpolant (linear is a sensitivity variant); valid masks are exact.
Relative-phase canonicalization applies only where the physical support is
observed.  Every grid parameter is explicit; nothing is hardcoded here.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def resample_pchip(
    time_ms: np.ndarray, signal_uv: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample to n_points over the observed support using PCHIP."""
    if time_ms.size < 2:
        raise ValueError("cannot resample a single point")
    if time_ms.size == n_points and np.allclose(time_ms, np.linspace(time_ms[0], time_ms[-1], n_points)):
        return time_ms.copy(), signal_uv.copy()
    grid = np.linspace(time_ms[0], time_ms[-1], n_points)
    interp = PchipInterpolator(time_ms, signal_uv, extrapolate=False)
    return grid, interp(grid)


def canonicalize_relative_phase(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    t_pos: float,
    t_neg: float,
    seg_range: tuple[float, float],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]:
    """Resample a positive-to-negative transition in relative phase.

    s = (t - t_pos) / (t_neg - t_pos); output grid is `seg_range` clipped to
    the observed support so no extrapolation occurs.  Returns canonical time,
    canonical signal, canonicalization type, and flags.
    """
    if t_neg <= t_pos:
        raise ValueError(f"invalid landmark order: t_pos={t_pos}, t_neg={t_neg}")
    duration = t_neg - t_pos
    s = (time_ms - t_pos) / duration
    t0, t1 = float(time_ms[0]), float(time_ms[-1])
    s0 = max(seg_range[0], (t0 - t_pos) / duration)
    s1 = min(seg_range[1], (t1 - t_pos) / duration)
    if s1 <= s0:
        raise ValueError("relative-phase window has no observed support")
    grid = np.linspace(s0, s1, n_points)
    flags: tuple[str, ...] = ()
    if s0 > seg_range[0]:
        flags += ("truncated-low",)
    if s1 < seg_range[1]:
        flags += ("truncated-high",)
    if time_ms.size == n_points and np.allclose(grid, s, atol=1e-9):
        return grid, signal_uv.copy(), "relative_phase", flags
    interp = PchipInterpolator(s, signal_uv, extrapolate=False)
    return grid, interp(grid), "relative_phase", flags


def valid_mask_from_grid(
    source_time: np.ndarray,
    source_valid: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Exact validity mask on a resampled grid (no extrapolation allowed)."""
    t0, t1 = source_time[0], source_time[-1]
    return source_valid.astype(bool) if grid.size == source_valid.size else np.ones_like(grid, dtype=bool) & (grid >= t0) & (grid <= t1)

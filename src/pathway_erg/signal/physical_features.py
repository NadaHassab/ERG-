"""Physical feature extraction per component segment.

Candidate features (plan Section 9.4): duration, minimum/maximum/peak-to-peak,
positive and negative peak latency, area above/below local reference, rising
and falling maximum slope, log positive/negative variation mass, landmark
confidence, and truncation/fallback masks.

Segments reaching this stage are finite by construction (hard validity +
exact resampling); any non-finite input raises instead of being patched.

Never include age, sex, site, diagnosis, participant ID, file path, or
label-coded metadata here.
"""

from __future__ import annotations

import numpy as np

from ..constants import MASS_EPSILON
from .smoothing import derivative_spacing_aware


def physical_features(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    confidence: float,
    truncation_flags: tuple[str, ...],
) -> dict[str, float]:
    """Compute deterministic scalar features for one segment."""
    sig = np.asarray(signal_uv, dtype=float)
    t = np.asarray(time_ms, dtype=float)
    if not np.all(np.isfinite(sig)):
        raise ValueError("non-finite signal in physical features")
    if not np.all(np.isfinite(t)):
        raise ValueError("non-finite time in physical features")
    if sig.size < 2:
        raise ValueError("segment too short for physical features")
    duration = float(t[-1] - t[0])
    vmin, vmax = float(sig.min()), float(sig.max())
    peak_to_peak = vmax - vmin
    i_min, i_max = int(np.argmin(sig)), int(np.argmax(sig))
    mean_level = float(np.mean(sig))
    above = float(np.trapezoid(np.maximum(sig - mean_level, 0.0), t))
    below = float(np.trapezoid(np.maximum(mean_level - sig, 0.0), t))
    derivative = derivative_spacing_aware(t, sig)
    rising = float(derivative.max()) if derivative.size else 0.0
    falling = float(derivative.min()) if derivative.size else 0.0
    v_pos = np.maximum(derivative, 0.0)
    v_neg = np.maximum(-derivative, 0.0)
    m_pos = float(np.trapezoid(v_pos, t))
    m_neg = float(np.trapezoid(v_neg, t))
    return {
        "duration_ms": duration,
        "min_uv": vmin,
        "max_uv": vmax,
        "peak_to_peak_uv": peak_to_peak,
        "min_latency_ms": float(t[i_min]),
        "max_latency_ms": float(t[i_max]),
        "area_above_ref_uv_ms": above,
        "area_below_ref_uv_ms": below,
        "max_rising_slope_uv_per_ms": rising,
        "max_falling_slope_uv_per_ms": falling,
        "log_mass_pos": float(np.log(m_pos + MASS_EPSILON)),
        "log_mass_neg": float(np.log(m_neg + MASS_EPSILON)),
        "mass_pos": m_pos,
        "mass_neg": m_neg,
        "landmark_confidence": float(confidence),
        "truncated_low": float("truncated-low" in truncation_flags),
        "truncated_high": float("truncated-high" in truncation_flags),
        "fallback_used": float("fallback-window" in truncation_flags),
    }

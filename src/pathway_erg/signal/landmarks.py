"""Landmark detection for LEOP and PERG components.

Landmarks are detected from a lightly smoothed copy using explicit search
windows from the preprocessing config.  No window or pad is hardcoded here:
every search bound, separation, prominence floor, and disagreement limit
comes from `LandmarkDetectionConfig`.  A missed landmark reduces confidence
and records flags; it never deletes a participant.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..config import LandmarkDetectionConfig
from ..data.schemas import Dataset, FLASH_DATASETS, Landmark


def _window_indices(time_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((time_ms >= lo) & (time_ms <= hi))[0]


def _extreme(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    lo: float,
    hi: float,
    mode: str,
    min_prominence_frac: float,
) -> tuple[float | None, float | None, str, tuple[str, ...]]:
    """Find an extreme in [lo, hi].  Returns (time, amplitude, source, flags)."""
    idx = _window_indices(time_ms, lo, hi)
    if idx.size == 0:
        return None, None, "fallback", ("no-samples-in-window",)
    seg = signal_uv[idx]
    work = -seg if mode == "min" else seg
    amp = np.ptp(np.abs(seg)) if seg.size else 0.0
    prominence = max(amp * min_prominence_frac, np.finfo(float).eps)
    peaks, _props = find_peaks(work, prominence=prominence)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(work))])
        flags = ("no-prominence-peak",)
    else:
        flags = ()
    p = peaks[np.argmax(work[peaks])]
    if p == 0 or p == idx.size - 1:
        flags = flags + ("boundary-extreme",)
        return (
            float(time_ms[idx[p]]),
            float(seg[p]),
            "fallback",
            flags,
        )
    return (
        float(time_ms[idx[p]]),
        float(seg[p]),
        "automatic",
        flags,
    )


def _late_min(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    lo: float,
    hi: float,
    min_lo_ms: float,
    min_prominence_frac: float,
) -> tuple[float | None, float | None, str, tuple[str, ...]]:
    lo = max(lo, min_lo_ms)
    idx = _window_indices(time_ms, lo, hi)
    if idx.size < 5:
        return None, None, "fallback", ("late-support-too-short",)
    return _extreme(time_ms, signal_uv, lo, hi, "min", min_prominence_frac)


def detect_leops_landmarks(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    config: LandmarkDetectionConfig,
    supplied: dict | None = None,
) -> dict[str, Landmark]:
    """Detect a trough, b peak, and late trough for a flash ERG."""
    a_time, a_amp, a_src, a_flags = _extreme(
        time_ms,
        signal_uv,
        *config.leops_a_range,
        "min",
        config.leops_min_prominence_frac,
    )
    b_lo = (
        max(a_time + config.leops_min_separation_ms, config.leops_b_range[0])
        if a_time is not None
        else config.leops_b_range[0]
    )
    b_time, b_amp, b_src, b_flags = _extreme(
        time_ms,
        signal_uv,
        b_lo,
        config.leops_b_range[1],
        "max",
        config.leops_min_prominence_frac,
    )
    late_lo = (
        max(b_time + config.leops_late_separation_ms, config.leops_late_range[0])
        if b_time is not None
        else config.leops_late_range[0]
    )
    late_time, late_amp, late_src, late_flags = _late_min(
        time_ms,
        signal_uv,
        late_lo,
        config.leops_late_range[1],
        config.leops_late_range[0],
        config.leops_min_prominence_frac,
    )
    landmarks = {
        "a_trough": Landmark(
            "a_trough", a_time, a_amp, 1.0 if a_src == "automatic" else 0.0, a_src, a_flags
        ),
        "b_peak": Landmark(
            "b_peak", b_time, b_amp, 1.0 if b_src == "automatic" else 0.0, b_src, b_flags
        ),
        "late_trough": Landmark(
            "late_trough",
            late_time,
            late_amp,
            1.0 if late_src == "automatic" else 0.0,
            late_src,
            late_flags,
        ),
    }
    if supplied:
        for name, key in (("a_trough", "a_time_ms"), ("b_peak", "b_time_ms")):
            value = supplied.get(key)
            if value is None:
                continue
            lm = landmarks[name]
            if lm.time_ms is None:
                landmarks[name] = Landmark(
                    lm.name,
                    float(value),
                    lm.amplitude_uv,
                    0.5,
                    "metadata",
                    lm.flags + ("supplied-only",),
                )
                continue
            diff = abs(lm.time_ms - float(value))
            if diff > config.leops_max_disagreement_ms:
                landmarks[name] = Landmark(
                    lm.name,
                    lm.time_ms,
                    lm.amplitude_uv,
                    lm.confidence,
                    lm.source,
                    lm.flags + ("disagrees-with-supplied",),
                )
    return landmarks


def detect_perg_landmarks(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    config: LandmarkDetectionConfig,
) -> dict[str, Landmark]:
    """Detect N35, P50, and N95 for a PERG eye curve."""
    n35_time, n35_amp, n35_src, n35_flags = _extreme(
        time_ms,
        signal_uv,
        *config.perg_n35_range,
        "min",
        config.perg_min_prominence_frac,
    )
    p50_lo = (
        max(n35_time + config.perg_min_separation_ms, config.perg_p50_range[0])
        if n35_time is not None
        else config.perg_p50_range[0]
    )
    p50_time, p50_amp, p50_src, p50_flags = _extreme(
        time_ms,
        signal_uv,
        p50_lo,
        config.perg_p50_range[1],
        "max",
        config.perg_min_prominence_frac,
    )
    n95_lo = (
        max(p50_time + config.perg_late_separation_ms, config.perg_n95_range[0])
        if p50_time is not None
        else config.perg_n95_range[0]
    )
    n95_time, n95_amp, n95_src, n95_flags = _extreme(
        time_ms,
        signal_uv,
        n95_lo,
        config.perg_n95_range[1],
        "min",
        config.perg_min_prominence_frac,
    )
    return {
        "n35": Landmark(
            "n35", n35_time, n35_amp, 1.0 if n35_src == "automatic" else 0.0, n35_src, n35_flags
        ),
        "p50": Landmark(
            "p50", p50_time, p50_amp, 1.0 if p50_src == "automatic" else 0.0, p50_src, p50_flags
        ),
        "n95": Landmark(
            "n95", n95_time, n95_amp, 1.0 if n95_src == "automatic" else 0.0, n95_src, n95_flags
        ),
    }


def detect_landmarks(
    dataset: Dataset,
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    config: LandmarkDetectionConfig,
    supplied: dict | None = None,
) -> dict[str, Landmark]:
    if dataset in FLASH_DATASETS:
        return detect_leops_landmarks(time_ms, signal_uv, config, supplied)
    return detect_perg_landmarks(time_ms, signal_uv, config)

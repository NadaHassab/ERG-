"""Component segmentation from landmarks.

Builds broad physiological component windows (plan Sections 8.2-8.4) with all
geometry (pads, bounds, fallback windows, relative-phase range, late end)
taken explicitly from `SegmentationConfig`:

- L_EARLY_A: 0 ms .. a_time+pad, bounded by early_a_bound_ms.
- L_A_TO_B: a_time+pad[0] .. b_time+pad[1].
- L_LATE: b_time-pad .. late_end_ms; relative-phase canonicalized.
- L_OP: entire supplied OP support.
- P_EARLY: N35+pad[0] .. P50+pad[1].
- P_LATE: P50+pad[0] .. N95+pad[1] bounded by support; relative-phase
  canonicalized.

Relative-phase canonicalization uses s=(t-t_pos)/(t_neg-t_pos) over the
configured range resampled without extrapolation beyond observed support.
If late landmarks are invalid, an absolute-time segment is produced with a
distinct flag.
"""

from __future__ import annotations

import numpy as np

from ..config import SegmentationConfig
from ..data.schemas import ComponentID, Dataset, Landmark, Segment
from .resample import canonicalize_relative_phase


def _clip_range(lo: float, hi: float, t0: float, t1: float) -> tuple[float, float]:
    lo = max(lo, t0)
    hi = min(hi, t1)
    if hi <= lo:
        raise ValueError(f"empty segment window [{lo}, {hi}] within support [{t0}, {t1}]")
    return lo, hi


def _window(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    lo: float,
    hi: float,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _clip_range(lo, hi, t0, t1)
    mask = (time_ms >= lo) & (time_ms <= hi)
    return time_ms[mask], signal_uv[mask]


def make_leops_segments(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    landmarks: dict[str, Landmark],
    seg_cfg: SegmentationConfig,
    n_points: int,
) -> list[Segment]:
    cfg = seg_cfg.leops
    t0, t1 = float(time_ms[0]), float(time_ms[-1])
    a = landmarks["a_trough"]
    b = landmarks["b_peak"]
    late = landmarks["late_trough"]
    segments: list[Segment] = []

    if a.time_ms is not None:
        early_lo, early_hi = 0.0, min(a.time_ms + cfg.early_a_pad_ms, cfg.early_a_bound_ms)
    else:
        early_lo, early_hi = cfg.early_a_fallback_ms
    et, es = _window(time_ms, signal_uv, early_lo, early_hi, t0, t1)
    segments.append(
        Segment(
            component_id=ComponentID.L_EARLY_A,
            time_ms=et,
            signal_uv=es,
            canonical_time=et,
            canonical_signal=es,
            physical_features={},
            confidence=a.confidence,
            flags=("fallback-window",) if a.time_ms is None else a.flags,
        )
    )

    if a.time_ms is not None and b.time_ms is not None:
        atb_lo, atb_hi = a.time_ms + cfg.a_to_b_pad_ms[0], b.time_ms + cfg.a_to_b_pad_ms[1]
        confidence = min(a.confidence, b.confidence)
        flags = a.flags + b.flags
    else:
        atb_lo, atb_hi = cfg.a_to_b_fallback_ms
        confidence = 0.0
        flags = ("fallback-window",)
    tt, ts = _window(time_ms, signal_uv, atb_lo, atb_hi, t0, t1)
    segments.append(
        Segment(
            component_id=ComponentID.L_A_TO_B,
            time_ms=tt,
            signal_uv=ts,
            canonical_time=tt,
            canonical_signal=ts,
            physical_features={},
            confidence=confidence,
            flags=flags,
        )
    )

    if b.time_ms is not None:
        late_lo = b.time_ms - cfg.late_pad_ms
    else:
        late_lo = cfg.late_fallback_lo_ms
    late_hi = min(t1, seg_cfg.late_end_ms)
    lt, ls = _window(time_ms, signal_uv, late_lo, late_hi, t0, t1)
    if b.time_ms is not None and late.time_ms is not None:
        canonical_time, canonical_signal, canon_type, canon_flags = canonicalize_relative_phase(
            lt,
            ls,
            b.time_ms,
            late.time_ms,
            seg_range=seg_cfg.relative_phase_range,
            n_points=n_points,
        )
    else:
        canonical_time, canonical_signal, canon_type, _canon_flags = (
            lt,
            ls,
            "absolute",
            ("late-landmark-invalid",),
        )
    segments.append(
        Segment(
            component_id=ComponentID.L_LATE,
            time_ms=lt,
            signal_uv=ls,
            canonical_time=canonical_time,
            canonical_signal=canonical_signal,
            physical_features={},
            confidence=min(b.confidence, late.confidence) if b.time_ms is not None else 0.0,
            canonicalization_type=canon_type,
            flags=canon_flags,
        )
    )
    return segments


def make_op_segment(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    confidence: float,
) -> Segment:
    """L_OP uses the entire supplied OP support at the configured confidence."""
    return Segment(
        component_id=ComponentID.L_OP,
        time_ms=time_ms,
        signal_uv=signal_uv,
        canonical_time=time_ms,
        canonical_signal=signal_uv,
        physical_features={},
        confidence=confidence,
        flags=(),
    )


def make_perg_segments(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    landmarks: dict[str, Landmark],
    seg_cfg: SegmentationConfig,
    n_points: int,
) -> list[Segment]:
    cfg = seg_cfg.perg
    t0, t1 = float(time_ms[0]), float(time_ms[-1])
    n35 = landmarks["n35"]
    p50 = landmarks["p50"]
    n95 = landmarks["n95"]
    segments: list[Segment] = []

    if n35.time_ms is not None and p50.time_ms is not None:
        early_lo, early_hi = n35.time_ms + cfg.early_pad_ms[0], p50.time_ms + cfg.early_pad_ms[1]
        confidence = min(n35.confidence, p50.confidence)
        flags = n35.flags + p50.flags
    else:
        early_lo, early_hi = cfg.early_fallback_ms
        confidence = 0.0
        flags = ("fallback-window",)
    et, es = _window(time_ms, signal_uv, early_lo, early_hi, t0, t1)
    segments.append(
        Segment(
            component_id=ComponentID.P_EARLY,
            time_ms=et,
            signal_uv=es,
            canonical_time=et,
            canonical_signal=es,
            physical_features={},
            confidence=confidence,
            flags=flags,
        )
    )

    if p50.time_ms is not None and n95.time_ms is not None:
        late_lo = p50.time_ms + cfg.late_pad_ms[0]
        late_hi = min(t1, n95.time_ms + cfg.late_pad_ms[1])
        confidence = min(p50.confidence, n95.confidence)
        flags = p50.flags + n95.flags
    else:
        late_lo, late_hi = cfg.late_fallback_ms
        confidence = 0.0
        flags = ("fallback-window",)
    lt, ls = _window(time_ms, signal_uv, late_lo, late_hi, t0, t1)
    if p50.time_ms is not None and n95.time_ms is not None:
        canonical_time, canonical_signal, canon_type, canon_flags = canonicalize_relative_phase(
            lt,
            ls,
            p50.time_ms,
            n95.time_ms,
            seg_range=seg_cfg.relative_phase_range,
            n_points=n_points,
        )
    else:
        canonical_time, canonical_signal, canon_type, _canon_flags = (
            lt,
            ls,
            "absolute",
            ("late-landmark-invalid",),
        )
    segments.append(
        Segment(
            component_id=ComponentID.P_LATE,
            time_ms=lt,
            signal_uv=ls,
            canonical_time=canonical_time,
            canonical_signal=canonical_signal,
            physical_features={},
            confidence=confidence,
            canonicalization_type=canon_type,
            flags=flags,
        )
    )
    return segments


def make_segments(
    dataset: Dataset,
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    landmarks: dict[str, Landmark],
    seg_cfg: SegmentationConfig,
    n_points: int,
    op_time_ms: np.ndarray | None = None,
    op_signal_uv: np.ndarray | None = None,
) -> list[Segment]:
    """Build component segments for one waveform (or waveform+OP for LEOP)."""
    if dataset is Dataset.LEOP:
        segments = make_leops_segments(time_ms, signal_uv, landmarks, seg_cfg, n_points)
        if op_time_ms is not None and op_signal_uv is not None:
            segments.append(make_op_segment(op_time_ms, op_signal_uv, seg_cfg.op_default_confidence))
        return segments
    return make_perg_segments(time_ms, signal_uv, landmarks, seg_cfg, n_points)

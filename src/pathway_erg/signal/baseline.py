"""Baseline and offset handling.

Controls technical offsets without deleting amplitude (plan Section 12.9).

- LEOP primary: compute pre-stimulus median/MAD/slope and subtract the median
  only from the modeling copy; statistics are retained for QC.  The policy
  requires actual pre-stimulus support.
- PERG primary: no fixed baseline; source level is kept for the raw branch and
  derivative transport is offset-invariant.  Alternative policies
  (whole-trace median, robust detrend) are sensitivity variants only.

Every knob (policy, stimulus onset, MAD scale, detrend inlier multiple) is an
explicit parameter; there are no hidden defaults and no silent no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..constants import MAD_SCALE


class OffsetPolicyName(str, Enum):  # noqa: UP042
    NONE = "none"
    PRESTIMULUS_MEDIAN = "prestimulus_median"
    WHOLE_TRACE_MEDIAN = "whole_trace_median"
    ROBUST_TREND = "robust_trend"


@dataclass(frozen=True)
class BaselineStats:
    prestimulus_median_uv: float | None
    prestimulus_mad_uv: float | None
    prestimulus_slope_uv_per_ms: float | None
    whole_trace_median_uv: float | None
    policy: str

    def as_dict(self) -> dict:
        return {
            "prestimulus_median_uv": self.prestimulus_median_uv,
            "prestimulus_mad_uv": self.prestimulus_mad_uv,
            "prestimulus_slope_uv_per_ms": self.prestimulus_slope_uv_per_ms,
            "whole_trace_median_uv": self.whole_trace_median_uv,
            "policy": self.policy,
        }


def prestimulus_mask(time_ms: np.ndarray, t0_ms: float) -> np.ndarray:
    """True for samples at or before stimulus onset (negative-time support)."""
    return time_ms <= t0_ms


def compute_baseline_stats(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    t0_ms: float,
    policy: str,
    mad_multiple: float,
) -> BaselineStats:
    """Compute offset statistics without modifying the signal."""
    if policy not in {p.value for p in OffsetPolicyName}:
        raise ValueError(f"unknown offset policy {policy!r}")
    prestim = signal_uv[prestimulus_mask(time_ms, t0_ms)]
    median: float | None = None
    mad: float | None = None
    slope: float | None = None
    if prestim.size:
        median = float(np.median(prestim))
        if prestim.size >= 2:
            mad = float(MAD_SCALE * np.median(np.abs(prestim - median)))
            slope = float(np.polyfit(time_ms[prestimulus_mask(time_ms, t0_ms)], prestim, 1)[0])
        else:
            mad = 0.0
    whole = float(np.median(signal_uv))
    return BaselineStats(median, mad, slope, whole, policy)


def apply_offset(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    t0_ms: float,
    policy: str,
    mad_multiple: float,
) -> tuple[np.ndarray, BaselineStats]:
    """Return the modeling copy and the statistics used.

    Policies:
      - "prestimulus_median": subtract pre-stimulus median (LEOP primary);
        raises if there is no pre-stimulus support.
      - "none": return signal unchanged (PERG primary).
      - "whole_trace_median": subtract whole-trace median (sensitivity).
      - "robust_trend": subtract a robust linear trend fit on the whole trace
        (sensitivity); raises if fewer than three inliers remain.
    """
    if policy not in {p.value for p in OffsetPolicyName}:
        raise ValueError(f"unknown offset policy {policy!r}")
    if policy == "none":
        stats = compute_baseline_stats(time_ms, signal_uv, t0_ms, policy, mad_multiple)
        return signal_uv.astype(np.float64).copy(), stats
    stats = compute_baseline_stats(time_ms, signal_uv, t0_ms, policy, mad_multiple)
    out = signal_uv.astype(np.float64).copy()
    if policy == "prestimulus_median":
        if stats.prestimulus_median_uv is None:
            raise ValueError("prestimulus_median policy requires pre-stimulus support")
        out = out - stats.prestimulus_median_uv
    elif policy == "whole_trace_median":
        out = out - stats.whole_trace_median_uv
    elif policy == "robust_trend":
        med = np.median(signal_uv)
        mad = MAD_SCALE * np.median(np.abs(signal_uv - med))
        inlier = mad > 0 and np.abs(signal_uv - med) <= mad_multiple * mad
        if inlier.sum() < 3:
            raise ValueError("robust_trend: fewer than three inliers")
        slope, intercept = np.polyfit(time_ms[inlier], signal_uv[inlier], 1)
        out = out - (slope * time_ms + intercept)
    return out, stats

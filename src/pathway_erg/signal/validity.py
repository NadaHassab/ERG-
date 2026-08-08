"""Hard technical validity.

Rejects only structurally unusable curves *before* fold-dependent QC.  Low
amplitude, unusual timing, or flat morphology may be disease biology and are
never rejected here.

Hard failures (plan Section 12.7): empty/unparseable arrays, time-amplitude
mismatch, below `hard_finite_fraction` finite values, unrepairable
non-monotonic time, more than one isolated non-finite gap, broken
subject/visit linkage, impossible eye/session linkage, or inadequate points
for interpolation.  At most one isolated non-finite value may be interpolated
and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    n_finite: int = 0
    n_samples: int = 0
    fixed_mask: np.ndarray | None = None  # True where a NaN was interpolated

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "n_finite": int(self.n_finite),
            "n_samples": int(self.n_samples),
        }


def check_hard_validity(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    hard_finite_fraction: float,
    max_isolated_nan_gaps: int,
) -> ValidityResult:
    """Validate one waveform against hard structural criteria."""
    if time_ms.ndim != 1 or signal_uv.ndim != 1:
        return ValidityResult(False, ("non-1d-array",))
    if time_ms.size == 0 or signal_uv.size == 0:
        return ValidityResult(False, ("empty-array",))
    if time_ms.shape != signal_uv.shape:
        return ValidityResult(False, ("length-mismatch",))
    if not np.all(np.isfinite(time_ms)):
        return ValidityResult(False, ("non-finite-time",))
    n = time_ms.size
    finite_mask = np.isfinite(signal_uv)
    n_finite = int(finite_mask.sum())
    if n_finite / n < hard_finite_fraction:
        return ValidityResult(False, ("finite-fraction",), n_finite, n)

    # monotonic time
    if np.any(np.diff(time_ms) <= 0):
        return ValidityResult(False, ("non-monotonic-time",), n_finite, n)

    # count isolated non-finite gaps (runs of NaN separated by finite values)
    if n_finite < n:
        runs: list[int] = []
        in_run = False
        for v in ~finite_mask:
            if v and not in_run:
                in_run = True
                runs.append(1)
            elif v:
                runs[-1] += 1
            else:
                in_run = False
        # allow at most `max_isolated_nan_gaps` runs of a single point each
        if len(runs) > max_isolated_nan_gaps or any(r > 1 for r in runs):
            return ValidityResult(False, ("too-many-nan-gaps",), n_finite, n)

    if n < 3:
        return ValidityResult(False, ("insufficient-points",), n_finite, n)

    fixed_mask = None
    if n_finite < n:
        fixed_mask = ~finite_mask

    return ValidityResult(True, (), n_finite, n, fixed_mask)


def interpolate_isolated_nan(
    time_ms: np.ndarray, signal_uv: np.ndarray, fixed_mask: np.ndarray | None
) -> np.ndarray:
    """Linearly interpolate flagged isolated non-finite values (in place copy)."""
    if fixed_mask is None:
        return signal_uv.copy()
    out = signal_uv.copy()
    for idx in np.where(fixed_mask)[0]:
        if idx > 0 and idx < out.size - 1:
            out[idx] = 0.5 * (out[idx - 1] + out[idx + 1])
        else:
            out[idx] = out[idx + 1] if idx == 0 else out[idx - 1]
    return out

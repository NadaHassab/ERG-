"""Signed derivative optimal transport descriptor (plan Section 7).

For a smoothed component x_s(t), compute the spacing-aware derivative
v(t)=dx_s/dt, split v into v+ and v- variation, normalize each sign's mass,
and represent each sign by its quantile map (inverse CDF) on a fixed
probability grid plus retained masses and validity flags.

Interpretation boundary: v+ and v- mean upward and downward voltage variation;
they do not imply retinal ON/OFF pathways (plan Section 7.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import MASS_EPSILON
from .smoothing import SmoothingConfig, derivative_spacing_aware, smooth_for_analysis

TRANSFORM_VERSION = "signed_derivative_ot_v2"

# The sign reference is declared: the offset policy (apply_offset) establishes
# a zero baseline before the derivative is taken, so v+/v- mean voltage
# variation above/below that declared zero reference.
DECLARED_REFERENCE = "zero_uv_after_offset_policy"


@dataclass(frozen=True)
class SignedOTResult:
    q_pos: np.ndarray  # [n_quantiles] quantile map (ms)
    q_neg: np.ndarray  # [n_quantiles] quantile map (ms)
    mass_pos: float
    mass_neg: float
    mass_pos_frac: float  # normalized +mass: mass_pos / (mass_pos + mass_neg)
    valid_pos: bool
    valid_neg: bool
    total_variation: float
    net_variation: float
    reference: str = DECLARED_REFERENCE
    transform_version: str = TRANSFORM_VERSION

    def to_vector(self) -> np.ndarray:
        """Fixed-order flat descriptor for classical models and caches."""
        return np.concatenate(
            [
                self.q_pos.astype(np.float64),
                self.q_neg.astype(np.float64),
                [np.log(self.mass_pos + MASS_EPSILON), np.log(self.mass_neg + MASS_EPSILON)],
                [self.mass_pos_frac],
                [self.total_variation, self.net_variation],
                [float(self.valid_pos), float(self.valid_neg)],
            ]
        )


def _quantile_map(
    time_ms: np.ndarray,
    density: np.ndarray,
    mass: float,
    n_quantiles: int,
    mass_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Inverse-CDF of a normalized density over time, on a fixed tau grid."""
    if mass <= mass_tolerance:
        return np.zeros(n_quantiles, dtype=np.float64), False
    dt = np.empty_like(density)
    dt[0] = time_ms[1] - time_ms[0]
    dt[1:] = np.diff(time_ms)
    cdf = np.cumsum(density * dt) / mass
    cdf = np.clip(cdf, 0.0, 1.0)
    tau = (np.arange(n_quantiles) + 0.5) / n_quantiles
    q = np.interp(tau, cdf, time_ms)
    return q, True


def signed_derivative_ot(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    median_dt_ms: float,
    smoothing: SmoothingConfig,
    n_quantiles: int,
    mass_tolerance: float,
) -> SignedOTResult:
    """Compute the signed derivative OT descriptor for one component.

    time_ms and signal_uv are the local component window (raw values);
    smoothing is applied on a copy so the raw branch is untouched.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    signal_uv = np.asarray(signal_uv, dtype=float)
    if time_ms.size != signal_uv.size or time_ms.size < 3:
        raise ValueError("signed OT requires aligned arrays with >= 3 points")
    if not np.all(np.isfinite(time_ms)):
        raise ValueError("non-finite timestamps in signed OT")
    if np.any(np.diff(time_ms) <= 0):
        raise ValueError("timestamps must be increasing")
    if not np.all(np.isfinite(signal_uv)):
        raise ValueError("non-finite signal in signed OT")

    smoothed = smooth_for_analysis(time_ms, signal_uv, median_dt_ms, smoothing)
    v = derivative_spacing_aware(time_ms, smoothed)
    v_pos = np.maximum(v, 0.0)
    v_neg = np.maximum(-v, 0.0)
    mass_pos = float(np.trapezoid(v_pos, time_ms))
    mass_neg = float(np.trapezoid(v_neg, time_ms))
    total_variation = mass_pos + mass_neg
    mass_pos_frac = mass_pos / total_variation if total_variation > mass_tolerance else 0.0
    q_pos, valid_pos = _quantile_map(time_ms, v_pos, mass_pos, n_quantiles, mass_tolerance)
    q_neg, valid_neg = _quantile_map(time_ms, v_neg, mass_neg, n_quantiles, mass_tolerance)
    net_variation = mass_pos - mass_neg
    return SignedOTResult(
        q_pos=q_pos,
        q_neg=q_neg,
        mass_pos=mass_pos,
        mass_neg=mass_neg,
        mass_pos_frac=mass_pos_frac,
        valid_pos=valid_pos,
        valid_neg=valid_neg,
        total_variation=total_variation,
        net_variation=net_variation,
    )


def signed_ot_distance(
    a: SignedOTResult,
    b: SignedOTResult,
    mass_weight: float,
) -> float:
    """Signed OT distance (plan Section 7.4).

    D2 = W2^2(p+,q+) + W2^2(p-,q-) + mass_weight * ||log(m_i+eps)-log(m_j+eps)||^2
    """
    if a.transform_version != b.transform_version:
        raise ValueError("cannot compare different transform versions")
    w2_pos = float(np.mean((a.q_pos - b.q_pos) ** 2)) if a.valid_pos and b.valid_pos else 0.0
    w2_neg = float(np.mean((a.q_neg - b.q_neg) ** 2)) if a.valid_neg and b.valid_neg else 0.0
    log_mass_diff = (np.log(a.mass_pos + MASS_EPSILON) - np.log(b.mass_pos + MASS_EPSILON)) ** 2 + (
        np.log(a.mass_neg + MASS_EPSILON) - np.log(b.mass_neg + MASS_EPSILON)
    ) ** 2
    return w2_pos + w2_neg + mass_weight * float(log_mass_diff)


def signed_ot_w1(a: SignedOTResult, b: SignedOTResult) -> float:
    """W1 approximation from quantile maps (used for direct-Wasserstein tests)."""
    if not (a.valid_pos and b.valid_pos):
        return float("nan")
    return float(np.mean(np.abs(a.q_pos - b.q_pos)))

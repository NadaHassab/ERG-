"""Probability calibration (plan §14.10, Module 21.18).

Fit a temperature (or logistic) calibrator on inner out-of-fold
predictions only, then apply to outer-test logits (plan §14.10: "Fit
temperature or logistic calibration on inner out-of-fold predictions
only, then apply to outer-test logits").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import ECE_BINS
from .metrics import expected_calibration_error


@dataclass(frozen=True)
class Calibrator:
    """Fitted temperature calibrator: p = sigmoid(logit / T)."""

    temperature: float
    fit_n: int
    fit_ece: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Map logits to calibrated probabilities."""
        logits = np.asarray(logits, dtype=float)
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


def _binary_cross_entropy(temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits / temp
    return float(np.mean(np.logaddexp(0.0, z) - z * labels))


def fit_calibrator(
    inner_oof_logits: np.ndarray,
    inner_oof_labels: np.ndarray,
    temperature0: float = 1.0,
) -> Calibrator:
    """Temperature calibration on inner OOF predictions (plan §14.10).

    Raises ``ValueError`` when calibration would be degenerate (missing
    class or no usable OOF samples).
    """
    logits = np.asarray(inner_oof_logits, dtype=float)
    labels = np.asarray(inner_oof_labels, dtype=float)
    if len(logits) != len(labels):
        raise ValueError("logits/labels length mismatch")
    if len(logits) < 2:
        raise ValueError("need at least 2 inner OOF samples for calibration")
    classes = set(np.unique(labels))
    if classes != {0.0, 1.0}:
        raise ValueError(f"calibration requires both classes; got {sorted(classes)}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("inner OOF logits must be finite")

    t = float(temperature0)
    lr = 0.1
    best = _binary_cross_entropy(t, logits, labels)
    for _ in range(200):
        grad = _ce_gradient(t, logits, labels)
        t_new = t - lr * grad
        if t_new <= 1e-3:
            t_new = 1e-3
        ce = _binary_cross_entropy(t_new, logits, labels)
        if ce < best:
            best = ce
            t = t_new
            lr = min(lr * 1.05, 1.0)
        else:
            lr *= 0.5
        if lr < 1e-6:
            break

    prob = 1.0 / (1.0 + np.exp(-logits / t))
    ece = expected_calibration_error(labels, prob, bins=ECE_BINS)
    return Calibrator(temperature=t, fit_n=len(logits), fit_ece=float(ece))


def _ce_gradient(t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits / t
    p = 1.0 / (1.0 + np.exp(-z))
    # d/dT of mean[logaddexp(0, z) - z*y] wrt T
    return float(np.mean((p - labels) * logits / (t * t)))

"""Fold-safe binary evaluation metrics with cluster bootstrap (plan Section 18).

Conventions
-----------
- Metrics are computed at the supervised-unit level: one prediction per LEOP
  participant and one per PERG visit.
- Confidence intervals come from stratified cluster bootstrap with at least
  2,000 replicates (plan Section 18.3).  LEOP cluster = participant; PERG
  cluster = canonical subject, with visits drawn together.
- The five outer folds are never treated as independent samples.
- No silent fallbacks: metrics that are undefined for a given sample are
  reported as None together with the explicit reason (missing class), never
  replaced by a dummy value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from ..constants import BOOTSTRAP_SEED, DEFAULT_CONFIDENCE, DEFAULT_N_BOOTSTRAP_REPS, ECE_BINS

# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def _check_both_classes(y_true: np.ndarray) -> tuple[set[int], int]:
    classes = set(np.unique(y_true))
    if len(classes) != 2 or not {0, 1}.issubset(classes):
        raise ValueError(
            "binary_metrics requires both classes 0 and 1 to be present; "
            f"got classes {sorted(classes)}"
        )
    return classes, int((y_true == 1).sum())


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = ECE_BINS) -> float:
    """Binned expected calibration error with explicit bin count."""
    if bins < 2:
        raise ValueError(f"ece_bins must be >= 2, got {bins}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return total


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All point metrics at one operating point (thresholded at `threshold`)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_prob={len(y_prob)}")
    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    _, n_pos = _check_both_classes(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob, bins=ECE_BINS),        "n_positive": n_pos,
        "n_total": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Stratified cluster bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n_replicates_used: int
    n_replicates_skipped: int
    n_clusters: int


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clusters: np.ndarray,
    metric: str = "roc_auc",
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    """Percentile interval from a stratified cluster bootstrap.

    Clusters are resampled with replacement *within* each label class, so
    every replicate keeps at least one cluster of each class whenever the
    original sample allows it.  Replicates where a class is absent are
    skipped and counted explicitly (never silently dropped from the report).
    """
    if n_reps < 100:
        raise ValueError(f"n_reps must be >= 100, got {n_reps}")
    if not (0.5 <= confidence < 1.0):
        raise ValueError(f"confidence must be in [0.5, 1), got {confidence}")
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    clusters = np.asarray(clusters)
    if not (len(y_true) == len(y_prob) == len(clusters)):
        raise ValueError("y_true, y_prob, and clusters must have equal length")
    _check_both_classes(y_true)

    unique = np.unique(clusters)
    group_pos = {}
    for c in unique:
        mask = clusters == c
        group_pos[c] = (mask, y_true[mask][0], y_prob[mask])

    class0 = [c for c in unique if group_pos[c][1] == 0]
    class1 = [c for c in unique if group_pos[c][1] == 1]
    if not class0 or not class1:
        raise ValueError("both label classes must have at least one cluster")

    rng = np.random.default_rng(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(n_reps):
        chosen = list(rng.choice(class0, size=len(class0), replace=True)) + list(
            rng.choice(class1, size=len(class1), replace=True)
        )
        yt: list[float] = []
        yp: list[float] = []
        for c in chosen:
            mask, label, prob = group_pos[c]
            yt.extend([label] * int(mask.sum()))
            yp.extend(prob.tolist())
        yt_a, yp_a = np.asarray(yt, dtype=float), np.asarray(yp, dtype=float)
        if len(set(np.unique(yt_a))) != 2:
            skipped += 1
            continue
        values.append(_metric_value(metric, yt_a, yp_a))

    if len(values) < 2:
        raise ValueError(f"too few valid bootstrap replicates ({len(values)}) for metric {metric!r}")
    values = np.asarray(values)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))
    return BootstrapResult(
        metric=metric,
        mean=float(np.mean(values)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_replicates_used=len(values),
        n_replicates_skipped=skipped,
        n_clusters=len(unique),
    )


def _metric_value(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_prob))
    if metric == "balanced_accuracy":
        y_pred = (y_prob >= 0.5).astype(int)
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"unsupported bootstrap metric {metric!r}")


# ---------------------------------------------------------------------------
# Prediction-table level reports (plan Module 21.18, §18.1/18.3/18.8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricReport:
    """Full point-metric report at the supervised-unit level (plan §18.1)."""

    n_units: int
    n_clusters: int
    n_positive: int
    metrics: dict[str, float]
    endpoint: str

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]


def _validate_prediction_table(
    prediction_table: pd.DataFrame,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "y_true" not in prediction_table.columns or "y_prob" not in prediction_table.columns:
        raise ValueError("prediction_table needs y_true and y_prob columns")
    if "cluster" not in prediction_table.columns:
        raise ValueError("prediction_table needs a cluster column (plan §18.3)")
    y_true = prediction_table["y_true"].astype(float).to_numpy()
    y_prob = prediction_table["y_prob"].astype(float).to_numpy()
    if len(y_true) != len(y_prob):
        raise ValueError("y_true / y_prob length mismatch")
    return y_true, y_prob


def evaluate_predictions(
    prediction_table: pd.DataFrame,
    endpoint: str = "roc_auc",
    threshold: float = 0.5,
) -> MetricReport:
    """All point metrics at unit level (plan §18.1) with sample-size honesty.

    ``prediction_table`` columns: ``y_true`` (0/1), ``y_prob`` (calibrated
    probability or logit), ``cluster`` (participant id for LEOP, canonical
    subject for PERG).  Raises on single-class input (no silent dummy).
    """
    y_true, y_prob = _validate_prediction_table(prediction_table, endpoint)
    m = binary_metrics(y_true, y_prob, threshold=threshold)
    n_clusters = int(prediction_table["cluster"].nunique())
    return MetricReport(
        n_units=len(y_true),
        n_clusters=n_clusters,
        n_positive=int((y_true == 1).sum()),
        metrics=m,
        endpoint=endpoint,
    )


@dataclass(frozen=True)
class BootstrapReport:
    """Cluster-bootstrap CIs for a prediction table (plan §18.3)."""

    metric: str
    point: float
    mean: float
    ci_low: float
    ci_high: float
    n_replicates_used: int
    n_replicates_skipped: int
    n_clusters: int


def cluster_bootstrap(
    prediction_table: pd.DataFrame,
    cluster_col: str = "cluster",
    metric: str = "roc_auc",
    seed: int = BOOTSTRAP_SEED,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapReport:
    """Stratified cluster bootstrap on a prediction table (plan §18.3)."""
    y_true, y_prob = _validate_prediction_table(prediction_table, metric)
    clusters = prediction_table[cluster_col].to_numpy()
    r = cluster_bootstrap_ci(
        y_true, y_prob, clusters, metric=metric,
        n_reps=n_reps, seed=seed, confidence=confidence,
    )
    return BootstrapReport(
        metric=metric,
        point=_metric_value(metric, y_true, y_prob),
        mean=r.mean, ci_low=r.ci_low, ci_high=r.ci_high,
        n_replicates_used=r.n_replicates_used,
        n_replicates_skipped=r.n_replicates_skipped,
        n_clusters=r.n_clusters,
    )


def _validate_both_classes(y: np.ndarray) -> None:
    if len(set(np.unique(y))) != 2 or not {0, 1}.issubset(set(np.unique(y))):
        raise ValueError(
            f"both classes 0 and 1 required; got {sorted(set(np.unique(y)))}"
        )

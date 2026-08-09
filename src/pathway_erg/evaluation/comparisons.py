"""Paired comparisons at the clustered unit level (plan §18.4, Module 21.18).

Paired cluster bootstrap of the *difference* in a metric between two
predictors on the same units: clusters are resampled once and both
models scored on the same replicate so within-unit dependence is kept
(plan §18.4: "paired cluster bootstrap of metric differences").

The primary comparison family (plan §18.5: pathway vs separate, correct
vs wrong graph) uses this; Holm correction is applied at the family
level by the reporting stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import BOOTSTRAP_SEED, DEFAULT_CONFIDENCE, DEFAULT_N_BOOTSTRAP_REPS
from .metrics import _metric_value, _validate_both_classes


@dataclass(frozen=True)
class ComparisonReport:
    """Paired difference with a percentile CI (plan §18.4)."""

    metric: str
    diff_point: float        # M_A - M_B on the observed units
    diff_mean: float         # mean over replicates
    ci_low: float            # percentile CI of the difference
    ci_high: float
    p_value: float           # two-sided sign-flip permutation p (unit level)
    n_replicates_used: int
    n_replicates_skipped: int
    n_clusters: int
    n_units: int


def _paired_frame(pred_a: pd.DataFrame, pred_b: pd.DataFrame, cluster_col: str):
    """Validate that A and B are the exact same units in the same order.

    Contract (plan §18.2): one row per supervised unit in both tables;
    the tables must have identical unit ids and equal length.  Pairing is
    positional after id-order verification, so repeated units can never
    cross-pair.
    """
    if "y_true" not in pred_a or "y_true" not in pred_b:
        raise ValueError("both tables need y_true")
    if cluster_col not in pred_a or cluster_col not in pred_b:
        raise ValueError(f"both tables need {cluster_col!r}")
    if len(pred_a) != len(pred_b):
        raise ValueError(
            f"unmatched unit counts: A={len(pred_a)} B={len(pred_b)} — "
            "paired comparison needs the exact same units"
        )
    pair_col = "unit_id" if "unit_id" in pred_a and "unit_id" in pred_b else cluster_col
    if pair_col == cluster_col and pred_a[cluster_col].duplicated().any():
        raise ValueError(
            "repeated clusters require an explicit unit_id column for exact pairing"
        )
    ids_a = pred_a[pair_col].to_numpy()
    ids_b = pred_b[pair_col].to_numpy()
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("unit sets differ between A and B")
    if not np.array_equal(
        pred_a[cluster_col].to_numpy(), pred_b[cluster_col].to_numpy()
    ):
        raise ValueError("cluster assignments differ between A and B")
    return pred_a, pred_b


def paired_compare(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    cluster_col: str = "cluster",
    metric: str = "roc_auc",
    seed: int = BOOTSTRAP_SEED,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    n_perm: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ComparisonReport:
    """Paired cluster bootstrap of the metric difference M_A - M_B.

    Rows of ``pred_a``/``pred_b`` are the *same* units in the same order
    (LEOP participants or PERG visits) with a ``cluster`` column; both
    tables must cover the exact same units (plan §18.4, exact ID pairing).
    """
    pa, pb = _paired_frame(pred_a, pred_b, cluster_col)
    y_true = pa["y_true"].astype(float).to_numpy()
    p_a = pa["y_prob"].astype(float).to_numpy()
    p_b = pb["y_prob"].astype(float).to_numpy()
    clusters = pa[cluster_col].to_numpy()
    _validate_both_classes(y_true)
    if not (len(y_true) == len(p_a) == len(p_b)):
        raise ValueError("length mismatch after pairing")

    unique = np.unique(clusters)
    grp = {c: (clusters == c) for c in unique}
    class0 = [c for c in unique if y_true[grp[c]][0] == 0]
    class1 = [c for c in unique if y_true[grp[c]][0] == 1]
    if not class0 or not class1:
        raise ValueError("both label classes need at least one cluster")

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    skipped = 0
    for _ in range(n_reps):
        chosen = list(rng.choice(class0, size=len(class0), replace=True)) + list(
            rng.choice(class1, size=len(class1), replace=True)
        )
        yt: list[float] = []
        pa: list[float] = []
        pb: list[float] = []
        for c in chosen:
            m = grp[c]
            yt.extend(y_true[m].tolist())
            pa.extend(p_a[m].tolist())
            pb.extend(p_b[m].tolist())
        yt_a, pa_a, pb_a = map(np.asarray, (yt, pa, pb))
        if len(set(np.unique(yt_a))) != 2:
            skipped += 1
            continue
        diffs.append(_metric_value(metric, yt_a, pa_a) - _metric_value(metric, yt_a, pb_a))

    if len(diffs) < 2:
        raise ValueError(f"too few valid paired replicates ({len(diffs)})")
    diffs = np.asarray(diffs)
    alpha = (1.0 - confidence) / 2.0

    # unit-level two-sided sign-flip permutation test on observed difference
    pv = _permutation_pvalue(y_true, p_a, p_b, metric, rng, n_perm)

    return ComparisonReport(
        metric=metric,
        diff_point=_metric_value(metric, y_true, p_a) - _metric_value(metric, y_true, p_b),
        diff_mean=float(np.mean(diffs)),
        ci_low=float(np.quantile(diffs, alpha)),
        ci_high=float(np.quantile(diffs, 1.0 - alpha)),
        p_value=pv,
        n_replicates_used=len(diffs),
        n_replicates_skipped=skipped,
        n_clusters=len(unique),
        n_units=len(y_true),
    )


def _permutation_pvalue(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    """Two-sided sign-flip p-value of M_A - M_B (plan §18.4)."""
    obs = _metric_value(metric, y_true, p_a) - _metric_value(metric, y_true, p_b)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(p_a)) < 0.5
        pa_s = np.where(swap, p_b, p_a)
        pb_s = np.where(swap, p_a, p_b)
        d = _metric_value(metric, y_true, pa_s) - _metric_value(metric, y_true, pb_s)
        if abs(d) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)

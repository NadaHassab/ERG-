"""Metrics: point metrics and cluster bootstrap (plan Section 18)."""

from __future__ import annotations

import numpy as np
import pytest

from pathway_erg.evaluation.metrics import (
    binary_metrics,
    cluster_bootstrap_ci,
    expected_calibration_error,
)


def test_binary_metrics_hand_computed():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1, 0.6, 0.2])
    m = binary_metrics(y, p, threshold=0.5)
    assert m["n_total"] == 6
    assert m["n_positive"] == 3
    assert np.isclose(m["balanced_accuracy"], 1.0)  # sens 3/3, spec 3/3
    assert np.isclose(m["sensitivity"], 1.0)
    assert np.isclose(m["specificity"], 1.0)
    assert np.isclose(m["brier"], np.mean((np.array([1, 1, 0, 0, 1, 0]) - np.array([0.9, 0.8, 0.3, 0.1, 0.6, 0.2])) ** 2))
    assert m["roc_auc"] > 0.8


def test_binary_metrics_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        binary_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))


def test_ece_perfect_calibration_is_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 2000)
    y = (rng.uniform(size=2000) < p).astype(float)
    assert expected_calibration_error(y, p, bins=10) < 0.05


def test_ece_miscalibration_large():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    assert expected_calibration_error(y, p, bins=10) > 0.4


def test_cluster_bootstrap_perfect_predictor():
    rng = np.random.default_rng(3)
    clusters = np.arange(40)
    y = rng.integers(0, 2, 40).astype(float)
    p = y.astype(float)
    res = cluster_bootstrap_ci(y, p, clusters, metric="roc_auc", n_reps=500, seed=1)
    assert res.mean == 1.0
    assert res.ci_low == 1.0 and res.ci_high == 1.0
    assert res.n_replicates_skipped == 0


def test_cluster_bootstrap_deterministic():
    rng = np.random.default_rng(9)
    clusters = np.repeat(np.arange(30), 2)
    y = rng.integers(0, 2, len(clusters)).astype(float)
    p = rng.uniform(size=len(clusters))
    a = cluster_bootstrap_ci(y, p, clusters, n_reps=300, seed=42)
    b = cluster_bootstrap_ci(y, p, clusters, n_reps=300, seed=42)
    assert a.mean == b.mean and a.ci_low == b.ci_low and a.ci_high == b.ci_high


def test_cluster_bootstrap_resamples_clusters_together():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    clusters = np.array(["a", "a", "a", "b", "b", "b"])
    res = cluster_bootstrap_ci(y, p, clusters, n_reps=200, seed=5)
    assert res.n_clusters == 2
    assert res.mean == 1.0


def test_cluster_bootstrap_guards():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    with pytest.raises(ValueError, match="equal length"):
        cluster_bootstrap_ci(y, p, np.array(["a", "b", "c"]), n_reps=200, seed=1)
    with pytest.raises(ValueError, match="n_reps"):
        cluster_bootstrap_ci(y, p, np.array(["a", "b", "c", "d"]), n_reps=10, seed=1)

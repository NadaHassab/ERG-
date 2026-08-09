"""Evaluation/statistics tests (plan Module 21.18).

Plan-mandated tests: known metrics, repeated-subject bootstrap, exact ID
pairing, degenerate class handling, calibration separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.evaluation.calibration import Calibrator, fit_calibrator
from pathway_erg.evaluation.comparisons import paired_compare
from pathway_erg.evaluation.metrics import (
    cluster_bootstrap,
    evaluate_predictions,
)


def make_table(
    n_clusters: int = 20,
    repeat: int = 3,
    seed: int = 0,
    auroc: float = 0.9,
) -> pd.DataFrame:
    """Prediction table with repeated observations per cluster."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for c in range(n_clusters):
        y = int(c % 2)
        labels.append(y)
        z = rng.normal(1.6 * auroc if y else -1.6 * auroc)
        for _ in range(repeat):
            rows.append({"cluster": f"c{c}", "y_true": y, "y_prob": 1 / (1 + np.exp(-z))})
    return pd.DataFrame(rows)


def test_evaluate_predictions_known_auroc():
    # perfect separation -> AUROC 1.0
    t = pd.DataFrame(
        {
            "cluster": ["a", "b", "c", "d"],
            "y_true": [1, 1, 0, 0],
            "y_prob": [0.99, 0.98, 0.01, 0.02],
        }
    )
    r = evaluate_predictions(t)
    assert r.metrics["roc_auc"] == pytest.approx(1.0)
    assert r.n_units == 4
    assert r.n_clusters == 4
    assert r.metrics["brier"] < 0.01


def test_evaluate_predictions_requires_cluster():
    t = pd.DataFrame({"y_true": [0, 1], "y_prob": [0.1, 0.9]})
    with pytest.raises(ValueError, match="cluster"):
        evaluate_predictions(t)


def test_evaluate_predictions_degenerate_class():
    t = pd.DataFrame(
        {"cluster": ["a", "b"], "y_true": [1, 1], "y_prob": [0.9, 0.8]}
    )
    with pytest.raises(ValueError):
        evaluate_predictions(t)


def test_evaluate_predictions_rejects_invalid_probability_and_duplicate_unit():
    bad = pd.DataFrame(
        {
            "unit_id": ["u1", "u2"],
            "cluster": ["a", "b"],
            "y_true": [0, 1],
            "y_prob": [0.1, 1.2],
        }
    )
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_predictions(bad)
    duplicate = bad.assign(unit_id=["u1", "u1"], y_prob=[0.1, 0.9])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_predictions(duplicate)


def test_cluster_bootstrap_ci_plausible():
    t = make_table(n_clusters=40, repeat=5, seed=3)
    r = cluster_bootstrap(t, n_reps=300, seed=1)
    assert 0.0 <= r.ci_low <= r.ci_high <= 1.0
    assert r.n_clusters == 40
    assert r.n_replicates_used > 250
    assert r.point >= r.ci_low - 1e-9 and r.point <= r.ci_high + 1e-9


def test_paired_compare_exact_id_pairing():
    # one prediction per unit (plan §18.2); A and B cover the same units
    t = make_table(n_clusters=30, repeat=1, seed=5)
    t2 = t.copy()
    t2["y_prob"] = 1.0 - t2["y_prob"]  # inverted predictions -> worse
    r = paired_compare(t, t2, n_reps=200, n_perm=100, seed=2)
    assert r.n_units == len(t)
    assert r.n_clusters == 30
    assert r.diff_point > 0  # A (good) beats B (inverted)
    assert r.p_value < 0.01


def test_paired_compare_rejects_mismatched_units():
    a = make_table(n_clusters=10, seed=1)
    b = make_table(n_clusters=11, seed=2)
    with pytest.raises(ValueError, match="unmatched"):
        paired_compare(a, b)


def test_paired_compare_rejects_different_clusters():
    a = make_table(n_clusters=10, repeat=1, seed=1)
    b = make_table(n_clusters=10, repeat=1, seed=2)
    b["cluster"] = [f"x{i}" for i in range(10)]  # entirely different ids
    with pytest.raises(ValueError, match="unit sets differ"):
        paired_compare(a, b)


def test_calibrator_sharpens_perfect_logits():
    c = fit_calibrator(np.array([-5.0, -5.0, 5.0, 5.0]), np.array([0, 0, 1, 1]))
    assert 0.0 < c.temperature < 1.0
    assert c.apply(np.array([5.0]))[0] > 0.99


def test_calibrator_corrects_overconfidence():
    rng = np.random.default_rng(0)
    logits = np.concatenate([rng.normal(3.0, 0.5, 200), rng.normal(-3.0, 0.5, 200)])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    c = fit_calibrator(logits, labels)
    p = c.apply(logits)
    # after calibration, predicted prob ~= label rate within tolerance
    assert abs(p[labels == 1].mean() - 1.0) < 0.15
    assert abs(p[labels == 0].mean() - 0.0) < 0.15
    assert c.fit_ece < 0.15


def test_calibrator_degenerate_rejection():
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator(np.array([1.0, 2.0]), np.array([1, 1]))
    with pytest.raises(ValueError, match="finite"):
        fit_calibrator(np.array([np.nan, 2.0]), np.array([0, 1]))


def test_calibrator_apply_shape():
    c = Calibrator(temperature=1.3, fit_n=10, fit_ece=0.1)
    out = c.apply(np.array([-2.0, 0.0, 2.0]))
    assert out.shape == (3,)
    assert ((out > 0) & (out < 1)).all()

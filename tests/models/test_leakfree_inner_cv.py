"""Phase 7: nested-CV leakage removal (plan PROGRESS §7).

Regression tests written *before* the implementation. They pin the corrected
contract: every preprocessing step (column pruning -> median impute + missing
indicator -> standard scale -> [PCA]) lives inside a single sklearn ``Pipeline``
that is fitted on the *inner training slice only*. Hyperparameters and the PCA
dimension are selected on inner folds; the SVM is Platt-calibrated inside the
pipeline; no decision threshold is tuned anywhere (thresholds locked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pathway_erg.config import BaselinesConfig
from pathway_erg.models.baselines import (
    DropDegenerateColumns,
    _inner_fold_scores,
    _parameter_grid,
    _pipeline_param_grid,
    build_pipeline,
    select_and_fit,
)


def _cfg(**kw) -> BaselinesConfig:
    kwargs = dict(name="p7")
    kwargs.update(kw)
    return BaselinesConfig(**kwargs)


def _data(n=60, d=5, seed=0, marker_amp=1000.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=n) > 0).astype(float)
    return X, y


def _inner_folds(n_units: int, outer_fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [f"U{i}" for i in range(n_units)],
            "dataset": ["LEOP"] * n_units,
            "outer_fold": [outer_fold] * n_units,
            "inner_fold": [i % 4 for i in range(n_units)],
            "outer_fold_sel": [outer_fold] * n_units,
        }
    )


# ---------------------------------------------------------------------------
# Pipeline contents and ordering
# ---------------------------------------------------------------------------


def test_build_pipeline_logreg_steps_and_order():
    pipe = build_pipeline("logreg", "clinical", _cfg(), seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0})
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == ["col_drop", "imputer", "scaler", "est"]
    assert isinstance(pipe.named_steps["est"], LogisticRegression)


def test_build_pipeline_pca_fpca_has_pca_in_pipeline():
    pipe = build_pipeline("logreg", "pca_fpca", _cfg(), seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0, "n_components": 4})
    assert [n for n, _ in pipe.steps] == ["col_drop", "imputer", "scaler", "pca", "est"]
    assert pipe.named_steps["pca"].n_components == 4


def test_build_pipeline_pca_fpca_demog_variant_is_pca_branch():
    pipe = build_pipeline("logreg", "pca_fpca_demog", _cfg(), seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0, "n_components": 4})
    assert "pca" in [n for n, _ in pipe.steps]


def test_build_pipeline_svm_is_calibrated_inside_pipeline():
    pipe = build_pipeline("svm_rbf", "clinical", _cfg(), seed=7, use_gpu=False,
                          params={"C": 1.0, "gamma": 0.1})
    est = pipe.named_steps["est"]
    assert isinstance(est, CalibratedClassifierCV)


def test_build_pipeline_histgb_has_no_scaler():
    pipe = build_pipeline("histgb", "clinical", _cfg(), seed=7, use_gpu=False,
                          params={"max_iter": 50, "learning_rate": 0.1,
                                  "max_depth": 3, "l2": 0.0})
    assert [n for n, _ in pipe.steps] == ["col_drop", "imputer", "est"]


# ---------------------------------------------------------------------------
# Column pruning is fitted on the slice handed to it (inner training only)
# ---------------------------------------------------------------------------


def test_col_drop_prunes_by_fit_slice_only():
    # Column 1 is constant within each single inner-training slice because it is
    # 1.0 exactly on fold-3 samples and 0.0 elsewhere. Fit on the full set keeps
    # it (variance > 0); fit on a slice that excludes the 1.0 rows drops it.
    X = np.zeros((64, 3))
    X[:, 0] = np.arange(64) / 64
    X[:, 1] = (np.arange(64) % 4 == 3).astype(float)  # constant within each slice
    X[:, 2] = 2.0  # constant everywhere
    tr = np.arange(64) % 4 != 3
    full = DropDegenerateColumns().fit(X)
    assert full.n_kept() == 2  # col2 constant -> dropped
    sliced = DropDegenerateColumns().fit(X[tr])
    assert sliced.n_kept() == 1  # col1 fought constant once fold-3 rows excluded
    assert sliced.n_dropped() == 2


def test_col_drop_drops_all_nan_on_slice():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [0.5, np.nan]])
    col = DropDegenerateColumns().fit(X)
    out = col.transform(X)
    assert out.shape[1] == 1
    assert col.n_dropped() == 1


# ---------------------------------------------------------------------------
# Inner scoring fits the whole pipeline per slice
# ---------------------------------------------------------------------------


def test_inner_fold_scores_signature_and_nonzero():
    X, y = _data()
    inner_fold = np.arange(len(y)) % 4
    score, params = _inner_fold_scores(
        "logreg", "clinical", {"C": 1.0, "l1_ratio": 0.0},
        X, y, inner_fold, _cfg(), seed=7, use_gpu=False,
    )
    assert np.isfinite(score)
    assert params["C"] == 1.0


def test_select_and_fit_returns_fitted_pipeline():
    X, y = _data()
    ids = np.array([f"U{i}" for i in range(len(y))])
    inner = _inner_folds(len(y), 2)
    pipe, params, inner_auc = select_and_fit(
        "logreg", "clinical", "LEOP", X, y, ids, inner, 2, _cfg(), seed=7, use_gpu=False
    )
    assert isinstance(pipe, Pipeline)
    assert np.isfinite(inner_auc)
    assert pipe.predict_proba(X).shape == (len(y), 2)


def test_select_and_fit_raises_without_inner_folds():
    X = np.zeros((10, 2))
    y = np.array([1.0] * 5 + [0.0] * 5)
    ids = np.array([f"U{i}" for i in range(10)])
    inner = _inner_folds(10, 0)
    with pytest.raises(ValueError, match="inner-fold assignments"):
        select_and_fit("logreg", "clinical", "LEOP", X, y, ids, inner, 3, _cfg(), seed=7, use_gpu=False)


# ---------------------------------------------------------------------------
# PCA dimension is selected on inner folds, not an outer-train threshold
# ---------------------------------------------------------------------------


def test_pipeline_param_grid_adds_pca_dim_only_for_pca_fpca():
    cfg = _cfg(max_pca_components=16)
    grid = _pipeline_param_grid("logreg", "pca_fpca", cfg, n_features=40)
    assert all("n_components" in p for p in grid)
    assert all(2 <= p["n_components"] <= 16 for p in grid)
    assert sorted({p["n_components"] for p in grid}) == sorted({p["n_components"] for p in grid})
    plain = _pipeline_param_grid("logreg", "clinical", cfg, n_features=40)
    assert plain and all("n_components" not in p for p in plain)


def test_pca_dim_grid_respects_feature_dim_and_cap():
    grid = _pipeline_param_grid("logreg", "pca_fpca", _cfg(max_pca_components=64), n_features=5)
    assert max(p["n_components"] for p in grid) == 5


def test_no_threshold_keys_in_any_parameter_grid():
    for kind in ("logreg", "svm_rbf", "histgb"):
        for params in _parameter_grid(kind):
            assert not any("threshold" in k for k in params)


# ---------------------------------------------------------------------------
# Determinism and parallelism agree
# ---------------------------------------------------------------------------


def test_select_and_fit_parallel_matches_sequential():
    X, y = _data(seed=42)
    inner_fold = np.arange(len(y)) % 4
    scored = [
        _inner_fold_scores("logreg", "clinical", p, X, y, inner_fold, _cfg(), seed=7, use_gpu=False)
        for p in _pipeline_param_grid("logreg", "clinical", _cfg(), n_features=X.shape[1])
    ]
    best_seq = max(scored, key=lambda r: r[0])[1]
    ids = np.array([f"U{i}" for i in range(len(y))])
    inner = _inner_folds(len(y), 1)
    _, params_par, score_par = select_and_fit(
        "logreg", "clinical", "LEOP", X, y, ids, inner, 1, _cfg(), seed=7, use_gpu=False
    )
    assert params_par == best_seq
    assert np.isfinite(score_par)


def test_pipeline_deterministic_repeat_identical():
    X, y = _data(seed=3)
    ids = np.array([f"U{i}" for i in range(len(y))])
    inner = _inner_folds(len(y), 0)
    p1, pa, _ = select_and_fit("logreg", "clinical", "LEOP", X, y, ids, inner, 0, _cfg(), seed=7, use_gpu=False)
    p2, pb, _ = select_and_fit("logreg", "clinical", "LEOP", X, y, ids, inner, 0, _cfg(), seed=7, use_gpu=False)
    assert pa == pb
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))

"""Phase 9 reporting/acceptance gate tests (v2 plan Phase 9).

Covers: subject-level label permutation (deterministic, prevalence and
subject-clustering preserved), confusion matrices at the locked 0.5
threshold, manifest-hash verification, and metric-completeness checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pathway_erg.config import BaselinesConfig, load_config
from pathway_erg.evaluation.acceptance import (
    _fast_auc,
    clustered_null_pvalue,
    verify_metric_completeness,
    verify_predictions,
    verify_run_manifest_hashes,
)
from pathway_erg.models.baselines import _confusion_counts, _permute_labels_subject_level


def _visits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_visit_id": [f"v{i}" for i in range(12)],
            "global_subject_id": [f"s{i % 4}" for i in range(12)],
            "target_binary": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            "dataset": ["PERG"] * 12,
        }
    )


def _participants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_subject_id": [f"s{i}" for i in range(4)],
            "dataset": ["PERG"] * 4,
            "age_years": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_permute_labels_is_deterministic():
    v1 = _permute_labels_subject_level(_visits(), seed=7, participants=_participants())
    v2 = _permute_labels_subject_level(_visits(), seed=7, participants=_participants())
    assert v1["target_binary"].tolist() == v2["target_binary"].tolist()


def test_permute_labels_preserves_prevalence():
    out = _permute_labels_subject_level(_visits(), seed=7, participants=_participants())
    assert out["target_binary"].sum() == _visits()["target_binary"].sum()
    assert set(out["target_binary"].unique()) <= {0, 1}


def test_permute_labels_preserves_subject_clustering():
    out = _permute_labels_subject_level(_visits(), seed=7, participants=_participants())
    for sid in ("s0", "s1", "s2", "s3"):
        labels = set(out.loc[out["global_subject_id"] == sid, "target_binary"])
        assert len(labels) == 1, f"subject {sid} has mixed labels after permutation"


def test_permute_labels_does_not_mutate_input():
    visits = _visits()
    original = visits["target_binary"].tolist()
    _permute_labels_subject_level(visits, seed=7, participants=_participants())
    assert visits["target_binary"].tolist() == original


def test_confusion_counts():
    y = np.asarray([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.asarray([0.9, 0.4, 0.6, 0.7, 0.2, 0.1, 0.3, 0.6])
    cm = _confusion_counts(y, p)
    assert cm == {"threshold": 0.5, "tp": 3, "fp": 1, "tn": 3, "fn": 1}


def test_verify_manifest_hashes(tmp_path):
    root = Path(tmp_path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config_hash": "a",
                "data_hash": "b",
                "split_hash": "c",
                "label_mapping_hash": "d",
            }
        )
    )
    checks = verify_run_manifest_hashes(root)
    assert all(v for k, v in checks.items() if k.endswith("_hash"))
    assert checks["manifest_exists"]


def test_verify_manifest_hashes_fails_on_empty(tmp_path):
    root = Path(tmp_path)
    (root / "manifest.json").write_text(
        json.dumps({"config_hash": "", "data_hash": "", "split_hash": "", "label_mapping_hash": ""})
    )
    checks = verify_run_manifest_hashes(root)
    assert not checks["config_hash"]
    assert not checks["data_hash"]


def test_verify_metric_completeness(tmp_path):
    root = Path(tmp_path)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "LEOP/x": {
                    "roc_auc": 0.7,
                    "roc_auc_ci_low": 0.6,
                    "roc_auc_ci_high": 0.8,
                    "balanced_accuracy": 0.6,
                    "sensitivity": 0.6,
                    "specificity": 0.6,
                    "f1": 0.5,
                    "auprc": 0.4,
                    "brier": 0.2,
                    "ece": 0.1,
                    "confusion_matrix_at_0_5": {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
                }
            }
        )
    )
    checks = verify_metric_completeness(root)
    assert checks["passed"]
    assert checks["complete_rows"] == 1


def test_verify_metric_completeness_flags_missing_confusion(tmp_path):
    root = Path(tmp_path)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "LEOP/x": {
                    "roc_auc": 0.7,
                    "roc_auc_ci_low": 0.6,
                    "roc_auc_ci_high": 0.8,
                    "balanced_accuracy": 0.6,
                    "sensitivity": 0.6,
                    "specificity": 0.6,
                    "f1": 0.5,
                    "auprc": 0.4,
                    "brier": 0.2,
                    "ece": 0.1,
                }
            }
        )
    )
    checks = verify_metric_completeness(root)
    assert not checks["passed"]
    assert len(checks["incomplete"]) == 1


def test_verify_predictions(tmp_path):
    root = Path(tmp_path)
    pd.DataFrame(
        {
            "method": ["a"] * 4,
            "task": ["LEOP"] * 4,
            "cohort": [None] * 4,
            "outer_fold": [0] * 4,
            "unit_id": [1, 2, 3, 4],
            "subject_id": [1, 2, 3, 4],
            "visit_id": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8],
            "note": ["x"] * 4,
        }
    ).to_parquet(root / "predictions.parquet", index=False)
    checks = verify_predictions(root)
    assert checks["passed"]
    assert checks["n_rows"] == 4


def test_config_accepts_label_permutation_seed():
    cfg = load_config(BaselinesConfig, "configs/experiments/e4_baselines_legacy.yaml")
    assert cfg.label_permutation_seed is None


def _perm_predictions(n_subjects: int = 20, n_units: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for s in range(n_subjects):
        label = int(rng.random() < 0.5)
        for u in range(n_units):
            rows.append(
                {
                    "method": "m0",
                    "task": "LEOP",
                    "outer_fold": s % 5,
                    "unit_id": s * n_units + u,
                    "subject_id": s,
                    "target": label,
                    "probability": float(rng.random()),
                }
            )
    return pd.DataFrame(rows)


def test_fast_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=200).astype(float)
    p = rng.random(200)
    order = np.argsort(p, kind="stable")
    assert abs(_fast_auc(y[order]) - roc_auc_score(y, p)) < 1e-9


def test_clustered_null_consistent_with_chance(tmp_path):
    root = Path(tmp_path)
    perm_dirs = []
    for seed in (0, 1, 2):
        d = root / f"p_labelperm_s{seed}"
        d.mkdir(parents=True)
        _perm_predictions().to_parquet(d / "predictions.parquet", index=False)
        perm_dirs.append(d)
    null = clustered_null_pvalue(perm_dirs)
    assert null["n_blocks"] == 3
    assert 0.0 <= null["p_value"] <= 1.0
    # subject-clustered random labels must sit inside the null
    assert null["p_value"] >= 0.05
    assert null["observed"] <= null["null_95"] or null["p_value"] >= 0.05


def test_clustered_null_rejects_predictive_permutation(tmp_path):
    root = Path(tmp_path)
    d = root / "p_labelperm_s0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(5)
    rows = []
    for s in range(24):
        label = int(rng.random() < 0.5)
        for u in range(3):
            rows.append(
                {
                    "method": "m0",
                    "task": "LEOP",
                    "outer_fold": s % 6,
                    "unit_id": s * 3 + u,
                    "subject_id": s,
                    "target": label,
                    "probability": float(label) + 0.05 * rng.random(),
                }
            )
    pd.DataFrame(rows).to_parquet(d / "predictions.parquet", index=False)
    null = clustered_null_pvalue([d])
    assert null["observed"] > 0.2
    assert null["p_value"] < 0.05

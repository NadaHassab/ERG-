"""Phase 0: legacy freeze regression tests.

Guards the frozen legacy pipeline (models/legacy_baselines.py) and the
versioned-output contract:

- legacy:true configs dispatch to the frozen snapshot;
- outputs land in <results>/<cfg.output_subdir>/, never overwriting the
  original artifacts/results/baselines/ directory;
- a minimal legacy run is numerically deterministic (same seed -> same
  out-of-fold predictions).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pathway_erg.config import BaselinesConfig, load_config
from pathway_erg.models.baselines import run_baselines, write_baselines_artifacts


def _tiny_config(tmp_path: Path, *, legacy: bool, output_subdir: str) -> BaselinesConfig:
    return BaselinesConfig(
        name="phase0_test",
        fold_version="v1",
        datasets=("LEOP",),
        e0_methods=("prevalence",),
        e4_methods=("clinical",),
        demographic_methods=(),
        models=("logreg",),
        outer_folds=(0,),
        n_bootstrap_reps=100,
        bootstrap_seed=424242,
        confidence=0.95,
        pca_variance=0.95,
        max_pca_components=32,
        seed=777,
        legacy=legacy,
        output_subdir=output_subdir,
    )


@pytest.fixture(scope="module")
def data_config():
    from pathway_erg.config import DataConfig, load_config

    return load_config(DataConfig, "configs/data/local.yaml")


def test_legacy_config_loads_with_legacy_flag():
    cfg = load_config(BaselinesConfig, "configs/experiments/e4_baselines_legacy.yaml")
    assert cfg.legacy is True
    assert cfg.output_subdir == "baselines_legacy_v1"
    active = load_config(BaselinesConfig, "configs/experiments/e4_baselines.yaml")
    assert active.legacy is False
    assert active.output_subdir == "baselines_v2"


def test_legacy_dispatch_is_deterministic(tmp_path, data_config):
    cfg = _tiny_config(tmp_path, legacy=True, output_subdir="phase0_legacy")
    results = run_baselines(cfg, data_config)
    preds = results.predictions
    assert len(preds) > 0
    assert preds["task"].eq("LEOP").all()
    assert preds["method"].nunique() == 4  # prevalence + 3 clinical models
    # rerun: identical out-of-fold predictions (determinism)
    again = run_baselines(cfg, data_config)
    pd.testing.assert_frame_equal(preds, again.predictions)


def test_legacy_writes_to_versioned_subdir(tmp_path, data_config):
    artifact_root = tmp_path / "artifacts"
    cfg = _tiny_config(tmp_path, legacy=True, output_subdir="phase0_legacy")
    results = run_baselines(cfg, data_config)
    report = write_baselines_artifacts(artifact_root, results, cfg)
    out = Path(report)
    assert str(artifact_root / "results" / "phase0_legacy") in str(out)
    assert (artifact_root / "results" / "phase0_legacy" / "predictions.parquet").is_file()
    assert (artifact_root / "results" / "phase0_legacy" / "metrics.json").is_file()
    assert (artifact_root / "results" / "phase0_legacy" / "manifest.json").is_file()
    metrics = json.loads(
        (artifact_root / "results" / "phase0_legacy" / "metrics.json").read_text()
    )
    assert "LEOP/prevalence" in metrics
    assert "LEOP/clinical_logreg" in metrics


def test_original_baseline_dir_not_touched(tmp_path, data_config):
    """The writer must never write into the original results/baselines dir."""
    artifact_root = tmp_path / "artifacts"
    cfg = _tiny_config(tmp_path, legacy=True, output_subdir="phase0_legacy")
    results = run_baselines(cfg, data_config)
    write_baselines_artifacts(artifact_root, results, cfg)
    original = artifact_root / "results" / "baselines"
    assert not original.exists() or not any(original.iterdir())


def test_legacy_snapshot_import_is_self_contained():
    """legacy_baselines.py must not import from the (changing) baselines.py."""
    import inspect

    import pathway_erg.models.legacy_baselines as legacy

    src = inspect.getsource(legacy)
    assert "from .baselines import" not in src
    assert "from . import baselines" not in src


def test_legacy_predictions_match_recorded_reference(tmp_path, data_config):
    """Phase-0 acceptance: legacy is numerically frozen.

    Fixture recorded 2026-08-02 from the frozen snapshot on a minimal config
    (LEOP, fold 0 only, e0=prevalence, e4=clinical, fixed seeds). The legacy
    module always runs its three model variants, so all four keys must
    reproduce bit-for-bit (the pipeline is deterministic and the module is
    frozen).
    """
    from pathway_erg.models.legacy_baselines import run_baselines as run_legacy

    cfg = _tiny_config(tmp_path, legacy=True, output_subdir="phase0_reference")
    results = run_legacy(cfg, data_config)
    expected = {
        "LEOP/clinical_histgb": (0.6656249999999999, 0.596875),
        "LEOP/clinical_logreg": (0.6458333333333334, 0.5770833333333334),
        "LEOP/clinical_svm_rbf": (0.6333333333333333, 0.5177083333333333),
        "LEOP/prevalence": (0.5, 0.5),
    }
    assert set(results.metrics) == set(expected)
    for method, (auc, bal_acc) in expected.items():
        m = results.metrics[method]
        assert abs(m["roc_auc"] - auc) < 1e-9
        assert abs(m["balanced_accuracy"] - bal_acc) < 1e-9
        assert m["n_total"] == 47

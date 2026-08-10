"""Probe battery tests (plan E12, item 22)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.data.datasets import ComponentDataset, LoadedCaches
from pathway_erg.evaluation.probes import (
    ProbeResult,
    encode_component_frame,
    evaluate_probe,
    load_model_from_checkpoint,
    probe_targets,
    run_probe_battery,
    save_probe_report,
)
from pathway_erg.models.path_erg import build_model
from pathway_erg.training.separate import (
    SeparateTrainingConfig,
    build_stage_model,
)


@pytest.fixture(scope="module")
def caches():
    return LoadedCaches("artifacts")


def test_probe_targets_have_expected_shape_and_values(caches):
    rows = list(ComponentDataset(caches))[:20]
    targets = probe_targets(rows)
    assert targets["component_identity"].shape == (20,)
    assert set(np.unique(targets["component_identity"]).tolist()) <= {
        float(i) for i in range(6)
    }
    assert set(np.unique(targets["dataset_identity"]).tolist()) <= {0.0, 1.0}
    assert targets["peak_to_peak"].shape == (20,)
    assert np.isfinite(targets["flash_intensity"]).sum() <= 20


def test_encode_component_frame_stream_shapes(caches):
    ds = ComponentDataset(caches, dataset="LEOP", outer_folds={0, 1, 2})
    model = build_model()
    frame = encode_component_frame(model, list(ds)[:16], batch_size=8)
    assert frame.n() == 16
    assert frame.streams["fused"].shape == (16, 128)
    assert set(frame.streams) == {"fused"}
    assert len(frame.unit_ids) == 16


def test_encode_component_frame_routed_streams(caches):
    ds = ComponentDataset(caches, dataset="PERG", outer_folds={0, 1, 2})
    model = build_stage_model(
        SeparateTrainingConfig(name="t", routing_graph="correct"), seed=7
    )
    frame = encode_component_frame(model, list(ds)[:8], batch_size=8)
    assert set(frame.streams) == {"fused", "shared", "private"}
    assert frame.streams["shared"].shape == (8, 64)
    assert frame.streams["private"].shape == (8, 64)


def test_evaluate_probe_class_binary():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] > 0).astype(int)
    train, test = slice(0, 150), slice(150, 300)
    clusters = np.repeat(np.arange(150), 1).astype(str)
    result = evaluate_probe(
        X[train], y[train], X[test], y[test],
        kind="class", clusters=clusters, n_reps=30, seed=1,
    )
    assert result.metric == "roc_auc"
    assert 0.0 <= result.value <= 1.0
    assert result.value > 0.9
    assert result.ci_low is not None and result.ci_low <= result.value
    assert result.ci_high is not None and result.ci_high >= result.value


def test_evaluate_probe_class_multiclass():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 3, size=600)
    train, test = slice(0, 400), slice(400, 600)
    clusters = np.arange(200).astype(str)
    result = evaluate_probe(
        X[train], y[train], X[test], y[test],
        kind="class", clusters=clusters, n_reps=20, seed=2,
    )
    assert result.metric == "macro_ovr_auroc"
    assert 0.0 <= result.value <= 1.0


def test_evaluate_probe_regression():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 8))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=400)
    train, test = slice(0, 200), slice(200, 400)
    clusters = np.arange(200).astype(str)
    result = evaluate_probe(
        X[train], y[train], X[test], y[test],
        kind="reg", clusters=clusters, n_reps=20, seed=4,
    )
    assert result.metric == "pearson_r"
    assert result.value > 0.95
    assert result.n_train == 200 and result.n_eval == 200


def test_run_probe_battery_fold_safety(caches):
    model = build_stage_model(SeparateTrainingConfig(name="t"), seed=7)
    results = run_probe_battery(
        model, caches, test_fold=0, outer_folds=(0, 1), n_reps=10, seed=1,
        batch_size=64,
    )
    assert results
    for r in results:
        assert r.frame in {"all", "LEOP", "PERG"}
        assert r.stream == "fused"
        assert r.n_eval >= 2
        assert 0.0 <= r.value <= 1.0 if r.kind == "class" else np.isfinite(r.value)


def test_run_probe_battery_rejects_test_fold_outside_outer_folds(caches):
    model = build_stage_model(SeparateTrainingConfig(name="t"), seed=7)
    with pytest.raises(ValueError):
        run_probe_battery(model, caches, test_fold=9, outer_folds=(0, 1), n_reps=2)


def test_load_model_from_checkpoint_roundtrip(caches, tmp_path):
    cfg = SeparateTrainingConfig(name="ckpt", routing_graph="correct")
    model = build_stage_model(cfg, seed=7)
    path = tmp_path / "final.pt"
    torch.save(
        {"model": model.state_dict(), "experiment": cfg.__dict__, "seed": 7},
        path,
    )
    loaded, loaded_cfg, seed = load_model_from_checkpoint(path)
    assert seed == 7
    assert loaded_cfg.routing_graph == "correct"
    assert loaded.router is not None
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_save_probe_report_writes_parquet_and_json(tmp_path):
    results = [
        ProbeResult(
            frame="all", stream="fused", target="component_identity",
            kind="class", metric="macro_ovr_auroc", value=0.8,
            ci_low=0.7, ci_high=0.9, n_train=100, n_eval=20, n_units_eval=10,
        )
    ]
    out = save_probe_report(results, tmp_path / "probe_battery_fold0.parquet")
    assert out.is_file()
    assert out.with_suffix(".json").is_file()
    import json

    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["n_probes"] == 1

"""Plan §11.4 FLINDERS routed-token calibration tests (headless, synthetic)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pathway_erg.evaluation.flinders_routed import (
    FlindersRoutedConfig,
    _median_dim_ks,
    _subject_mean_tokens,
    load_ssl_checkpoint,
    run_flinders_routed_calibration,
)
from pathway_erg.models.path_erg import ModelConfig, build_model
from pathway_erg.training.ssl import SSLConfig

from tests._ext_synth import (
    build_synthetic_external,
    build_synthetic_external_splits,
    build_synthetic_v1_splits,
    build_synthetic_v4,
    external_fold_config,
    pre_cfg,
    write_synthetic_tables,
)


def _ssl_cfg() -> SSLConfig:
    return SSLConfig(
        name="synthetic_ssl",
        fold_version="v1",
        routing_graph="correct",
        outer_folds=(0, 1, 2),
        exclude_fold=0,
        leop_batch=4,
        perg_batch=4,
        epochs=1,
        lr=0.0001,
        weight_decay=0.05,
        warmup_epochs=0,
        grad_clip=1.0,
        seed=1001,
        device="cpu",
        mask_len=8,
        ssl_dim=32,
        lambda_mask=1.0,
        lambda_view=0.25,
        lambda_aug=0.25,
        lambda_geom=0.1,
        lambda_prior=0.01,
        gate_prior=0.75,
        output_subdir="ssl_pretrain_external_v1",
        log_every=10,
        ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"),
        domain_batches={"LEOP": 4, "PERG": 4, "URFU": 4, "FLINDERS": 4},
        plan_per_epoch=True,
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )


def _write_checkpoint(path: Path, cfg: SSLConfig, fold: int) -> None:
    model = build_model(
        ModelConfig(
            routing_graph=cfg.routing_graph,
            stems_seed=cfg.seed,
            agg_seed=cfg.seed,
            head_seed=cfg.seed,
        )
    )
    payload = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "exclude_fold": fold,
        "train_folds": [f for f in cfg.outer_folds if f != fold],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    write_synthetic_tables(root)
    pre = pre_cfg()
    build_synthetic_v4(root, pre)
    build_synthetic_external(root, pre)
    build_synthetic_v1_splits(root)
    build_synthetic_external_splits(root, external_fold_config())
    return root


def test_load_ssl_checkpoint_validates_fold(tmp_path):
    checkpoint = tmp_path / "final.pt"
    _write_checkpoint(checkpoint, _ssl_cfg(), 0)
    model, cfg = load_ssl_checkpoint(checkpoint, 0)
    assert cfg.exclude_fold == 0
    assert "FLINDERS" in cfg.ssl_datasets
    assert model.router is not None
    with pytest.raises(ValueError, match="fold mismatch"):
        load_ssl_checkpoint(checkpoint, 1)


def test_load_ssl_checkpoint_requires_external_binding(tmp_path):
    cfg = _ssl_cfg()
    cfg = SSLConfig(
        **{**asdict(cfg), "external_bindings": (), "external_fold_version": None}
    )
    checkpoint = tmp_path / "final.pt"
    _write_checkpoint(checkpoint, cfg, 0)
    with pytest.raises(ValueError, match="external binding"):
        load_ssl_checkpoint(checkpoint, 0)


def test_load_ssl_checkpoint_forbids_flinders_head(tmp_path):
    cfg = _ssl_cfg()
    checkpoint = tmp_path / "final.pt"
    _write_checkpoint(checkpoint, cfg, 0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["heads.FLINDERS.net.0.weight"] = payload["model"]["heads.LEOP.net.0.weight"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="forbidden FLINDERS head"):
        load_ssl_checkpoint(checkpoint, 0)


def test_flinders_routed_calibration_writes_report(tmp_path):
    root = _synthetic_root(tmp_path)
    checkpoint = tmp_path / "ckpts" / "fold0" / "final.pt"
    _write_checkpoint(checkpoint, _ssl_cfg(), 0)
    config = FlindersRoutedConfig(
        output_subdir="external_v1/flinders_calibration",
        n_boot_reps=10,
        seed=7,
        protocols=("LA3", "9_step"),
    )
    report = run_flinders_routed_calibration(root, checkpoint, 0, config, device="cpu")

    assert report["exclude_fold"] == 0
    assert report["n_flinders_components"] > 0
    assert "fused" in report["per_stream"]
    out_dir = root / "results" / config.output_subdir
    assert (out_dir / "calibration_report.json").is_file()
    assert (out_dir / "COMPLETE").is_file()
    tokens = pd.read_parquet(out_dir / "routed_tokens.parquet")
    assert set(tokens["dataset"].unique()) >= {"FLINDERS", "LEOP"}
    assert "token" in tokens.columns
    for rows in report["per_stream"].values():
        if isinstance(rows, list):
            for row in rows:
                assert row["ks_median_dim"] >= 0.0
                assert row["ci_low"] <= row["ci_high"]
                assert row["n_flinders_subjects"] >= 1


def test_median_dim_ks_identical_and_distinct():
    x = np.zeros((10, 4), dtype=np.float64)
    assert _median_dim_ks(x, x) == 0.0
    y = np.concatenate([np.full((10, 2), 10.0), x[:, 2:]], axis=1)
    assert _median_dim_ks(x, y) > 0.0


def test_subject_mean_tokens_aggregates_per_subject_protocol():
    tokens = pd.DataFrame(
        {
            "subject_id": ["f0", "f0", "f1"],
            "protocol": ["LA3", "LA3", "LA3"],
            "stream": ["fused", "fused", "fused"],
            "token": [
                np.ones(2, dtype=np.float32),
                np.full(2, 3.0, dtype=np.float32),
                np.full(2, 5.0, dtype=np.float32),
            ],
        }
    )
    means = _subject_mean_tokens(tokens)
    assert "fused" in means
    frame = means["fused"]
    assert len(frame) == 2
    assert np.allclose(frame.loc[frame["subject_id"].eq("f0"), "token"].iloc[0], 2.0)
    assert np.allclose(frame.loc[frame["subject_id"].eq("f1"), "token"].iloc[0], 5.0)

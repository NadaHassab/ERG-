"""Separate-training runner tests (plan §35 item 18)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from pathway_erg.config import DataConfig, LeopsDataConfig, PergDataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.calibration import _binary_cross_entropy, _ce_gradient
from pathway_erg.models.path_erg import build_model
from pathway_erg.training.separate import (
    SeparateTrainingConfig,
    build_stage_model,
    build_task_bags,
    inner_partition,
    load_inner_map,
    outer_partition,
    predict_bags,
    run_outer_seed,
)
from pathway_erg.training.trainer import TrainConfig, Trainer, TrainLog


@pytest.fixture(scope="module")
def caches():
    return LoadedCaches("artifacts")


def test_primary_leop_bags_are_nine_step_only(caches):
    bags = build_task_bags(caches, "LEOP", "primary_nine_step")
    assert len(bags) == 160
    assert all(b.target_binary in {0, 1} for b in bags)
    assert {c.protocol for b in bags for c in b.components} == {"9_step"}


def test_build_stage_model_respects_routing_graph():
    plain = build_stage_model(SeparateTrainingConfig(name="t"), seed=7)
    assert plain.router is None
    routed = build_stage_model(
        SeparateTrainingConfig(name="t", routing_graph="correct"), seed=7
    )
    assert routed.router is not None
    assert routed.router.graph.name == "correct"
    wrong = build_stage_model(
        SeparateTrainingConfig(name="t", routing_graph="wrong"), seed=7
    )
    counts = lambda m: sum(p.numel() for p in m.parameters())
    assert counts(plain) < counts(routed)  # router adds parameters
    assert counts(routed) == counts(wrong)  # controls share parameter counts
    with pytest.raises(ValueError):
        build_stage_model(SeparateTrainingConfig(name="t", routing_graph="bogus"), seed=7)


def test_perg_bags_preserve_subject_and_visit_ids(caches):
    bags = build_task_bags(caches, "PERG")
    assert len(bags) == 336
    assert len({b.subject_id for b in bags}) == 304
    assert all(b.visit_id == b.unit_id for b in bags)


@pytest.mark.parametrize("task", ["LEOP", "PERG"])
def test_nested_partitions_cover_once_without_subject_leakage(caches, task):
    bags = build_task_bags(caches, task)
    outer_train, outer_test = outer_partition(bags, 0)
    assert not ({b.subject_id for b in outer_train} & {b.subject_id for b in outer_test})
    inner_map = load_inner_map("artifacts", "v1", task, 0)
    validation_units: list[str] = []
    for fold in sorted(set(inner_map.values())):
        train, valid = inner_partition(outer_train, inner_map, fold)
        assert not ({b.subject_id for b in train} & {b.subject_id for b in valid})
        validation_units.extend(b.unit_id for b in valid)
    assert sorted(validation_units) == sorted(b.unit_id for b in outer_train)


def test_trainer_rejects_subject_overlap(caches):
    bags = build_task_bags(caches, "PERG")
    repeated_subject = next(
        s for s in {b.subject_id for b in bags}
        if sum(b.subject_id == s for b in bags) > 1
    )
    visits = [b for b in bags if b.subject_id == repeated_subject]
    trainer = Trainer(build_model(), TrainConfig(task="PERG", epochs=1))
    with pytest.raises(ValueError, match="subject leakage"):
        trainer.fit([visits[0]], [visits[1]])


def test_explicit_partition_one_step_smoke(caches):
    bags = build_task_bags(caches, "PERG")
    outer_train, _ = outer_partition(bags, 0)
    inner_map = load_inner_map("artifacts", "v1", "PERG", 0)
    train, valid = inner_partition(outer_train, inner_map, 0)

    def balanced(items, n=2):
        return (
            [b for b in items if b.target_binary == 0][:n]
            + [b for b in items if b.target_binary == 1][:n]
        )

    model = build_model()
    log = Trainer(
        model,
        TrainConfig(
            task="PERG",
            outer_fold=0,
            epochs=1,
            batch_size=4,
            max_steps_per_epoch=1,
            seed=7,
        ),
    ).fit(balanced(train), balanced(valid))
    assert len(log.train_loss) == 1
    assert np.isfinite(log.train_loss[0])
    assert log.best_epoch == 0


def test_predict_bags_is_one_row_per_unit(caches):
    bags = build_task_bags(caches, "PERG")[:4]
    pred = predict_bags(build_model().eval(), bags, "PERG", batch_size=2)
    assert pred["unit_id"].is_unique
    assert pred["subject_id"].notna().all()
    assert len(pred) == 4


def test_temperature_gradient_matches_finite_difference():
    logits = np.array([-2.0, -0.5, 1.0, 3.0])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    t = 1.3
    eps = 1e-5
    finite = (
        _binary_cross_entropy(t + eps, logits, labels)
        - _binary_cross_entropy(t - eps, logits, labels)
    ) / (2 * eps)
    assert _ce_gradient(t, logits, labels) == pytest.approx(finite, abs=1e-6)


def test_outer_seed_writes_staged_checkpoints(caches, tmp_path, monkeypatch):
    split_dir = tmp_path / "data" / "splits"
    split_dir.mkdir(parents=True)
    for name in ("outer_folds_v1.parquet", "inner_folds_v1.parquet"):
        shutil.copy(Path("artifacts/data/splits") / name, split_dir / name)
    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    shutil.copy(
        "artifacts/data/manifests/component_cache_manifest_v4.json",
        manifest_dir / "component_cache_manifest_v4.json",
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch, task):
            return self.bias.expand(len(batch["label"]))

    class FakeTrainer:
        def __init__(self, model, config):
            self.model = model

        def fit(self, train_bags, val_bags):
            return TrainLog(
                train_loss=[0.69], val_auc=[0.5], best_epoch=0, best_val_auc=0.5
            )

    import pathway_erg.training.separate as separate

    monkeypatch.setattr(separate, "build_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(separate, "Trainer", FakeTrainer)
    data_cfg = DataConfig(
        leops=LeopsDataConfig(json_root="unused", xlsx_path="unused"),
        perg=PergDataConfig(root="unused", metadata_csv="unused"),
        artifact_root=str(tmp_path),
    )
    cfg = SeparateTrainingConfig(
        name="test",
        tasks=("LEOP",),
        outer_folds=(0,),
        seeds=(7,),
        epochs=1,
        output_subdir="checkpoint_test",
        n_bootstrap_reps=100,
    )
    bags = build_task_bags(caches, "LEOP")
    pred, run_dir = run_outer_seed(cfg, data_cfg, caches, "LEOP", bags, 0, 7)
    assert (run_dir / "COMPLETE").is_file()
    assert (run_dir / "final.pt").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert len(list(run_dir.glob("inner_fold_*.pt"))) == 4
    assert pred["unit_id"].is_unique
    assert len(pred) == sum(b.outer_fold == 0 for b in bags)

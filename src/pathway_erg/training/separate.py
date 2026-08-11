"""Leakage-safe separate-training neural baseline (plan §35 item 18).

Method name: ``separate_raw_ot_hierarchical_v1``.  This is intentionally
narrower than the future complete PATH model: fresh raw/OT hierarchical
models are trained independently for LEOP and PERG, with no cross-domain
pretraining or shared checkpoint.

For every task / outer fold / seed:

1. hold out ``outer_fold == f``;
2. train four fresh inner models on ``inner_fold != j`` and predict only
   ``inner_fold == j``;
3. fit temperature calibration on the concatenated inner OOF logits;
4. refit a fresh model on all outer-train bags for the selected epoch count;
5. predict outer test once and write a complete run checkpoint.

The ``COMPLETE`` marker is written last.  Interrupted runs therefore never
look complete, while each finished inner fold has its own checkpoint and OOF
prediction table for recovery/audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import DataConfig
from ..constants import INNER_FOLDS_TEMPLATE, OUTER_FOLDS_TEMPLATE
from ..data.collate import collate_bag_units
from ..data.datasets import BagUnit, LoadedCaches, build_bags
from ..data.urfu_labels import require_urfu_labels_signed_off
from ..evaluation.calibration import fit_calibrator
from ..evaluation.metrics import binary_metrics, cluster_bootstrap_ci
from ..models.path_erg import ModelConfig, build_model
from ..provenance import RunManifest, git_revision, sha256_file, sha256_text
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths
from .trainer import TrainConfig, Trainer, TrainLog

METHOD = "separate_raw_ot_hierarchical_v1"


@dataclass(frozen=True)
class SeparateTrainingConfig:
    """Typed experiment config for the separate neural baseline."""

    name: str
    method: str = METHOD
    fold_version: str = "v1"
    tasks: tuple[str, ...] = ("LEOP", "PERG")
    leop_cohort: str = "primary_nine_step"
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (1001, 2002, 3003)
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 25
    grad_clip: float = 1.0
    device: str = "cpu"
    max_steps_per_epoch: int | None = None
    output_subdir: str = "separate_raw_ot_hierarchical_v1"
    n_bootstrap_reps: int = 2000
    bootstrap_seed: int = 424242
    confidence: float = 0.95
    init_ssl: str | None = None
    routing_graph: str | None = None
    label_frac: float = 1.0
    subset_seed: int = 9001
    resume: bool = False
    external_bindings: tuple[str, ...] = ()
    external_fold_version: str | None = None


@dataclass
class SeparateResults:
    predictions_by_seed: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float | int | None]]
    run_dirs: list[str] = field(default_factory=list)


def build_task_bags(
    caches: LoadedCaches,
    task: str,
    leop_cohort: str = "primary_nine_step",
) -> list[BagUnit]:
    """Labeled bags with cohort filtering applied before bag construction."""
    if task == "FLINDERS":
        raise ValueError(
            "the FLINDERS supervised head is forbidden (no labels exist; "
            "plan integration §11.2)"
        )
    if task not in {"LEOP", "PERG", "URFU"}:
        raise ValueError(f"unknown task {task!r}")
    allowed: set[str] | None = None
    if task == "LEOP":
        if leop_cohort == "primary_nine_step":
            rec = caches.recordings
            allowed = set(
                rec.loc[
                    (rec["dataset"] == "LEOP") & (rec["protocol"] == "9_step"),
                    "global_recording_id",
                ].astype(str)
            )
        elif leop_cohort != "secondary_all_protocols":
            raise ValueError(f"unknown LEOP cohort {leop_cohort!r}")
    elif task == "URFU":
        require_urfu_labels_signed_off()
    bags = build_bags(caches, task, allowed_recording_ids=allowed)
    return [b for b in bags if b.target_binary is not None]


def stratified_subset(
    bags: list[BagUnit], fraction: float, seed: int
) -> list[BagUnit]:
    """Grouped stratified subset of training units (plan E9).

    Samples whole subjects so every PERG visit of a chosen subject stays
    together and the frozen inner partition remains nested.  Strata are
    class-balanced: each target class is sampled at ``fraction`` of its
    subjects.  ``fraction == 1.0`` returns the input unchanged and draws
    no randomness.  Deterministic for a fixed seed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"label fraction must be in (0, 1], got {fraction!r}")
    if fraction == 1.0:
        return list(bags)
    subjects: dict[str, list[BagUnit]] = {}
    for bag in bags:
        subjects.setdefault(bag.subject_id, []).append(bag)
    labels: dict[str, int] = {}
    for sid, unit_bags in subjects.items():
        values = {b.target_binary for b in unit_bags}
        if len(values) != 1:
            raise ValueError(f"subject {sid!r} has inconsistent targets {values}")
        labels[sid] = values.pop()
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for label in sorted(set(labels.values())):
        pool = sorted(s for s, v in labels.items() if v == label)
        n = max(1, int(round(fraction * len(pool))))
        chosen.extend(rng.choice(pool, size=n, replace=False).tolist())
    out = [b for b in bags if b.subject_id in set(chosen)]
    if not out:
        raise ValueError("empty stratified subset")
    return out


def outer_partition(
    bags: list[BagUnit], outer_fold: int
) -> tuple[list[BagUnit], list[BagUnit]]:
    """Outer train/test partition; the test subjects are never optimized."""
    train = [b for b in bags if b.outer_fold != outer_fold]
    test = [b for b in bags if b.outer_fold == outer_fold]
    if not train or not test:
        raise ValueError(f"empty outer partition for fold {outer_fold}")
    _assert_subject_disjoint(train, test, "outer train/test")
    return train, test


def load_inner_map(
    artifact_root: str | Path,
    fold_version: str,
    task: str,
    outer_fold: int,
) -> dict[str, int]:
    """Subject -> inner fold for one outer-training partition."""
    path = (
        Path(artifact_root)
        / "data"
        / "splits"
        / INNER_FOLDS_TEMPLATE.format(version=fold_version)
    )
    inner = pd.read_parquet(path)
    rows = inner[
        (inner["dataset"] == task) & (inner["outer_fold_sel"] == outer_fold)
    ]
    if rows.empty:
        raise ValueError(f"no inner assignments for {task} outer fold {outer_fold}")
    if rows["unit_id"].duplicated().any():
        raise ValueError("duplicate subject in inner assignments")
    return {
        str(r.unit_id): int(r.inner_fold)
        for r in rows.itertuples(index=False)
    }


def inner_partition(
    outer_train: list[BagUnit], inner_map: dict[str, int], inner_fold: int
) -> tuple[list[BagUnit], list[BagUnit]]:
    """Inner train/validation split grouped by canonical subject."""
    missing = {b.subject_id for b in outer_train} - set(inner_map)
    if missing:
        raise ValueError(f"outer-train subjects missing inner assignment: {sorted(missing)[:5]}")
    train = [b for b in outer_train if inner_map[b.subject_id] != inner_fold]
    valid = [b for b in outer_train if inner_map[b.subject_id] == inner_fold]
    if not train or not valid:
        raise ValueError(f"empty inner partition for fold {inner_fold}")
    _assert_subject_disjoint(train, valid, "inner train/validation")
    return train, valid


def predict_bags(
    model: torch.nn.Module,
    bags: list[BagUnit],
    task: str,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Exactly one logit per supervised bag/unit."""
    model.eval()
    rows: list[dict[str, object]] = []
    for start in range(0, len(bags), batch_size):
        chunk = bags[start : start + batch_size]
        batch = collate_bag_units(chunk)
        with torch.no_grad():
            logits = model(batch, task).detach().cpu().numpy()
        for bag, logit in zip(chunk, logits, strict=True):
            rows.append(
                {
                    "unit_id": bag.unit_id,
                    "subject_id": bag.subject_id,
                    "visit_id": bag.visit_id,
                    "target": int(bag.target_binary),
                    "logit": float(logit),
                    "probability": float(1.0 / (1.0 + np.exp(-logit))),
                }
            )
    out = pd.DataFrame(rows)
    if out["unit_id"].duplicated().any():
        raise ValueError("duplicate prediction for a supervised unit")
    return out


def build_stage_model(
    cfg: SeparateTrainingConfig, seed: int
) -> torch.nn.Module:
    """Fresh model, optionally routed and initialized from joint SSL."""
    from .finetune import freeze_encoders, init_from_ssl

    model = build_model(
        ModelConfig(
            stems_seed=seed,
            agg_seed=seed,
            head_seed=seed,
            routing_graph=cfg.routing_graph,
        )
    )
    if cfg.init_ssl:
        init_from_ssl(model, cfg.init_ssl)
        freeze_encoders(model, freeze=True)
    return model


def run_dir_for(
    data_root: str | Path,
    cfg: SeparateTrainingConfig,
    task: str,
    outer_fold: int,
    seed: int,
) -> Path:
    """Artifact dir for one task/fold/seed run (mirrors run_outer_seed)."""
    run_id = f"{cfg.method}-{task.lower()}-fold{outer_fold}-seed{seed}"
    return (
        Path(data_root) / "results" / cfg.output_subdir / "runs" / run_id
    )


def run_outer_seed(
    cfg: SeparateTrainingConfig,
    data_cfg: DataConfig,
    caches: LoadedCaches,
    task: str,
    bags: list[BagUnit],
    outer_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, Path]:
    """Run one task/outer-fold/seed and write staged checkpoints."""
    outer_train, outer_test = outer_partition(bags, outer_fold)
    if cfg.label_frac < 1.0:
        outer_train = stratified_subset(
            outer_train, cfg.label_frac, cfg.subset_seed
        )
    inner_map = load_inner_map(
        data_cfg.artifact_root, cfg.fold_version, task, outer_fold
    )
    inner_folds = sorted(set(inner_map.values()))
    covered: list[str] = []
    oof: list[pd.DataFrame] = []
    best_epochs: list[int] = []

    run_dir = run_dir_for(data_cfg.artifact_root, cfg, task, outer_fold, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError(f"completed run already exists: {run_dir}")

    for inner_fold in inner_folds:
        train_bags, valid_bags = inner_partition(
            outer_train, inner_map, inner_fold
        )
        model = build_stage_model(cfg, seed)
        trainer = Trainer(
            model,
            _train_config(cfg, task, outer_fold, seed),
        )
        log = trainer.fit(train_bags, valid_bags)
        pred = predict_bags(model, valid_bags, task, cfg.batch_size)
        pred["inner_fold"] = inner_fold
        oof.append(pred)
        covered.extend(pred["subject_id"].astype(str))
        best_epochs.append((log.best_epoch or 0) + 1)
        _write_checkpoint(
            run_dir / f"inner_fold_{inner_fold}.pt",
            model,
            cfg,
            task,
            outer_fold,
            seed,
            log,
            train_bags,
            valid_bags,
        )
        pred.to_parquet(run_dir / f"inner_oof_fold_{inner_fold}.parquet", index=False)

    expected = sorted({b.subject_id for b in outer_train})
    if sorted(covered) != sorted(
        [b.subject_id for b in outer_train]
    ):
        raise ValueError("inner OOF coverage is not exactly one row per outer-train bag")
    if sorted(set(covered)) != expected:
        raise ValueError("inner OOF subject coverage mismatch")

    inner_oof = pd.concat(oof, ignore_index=True)
    calibrator = fit_calibrator(
        inner_oof["logit"].to_numpy(), inner_oof["target"].to_numpy()
    )
    inner_oof.to_parquet(run_dir / "inner_oof.parquet", index=False)
    (run_dir / "calibrator.json").write_text(
        json.dumps(asdict(calibrator), indent=2, sort_keys=True)
    )

    selected_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final_model = build_stage_model(cfg, seed)
    final_cfg = _train_config(cfg, task, outer_fold, seed)
    final_cfg.epochs = selected_epochs
    final_cfg.patience = selected_epochs + 1
    final_log = Trainer(final_model, final_cfg).fit(outer_train, [])
    pred = predict_bags(final_model, outer_test, task, cfg.batch_size)
    pred["calibrated_probability"] = calibrator.apply(pred["logit"].to_numpy())
    pred["method"] = cfg.method
    pred["task"] = task
    pred["cohort"] = cfg.leop_cohort if task == "LEOP" else "all_visits"
    pred["label_frac"] = cfg.label_frac
    pred["outer_fold"] = outer_fold
    pred["seed"] = seed
    pred["checkpoint"] = str(run_dir / "final.pt")
    pred["note"] = (
        "fresh single-task model; no shared pretraining"
        if not cfg.init_ssl
        else f"init from joint SSL {cfg.init_ssl}"
    )
    if cfg.routing_graph:
        pred["note"] += f"; routing graph={cfg.routing_graph}"
    if cfg.label_frac < 1.0:
        pred["note"] += f"; label_frac={cfg.label_frac}"
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_checkpoint(
        run_dir / "final.pt",
        final_model,
        cfg,
        task,
        outer_fold,
        seed,
        final_log,
        outer_train,
        outer_test,
    )
    _write_run_manifest(cfg, data_cfg, run_dir, task, outer_fold, seed)
    complete.write_text("complete\n")
    return pred, run_dir


def run_separate_training(
    cfg: SeparateTrainingConfig, data_cfg: DataConfig
) -> SeparateResults:
    """Run all configured task/fold/seed separate baselines."""
    caches = LoadedCaches(
        data_cfg.artifact_root,
        fold_version=cfg.fold_version,
        external_bindings=cfg.external_bindings,
        external_fold_version=cfg.external_fold_version,
    )
    all_predictions: list[pd.DataFrame] = []
    run_dirs: list[str] = []
    for task in cfg.tasks:
        bags = build_task_bags(caches, task, cfg.leop_cohort)
        for outer_fold in cfg.outer_folds:
            for seed in cfg.seeds:
                run_dir = run_dir_for(
                    data_cfg.artifact_root, cfg, task, outer_fold, seed
                )
                if cfg.resume and run_dir.joinpath("COMPLETE").exists():
                    loaded = run_dir / "predictions.parquet"
                    if not loaded.exists():
                        raise FileNotFoundError(
                            f"resume: {run_dir} is COMPLETE but has no "
                            f"{loaded.name}; cannot rebuild the ensemble"
                        )
                    loaded_pred = pd.read_parquet(loaded)
                    if "label_frac" in loaded_pred.columns:
                        loaded_pred["label_frac"] = loaded_pred["label_frac"].fillna(
                            cfg.label_frac
                        )
                    else:
                        loaded_pred["label_frac"] = cfg.label_frac
                    all_predictions.append(loaded_pred)
                    run_dirs.append(str(run_dir))
                    continue
                pred, run_dir = run_outer_seed(
                    cfg, data_cfg, caches, task, bags, outer_fold, seed
                )
                all_predictions.append(pred)
                run_dirs.append(str(run_dir))
    if not all_predictions:
        raise ValueError("no runs to execute (resume found every run COMPLETE)")
    by_seed = pd.concat(all_predictions, ignore_index=True)
    predictions = _ensemble_predictions(by_seed)
    _assert_ensemble_uniqueness(predictions)
    metrics = _metrics_by_task(predictions, cfg)
    out_dir = Path(data_cfg.artifact_root) / "results" / cfg.output_subdir
    by_seed.to_parquet(out_dir / "predictions_by_seed.parquet", index=False)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    return SeparateResults(by_seed, predictions, metrics, run_dirs)


def _train_config(
    cfg: SeparateTrainingConfig, task: str, outer_fold: int, seed: int
) -> TrainConfig:
    return TrainConfig(
        task=task,
        outer_fold=outer_fold,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        grad_clip=cfg.grad_clip,
        patience=cfg.patience,
        seed=seed,
        device=cfg.device,
        max_steps_per_epoch=cfg.max_steps_per_epoch,
    )


def _assert_subject_disjoint(
    left: list[BagUnit], right: list[BagUnit], label: str
) -> None:
    overlap = {b.subject_id for b in left} & {b.subject_id for b in right}
    if overlap:
        raise ValueError(f"{label} subject leakage: {sorted(overlap)[:5]}")


def _write_checkpoint(
    path: Path,
    model: torch.nn.Module,
    cfg: SeparateTrainingConfig,
    task: str,
    outer_fold: int,
    seed: int,
    log: TrainLog,
    train_bags: list[BagUnit],
    eval_bags: list[BagUnit],
) -> None:
    payload = {
        "model": model.state_dict(),
        "experiment": asdict(cfg),
        "task": task,
        "outer_fold": outer_fold,
        "seed": seed,
        "log": asdict(log),
        "train_unit_ids": [b.unit_id for b in train_bags],
        "eval_unit_ids": [b.unit_id for b in eval_bags],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _write_run_manifest(
    cfg: SeparateTrainingConfig,
    data_cfg: DataConfig,
    run_dir: Path,
    task: str,
    outer_fold: int,
    seed: int,
) -> None:
    root = Path(data_cfg.artifact_root)
    outer = root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=cfg.fold_version)
    inner = root / "data" / "splits" / INNER_FOLDS_TEMPLATE.format(version=cfg.fold_version)
    manifest = RunManifest(kind="separate_neural", name=run_dir.name)
    manifest.config_hash = sha256_text(json.dumps(asdict(cfg), sort_keys=True))
    manifest.data_hash = sha256_file(
        cache_paths(root, CACHE_SCHEMA_VERSION)["manifest"]
    )
    manifest.split_hash = sha256_text(sha256_file(outer) + sha256_file(inner))
    manifest.code_revision = git_revision(Path.cwd())
    manifest.extra = {
        "method": cfg.method,
        "task": task,
        "outer_fold": outer_fold,
        "seed": seed,
        "checkpoint": "final.pt",
        "label_frac": cfg.label_frac,
    }
    manifest.write_atomic(run_dir / "run_manifest.json")


def _ensemble_predictions(by_seed: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method", "task", "cohort", "label_frac", "outer_fold",
        "unit_id", "subject_id", "visit_id", "target",
    ]
    grouped = by_seed.groupby(keys, dropna=False, as_index=False)
    out = grouped.agg(
        logit=("logit", "mean"),
        probability=("probability", "mean"),
        calibrated_probability=("calibrated_probability", "mean"),
    )
    out["seed"] = "ensemble"
    return out


def _assert_ensemble_uniqueness(predictions: pd.DataFrame) -> None:
    """Refuse to write metrics from an ensembled table with duplicate units.

    A split identifier (e.g. a NaN vs 1.0 ``label_frac``) can silently split
    one unit across two ensemble rows; this guard makes that loud.
    """
    dup = predictions.duplicated(subset=["task", "outer_fold", "unit_id"])
    if dup.any():
        raise ValueError(
            "ensemble produced duplicate unit rows (identifier mismatch "
            "across seeds?); refusing to write wrong metrics"
        )
    expect = predictions.groupby(["task", "outer_fold"])["unit_id"].nunique()
    total = predictions.groupby("task").size()
    if (total != expect.groupby("task").sum()).any():
        raise ValueError("ensemble unit count does not match fold sums")


def _metrics_by_task(
    predictions: pd.DataFrame, cfg: SeparateTrainingConfig
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for task, frame in predictions.groupby("task"):
        label_counts = frame.groupby("subject_id")["target"].nunique()
        if (label_counts > 1).any():
            raise ValueError(f"{task} has inconsistent labels within subject cluster")
        y = frame["target"].to_numpy(dtype=float)
        p = frame["calibrated_probability"].to_numpy(dtype=float)
        point = binary_metrics(y, p)
        ci = cluster_bootstrap_ci(
            y,
            p,
            frame["subject_id"].to_numpy(),
            metric="roc_auc",
            n_reps=cfg.n_bootstrap_reps,
            seed=cfg.bootstrap_seed,
            confidence=cfg.confidence,
        )
        point.update(
            {
                "roc_auc_ci_low": ci.ci_low,
                "roc_auc_ci_high": ci.ci_high,
                "n_clusters": ci.n_clusters,
            }
        )
        out[str(task)] = point
    return out

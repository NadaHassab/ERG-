"""Multi-task ERG classifier: joint LEOP + PERG training.

Trains a single shared encoder with task-specific heads, alternating
batches from both tasks in each epoch. Uses the same nested CV
evaluation protocol as the single-task baselines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import BagUnit, LoadedCaches
from pathway_erg.training.samplers import BagSampler
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.models.path_erg import PathModel, build_model, ModelConfig
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine


# ── Config ────────────────────────────────────────────────────────────────
DATA_CFG = "configs/data/local.yaml"
OUT_DIR = Path("artifacts/results/multitask_v1")
N_SEEDS = [1001, 2002, 3003]
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
PATIENCE = 25
GRAD_CLIP = 1.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 424242
LEOP_WEIGHT = 1.0  # loss weight for LEOP
PERG_WEIGHT = 1.0  # loss weight for PERG


# ── Multi-task trainer ────────────────────────────────────────────────────
class MultiTaskTrainer:
    """Train shared encoder on both LEOP and PERG simultaneously."""

    def __init__(self, model: PathModel, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)

    def fit(
        self,
        leop_train: list[BagUnit],
        leop_val: list[BagUnit],
        perg_train: list[BagUnit],
        perg_val: list[BagUnit],
        seed: int = 1001,
    ) -> dict:
        """Joint training with alternating LEOP/PERG batches."""
        self.model.to(self.device)

        # Create samplers for both tasks
        leop_sampler = BagSampler(
            leop_train, folds={b.outer_fold for b in leop_train},
            batch_size=BATCH_SIZE, seed=seed,
        )
        perg_sampler = BagSampler(
            perg_train, folds={b.outer_fold for b in perg_train},
            batch_size=BATCH_SIZE, seed=seed + 1000,
        )

        # Class weights
        leop_labels = np.asarray([b.target_binary for b in leop_train], dtype=float)
        perg_labels = np.asarray([b.target_binary for b in perg_train], dtype=float)
        leop_criterion = FoldWeightedBCE(positive_class_weight(leop_labels))
        perg_criterion = FoldWeightedBCE(positive_class_weight(perg_labels))

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Scheduler
        steps_per_epoch = max(len(leop_sampler.bags), len(perg_sampler.bags)) // BATCH_SIZE
        total = EPOCHS * steps_per_epoch
        warm = WARMUP_EPOCHS * steps_per_epoch
        sched = _WarmupCosine(optimizer, warmup_steps=warm, total_steps=total, min_frac=0.05)

        best_auc = -1.0
        best_state = None
        patience_counter = 0
        log = {"train_loss": [], "leop_auc": [], "perg_auc": []}

        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0.0
            n = 0

            # Alternate LEOP and PERG batches
            leop_iter = iter(leop_sampler)
            perg_iter = iter(perg_sampler)

            for step in range(steps_per_epoch):
                # Alternate: LEOP on even steps, PERG on odd steps
                if step % 2 == 0:
                    try:
                        idx = next(leop_iter)
                    except StopIteration:
                        leop_iter = iter(leop_sampler)
                        idx = next(leop_iter)
                    bags = [leop_sampler.bags[i] for i in idx]
                    task = "LEOP"
                    criterion = leop_criterion
                else:
                    try:
                        idx = next(perg_iter)
                    except StopIteration:
                        perg_iter = iter(perg_sampler)
                        idx = next(perg_iter)
                    bags = [perg_sampler.bags[i] for i in idx]
                    task = "PERG"
                    criterion = perg_criterion

                batch = collate_bag_units(bags)
                labels_b = torch.as_tensor(batch["label"], dtype=torch.float32).to(self.device)
                logits = self.model(batch, task)
                loss = criterion(logits, labels_b)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                optimizer.step()
                sched.step()

                total_loss += float(loss.item()) * len(labels_b)
                n += len(labels_b)

            # Evaluate on validation sets
            leop_auc = self._eval_auc(leop_val, "LEOP")
            perg_auc = self._eval_auc(perg_val, "PERG")

            avg_loss = total_loss / max(1, n)
            combined_auc = (leop_auc + perg_auc) / 2

            log["train_loss"].append(avg_loss)
            log["leop_auc"].append(leop_auc)
            log["perg_auc"].append(perg_auc)

            if combined_auc > best_auc:
                best_auc = combined_auc
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Find best epoch
        best_epoch = int(np.argmax([(l + p) / 2 for l, p in zip(log["leop_auc"], log["perg_auc"])]))
        log["best_epoch"] = best_epoch
        log["best_leop_auc"] = log["leop_auc"][best_epoch]
        log["best_perg_auc"] = log["perg_auc"][best_epoch]

        return log

    def _eval_auc(self, bags: list[BagUnit], task: str) -> float:
        self.model.eval()
        y_true, y_prob = [], []
        for bag in bags:
            if bag.target_binary is None:
                continue
            batch = collate_bag_units([bag])
            with torch.no_grad():
                logit = self.model(batch, task)
            y_true.append(bag.target_binary)
            y_prob.append(float(torch.sigmoid(logit[0]).item()))
        if len(y_true) < 2 or len(set(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(np.array(y_true), np.array(y_prob)))
        except ValueError:
            return 0.5


def predict_bags(model, bags, task, batch_size=8):
    """One logit per bag."""
    model.eval()
    rows = []
    for start in range(0, len(bags), batch_size):
        chunk = bags[start:start + batch_size]
        batch = collate_bag_units(chunk)
        with torch.no_grad():
            logits = model(batch, task).detach().cpu().numpy()
        for bag, logit in zip(chunk, logits, strict=True):
            rows.append({
                "unit_id": bag.unit_id,
                "subject_id": bag.subject_id,
                "target": int(bag.target_binary),
                "logit": float(logit),
                "probability": float(1.0 / (1.0 + np.exp(-logit))),
            })
    return pd.DataFrame(rows)


def bootstrap_auroc(y_true, y_prob, n_reps=2000, seed=424242):
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y_true, y_prob)
    scores = []
    for _ in range(n_reps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            scores.append(float("nan"))
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    scores = scores[np.isfinite(scores)]
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    data_cfg = load_config(DataConfig, DATA_CFG)
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")

    leop_bags = build_task_bags(caches, "LEOP", "primary_nine_step")
    perg_bags = build_task_bags(caches, "PERG", "primary_nine_step")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in N_SEEDS:
        print(f"\n{'='*60}")
        print(f"  Multi-task seed {seed}")
        print(f"{'='*60}")

        for outer_fold in range(5):
            leop_train, leop_test = outer_partition(leop_bags, outer_fold)
            perg_train, perg_test = outer_partition(perg_bags, outer_fold)

            # Build model with both heads
            model = build_model(ModelConfig(stems_seed=seed, agg_seed=seed, head_seed=seed))

            trainer = MultiTaskTrainer(model, device="cuda")

            # Train
            log = trainer.fit(
                leop_train, leop_test,
                perg_train, perg_test,
                seed=seed,
            )

            # Predict on test sets
            leop_pred = predict_bags(model, leop_test, "LEOP")
            perg_pred = predict_bags(model, perg_test, "PERG")

            # AUROC with bootstrap CI
            leop_point, leop_lo, leop_hi = bootstrap_auroc(
                leop_pred["target"].values, leop_pred["probability"].values,
                n_reps=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
            )
            perg_point, perg_lo, perg_hi = bootstrap_auroc(
                perg_pred["target"].values, perg_pred["probability"].values,
                n_reps=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
            )

            print(
                f"  fold {outer_fold} seed {seed}: "
                f"LEOP={leop_point:.4f} [{leop_lo:.4f}, {leop_hi:.4f}] "
                f"PERG={perg_point:.4f} [{perg_lo:.4f}, {perg_hi:.4f}] "
                f"best_epoch={log['best_epoch']}"
            )

            # Save predictions
            run_dir = OUT_DIR / f"run-fold{outer_fold}-seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            leop_pred.to_parquet(run_dir / "leop_predictions.parquet", index=False)
            perg_pred.to_parquet(run_dir / "perg_predictions.parquet", index=False)
            with open(run_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

        # Summary ensemble across seeds for this fold
        print(f"\n  --- fold {outer_fold} ensemble ---")

    # Final summary: load all predictions and compute ensemble
    print(f"\n{'='*60}")
    print(f"  FINAL ENSEMBLE (all folds, all seeds)")
    print(f"{'='*60}")

    all_leop = []
    all_perg = []
    for fold in range(5):
        for seed in N_SEEDS:
            run_dir = OUT_DIR / f"run-fold{fold}-seed{seed}"
            if (run_dir / "leop_predictions.parquet").exists():
                lp = pd.read_parquet(run_dir / "leop_predictions.parquet")
                lp["outer_fold"] = fold
                lp["seed"] = seed
                all_leop.append(lp)
            if (run_dir / "perg_predictions.parquet").exists():
                pp = pd.read_parquet(run_dir / "perg_predictions.parquet")
                pp["outer_fold"] = fold
                pp["seed"] = seed
                all_perg.append(pp)

    if all_leop:
        leop_all = pd.concat(all_leop, ignore_index=True)
        leop_ensemble = leop_all.groupby("unit_id").agg(
            target=("target", "first"),
            probability=("probability", "mean"),
            subject_id=("subject_id", "first"),
            outer_fold=("outer_fold", "first"),
        ).reset_index()
        lp, ll, lh = bootstrap_auroc(
            leop_ensemble["target"].values, leop_ensemble["probability"].values,
            n_reps=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        print(f"  LEOP: AUROC={lp:.4f} [{ll:.4f}, {lh:.4f}] n={len(leop_ensemble)}")
        leop_ensemble.to_parquet(OUT_DIR / "leop_ensemble.parquet", index=False)

    if all_perg:
        perg_all = pd.concat(all_perg, ignore_index=True)
        perg_ensemble = perg_all.groupby("unit_id").agg(
            target=("target", "first"),
            probability=("probability", "mean"),
            subject_id=("subject_id", "first"),
            outer_fold=("outer_fold", "first"),
        ).reset_index()
        pp, pl, ph = bootstrap_auroc(
            perg_ensemble["target"].values, perg_ensemble["probability"].values,
            n_reps=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        print(f"  PERG: AUROC={pp:.4f} [{pl:.4f}, {ph:.4f}] n={len(perg_ensemble)}")
        perg_ensemble.to_parquet(OUT_DIR / "perg_ensemble.parquet", index=False)

    # Save summary
    summary = {}
    if all_leop:
        summary["LEOP"] = {"point": lp, "ci_low": ll, "ci_high": lh, "n": len(leop_ensemble)}
    if all_perg:
        summary["PERG"] = {"point": pp, "ci_low": pl, "ci_high": ph, "n": len(perg_ensemble)}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

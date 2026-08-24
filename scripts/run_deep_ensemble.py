"""Deep Ensemble ERG Classifier

Trains N independent models from different seeds and averages predictions.
Unlike the existing 3-seed ensemble, this uses 10 seeds for a proper deep ensemble.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_attention_erg import (
    AttentionERGClassifier,
    train_one_task,
    predict_bags,
    bootstrap_auroc,
)
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.separate import build_task_bags, outer_partition

N_SEEDS = 10
SEEDS = list(range(1001, 1001 + N_SEEDS * 100, 100))  # [1001, 1101, 1201, ..., 1901]
OUT_DIR = Path("artifacts/results/deep_ensemble_v1")


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Deep Ensemble ({N_SEEDS} seeds)")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue

                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = AttentionERGClassifier(seed=seed)
                log = train_one_task(model, train_bags, test_bags, task, seed, DEVICE)
                pred = predict_bags(model, test_bags, task, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(
                    pred["target"].values, pred["probability"].values
                )
                print(
                    f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                    f"n={len(pred)} best_epoch={log['best_epoch']}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(pred_file, index=False)

        # Deep ensemble: average across ALL seeds for each fold
        print(f"\n  --- {task} DEEP ENSEMBLE ---")
        fold_results = []
        for fold in range(5):
            all_fold_preds = []
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    df["seed"] = seed
                    all_fold_preds.append(df)

            if not all_fold_preds:
                continue

            all_df = pd.concat(all_fold_preds, ignore_index=True)
            ensemble = all_df.groupby("unit_id").agg(
                target=("target", "first"),
                probability=("probability", "mean"),
                subject_id=("subject_id", "first"),
            ).reset_index()
            point, ci_lo, ci_hi = bootstrap_auroc(
                ensemble["target"].values, ensemble["probability"].values
            )
            print(
                f"  fold {fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                f"n={len(ensemble)} seeds={len(all_fold_preds) // len(ensemble) * len(ensemble) // len(ensemble)}"
            )
            fold_results.append({"fold": fold, "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(ensemble)})

        if fold_results:
            mean_auc = np.mean([r["point"] for r in fold_results])
            std_auc = np.std([r["point"] for r in fold_results])
            print(f"\n  Mean across folds: {mean_auc:.4f} ± {std_auc:.4f}")

            with open(OUT_DIR / f"{task.lower()}_deep_ensemble_metrics.json", "w") as f:
                json.dump({"fold_results": fold_results, "mean": mean_auc, "std": std_auc}, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

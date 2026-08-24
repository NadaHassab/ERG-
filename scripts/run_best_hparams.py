"""Run best hyperparameter configs across all 5 folds.

Best configs from sweep:
  LEOP: d256_lr3e4 (0.902 on fold 0)
  PERG: d128_lr5e4 (0.846 on fold 0)
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

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/best_hparams_v1")

BEST_CONFIGS = {
    "LEOP": {"d_model": 256, "lr": 3e-4, "label": "d256_lr3e4"},
    "PERG": {"d_model": 128, "lr": 5e-4, "label": "d128_lr5e4"},
}


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    all_metrics = {}
    for task in ["LEOP", "PERG"]:
        cfg = BEST_CONFIGS[task]
        print(f"\n{'='*60}")
        print(f"  {task} — Best Config: {cfg['label']} (d_model={cfg['d_model']}, lr={cfg['lr']})")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")
        fold_results = []

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    df = pd.read_parquet(pred_file)
                    pt, ci_lo, ci_hi = bootstrap_auroc(df["target"].values, df["probability"].values)
                    print(f"  fold {outer_fold}: EXISTS AUROC={pt:.4f}")
                    fold_results.append({"fold": outer_fold, "seed": seed, "auroc": pt})
                    continue

                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = AttentionERGClassifier(seed=seed, d_model=cfg["d_model"])
                log = train_one_task(model, train_bags, test_bags, task, seed, DEVICE, lr=cfg["lr"])
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
                fold_results.append({"fold": outer_fold, "seed": seed, "auroc": point})

        if fold_results:
            df_r = pd.DataFrame(fold_results)
            by_fold = df_r.groupby("fold")["auroc"].agg(["mean", "std"])
            mean_auc = by_fold["mean"].mean()
            std_auc = by_fold["mean"].std()
            print(f"\n  Per-fold mean: {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"  Per-fold breakdown:")
            for fold in range(5):
                fold_data = df_r[df_r["fold"] == fold]["auroc"]
                print(f"    fold {fold}: {fold_data.mean():.4f}")
            all_metrics[task] = {
                "mean": float(mean_auc),
                "std": float(std_auc),
                "per_fold": by_fold.to_dict(),
            }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

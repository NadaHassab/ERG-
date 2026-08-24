"""Hyperparameter sweep for Attention ERG classifier.

Tests wider d_model, different learning rates on fold 0.
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

SEED = 1001
OUT_DIR = Path("artifacts/results/hparams_v1")


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    sweep_configs = [
        {"d_model": 64,  "lr": 3e-4, "label": "d64_lr3e4"},
        {"d_model": 128, "lr": 1e-3, "label": "d128_lr1e3"},
        {"d_model": 256, "lr": 3e-4, "label": "d256_lr3e4"},
        {"d_model": 256, "lr": 1e-3, "label": "d256_lr1e3"},
        {"d_model": 128, "lr": 1e-4, "label": "d128_lr1e4"},
        {"d_model": 64,  "lr": 1e-3, "label": "d64_lr1e3"},
        {"d_model": 256, "lr": 5e-4, "label": "d256_lr5e4"},
        {"d_model": 128, "lr": 5e-4, "label": "d128_lr5e4"},
    ]

    all_results = {}
    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Attention Model Hyperparameter Sweep (fold 0)")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")
        train_bags, test_bags = outer_partition(bags, 0)

        results = []
        for cfg in sweep_configs:
            print(f"\n  --- {cfg['label']} ---")
            model = AttentionERGClassifier(seed=SEED, d_model=cfg["d_model"])
            log = train_one_task(model, train_bags, test_bags, task, SEED, DEVICE, lr=cfg["lr"])
            pred = predict_bags(model, test_bags, task, DEVICE)
            point, ci_lo, ci_hi = bootstrap_auroc(
                pred["target"].values, pred["probability"].values
            )
            print(
                f"  AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                f"n={len(pred)} best_epoch={log['best_epoch']}"
            )
            results.append({
                "config": cfg["label"],
                "d_model": cfg["d_model"],
                "lr": cfg["lr"],
                "auroc": point,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "best_epoch": log["best_epoch"],
            })

        all_results[task] = results
        best = max(results, key=lambda x: x["auroc"])
        print(f"\n  Best: {best['config']} AUROC={best['auroc']:.4f}")

    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

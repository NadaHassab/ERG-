"""Test-Time Augmentation (TTA) for ERG Classifier

Trains standard models, then applies augmentations at test time and averages predictions.
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
    bootstrap_auroc,
)
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.separate import build_task_bags, outer_partition


SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/tta_v1")
N_TTA = 10


def augment_signal(sig_tensor, rng):
    """Augment signal tensor (1, L, 1, 128) with time shift + noise."""
    s = sig_tensor.clone()
    L = s.shape[1]
    # Time shift (circular, up to 10% of length)
    shift = int(rng.integers(-L // 10, L // 10))
    s = torch.roll(s, shifts=shift, dims=1)
    # Gaussian noise
    noise = torch.randn_like(s) * 0.02
    s = s + noise
    return s


def predict_bags_tta(model, bags, task, n_tta=N_TTA, device="cuda"):
    model.eval()
    rows = []
    rng = np.random.default_rng(42)
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_bag_units([bag])
        sig_base = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)  # (1, L, 1, 128)
        vmask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        probs = []
        with torch.no_grad():
            logit, _ = model(sig_base, vmask)
            probs.append(float(torch.sigmoid(logit[0]).item()))
            for _ in range(n_tta - 1):
                sig_aug = augment_signal(sig_base, rng)
                logit, _ = model(sig_aug, vmask)
                probs.append(float(torch.sigmoid(logit[0]).item()))
        rows.append({
            "unit_id": bag.unit_id,
            "subject_id": bag.subject_id,
            "target": int(bag.target_binary),
            "probability": float(np.mean(probs)),
            "probability_std": float(np.std(probs)),
        })
    return pd.DataFrame(rows)


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Test-Time Augmentation ({N_TTA} augmented copies)")
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
                pred_tta = predict_bags_tta(model, test_bags, task, N_TTA, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(
                    pred_tta["target"].values, pred_tta["probability"].values
                )
                print(
                    f"  fold {outer_fold}: TTA AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                    f"n={len(pred_tta)} best_epoch={log['best_epoch']}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                pred_tta.to_parquet(pred_file, index=False)

        # Compare with baseline
        attn_dir = Path("artifacts/results/attention_erg_v1")
        baseline_aucs = []
        for fold in range(5):
            for seed in SEEDS:
                p = attn_dir / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    pt, _, _ = bootstrap_auroc(df["target"].values, df["probability"].values)
                    baseline_aucs.append(pt)

        tta_aucs = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    pt, _, _ = bootstrap_auroc(df["target"].values, df["probability"].values)
                    tta_aucs.append(pt)

        if baseline_aucs and tta_aucs:
            bl_mean = np.mean(baseline_aucs)
            tta_mean = np.mean(tta_aucs)
            print(f"\n  Baseline mean: {bl_mean:.4f}")
            print(f"  TTA mean:      {tta_mean:.4f}")
            print(f"  Delta:         {tta_mean - bl_mean:+.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

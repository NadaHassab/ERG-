"""Train MultiDomain+CWT on WGAN-GP augmented data.

Same logic as run_cgan_train.py but loads from wgan_gp_v1 directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_cwt import (
    MultiDomainCWT, train_model, predict_model, bootstrap_auroc,
)
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset
from scripts.run_cwt_erg import precompute_scalograms
from scripts.run_cgan_train import create_synthetic_bags
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.separate import build_task_bags, outer_partition

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/wgan_gp_trained_v1")
N_SCALES = 16
MAX_SYNTH_PER_CLASS = 100


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    synth_dir = Path("artifacts/results/wgan_gp_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — MultiDomain+CWT on WGAN-GP Augmented Data")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        for seed in SEEDS:
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue

                sig_path = synth_dir / task.lower() / f"run-fold{outer_fold}-seed{seed}" / "synth_signals.npy"
                lab_path = synth_dir / task.lower() / f"run-fold{outer_fold}-seed{seed}" / "synth_labels.npy"
                if not sig_path.exists():
                    print(f"  fold {outer_fold}: NO DATA (skip)")
                    continue

                synth_sigs = np.load(sig_path)
                synth_labs = np.load(lab_path)
                train_bags, test_bags = outer_partition(bags, outer_fold)

                keep = []
                for cls in [0, 1]:
                    ci = np.where(synth_labs == cls)[0]
                    if len(ci) > MAX_SYNTH_PER_CLASS:
                        ci = np.random.RandomState(seed).choice(ci, MAX_SYNTH_PER_CLASS, replace=False)
                    keep.extend(ci.tolist())
                keep = sorted(keep)
                synth_sigs = synth_sigs[keep]
                synth_labs = synth_labs[keep]

                synth_bags = create_synthetic_bags(synth_sigs, synth_labs, train_bags, task)
                augmented = list(train_bags) + synth_bags
                print(f"  fold {outer_fold}: {len(train_bags)} real + {len(synth_bags)} synth")

                all_bags = list(bags) + synth_bags
                ds = MultidomainERGDataset(all_bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)
                scal_cache = precompute_scalograms(list(bags) + synth_bags, N_SCALES)

                model = MultiDomainCWT(seed=seed)
                log = train_model(model, augmented, test_bags, ds, scal_cache, seed, DEVICE)
                pred = predict_model(model, ds, scal_cache, test_bags, DEVICE)
                probs = pred["probability"].values
                if np.any(np.isnan(probs)):
                    print(f"  fold {outer_fold}: NaN (skip)")
                    continue
                pt, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, probs)
                print(f"  fold {outer_fold}: AUROC={pt:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] best_epoch={log['best_epoch']}")
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(run_dir / "predictions.parquet", index=False)

    for task in ["LEOP", "PERG"]:
        aucs = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    pt, _, _ = bootstrap_auroc(df["target"].values, df["probability"].values)
                    aucs.append(pt)
        if aucs:
            by_fold = [np.mean([aucs[i] for i in range(f, len(aucs), 5)]) for f in range(5)]
            print(f"\n  {task} per-fold mean: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")


if __name__ == "__main__":
    main()

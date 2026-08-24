"""Mixup Augmentation for ERG Classification.

Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
Simple interpolation between training samples. Often improves
generalization without the complexity of GANs.

No data leakage: mixup applied only to training bags.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_cwt import (
    MultiDomainCWT, collate_multidomain_cwt, predict_model, bootstrap_auroc,
    BagSampler, _WarmupCosine,
)
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset
from scripts.run_cwt_erg import precompute_scalograms
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches, BagUnit, ComponentRow
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.evaluation.metrics import roc_auc_score

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/mixup_v1")
N_SCALES = 16
MIXUP_ALPHA = 0.4


def train_model_mixup(model, train_bags, val_bags, dataset, scal_cache,
                      seed, device, lr=1e-4, alpha=MIXUP_ALPHA):
    model.to(device)
    rng = np.random.RandomState(seed)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(sampler.bags) // 8)
    total = 200 * steps_per_epoch
    warm = 5 * steps_per_epoch
    sched = _WarmupCosine(optimizer, warmup_steps=warm, total_steps=total, min_frac=0.05)

    best_auc = -1.0
    best_state = None
    patience = 0

    def eval_auc():
        model.eval()
        yt, yp = [], []
        for bag in val_bags:
            if bag.target_binary is None:
                continue
            batch = collate_multidomain_cwt([bag], dataset, scal_cache)
            with torch.no_grad():
                logit, _ = model(
                    batch["signal"].to(device), batch["valid_mask"].to(device),
                    batch["ot"].to(device), batch["spectral"].to(device),
                    batch["vmd"].to(device), batch["physical"].to(device),
                    batch["cwt"].to(device), batch["comp_mask"].to(device),
                )
            yt.append(bag.target_binary)
            yp.append(float(torch.sigmoid(logit[0]).item()))
        if len(yt) < 2 or len(set(yt)) < 2:
            return 0.5
        return float(roc_auc_score(np.array(yt), np.array(yp)))

    model.train()
    for epoch in range(200):
        for step, idx in enumerate(sampler):
            if step >= steps_per_epoch:
                break
            bags_batch = [sampler.bags[i] for i in idx]
            batch = collate_multidomain_cwt(bags_batch, dataset, scal_cache)
            labels_b = batch["label"].to(device)

            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            perm = torch.randperm(len(bags_batch))
            batch2 = collate_multidomain_cwt([sampler.bags[i] for i in perm], dataset, scal_cache)

            logits1, _ = model(
                batch["signal"].to(device), batch["valid_mask"].to(device),
                batch["ot"].to(device), batch["spectral"].to(device),
                batch["vmd"].to(device), batch["physical"].to(device),
                batch["cwt"].to(device), batch["comp_mask"].to(device),
            )
            logits2, _ = model(
                batch2["signal"].to(device), batch2["valid_mask"].to(device),
                batch2["ot"].to(device), batch2["spectral"].to(device),
                batch2["vmd"].to(device), batch2["physical"].to(device),
                batch2["cwt"].to(device), batch2["comp_mask"].to(device),
            )
            labels2 = batch2["label"].to(device)

            loss = lam * criterion(logits1, labels_b) + (1 - lam) * criterion(logits2, labels2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()

        val_auc = eval_auc()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": epoch, "best_val_auc": best_auc}


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Mixup Augmentation (alpha={MIXUP_ALPHA})")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                ds = MultidomainERGDataset(bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)
                scal_cache = precompute_scalograms(bags, N_SCALES)

                model = MultiDomainCWT(seed=seed)
                log = train_model_mixup(model, train_bags, test_bags, ds, scal_cache, seed, DEVICE)
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

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

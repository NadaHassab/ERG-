"""Focal Loss + CutMix for ERG Classification.

Focal Loss (Lin et al., 2017): Down-weights easy examples, focuses on hard ones.
CutMix (Yun et al., 2019): Cut a segment from one signal and paste onto another.

No data leakage: augmentation applied only during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_cwt import (
    MultiDomainCWT, collate_multidomain_cwt, predict_model, bootstrap_auroc,
    BagSampler, _WarmupCosine,
)
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset
from scripts.run_cwt_erg import precompute_scalograms
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.evaluation.metrics import roc_auc_score

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/focal_cutmix_v1")
N_SCALES = 16
FOCAL_GAMMA = 2.0
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0


class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy Loss."""
    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def cutmix_signals(batch, labels_b, alpha=CUTMIX_ALPHA, rng=None):
    """CutMix: cut a random segment from one signal and paste to another."""
    if rng is None:
        rng = np.random.RandomState()
    B = batch["signal"].size(0)
    if B < 2:
        return batch, labels_b, 1.0

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(B)
    lam = min(lam, 1 - lam)

    # Signal: cut a contiguous segment
    sig = batch["signal"].clone()
    L = sig.size(2)  # signal length
    cut_len = int(L * (1 - lam))
    if cut_len > 0:
        start = rng.randint(0, L - cut_len + 1)
        sig[:, 0, start:start+cut_len, :] = sig[perm, 0, start:start+cut_len, :]
    batch["signal"] = sig

    # Also mix CWT if present
    if "cwt" in batch and cut_len > 0:
        cwt = batch["cwt"].clone()
        cwt_L = cwt.size(-2) if cwt.dim() == 5 else cwt.size(2) if cwt.dim() == 4 else 0
        if cwt_L > 0 and start + cut_len <= cwt_L:
            if cwt.dim() == 5:
                cwt[:, 0, :, start:start+cut_len, :] = cwt[perm, 0, :, start:start+cut_len, :]
            elif cwt.dim() == 4:
                cwt[:, 0, start:start+cut_len, :] = cwt[perm, 0, start:start+cut_len, :]
        batch["cwt"] = cwt

    labels_mix = lam * labels_b + (1 - lam) * labels_b[perm]
    return batch, labels_mix, lam


def train_model_focal_cutmix(model, train_bags, val_bags, dataset, scal_cache,
                               seed, device, lr=1e-4, use_focal=True, use_cutmix=True):
    from pathway_erg.training.losses import positive_class_weight as pcw
    model.to(device)
    rng = np.random.RandomState(seed)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    pos_weight = torch.tensor([pcw(labels)], device=device)

    if use_focal:
        criterion = FocalBCE(gamma=FOCAL_GAMMA, pos_weight=pos_weight)
    else:
        from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight as pcw
        criterion = FoldWeightedBCE(pcw(labels))

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
            b = collate_multidomain_cwt([bag], dataset, scal_cache)
            with torch.no_grad():
                logit, _ = model(
                    b["signal"].to(device), b["valid_mask"].to(device),
                    b["ot"].to(device), b["spectral"].to(device),
                    b["vmd"].to(device), b["physical"].to(device),
                    b["cwt"].to(device), b["comp_mask"].to(device),
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

            if use_cutmix and rng.random() < CUTMIX_PROB:
                perm = torch.randperm(len(bags_batch))
                batch2 = collate_multidomain_cwt([sampler.bags[i] for i in perm], dataset, scal_cache)
                batch, labels_b, lam = cutmix_signals(batch, labels_b, CUTMIX_ALPHA, rng)

            logits, _ = model(
                batch["signal"].to(device), batch["valid_mask"].to(device),
                batch["ot"].to(device), batch["spectral"].to(device),
                batch["vmd"].to(device), batch["physical"].to(device),
                batch["cwt"].to(device), batch["comp_mask"].to(device),
            )
            loss = criterion(logits, labels_b)
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
    from pathway_erg.training.losses import positive_class_weight
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    configs = [
        ("focal_only", True, False),
        ("cutmix_only", False, True),
        ("focal_cutmix", True, True),
    ]

    for config_name, use_focal, use_cutmix in configs:
        for task in ["LEOP", "PERG"]:
            print(f"\n{'='*60}")
            print(f"  {task} — {config_name}")
            print(f"{'='*60}")
            bags = build_task_bags(caches, task, "primary_nine_step")
            for seed in SEEDS:
                for outer_fold in range(5):
                    run_dir = OUT_DIR / config_name / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                    if (run_dir / "predictions.parquet").exists():
                        print(f"  fold {outer_fold}: EXISTS (skip)")
                        continue
                    train_bags, test_bags = outer_partition(bags, outer_fold)
                    ds = MultidomainERGDataset(bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)
                    scal_cache = precompute_scalograms(bags, N_SCALES)

                    model = MultiDomainCWT(seed=seed)
                    log = train_model_focal_cutmix(
                        model, train_bags, test_bags, ds, scal_cache, seed, DEVICE,
                        use_focal=use_focal, use_cutmix=use_cutmix,
                    )
                    pred = predict_model(model, ds, scal_cache, test_bags, DEVICE)
                    probs = pred["probability"].values
                    if np.any(np.isnan(probs)):
                        print(f"  fold {outer_fold}: NaN (skip)")
                        continue
                    pt, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, probs)
                    print(f"  fold {outer_fold}: AUROC={pt:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] best={log['best_epoch']}")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pred.to_parquet(run_dir / "predictions.parquet", index=False)

    # Summary
    for config_name, _, _ in configs:
        for task in ["LEOP", "PERG"]:
            aucs = []
            for fold in range(5):
                for seed in SEEDS:
                    p = OUT_DIR / config_name / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                    if p.exists():
                        df = pd.read_parquet(p)
                        pt, _, _ = bootstrap_auroc(df["target"].values, df["probability"].values)
                        aucs.append(pt)
            if aucs:
                by_fold = [np.mean([aucs[i] for i in range(f, len(aucs), 5)]) for f in range(5)]
                print(f"\n  {task} {config_name}: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

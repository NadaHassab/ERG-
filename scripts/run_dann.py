"""Domain-Adversarial Neural Network (DANN) for LEOP-PERG adaptation.

Wraps the existing MultiDomainCWT encoder. Adds gradient reversal layer
for task-invariant feature learning. Two task classifiers + one domain
classifier that the encoder tries to fool.

No data leakage: fold partitioning maintained, both tasks use same fold splits.
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
    MultiDomainCWT, collate_multidomain_cwt, bootstrap_auroc,
    BagSampler, N_SCALES,
)
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset
from scripts.run_cwt_erg import precompute_scalograms
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine
from pathway_erg.evaluation.metrics import roc_auc_score

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/dann_v1")


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DANN(nn.Module):
    """MultiDomainCWT encoder + GRL domain classifier."""
    def __init__(self, base_model: MultiDomainCWT, d_model=128, grl_alpha=0.0):
        super().__init__()
        self.encoder = base_model
        self.grl_alpha = grl_alpha
        # Domain classifier: predicts LEOP(0) vs PERG(1)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, signal, vmask, ot, spectral, vmd, physical, cwt, comp_mask=None):
        B, L = signal.shape[:2]

        # Extract per-component features from encoder (before head)
        sig_flat = signal.reshape(B * L, 1, -1)
        sig_feat = self.encoder.signal_proj(self.encoder.signal_cnn(sig_flat).squeeze(-1))
        ot_feat = self.encoder.ot_mlp(ot.reshape(B * L, -1))
        spec_feat = self.encoder.spectral_mlp(spectral.reshape(B * L, -1))
        vmd_feat = self.encoder.vmd_mlp(vmd.reshape(B * L, -1))
        phys_feat = self.encoder.physical_mlp(physical.reshape(B * L, -1))
        cwt_flat = cwt.reshape(B * L, N_SCALES, -1)
        cwt_feat = self.encoder.cwt_proj(self.encoder.cwt_cnn(cwt_flat).squeeze(-1))

        # Gated fusion
        concat = torch.cat([sig_feat, ot_feat, spec_feat, vmd_feat, phys_feat, cwt_feat], dim=-1)
        weights = self.encoder.gate(concat)
        fused = (weights[:, 0:1] * sig_feat + weights[:, 1:2] * ot_feat +
                 weights[:, 2:3] * spec_feat + weights[:, 3:4] * vmd_feat +
                 weights[:, 4:5] * phys_feat + weights[:, 5:6] * cwt_feat)
        tokens = fused.reshape(B, L, -1)

        if comp_mask is None:
            comp_mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)
        tokens = self.encoder.transformer(tokens)
        attn_scores = self.encoder.attn_scorer(tokens).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~comp_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~comp_mask, 0.0)
        pooled = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Task classifier (original head)
        task_logit = self.encoder.head(pooled).squeeze(-1)

        # Domain classifier with GRL
        pooled_rev = GradientReversal.apply(pooled, self.grl_alpha)
        domain_logit = self.domain_head(pooled_rev).squeeze(-1)

        return task_logit, domain_logit, pooled


def eval_auc(model, dataset, scal_cache, test_bags, device):
    model.eval()
    yt, yp = [], []
    for bag in test_bags:
        if bag.target_binary is None:
            continue
        b = collate_multidomain_cwt([bag], dataset, scal_cache)
        with torch.no_grad():
            task_logit, _, _ = model(
                b["signal"].to(device), b["valid_mask"].to(device),
                b["ot"].to(device), b["spectral"].to(device),
                b["vmd"].to(device), b["physical"].to(device),
                b["cwt"].to(device), b["comp_mask"].to(device),
            )
        yt.append(bag.target_binary)
        yp.append(float(torch.sigmoid(task_logit[0]).item()))
    if len(yt) < 2 or len(set(yt)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(yt), np.array(yp)))


def train_dann_both_tasks(ds, scal_cache, leop_bags, perg_bags,
                           leop_test, perg_test, seed, device,
                           lr=1e-4, n_epochs=200, lambda_domain=1.0):
    torch.manual_seed(seed)
    base = MultiDomainCWT(seed=seed)
    model = DANN(base, grl_alpha=0.0).to(device)

    params_task = list(model.encoder.parameters())
    params_domain = list(model.domain_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": params_task, "lr": lr},
        {"params": params_domain, "lr": lr * 2},
    ], weight_decay=1e-4)

    bce = nn.BCEWithLogitsLoss()
    steps_per_epoch = 20
    total = n_epochs * steps_per_epoch
    sched = _WarmupCosine(optimizer, warmup_steps=5*steps_per_epoch,
                          total_steps=total, min_frac=0.05)

    best_combined = -1.0
    best_state = None
    patience = 0

    model.train()
    for epoch in range(n_epochs):
        p = epoch / n_epochs
        grl_alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * lambda_domain
        model.grl_alpha = grl_alpha

        for step in range(steps_per_epoch):
            n_leop = min(8, len(leop_bags))
            n_perg = min(8, len(perg_bags))
            leop_idx = np.random.choice(len(leop_bags), n_leop, replace=False)
            perg_idx = np.random.choice(len(perg_bags), n_perg, replace=False)

            lb = collate_multidomain_cwt([leop_bags[i] for i in leop_idx], ds, scal_cache)
            pb = collate_multidomain_cwt([perg_bags[i] for i in perg_idx], ds, scal_cache)

            # Pad to same L
            def pad_to_max(t_list, dim=1):
                maxL = max(t.size(dim) for t in t_list)
                padded = []
                for t in t_list:
                    ps = maxL - t.size(dim)
                    if ps > 0:
                        shape = list(t.shape); shape[dim] = ps
                        t = torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=dim)
                    padded.append(t)
                return torch.cat(padded, dim=0)

            sig = pad_to_max([lb["signal"], pb["signal"]]).to(device)
            vm = pad_to_max([lb["valid_mask"], pb["valid_mask"]]).to(device)
            ot = pad_to_max([lb["ot"], pb["ot"]]).to(device)
            sp = pad_to_max([lb["spectral"], pb["spectral"]]).to(device)
            vm2 = pad_to_max([lb["vmd"], pb["vmd"]]).to(device)
            ph = pad_to_max([lb["physical"], pb["physical"]]).to(device)
            cw = pad_to_max([lb["cwt"], pb["cwt"]]).to(device)
            cm = pad_to_max([lb["comp_mask"], pb["comp_mask"]]).to(device)

            task_labels = torch.cat([lb["label"], pb["label"]]).to(device)
            domain_labels = torch.cat([
                torch.zeros(n_leop), torch.ones(n_perg)
            ]).to(device)

            task_logit, domain_logit, _ = model(sig, vm, ot, sp, vm2, ph, cw, cm)

            task_loss = bce(task_logit, task_labels)
            dom_loss = bce(domain_logit, domain_labels)
            loss = task_loss + lambda_domain * dom_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()

        if (epoch + 1) % 10 == 0:
            leop_auc = eval_auc(model, ds, scal_cache, leop_test, device)
            perg_auc = eval_auc(model, ds, scal_cache, perg_test, device)
            combined = (leop_auc + perg_auc) / 2
            print(f"    epoch {epoch+1}: LEOP={leop_auc:.4f} PERG={perg_auc:.4f} GRL={grl_alpha:.3f}")
            if combined > best_combined:
                best_combined = combined
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epoch": epoch, "best_combined": best_combined}


def predict_dann(model, test_bags, dataset, scal_cache, device):
    model.eval()
    rows = []
    for bag in test_bags:
        if bag.target_binary is None:
            continue
        b = collate_multidomain_cwt([bag], dataset, scal_cache)
        with torch.no_grad():
            task_logit, _, _ = model(
                b["signal"].to(device), b["valid_mask"].to(device),
                b["ot"].to(device), b["spectral"].to(device),
                b["vmd"].to(device), b["physical"].to(device),
                b["cwt"].to(device), b["comp_mask"].to(device),
            )
        rows.append({
            "bag_id": bag.unit_id,
            "target": bag.target_binary,
            "probability": float(torch.sigmoid(task_logit[0]).item()),
        })
    return pd.DataFrame(rows)


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    leop_bags = build_task_bags(caches, "LEOP", "primary_nine_step")
    perg_bags = build_task_bags(caches, "PERG", "primary_nine_step")

    all_bags = leop_bags + perg_bags
    ds = MultidomainERGDataset(all_bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)
    scal_cache = precompute_scalograms(all_bags, N_SCALES)

    for seed in SEEDS:
        print(f"\n  === seed {seed} ===")
        for outer_fold in range(5):
            leop_run = OUT_DIR / "leop" / f"run-fold{outer_fold}-seed{seed}/predictions.parquet"
            perg_run = OUT_DIR / "perg" / f"run-fold{outer_fold}-seed{seed}/predictions.parquet"
            if leop_run.exists() and perg_run.exists():
                print(f"  fold {outer_fold}: EXISTS (skip)")
                continue

            leop_train, leop_test = outer_partition(leop_bags, outer_fold)
            perg_train, perg_test = outer_partition(perg_bags, outer_fold)

            print(f"  fold {outer_fold}: training DANN...")
            model, log = train_dann_both_tasks(
                ds, scal_cache, leop_train, perg_train,
                leop_test, perg_test, seed, DEVICE,
            )

            for task, test_bags in [("LEOP", leop_test), ("PERG", perg_test)]:
                pred = predict_dann(model, test_bags, ds, scal_cache, DEVICE)
                if pred["probability"].isna().any():
                    print(f"  fold {outer_fold} {task}: NaN (skip)")
                    continue
                pt, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, pred["probability"].values)
                print(f"  fold {outer_fold} {task}: AUROC={pt:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
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
            print(f"\n  {task} DANN: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

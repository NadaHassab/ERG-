"""MultiDomain Fusion + CWT Combined (6 domains)

Extends MultiDomain Fusion with CWT scalogram as a 6th domain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_fusion import (
    MultidomainERGDataset, collate_multidomain,
    load_extra_features, bootstrap_auroc,
)
from scripts.run_cwt_erg import compute_cwt_scalogram
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/multidomain_cwt_v1")
N_SCALES = 16


class MultiDomainCWT(nn.Module):
    """Multi-domain fusion with CWT as 6th domain."""

    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)

        # Branch 1: CNN on raw signal
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.signal_proj = nn.Linear(64, d_model)

        # Branch 2: MLP on OT
        self.ot_mlp = nn.Sequential(
            nn.Linear(135, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, d_model),
        )

        # Branch 3: MLP on spectral
        self.spectral_mlp = nn.Sequential(
            nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, d_model),
        )

        # Branch 4: MLP on VMD
        self.vmd_mlp = nn.Sequential(
            nn.Linear(80, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, d_model),
        )

        # Branch 5: MLP on physical
        self.physical_mlp = nn.Sequential(
            nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d_model),
        )

        # Branch 6: CNN on CWT scalogram
        self.cwt_cnn = nn.Sequential(
            nn.Conv1d(N_SCALES, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.cwt_proj = nn.Linear(32, d_model)

        # Gated fusion (6 domains)
        self.gate = nn.Sequential(nn.Linear(d_model * 6, 6), nn.Softmax(dim=-1))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_scorer = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64),
            nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, signal, vmask, ot, spectral, vmd, physical, cwt, comp_mask=None):
        B, L = signal.shape[:2]

        # Branch 1: signal CNN
        sig_flat = signal.reshape(B * L, 1, -1)
        sig_feat = self.signal_proj(self.signal_cnn(sig_flat).squeeze(-1))

        # Branch 2: OT
        ot_feat = self.ot_mlp(ot.reshape(B * L, -1))

        # Branch 3: spectral
        spec_feat = self.spectral_mlp(spectral.reshape(B * L, -1))

        # Branch 4: VMD
        vmd_feat = self.vmd_mlp(vmd.reshape(B * L, -1))

        # Branch 5: physical
        phys_feat = self.physical_mlp(physical.reshape(B * L, -1))

        # Branch 6: CWT
        cwt_flat = cwt.reshape(B * L, N_SCALES, -1)
        cwt_feat = self.cwt_proj(self.cwt_cnn(cwt_flat).squeeze(-1))

        # Gated fusion
        concat = torch.cat([sig_feat, ot_feat, spec_feat, vmd_feat, phys_feat, cwt_feat], dim=-1)
        weights = self.gate(concat)
        fused = (weights[:, 0:1] * sig_feat + weights[:, 1:2] * ot_feat +
                 weights[:, 2:3] * spec_feat + weights[:, 3:4] * vmd_feat +
                 weights[:, 4:5] * phys_feat + weights[:, 5:6] * cwt_feat)
        tokens = fused.reshape(B, L, -1)

        if comp_mask is None:
            comp_mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)
        tokens = self.transformer(tokens)
        attn_scores = self.attn_scorer(tokens).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~comp_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~comp_mask, 0.0)
        pooled = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn_weights


def collate_multidomain_cwt(bags, dataset, scal_cache):
    """Collate with all 6 domains."""
    B = len(bags)
    L = max(len(bag.components) for bag in bags)

    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    valid_mask = np.zeros((B, L, 128), dtype=bool)
    ot = np.zeros((B, L, 135), dtype=np.float32)
    physical = np.zeros((B, L, 8), dtype=np.float32)
    spectral = np.zeros((B, L, 10), dtype=np.float32)
    vmd = np.zeros((B, L, 80), dtype=np.float32)
    cwt = np.zeros((B, L, N_SCALES, 128), dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)

    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i, j, 0, :] = comp.signal
            valid_mask[i, j, :] = comp.signal_mask
            ot[i, j, :] = comp.ot_vector
            physical[i, j, :] = comp.physical
            spectral[i, j, :] = dataset.get_spectral(comp)
            vmd[i, j, :] = dataset.get_vmd(comp)
            key = comp.global_component_id
            if key in scal_cache:
                cwt[i, j, :, :] = scal_cache[key]
            comp_mask[i, j] = True

    return {
        "signal": torch.as_tensor(signal),
        "valid_mask": torch.as_tensor(valid_mask),
        "ot": torch.as_tensor(ot),
        "physical": torch.as_tensor(physical),
        "spectral": torch.as_tensor(spectral),
        "vmd": torch.as_tensor(vmd),
        "cwt": torch.as_tensor(cwt),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, dataset, scal_cache, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
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
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_model(model, train_bags, val_bags, dataset, scal_cache, seed, device, lr=1e-4):
    model.to(device)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags}, batch_size=8, seed=seed)
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
    model.train()
    for epoch in range(200):
        for step, idx in enumerate(sampler):
            if step >= steps_per_epoch:
                break
            bags_batch = [sampler.bags[i] for i in idx]
            batch = collate_multidomain_cwt(bags_batch, dataset, scal_cache)
            labels_b = batch["label"].to(device)
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
        val_auc = eval_auc(model, dataset, scal_cache, val_bags, device)
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


def predict_model(model, dataset, scal_cache, bags, device):
    model.eval()
    rows = []
    for bag in bags:
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
        rows.append({
            "unit_id": bag.unit_id, "subject_id": bag.subject_id,
            "target": int(bag.target_binary),
            "probability": float(torch.sigmoid(logit[0]).item()),
        })
    return pd.DataFrame(rows)


from pathway_erg.evaluation.metrics import roc_auc_score


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — MultiDomain + CWT Fusion")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")
        ds = MultidomainERGDataset(bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)

        # Precompute CWT
        from scripts.run_cwt_erg import precompute_scalograms
        scal_cache = precompute_scalograms(bags, N_SCALES)
        print(f"  Precomputed {len(scal_cache)} CWT scalograms")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = MultiDomainCWT(seed=seed)
                log = train_model(model, train_bags, test_bags, ds, scal_cache, seed, DEVICE)
                pred = predict_model(model, ds, scal_cache, test_bags, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, pred["probability"].values)
                print(f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={len(pred)} best_epoch={log['best_epoch']}")
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(pred_file, index=False)

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

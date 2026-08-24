"""Approach 2: CWT Scalogram ERG Classifier

Uses Continuous Wavelet Transform (Morlet) to create time-frequency scalograms.
Feeds scalograms through 1D CNN + attention for classification.
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

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/cwt_erg_v1")


def compute_cwt_scalogram(signal_1d, n_scales=16, wavelet="morl"):
    sig = signal_1d.copy()
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return np.zeros((n_scales, len(sig)), dtype=np.float32)
    if nan_mask.any():
        good = np.where(~nan_mask)[0]
        sig[nan_mask] = np.interp(np.where(nan_mask)[0], good, sig[good])
    scales = np.logspace(np.log10(2), np.log10(min(128, len(sig) // 2)), n_scales).astype(float)
    try:
        coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0)
        scalogram = np.abs(coeffs).astype(np.float32)
        for i in range(n_scales):
            s_max = scalogram[i].max()
            if s_max > 0:
                scalogram[i] /= s_max
    except Exception:
        scalogram = np.zeros((n_scales, len(sig)), dtype=np.float32)
    return scalogram


class CWTERGClassifier(nn.Module):
    def __init__(self, n_scales=16, signal_len=128, cnn_channels=64,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)
        self.scalogram_cnn = nn.Sequential(
            nn.Conv1d(n_scales, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.raw_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(cnn_channels + 32, d_model)
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

    def forward(self, scalograms, raw_signals, comp_mask=None):
        B, L = scalograms.shape[:2]
        n_scales = scalograms.shape[2]
        scal_flat = scalograms.reshape(B * L, n_scales, -1)
        scal_feat = self.scalogram_cnn(scal_flat).squeeze(-1)
        raw_flat = raw_signals.reshape(B * L, 1, -1)
        raw_feat = self.raw_cnn(raw_flat).squeeze(-1)
        combined = torch.cat([scal_feat, raw_feat], dim=-1)
        tokens = self.proj(combined).reshape(B, L, -1)
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


def precompute_scalograms(bags, n_scales=16):
    cache = {}
    for bag in bags:
        for comp in bag.components:
            key = comp.global_component_id
            if key not in cache:
                cache[key] = compute_cwt_scalogram(comp.signal, n_scales)
    return cache


def collate_cwt(bags, scal_cache, n_scales=16):
    B = len(bags)
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    scalograms = np.zeros((B, L, n_scales, 128), dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)
    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i, j, 0, :] = comp.signal
            key = comp.global_component_id
            if key in scal_cache:
                scalograms[i, j, :, :] = scal_cache[key]
            comp_mask[i, j] = True
    return {
        "signal": torch.as_tensor(signal),
        "scalograms": torch.as_tensor(scalograms),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, scal_cache, bags, n_scales, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_cwt([bag], scal_cache, n_scales)
        with torch.no_grad():
            logit, _ = model(batch["scalograms"].to(device), batch["signal"].to(device), batch["comp_mask"].to(device))
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_cwt(model, train_bags, val_bags, scal_cache, n_scales, seed, device, lr=1e-4):
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
            batch = collate_cwt(bags_batch, scal_cache, n_scales)
            scal = batch["scalograms"].to(device)
            sig = batch["signal"].to(device)
            cmask = batch["comp_mask"].to(device)
            labels_b = batch["label"].to(device)
            logits, _ = model(scal, sig, cmask)
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, scal_cache, val_bags, n_scales, device)
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


def bootstrap_auroc(y_true, y_prob, n_reps=2000, seed=424242):
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y_true, y_prob)
    scores = []
    for _ in range(n_reps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    N_SCALES = 16

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — CWT Scalogram ERG Classifier")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        all_scal = precompute_scalograms(bags, N_SCALES)
        print(f"  Precomputed {len(all_scal)} scalograms")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = CWTERGClassifier(n_scales=N_SCALES, seed=seed)
                log = train_cwt(model, train_bags, test_bags, all_scal, N_SCALES, seed, DEVICE)
                model.eval()
                y_true, y_prob = [], []
                for bag in test_bags:
                    if bag.target_binary is None:
                        continue
                    batch = collate_cwt([bag], all_scal, N_SCALES)
                    with torch.no_grad():
                        logit, _ = model(batch["scalograms"].to(DEVICE), batch["signal"].to(DEVICE), batch["comp_mask"].to(DEVICE))
                    y_true.append(bag.target_binary)
                    y_prob.append(float(torch.sigmoid(logit[0]).item()))
                pt, lo, hi = bootstrap_auroc(np.array(y_true), np.array(y_prob))
                print(f"  fold {outer_fold}: AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] n={len(y_true)} best_epoch={log['best_epoch']}")
                run_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"target": y_true, "probability": y_prob}).to_parquet(pred_file, index=False)

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

"""Approach 4: Cross-Component Phase Coherence + Attention ERG

Computes phase coherence between ERG component pairs within a bag.
Adds physiological coupling features to the attention-based classifier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/phase_coh_v1")
N_PAIR_FEATURES = 6  # coherence, amp_corr, time_delay, peak_phase_diff, energy_ratio, correlation


def compute_phase_features(sig_a, sig_b, mask_a, mask_b):
    """Compute phase/coherence features between two component signals."""
    a = sig_a.copy()
    b = sig_b.copy()
    na = np.isnan(a)
    nb = np.isnan(b)
    if na.all() or nb.all():
        return np.zeros(N_PAIR_FEATURES, dtype=np.float32)
    if na.any():
        good = np.where(~na)[0]
        a[na] = np.interp(np.where(na)[0], good, a[good])
    if nb.any():
        good = np.where(~nb)[0]
        b[nb] = np.interp(np.where(nb)[0], good, b[good])

    # Hilbert transform for instantaneous phase
    try:
        analytic_a = hilbert(a)
        analytic_b = hilbert(b)
        phase_a = np.angle(analytic_a)
        phase_b = np.angle(analytic_b)
        amp_a = np.abs(analytic_a)
        amp_b = np.abs(analytic_b)
    except Exception:
        return np.zeros(N_PAIR_FEATURES, dtype=np.float32)

    # Phase coherence (magnitude of mean resultant vector)
    phase_diff = phase_a - phase_b
    coherence = float(np.abs(np.mean(np.exp(1j * phase_diff))))

    # Amplitude correlation
    valid = mask_a & mask_b
    if valid.sum() < 3:
        amp_corr = 0.0
    else:
        amp_corr = float(np.corrcoef(amp_a[valid], amp_b[valid])[0, 1])
        if np.isnan(amp_corr):
            amp_corr = 0.0

    # Peak time delay (cross-correlation based)
    try:
        cc = np.correlate(a - a.mean(), b - b.mean(), mode="full")
        delay = float(np.argmax(cc) - (len(a) - 1))
    except Exception:
        delay = 0.0

    # Peak phase difference
    try:
        peak_a = np.argmax(amp_a)
        peak_b = np.argmax(amp_b)
        peak_phase_diff = float(phase_a[peak_a] - phase_b[peak_b])
    except Exception:
        peak_phase_diff = 0.0

    # Energy ratio
    energy_a = float(np.sum(a ** 2))
    energy_b = float(np.sum(b ** 2))
    energy_ratio = energy_a / (energy_b + 1e-12)

    # Pearson correlation
    try:
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    return np.array([coherence, amp_corr, delay / 128.0, peak_phase_diff, energy_ratio, corr], dtype=np.float32)


def precompute_phase_features(bags):
    """Precompute phase features for all component pairs within each bag."""
    cache = {}
    for bag in bags:
        comps = list(bag.components)
        n = len(comps)
        # For each component, compute features against the first component
        # and store as a per-component feature vector
        if n < 2:
            for comp in comps:
                cache[comp.global_component_id] = np.zeros(N_PAIR_FEATURES * max(1, n - 1), dtype=np.float32)
            continue
        for i, comp_i in enumerate(comps):
            features = []
            for j, comp_j in enumerate(comps):
                if i == j:
                    continue
                feat = compute_phase_features(comp_i.signal, comp_j.signal,
                                              comp_i.signal_mask, comp_j.signal_mask)
                features.append(feat)
            if features:
                cache[comp_i.global_component_id] = np.concatenate(features).astype(np.float32)
            else:
                cache[comp_i.global_component_id] = np.zeros(N_PAIR_FEATURES, dtype=np.float32)
    return cache


class PhaseCohERGClassifier(nn.Module):
    """Attention ERG + cross-component phase coherence features."""

    def __init__(self, signal_len=128, cnn_channels=64, d_model=128,
                 n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.cnn_proj = nn.Linear(cnn_channels, d_model)

        # Phase features projection
        self.phase_proj = nn.Sequential(
            nn.Linear(N_PAIR_FEATURES * 4, 32),  # up to 4 other components
            nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d_model),
        )

        # Gated fusion
        self.gate = nn.Sequential(nn.Linear(d_model * 2, 2), nn.Softmax(dim=-1))

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

    def forward(self, signal, vmask, phase_feats, comp_mask=None):
        B, L = signal.shape[:2]
        sig_flat = signal.reshape(B * L, 1, -1)
        vmask_flat = vmask.reshape(B * L, -1)
        cnn_out = self.cnn(sig_flat).squeeze(-1)
        sig_feat = self.cnn_proj(cnn_out)

        pf = phase_feats.reshape(B * L, -1)
        # Pad if needed
        if pf.shape[1] < N_PAIR_FEATURES * 4:
            pad = torch.zeros(pf.shape[0], N_PAIR_FEATURES * 4 - pf.shape[1], device=pf.device)
            pf = torch.cat([pf, pad], dim=-1)
        elif pf.shape[1] > N_PAIR_FEATURES * 4:
            pf = pf[:, :N_PAIR_FEATURES * 4]
        phase_feat = self.phase_proj(pf)

        concat = torch.cat([sig_feat, phase_feat], dim=-1)
        weights = self.gate(concat)
        fused = weights[:, 0:1] * sig_feat + weights[:, 1:2] * phase_feat
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


def collate_phase(bags, phase_cache, max_pairs=4):
    B = len(bags)
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    valid_mask = np.zeros((B, L, 128), dtype=bool)
    phase_dim = N_PAIR_FEATURES * max_pairs
    phase_feats = np.zeros((B, L, phase_dim), dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)
    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i, j, 0, :] = comp.signal
            valid_mask[i, j, :] = comp.signal_mask
            key = comp.global_component_id
            if key in phase_cache:
                feat = phase_cache[key]
                phase_feats[i, j, :len(feat)] = feat[:phase_dim]
            comp_mask[i, j] = True
    return {
        "signal": torch.as_tensor(signal),
        "valid_mask": torch.as_tensor(valid_mask),
        "phase_feats": torch.as_tensor(phase_feats),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, phase_cache, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_phase([bag], phase_cache)
        with torch.no_grad():
            logit, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device),
                             batch["phase_feats"].to(device), batch["comp_mask"].to(device))
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_phase(model, train_bags, val_bags, phase_cache, seed, device, lr=1e-4):
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
            batch = collate_phase(bags_batch, phase_cache)
            sig = batch["signal"].to(device)
            vmask = batch["valid_mask"].to(device)
            pf = batch["phase_feats"].to(device)
            cmask = batch["comp_mask"].to(device)
            labels_b = batch["label"].to(device)
            logits, _ = model(sig, vmask, pf, cmask)
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, phase_cache, val_bags, device)
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

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Phase Coherence ERG Classifier")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        phase_cache = precompute_phase_features(bags)
        print(f"  Precomputed phase features for {len(phase_cache)} components")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = PhaseCohERGClassifier(seed=seed)
                log = train_phase(model, train_bags, test_bags, phase_cache, seed, DEVICE)
                pred_y, pred_p = [], []
                model.eval()
                for bag in test_bags:
                    if bag.target_binary is None:
                        continue
                    batch = collate_phase([bag], phase_cache)
                    with torch.no_grad():
                        logit, _ = model(batch["signal"].to(DEVICE), batch["valid_mask"].to(DEVICE),
                                         batch["phase_feats"].to(DEVICE), batch["comp_mask"].to(DEVICE))
                    pred_y.append(bag.target_binary)
                    pred_p.append(float(torch.sigmoid(logit[0]).item()))
                pt, lo, hi = bootstrap_auroc(np.array(pred_y), np.array(pred_p))
                print(f"  fold {outer_fold}: AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] n={len(pred_y)} best_epoch={log['best_epoch']}")
                run_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"target": pred_y, "probability": pred_p}).to_parquet(pred_file, index=False)

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

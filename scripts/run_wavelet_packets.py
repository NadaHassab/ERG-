"""Approach 3: Wavelet Packet Decomposition + Statistics

Decomposes ERG signal into wavelet packets, computes per-packet statistics.
Richer frequency resolution than existing wavelet scattering.
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
OUT_DIR = Path("artifacts/results/wavepacket_v1")
WAVELET = "db4"
MAX_LEVEL = 4


def wavelet_packet_features(signal_1d, wavelet=WAVELET, max_level=MAX_LEVEL):
    """Compute wavelet packet features from signal."""
    sig = signal_1d.copy()
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return np.zeros(3 * (2 ** max_level), dtype=np.float32)
    if nan_mask.any():
        good = np.where(~nan_mask)[0]
        sig[nan_mask] = np.interp(np.where(nan_mask)[0], good, sig[good])

    features = []
    try:
        wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, maxlevel=max_level)
        nodes = [node.path for node in wp.get_level(max_level, order="freq")]
        for node_path in nodes:
            node = wp[node_path]
            coeff = node.data
            if len(coeff) == 0:
                features.extend([0.0, 0.0, 0.0])
                continue
            energy = float(np.sum(coeff ** 2))
            entropy = float(-np.sum((coeff ** 2 / (energy + 1e-12)) * np.log(coeff ** 2 / (energy + 1e-12) + 1e-12)))
            kurtosis = float(np.mean((coeff - coeff.mean()) ** 4) / (coeff.std() ** 4 + 1e-12))
            features.extend([np.log1p(entropy), np.log1p(energy), kurtosis])
    except Exception:
        features = [0.0] * (3 * (2 ** max_level))

    n_expected = 3 * (2 ** max_level)
    if len(features) < n_expected:
        features.extend([0.0] * (n_expected - len(features)))
    return np.asarray(features[:n_expected], dtype=np.float32)


FEAT_DIM = 3 * (2 ** MAX_LEVEL)  # 3 stats x 16 nodes = 48


class WavePacketERGClassifier(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, signal_len=128,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)

        self.raw_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )

        self.wp_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 64),
        )

        self.proj = nn.Linear(64 + 64, d_model)
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

    def forward(self, signal, wp_feats, comp_mask=None):
        B, L = signal.shape[:2]
        sig_flat = signal.reshape(B * L, 1, -1)
        raw_feat = self.raw_cnn(sig_flat).squeeze(-1)
        wp_flat = wp_feats.reshape(B * L, -1)
        wp_feat = self.wp_mlp(wp_flat)
        combined = torch.cat([raw_feat, wp_feat], dim=-1)
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


def precompute_wp_feats(bags):
    cache = {}
    for bag in bags:
        for comp in bag.components:
            key = comp.global_component_id
            if key not in cache:
                cache[key] = wavelet_packet_features(comp.signal)
    return cache


def collate_wp(bags, wp_cache):
    B = len(bags)
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    wp_feats = np.zeros((B, L, FEAT_DIM), dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)
    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i, j, 0, :] = comp.signal
            key = comp.global_component_id
            if key in wp_cache:
                wp_feats[i, j, :] = wp_cache[key]
            comp_mask[i, j] = True
    return {
        "signal": torch.as_tensor(signal),
        "wp_feats": torch.as_tensor(wp_feats),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, wp_cache, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_wp([bag], wp_cache)
        with torch.no_grad():
            logit, _ = model(batch["signal"].to(device), batch["wp_feats"].to(device), batch["comp_mask"].to(device))
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_wp(model, train_bags, val_bags, wp_cache, seed, device, lr=1e-4):
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
            batch = collate_wp(bags_batch, wp_cache)
            sig = batch["signal"].to(device)
            wp = batch["wp_feats"].to(device)
            cmask = batch["comp_mask"].to(device)
            labels_b = batch["label"].to(device)
            logits, _ = model(sig, wp, cmask)
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, wp_cache, val_bags, device)
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
        print(f"  {task} — Wavelet Packet ERG Classifier")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        wp_cache = precompute_wp_feats(bags)
        print(f"  Precomputed {len(wp_cache)} wavelet packet feature vectors (dim={FEAT_DIM})")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = WavePacketERGClassifier(seed=seed)
                log = train_wp(model, train_bags, test_bags, wp_cache, seed, DEVICE)
                model.eval()
                y_true, y_prob = [], []
                for bag in test_bags:
                    if bag.target_binary is None:
                        continue
                    batch = collate_wp([bag], wp_cache)
                    with torch.no_grad():
                        logit, _ = model(batch["signal"].to(DEVICE), batch["wp_feats"].to(DEVICE), batch["comp_mask"].to(DEVICE))
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

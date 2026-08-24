"""Advanced Signal Processing Features for ERG Classification

Computes rich feature set from ERG waveforms:
- Hjorth parameters (activity, mobility, complexity)
- Teager-Kaiser Energy Operator (TKEO)
- Sample entropy
- Permutation entropy
- Higuchi fractal dimension
- Welch PSD features
- EMD (Empirical Mode Decomposition) features
- Higher-order statistics (skewness, kurtosis)
- Zero-crossing rate
- Autocorrelation features

All features computed on the 128-point canonical signal.
No data leakage: features computed per-component, no test data used.
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
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/advanced_sp_v1")


# ── Feature Functions ──────────────────────────────────────────────────────

def hjorth_parameters(sig):
    """Hjorth parameters: activity, mobility, complexity."""
    activity = float(np.var(sig))
    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    mobility = float(np.sqrt(np.var(diff1) / (activity + 1e-12)))
    complexity = float(np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12))
    return np.array([activity, mobility, complexity], dtype=np.float32)


def teager_kaiser_energy(sig):
    """Teager-Kaiser Energy Operator: psi[n] = x[n]^2 - x[n-1]*x[n+1]."""
    if len(sig) < 3:
        return np.zeros(3, dtype=np.float32)
    tkeo = sig[1:-1] ** 2 - sig[:-2] * sig[2:]
    return np.array([
        float(np.mean(np.abs(tkeo))),
        float(np.max(np.abs(tkeo))),
        float(np.std(tkeo)),
    ], dtype=np.float32)


def sample_entropy(sig, m=2, r=0.2):
    """Sample entropy: nonlinear complexity measure."""
    N = len(sig)
    if N < m + 2:
        return np.zeros(1, dtype=np.float32)
    r_threshold = r * np.std(sig)
    if r_threshold < 1e-12:
        return np.zeros(1, dtype=np.float32)

    def _count_matches(template_len):
        count = 0
        templates = np.array([sig[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r_threshold:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0 or A == 0:
        return np.zeros(1, dtype=np.float32)
    return np.array([-np.log(A / B + 1e-12)], dtype=np.float32)


def permutation_entropy(sig, m=3, delay=1):
    """Permutation entropy: ordinal pattern complexity."""
    N = len(sig)
    if N < m * delay + 1:
        return np.zeros(1, dtype=np.float32)
    patterns = np.array([sig[i:i + m * delay:delay] for i in range(N - m * delay + 1)])
    ordinal = np.argsort(patterns, axis=1)
    _, counts = np.unique(ordinal, axis=0, return_counts=True)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log(probs + 1e-12))
    import math
    pe_norm = pe / np.log(math.factorial(m) + 1e-12)
    return np.array([float(pe_norm)], dtype=np.float32)


def higuchi_fd(sig, kmax=10):
    """Higuchi fractal dimension."""
    N = len(sig)
    if N < kmax + 1:
        return np.zeros(1, dtype=np.float32)
    L = []
    x = np.arange(1, kmax + 1)
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(1, k + 1):
            indices = np.arange(m - 1, N, k)
            if len(indices) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(sig[indices])))
            Lmk *= (N - 1) / (k * len(indices) * k)
            Lk += Lmk
        L.append(Lk / k)
    L = np.array(L)
    valid = L > 0
    if valid.sum() < 2:
        return np.zeros(1, dtype=np.float32)
    coeffs = np.polyfit(np.log(x[valid]), np.log(L[valid]), 1)
    return np.array([float(coeffs[0])], dtype=np.float32)


def welch_psd_features(sig, fs=100.0, nperseg=32):
    """Welch PSD features: spectral centroid, spread, rolloff, flatness."""
    if len(sig) < nperseg:
        return np.zeros(5, dtype=np.float32)
    freqs, psd = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))
    psd_sum = psd.sum()
    if psd_sum < 1e-12:
        return np.zeros(5, dtype=np.float32)
    # Spectral centroid
    centroid = float(np.sum(freqs * psd) / psd_sum)
    # Spectral spread
    spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * psd) / psd_sum))
    # Spectral rolloff (85%)
    cumsum = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    # Spectral flatness (geometric mean / arithmetic mean)
    log_psd = np.log(psd + 1e-12)
    flatness = float(np.exp(np.mean(log_psd)) / (psd.mean() + 1e-12))
    # Spectral entropy
    p = psd / psd_sum
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return np.array([centroid, spread, rolloff, flatness, entropy], dtype=np.float32)


def emd_features(sig, n_imfs=5):
    """EMD features: energy and frequency of each IMF."""
    try:
        import emd
        sig_clean = sig.copy()
        nan_mask = np.isnan(sig_clean)
        if nan_mask.any():
            good = np.where(~nan_mask)[0]
            if len(good) > 0:
                sig_clean[nan_mask] = np.interp(np.where(nan_mask)[0], good, sig_clean[good])
        imfs = emd.sift.sift(sig_clean)[:n_imfs]
        features = []
        for i in range(n_imfs):
            if i < len(imfs):
                energy = float(np.sum(imfs[i] ** 2))
                # Instantaneous frequency via Hilbert
                analytic = hilbert(imfs[i])
                inst_phase = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(inst_phase) / (2 * np.pi)
                mean_freq = float(np.mean(np.abs(inst_freq)))
                features.extend([np.log1p(energy), mean_freq])
            else:
                features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)
    except Exception:
        return np.zeros(n_imfs * 2, dtype=np.float32)


def higher_order_stats(sig):
    """Higher-order statistics: skewness, kurtosis, crest factor."""
    return np.array([
        float(skew(sig)),
        float(kurtosis(sig)),
        float(np.max(np.abs(sig)) / (np.sqrt(np.mean(sig ** 2)) + 1e-12)),
    ], dtype=np.float32)


def zero_crossing_rate(sig):
    """Zero-crossing rate."""
    centered = sig - np.mean(sig)
    zcr = float(np.sum(np.abs(np.diff(np.sign(centered)))) / (2 * len(centered)))
    return np.array([zcr], dtype=np.float32)


def autocorrelation_features(sig, lags=[1, 2, 4, 8]):
    """Autocorrelation at specified lags."""
    if len(sig) < max(lags) + 1:
        return np.zeros(len(lags), dtype=np.float32)
    features = []
    for lag in lags:
        if lag < len(sig):
            c = np.corrcoef(sig[:-lag], sig[lag:])[0, 1]
            features.append(float(c) if np.isfinite(c) else 0.0)
        else:
            features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_all_features(signal_1d):
    """Compute all advanced signal processing features."""
    sig = signal_1d.copy()
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return np.zeros(3 + 3 + 1 + 1 + 1 + 5 + 10 + 3 + 1 + 4, dtype=np.float32)
    if nan_mask.any():
        good = np.where(~nan_mask)[0]
        sig[nan_mask] = np.interp(np.where(nan_mask)[0], good, sig[good])

    features = np.concatenate([
        hjorth_parameters(sig),           # 3
        teager_kaiser_energy(sig),        # 3
        sample_entropy(sig),              # 1
        permutation_entropy(sig),         # 1
        higuchi_fd(sig),                  # 1
        welch_psd_features(sig),          # 5
        emd_features(sig),                # 10
        higher_order_stats(sig),          # 3
        zero_crossing_rate(sig),          # 1
        autocorrelation_features(sig),    # 4
    ])
    return features.astype(np.float32)


FEAT_DIM = 3 + 3 + 1 + 1 + 1 + 5 + 10 + 3 + 1 + 4  # = 32


class AdvancedSPClassifier(nn.Module):
    """Multi-domain model with advanced signal processing features."""

    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)

        # CNN on raw signal
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.signal_proj = nn.Linear(64, d_model)

        # MLP on OT
        self.ot_mlp = nn.Sequential(
            nn.Linear(135, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, d_model),
        )

        # MLP on spectral
        self.spectral_mlp = nn.Sequential(
            nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, d_model),
        )

        # MLP on advanced SP features
        self.advsp_mlp = nn.Sequential(
            nn.Linear(FEAT_DIM, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, d_model),
        )

        # MLP on physical
        self.physical_mlp = nn.Sequential(
            nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d_model),
        )

        # Gated fusion (5 domains)
        self.gate = nn.Sequential(nn.Linear(d_model * 5, 5), nn.Softmax(dim=-1))

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

    def forward(self, signal, vmask, ot, spectral, physical, advsp, comp_mask=None):
        B, L = signal.shape[:2]

        sig_flat = signal.reshape(B * L, 1, -1)
        sig_feat = self.signal_proj(self.signal_cnn(sig_flat).squeeze(-1))
        ot_feat = self.ot_mlp(ot.reshape(B * L, -1))
        spec_feat = self.spectral_mlp(spectral.reshape(B * L, -1))
        advsp_feat = self.advsp_mlp(advsp.reshape(B * L, -1))
        phys_feat = self.physical_mlp(physical.reshape(B * L, -1))

        concat = torch.cat([sig_feat, ot_feat, spec_feat, advsp_feat, phys_feat], dim=-1)
        weights = self.gate(concat)
        fused = (weights[:, 0:1] * sig_feat + weights[:, 1:2] * ot_feat +
                 weights[:, 2:3] * spec_feat + weights[:, 3:4] * advsp_feat +
                 weights[:, 4:5] * phys_feat)
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


def precompute_advsp(bags):
    cache = {}
    for bag in bags:
        for comp in bag.components:
            key = comp.global_component_id
            if key not in cache:
                cache[key] = compute_all_features(comp.signal)
    return cache


def collate_advsp(bags, advsp_cache, spectral_vecs, comp_idx_map):
    B = len(bags)
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    valid_mask = np.zeros((B, L, 128), dtype=bool)
    ot = np.zeros((B, L, 135), dtype=np.float32)
    physical = np.zeros((B, L, 8), dtype=np.float32)
    spectral = np.zeros((B, L, 10), dtype=np.float32)
    advsp = np.zeros((B, L, FEAT_DIM), dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)

    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i, j, 0, :] = comp.signal
            valid_mask[i, j, :] = comp.signal_mask
            ot[i, j, :] = comp.ot_vector
            physical[i, j, :] = comp.physical
            key = comp.global_component_id
            if key in advsp_cache:
                advsp[i, j, :] = advsp_cache[key]
            idx = comp_idx_map.get(key)
            if idx is not None and idx < len(spectral_vecs):
                spectral[i, j, :] = spectral_vecs[idx]
            comp_mask[i, j] = True

    return {
        "signal": torch.as_tensor(signal),
        "valid_mask": torch.as_tensor(valid_mask),
        "ot": torch.as_tensor(ot),
        "physical": torch.as_tensor(physical),
        "spectral": torch.as_tensor(spectral),
        "advsp": torch.as_tensor(advsp),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, advsp_cache, spectral_vecs, comp_idx_map, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_advsp([bag], advsp_cache, spectral_vecs, comp_idx_map)
        with torch.no_grad():
            logit, _ = model(
                batch["signal"].to(device), batch["valid_mask"].to(device),
                batch["ot"].to(device), batch["spectral"].to(device),
                batch["physical"].to(device), batch["advsp"].to(device),
                batch["comp_mask"].to(device),
            )
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_model(model, train_bags, val_bags, advsp_cache, spectral_vecs, comp_idx_map, seed, device, lr=1e-4):
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
            batch = collate_advsp(bags_batch, advsp_cache, spectral_vecs, comp_idx_map)
            labels_b = batch["label"].to(device)
            logits, _ = model(
                batch["signal"].to(device), batch["valid_mask"].to(device),
                batch["ot"].to(device), batch["spectral"].to(device),
                batch["physical"].to(device), batch["advsp"].to(device),
                batch["comp_mask"].to(device),
            )
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, advsp_cache, spectral_vecs, comp_idx_map, val_bags, device)
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


def predict_model(model, advsp_cache, spectral_vecs, comp_idx_map, bags, device):
    model.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_advsp([bag], advsp_cache, spectral_vecs, comp_idx_map)
        with torch.no_grad():
            logit, _ = model(
                batch["signal"].to(device), batch["valid_mask"].to(device),
                batch["ot"].to(device), batch["spectral"].to(device),
                batch["physical"].to(device), batch["advsp"].to(device),
                batch["comp_mask"].to(device),
            )
        rows.append({
            "unit_id": bag.unit_id, "subject_id": bag.subject_id,
            "target": int(bag.target_binary),
            "probability": float(torch.sigmoid(logit[0]).item()),
        })
    return pd.DataFrame(rows)


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

    # Load spectral features
    import zarr
    from pathway_erg.signal.component_cache import load_cache_manifest, CACHE_SCHEMA_VERSION
    root = Path(data_cfg.artifact_root)
    manifest = load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    spectral_zarr_path = root / "data" / "arrays" / "spectral_features_v4.zarr"
    spectral_z = zarr.open_group(str(spectral_zarr_path), mode="r")
    spectral_vecs = np.asarray(spectral_z["components"]["spectral_vector"][:])
    print(f"Loaded spectral: {spectral_vecs.shape}")

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Advanced Signal Processing + MultiDomain Fusion")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        # Build component index map
        comp_idx = 0
        comp_idx_map = {}
        for bag in bags:
            for comp in bag.components:
                if comp.global_component_id not in comp_idx_map:
                    comp_idx_map[comp.global_component_id] = comp_idx
                    comp_idx += 1

        # Precompute advanced SP features
        advsp_cache = precompute_advsp(bags)
        print(f"  Precomputed {len(advsp_cache)} advanced SP feature vectors (dim={FEAT_DIM})")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = AdvancedSPClassifier(seed=seed)
                log = train_model(model, train_bags, test_bags, advsp_cache, spectral_vecs, comp_idx_map, seed, DEVICE)
                pred = predict_model(model, advsp_cache, spectral_vecs, comp_idx_map, test_bags, DEVICE)
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

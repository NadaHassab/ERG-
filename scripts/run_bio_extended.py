"""Biologically-inspired Extended Domains: STFT + FrFT + OP

Adds 3 new time-frequency domains to MultiDomain+CWT (6→9 domains):
  STFT  — Hamming-window spectrogram, time-localized spectra (novo vs CWT)
  FrFT  — Fractional Fourier, rotation in time-frequency plane (chirp basis)
  OP    — Oscillatory Potentials via bandpass 75-300Hz, inner-retina marker

Gated fusion over 9 domains. All transforms operate on 128-point canonical signals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import stft, butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_fusion import MultidomainERGDataset, load_extra_features, bootstrap_auroc
from scripts.run_cwt_erg import compute_cwt_scalogram
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine
from pathway_erg.evaluation.metrics import roc_auc_score

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/bio_extended_v1")
N_SCALES = 16
FS = 1000.0
FRFT_ORDERS = [0.2, 0.4, 0.6, 0.8, 1.0]
STFT_NPERSEG = 16


def _frft_matrix(N: int, alpha: float):
    t = np.arange(N) - N // 2
    u = np.arange(N) - N // 2
    cot_a = 1.0 / np.tan(alpha) if abs(np.sin(alpha)) > 1e-8 else 0.0
    csc_a = 1.0 / np.sin(alpha) if abs(np.sin(alpha)) > 1e-8 else 0.0
    A = np.sqrt(1 - 1j * cot_a) if abs(np.sin(alpha)) > 1e-8 else 1.0
    T, U = np.meshgrid(t, u)
    K = A * np.exp(1j * np.pi * (cot_a * (T ** 2 + U ** 2) - 2 * csc_a * U * T) / N)
    return K / np.sqrt(N)


_FRFT_MATRICES = {}


def get_frft_matrix(N, order):
    alpha = order * np.pi / 2
    key = (N, round(order, 4))
    if key not in _FRFT_MATRICES:
        if abs(order) < 1e-6:
            _FRFT_MATRICES[key] = np.eye(N, dtype=np.complex128)
        elif abs(order - 1.0) < 1e-6:
            _FRFT_MATRICES[key] = np.fft.fft(np.eye(N)) / np.sqrt(N)
        else:
            _FRFT_MATRICES[key] = _frft_matrix(N, alpha)
    return _FRFT_MATRICES[key]


def compute_frft_features(sig: np.ndarray):
    N = len(sig)
    feats = []
    for order in FRFT_ORDERS:
        K = get_frft_matrix(N, order)
        X = K @ sig.astype(np.complex128)
        mag = np.abs(X)
        feats.append(float(np.log1p(np.sum(mag ** 2))))
        feats.append(float(np.log1p(np.max(mag))))
    mag1 = np.abs(get_frft_matrix(N, 1.0) @ sig.astype(np.complex128))
    feats.append(float(np.sum(mag1[: N // 4]) / (np.sum(mag1) + 1e-8)))
    feats.append(float(np.sum(mag1[N // 4: N // 2]) / (np.sum(mag1) + 1e-8)))
    v = np.array(feats, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


def compute_stft_features(sig: np.ndarray):
    f, t_, Zxx = stft(sig.astype(np.float64), fs=FS, window="hamming", nperseg=STFT_NPERSEG, noverlap=STFT_NPERSEG // 2, nfft=STFT_NPERSEG, boundary="zeros", padded=True)
    mag = np.abs(Zxx)
    m = mag.mean(axis=1)
    m = np.log1p(m)
    if len(m) < N_SCALES:
        m = np.pad(m, (0, N_SCALES - len(m)))
    else:
        m = m[:N_SCALES]
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return m


_B_OP, _A_OP = butter(3, [75 / (FS / 2), 300 / (FS / 2)], btype="band")


def compute_op_features(sig: np.ndarray):
    try:
        op = filtfilt(_B_OP, _A_OP, sig.astype(np.float64))
    except Exception:
        op = np.zeros_like(sig, dtype=np.float64)
    total_e = float(np.sum(sig ** 2) + 1e-8)
    op_e = float(np.sum(op ** 2))
    ratio = op_e / total_e
    op_abs = np.abs(op)
    peak_e = float(np.max(op_abs))
    rms = float(np.sqrt(np.mean(op ** 2)))
    zc = int(np.sum(np.diff(np.signbit(op))))
    env = np.abs(op)
    env_peaks = int(np.sum((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:])))
    q75 = float(np.quantile(op_abs, 0.75))
    q25 = float(np.quantile(op_abs, 0.25))
    feats = np.array([ratio, peak_e, rms, float(zc) / len(sig), float(env_peaks) / len(sig), q75, q25, float(np.std(op))], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def precompute_bio_features(bags):
    stft_cache = {}
    frft_cache = {}
    op_cache = {}
    for bag in bags:
        for comp in bag.components:
            cid = comp.global_component_id
            if cid not in stft_cache:
                sig = comp.signal.astype(np.float64)
                stft_cache[cid] = compute_stft_features(sig)
                frft_cache[cid] = compute_frft_features(sig)
                op_cache[cid] = compute_op_features(sig)
    return stft_cache, frft_cache, op_cache


class MultiDomainBio(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)
        self.signal_cnn = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.signal_proj = nn.Linear(64, d_model)
        self.ot_mlp = nn.Sequential(nn.Linear(135, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d_model))
        self.spectral_mlp = nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, d_model))
        self.vmd_mlp = nn.Sequential(nn.Linear(80, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d_model))
        self.physical_mlp = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d_model))
        self.cwt_cnn = nn.Sequential(nn.Conv1d(N_SCALES, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.cwt_proj = nn.Linear(32, d_model)
        self.stft_mlp = nn.Sequential(nn.Linear(N_SCALES, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, d_model))
        self.frft_mlp = nn.Sequential(nn.Linear(12, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, d_model))
        self.op_mlp = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, d_model))
        self.gate = nn.Sequential(nn.Linear(d_model * 9, 9), nn.Softmax(dim=-1))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_scorer = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, signal, vmask, ot, spectral, vmd, physical, cwt, stft_f, frft_f, op_f, comp_mask=None):
        B, L = signal.shape[:2]
        sig_flat = signal.reshape(B * L, 1, -1)
        sig_feat = self.signal_proj(self.signal_cnn(sig_flat).squeeze(-1))
        ot_feat = self.ot_mlp(ot.reshape(B * L, -1))
        spec_feat = self.spectral_mlp(spectral.reshape(B * L, -1))
        vmd_feat = self.vmd_mlp(vmd.reshape(B * L, -1))
        phys_feat = self.physical_mlp(physical.reshape(B * L, -1))
        cwt_flat = cwt.reshape(B * L, N_SCALES, -1)
        cwt_feat = self.cwt_proj(self.cwt_cnn(cwt_flat).squeeze(-1))
        stft_feat = self.stft_mlp(stft_f.reshape(B * L, -1))
        frft_feat = self.frft_mlp(frft_f.reshape(B * L, -1))
        op_feat = self.op_mlp(op_f.reshape(B * L, -1))
        concat = torch.cat([sig_feat, ot_feat, spec_feat, vmd_feat, phys_feat, cwt_feat, stft_feat, frft_feat, op_feat], dim=-1)
        weights = self.gate(concat)
        fused = (weights[:, 0:1] * sig_feat + weights[:, 1:2] * ot_feat + weights[:, 2:3] * spec_feat + weights[:, 3:4] * vmd_feat + weights[:, 4:5] * phys_feat + weights[:, 5:6] * cwt_feat + weights[:, 6:7] * stft_feat + weights[:, 7:8] * frft_feat + weights[:, 8:9] * op_feat)
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


def collate_bio(bags, dataset, scal_cache, stft_cache, frft_cache, op_cache):
    B = len(bags)
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    valid_mask = np.zeros((B, L, 128), dtype=bool)
    ot = np.zeros((B, L, 135), dtype=np.float32)
    physical = np.zeros((B, L, 8), dtype=np.float32)
    spectral = np.zeros((B, L, 10), dtype=np.float32)
    vmd = np.zeros((B, L, 80), dtype=np.float32)
    cwt = np.zeros((B, L, N_SCALES, 128), dtype=np.float32)
    stft_f = np.zeros((B, L, N_SCALES), dtype=np.float32)
    frft_f = np.zeros((B, L, 12), dtype=np.float32)
    op_f = np.zeros((B, L, 8), dtype=np.float32)
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
            cid = comp.global_component_id
            if cid in scal_cache:
                cwt[i, j, :, :] = scal_cache[cid]
            if cid in stft_cache:
                stft_f[i, j, :] = stft_cache[cid]
            if cid in frft_cache:
                frft_f[i, j, :] = frft_cache[cid]
            if cid in op_cache:
                op_f[i, j, :] = op_cache[cid]
            comp_mask[i, j] = True
    return {"signal": torch.as_tensor(signal), "valid_mask": torch.as_tensor(valid_mask), "ot": torch.as_tensor(ot), "physical": torch.as_tensor(physical), "spectral": torch.as_tensor(spectral), "vmd": torch.as_tensor(vmd), "cwt": torch.as_tensor(cwt), "stft_f": torch.as_tensor(stft_f), "frft_f": torch.as_tensor(frft_f), "op_f": torch.as_tensor(op_f), "comp_mask": torch.as_tensor(comp_mask), "label": torch.as_tensor(labels)}


def eval_auc(model, dataset, scal_cache, stft_cache, frft_cache, op_cache, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_bio([bag], dataset, scal_cache, stft_cache, frft_cache, op_cache)
        with torch.no_grad():
            logit, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device), batch["ot"].to(device), batch["spectral"].to(device), batch["vmd"].to(device), batch["physical"].to(device), batch["cwt"].to(device), batch["stft_f"].to(device), batch["frft_f"].to(device), batch["op_f"].to(device), batch["comp_mask"].to(device))
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train_model(model, train_bags, val_bags, dataset, scal_cache, stft_cache, frft_cache, op_cache, seed, device, lr=1e-4):
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
            batch = collate_bio(bags_batch, dataset, scal_cache, stft_cache, frft_cache, op_cache)
            labels_b = batch["label"].to(device)
            logits, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device), batch["ot"].to(device), batch["spectral"].to(device), batch["vmd"].to(device), batch["physical"].to(device), batch["cwt"].to(device), batch["stft_f"].to(device), batch["frft_f"].to(device), batch["op_f"].to(device), batch["comp_mask"].to(device))
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, dataset, scal_cache, stft_cache, frft_cache, op_cache, val_bags, device)
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


def predict_model(model, dataset, scal_cache, stft_cache, frft_cache, op_cache, bags, device):
    model.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_bio([bag], dataset, scal_cache, stft_cache, frft_cache, op_cache)
        with torch.no_grad():
            logit, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device), batch["ot"].to(device), batch["spectral"].to(device), batch["vmd"].to(device), batch["physical"].to(device), batch["cwt"].to(device), batch["stft_f"].to(device), batch["frft_f"].to(device), batch["op_f"].to(device), batch["comp_mask"].to(device))
        rows.append({"unit_id": bag.unit_id, "subject_id": bag.subject_id, "target": int(bag.target_binary), "probability": float(torch.sigmoid(logit[0]).item())})
    return pd.DataFrame(rows)


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)
    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}\n  {task} — Bio Extended 9-Domain Fusion (STFT+FrFT+OP)\n{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        ds = MultidomainERGDataset(bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)
        from scripts.run_cwt_erg import precompute_scalograms
        scal_cache = precompute_scalograms(bags, N_SCALES)
        print(f"  CWT: {len(scal_cache)} scalograms")
        stft_cache, frft_cache, op_cache = precompute_bio_features(bags)
        print(f"  Bio features: STFT {len(stft_cache)}, FrFT {len(frft_cache)}, OP {len(op_cache)}")
        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)"); continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = MultiDomainBio(seed=seed)
                log = train_model(model, train_bags, test_bags, ds, scal_cache, stft_cache, frft_cache, op_cache, seed, DEVICE)
                pred = predict_model(model, ds, scal_cache, stft_cache, frft_cache, op_cache, test_bags, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, pred["probability"].values)
                print(f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={len(pred)} best_epoch={log['best_epoch']}")
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

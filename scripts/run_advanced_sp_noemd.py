"""Advanced SP + MultiDomain for PERG (no EMD, faster)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import zarr
from scipy.signal import welch
from scipy.stats import skew, kurtosis
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_advanced_sp import (
    hjorth_parameters, teager_kaiser_energy, sample_entropy,
    permutation_entropy, higuchi_fd, welch_psd_features,
    higher_order_stats, zero_crossing_rate, autocorrelation_features,
    FEAT_DIM, AdvancedSPClassifier, bootstrap_auroc,
)
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset, collate_multidomain
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/advanced_sp_v1")

def compute_no_emd(signal_1d):
    sig = signal_1d.copy()
    nan_mask = np.isnan(sig)
    if nan_mask.all(): return np.zeros(FEAT_DIM, dtype=np.float32)
    if nan_mask.any():
        good = np.where(~nan_mask)[0]
        sig[nan_mask] = np.interp(np.where(nan_mask)[0], good, sig[good])
    base = 3+3+1+1+1+5+3+1+4  # without EMD (10)
    feats = np.concatenate([
        hjorth_parameters(sig), teager_kaiser_energy(sig), sample_entropy(sig),
        permutation_entropy(sig), higuchi_fd(sig), welch_psd_features(sig),
        higher_order_stats(sig), zero_crossing_rate(sig), autocorrelation_features(sig),
    ])
    return feats.astype(np.float32)

NO_EMD_DIM = 22

class AdvSPNoEMD(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.signal_proj = nn.Linear(64, d_model)
        self.ot_mlp = nn.Sequential(nn.Linear(135, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d_model))
        self.spectral_mlp = nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, d_model))
        self.advsp_mlp = nn.Sequential(nn.Linear(NO_EMD_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, d_model))
        self.physical_mlp = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d_model))
        self.gate = nn.Sequential(nn.Linear(d_model * 5, 5), nn.Softmax(dim=-1))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_scorer = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, signal, vmask, ot, spectral, physical, advsp, comp_mask=None):
        B, L = signal.shape[:2]
        sig_feat = self.signal_proj(self.signal_cnn(signal.reshape(B*L,1,-1)).squeeze(-1))
        ot_feat = self.ot_mlp(ot.reshape(B*L,-1))
        spec_feat = self.spectral_mlp(spectral.reshape(B*L,-1))
        advsp_feat = self.advsp_mlp(advsp.reshape(B*L,-1))
        phys_feat = self.physical_mlp(physical.reshape(B*L,-1))
        concat = torch.cat([sig_feat, ot_feat, spec_feat, advsp_feat, phys_feat], dim=-1)
        weights = self.gate(concat)
        fused = weights[:,0:1]*sig_feat + weights[:,1:2]*ot_feat + weights[:,2:3]*spec_feat + weights[:,3:4]*advsp_feat + weights[:,4:5]*phys_feat
        tokens = fused.reshape(B,L,-1)
        if comp_mask is None: comp_mask = torch.ones(B,L,dtype=torch.bool,device=tokens.device)
        tokens = self.transformer(tokens)
        attn = self.attn_scorer(tokens).squeeze(-1).masked_fill(~comp_mask, float("-inf"))
        attn_w = F.softmax(attn, dim=-1).masked_fill(~comp_mask, 0.0)
        pooled = (tokens * attn_w.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn_w

def collate_advsp(bags, advsp_cache, spectral_vecs, comp_idx_map):
    B = len(bags); L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B,L,1,128), dtype=np.float32)
    valid_mask = np.zeros((B,L,128), dtype=bool)
    ot = np.zeros((B,L,135), dtype=np.float32)
    physical = np.zeros((B,L,8), dtype=np.float32)
    spectral = np.zeros((B,L,10), dtype=np.float32)
    advsp = np.zeros((B,L,NO_EMD_DIM), dtype=np.float32)
    comp_mask = np.zeros((B,L), dtype=bool)
    labels = np.full(B, np.nan, dtype=np.float64)
    for i, bag in enumerate(bags):
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        for j, comp in enumerate(bag.components):
            signal[i,j,0,:] = comp.signal
            valid_mask[i,j,:] = comp.signal_mask
            ot[i,j,:] = comp.ot_vector
            physical[i,j,:] = comp.physical
            key = comp.global_component_id
            if key in advsp_cache: advsp[i,j,:] = advsp_cache[key]
            idx = comp_idx_map.get(key)
            if idx is not None and idx < len(spectral_vecs): spectral[i,j,:] = spectral_vecs[idx]
            comp_mask[i,j] = True
    return {"signal": torch.as_tensor(signal), "valid_mask": torch.as_tensor(valid_mask), "ot": torch.as_tensor(ot),
            "physical": torch.as_tensor(physical), "spectral": torch.as_tensor(spectral), "advsp": torch.as_tensor(advsp),
            "comp_mask": torch.as_tensor(comp_mask), "label": torch.as_tensor(labels)}

def eval_auc(model, advsp_cache, spectral_vecs, comp_idx_map, bags, device):
    model.eval(); y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None: continue
        batch = collate_advsp([bag], advsp_cache, spectral_vecs, comp_idx_map)
        with torch.no_grad():
            logit, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device), batch["ot"].to(device),
                             batch["spectral"].to(device), batch["physical"].to(device), batch["advsp"].to(device), batch["comp_mask"].to(device))
        y_true.append(bag.target_binary); y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true)<2 or len(set(y_true))<2: return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))

def train_model(model, train_bags, val_bags, advsp_cache, spectral_vecs, comp_idx_map, seed, device, lr=1e-4):
    model.to(device)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags}, batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(sampler.bags)//8); total = 200*steps_per_epoch; warm = 5*steps_per_epoch
    sched = _WarmupCosine(optimizer, warmup_steps=warm, total_steps=total, min_frac=0.05)
    best_auc = -1.0; best_state = None; patience = 0
    model.train()
    for epoch in range(200):
        for step, idx in enumerate(sampler):
            if step >= steps_per_epoch: break
            batch = collate_advsp([sampler.bags[i] for i in idx], advsp_cache, spectral_vecs, comp_idx_map)
            labels_b = batch["label"].to(device)
            logits, _ = model(batch["signal"].to(device), batch["valid_mask"].to(device), batch["ot"].to(device),
                              batch["spectral"].to(device), batch["physical"].to(device), batch["advsp"].to(device), batch["comp_mask"].to(device))
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); sched.step()
        val_auc = eval_auc(model, advsp_cache, spectral_vecs, comp_idx_map, val_bags, device)
        if val_auc > best_auc: best_auc = val_auc; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= 25: break
    if best_state: model.load_state_dict(best_state)
    return {"best_epoch": epoch, "best_val_auc": best_auc}

from pathway_erg.evaluation.metrics import roc_auc_score

def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    for task in ["LEOP", "PERG"]:
        bags = build_task_bags(caches, task, "primary_nine_step")
        comp_idx = 0; comp_idx_map = {}
        for bag in bags:
            for comp in bag.components:
                if comp.global_component_id not in comp_idx_map:
                    comp_idx_map[comp.global_component_id] = comp_idx; comp_idx += 1
        advsp_cache = {}
        for bag in bags:
            for comp in bag.components:
                key = comp.global_component_id
                if key not in advsp_cache: advsp_cache[key] = compute_no_emd(comp.signal)
        print(f"{task}: {len(advsp_cache)} advsp vectors (no EMD, dim={NO_EMD_DIM})")

        for seed in SEEDS:
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists(): continue
                train_bags, test_bags = outer_partition(bags, outer_fold)
                model = AdvSPNoEMD(seed=seed)
                log = train_model(model, train_bags, test_bags, advsp_cache, spectral_vecs, comp_idx_map, seed, DEVICE)
                model.eval(); y_true, y_prob = [], []
                for bag in test_bags:
                    if bag.target_binary is None: continue
                    batch = collate_advsp([bag], advsp_cache, spectral_vecs, comp_idx_map)
                    with torch.no_grad():
                        logit, _ = model(batch["signal"].to(DEVICE), batch["valid_mask"].to(DEVICE), batch["ot"].to(DEVICE),
                                         batch["spectral"].to(DEVICE), batch["physical"].to(DEVICE), batch["advsp"].to(DEVICE), batch["comp_mask"].to(DEVICE))
                    y_true.append(bag.target_binary); y_prob.append(float(torch.sigmoid(logit[0]).item()))
                pt, lo, hi = bootstrap_auroc(np.array(y_true), np.array(y_prob))
                print(f"  {task} fold {outer_fold} seed {seed}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")
                run_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"target": y_true, "probability": y_prob}).to_parquet(run_dir / "predictions.parquet", index=False)

    for task in ["LEOP", "PERG"]:
        aucs = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p); pt, _, _ = bootstrap_auroc(df["target"].values, df["probability"].values); aucs.append(pt)
        if aucs:
            by_fold = [np.mean([aucs[i] for i in range(f, len(aucs), 5)]) for f in range(5)]
            print(f"\n  {task} per-fold mean: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")

if __name__ == "__main__":
    main()

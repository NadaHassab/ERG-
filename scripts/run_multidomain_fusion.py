"""Approach 1: Multi-Domain Fusion ERG Classifier

Combines ALL available features:
- Time domain: signal (128,) + mask (128,)
- OT domain: signed OT (135,)
- Spectral domain: FFT features (10,) — cached, not used before
- VMD domain: VMD features (80,) — cached, not used before
- Physical features: (8,)
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
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches, BagUnit, ComponentRow, build_bags
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.signal.vmd_cache import load_vmd_cache, vmd_cache_paths
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/multidomain_v1")


class MultiDomainFusion(nn.Module):
    """Multi-branch model fusing signal, OT, spectral, VMD, and physical features."""

    def __init__(
        self,
        signal_dim: int = 128,
        ot_dim: int = 135,
        spectral_dim: int = 10,
        vmd_dim: int = 80,
        physical_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 1001,
    ):
        super().__init__()
        torch.manual_seed(seed)

        # Branch 1: CNN on raw signal
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.signal_proj = nn.Linear(64, d_model)

        # Branch 2: MLP on OT features
        self.ot_mlp = nn.Sequential(
            nn.Linear(ot_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )

        # Branch 3: MLP on spectral features
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, d_model),
        )

        # Branch 4: MLP on VMD features
        self.vmd_mlp = nn.Sequential(
            nn.Linear(vmd_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )

        # Branch 5: MLP on physical features
        self.physical_mlp = nn.Sequential(
            nn.Linear(physical_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, d_model),
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 5, 5),
            nn.Softmax(dim=-1),
        )

        # Transformer self-attention over fused tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Attention pooling
        self.attn_scorer = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode_component(self, signal, vmask, ot, spectral, vmd, physical):
        """Encode a single component from all domains."""
        B = signal.shape[0]

        # Branch 1: CNN on signal (B, 1, 128) -> (B, d_model)
        sig = signal.unsqueeze(1) if signal.dim() == 2 else signal
        cnn_out = self.signal_cnn(sig).squeeze(-1)  # (B, 64)
        sig_feat = self.signal_proj(cnn_out)  # (B, d_model)

        # Branch 2: OT (B, 135) -> (B, d_model)
        ot_feat = self.ot_mlp(ot)

        # Branch 3: Spectral (B, 10) -> (B, d_model)
        spec_feat = self.spectral_mlp(spectral)

        # Branch 4: VMD (B, 80) -> (B, d_model)
        vmd_feat = self.vmd_mlp(vmd)

        # Branch 5: Physical (B, 8) -> (B, d_model)
        phys_feat = self.physical_mlp(physical)

        # Gated fusion
        concat = torch.cat([sig_feat, ot_feat, spec_feat, vmd_feat, phys_feat], dim=-1)
        weights = self.gate(concat)  # (B, 5)
        fused = (
            weights[:, 0:1] * sig_feat
            + weights[:, 1:2] * ot_feat
            + weights[:, 2:3] * spec_feat
            + weights[:, 3:4] * vmd_feat
            + weights[:, 4:5] * phys_feat
        )  # (B, d_model)

        return fused

    def forward(self, signal, vmask, ot, spectral, vmd, physical, comp_mask=None):
        """
        Args:
            signal: (B, L, 1, 128) or (B, 1, 128)
            vmask: (B, L, 128) or (B, 128)
            ot: (B, L, 135)
            spectral: (B, L, 10)
            vmd: (B, L, 80)
            physical: (B, L, 8)
            comp_mask: (B, L) bool, True for real components
        """
        if signal.dim() == 4:
            B, L = signal.shape[:2]
            signal = signal.squeeze(2)  # (B, L, 128)
        else:
            B, L = signal.shape[:2]

        # Flatten batch and length
        signal_flat = signal.reshape(B * L, -1)
        vmask_flat = vmask.reshape(B * L, -1)
        ot_flat = ot.reshape(B * L, -1)
        spec_flat = spectral.reshape(B * L, -1)
        vmd_flat = vmd.reshape(B * L, -1)
        phys_flat = physical.reshape(B * L, -1)

        # Encode each component
        fused = self.encode_component(signal_flat, vmask_flat, ot_flat, spec_flat, vmd_flat, phys_flat)
        fused = fused.reshape(B, L, -1)  # (B, L, d_model)

        # Transformer
        if comp_mask is None:
            comp_mask = torch.ones(B, L, dtype=torch.bool, device=fused.device)
        fused = self.transformer(fused)

        # Attention pooling
        attn_scores = self.attn_scorer(fused).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(~comp_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~comp_mask, 0.0)
        pooled = (fused * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        logits = self.head(pooled).squeeze(-1)  # (B,)
        return logits, attn_weights


class MultidomainERGDataset:
    """Dataset that loads spectral + VMD features alongside signal/OT/physical."""

    def __init__(self, bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names):
        self.bags = bags
        self.spectral_vecs = spectral_vecs
        self.vmd_vecs = vmd_vecs
        self.spectral_names = spectral_names
        self.vmd_names = vmd_names
        # Map global_component_id -> index in cache
        self._comp_idx = {}
        for i, bag in enumerate(bags):
            for comp in bag.components:
                if comp.global_component_id not in self._comp_idx:
                    self._comp_idx[comp.global_component_id] = len(self._comp_idx)

    def get_spectral(self, comp: ComponentRow) -> np.ndarray:
        idx = self._comp_idx.get(comp.global_component_id)
        if idx is None or idx >= len(self.spectral_vecs):
            return np.zeros(len(self.spectral_names), dtype=np.float32)
        return self.spectral_vecs[idx]

    def get_vmd(self, comp: ComponentRow) -> np.ndarray:
        idx = self._comp_idx.get(comp.global_component_id)
        if idx is None or idx >= len(self.vmd_vecs):
            return np.zeros(len(self.vmd_names), dtype=np.float32)
        return self.vmd_vecs[idx]


def collate_multidomain(bags, dataset):
    """Collate bags with all feature domains."""
    B = len(bags)
    L = max(len(bag.components) for bag in bags)

    signal = np.zeros((B, L, 1, 128), dtype=np.float32)
    valid_mask = np.zeros((B, L, 128), dtype=bool)
    ot = np.zeros((B, L, 135), dtype=np.float32)
    physical = np.zeros((B, L, 8), dtype=np.float32)
    spectral = np.zeros((B, L, 10), dtype=np.float32)
    vmd = np.zeros((B, L, 80), dtype=np.float32)
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
            comp_mask[i, j] = True

    return {
        "signal": torch.as_tensor(signal),
        "valid_mask": torch.as_tensor(valid_mask),
        "ot": torch.as_tensor(ot),
        "physical": torch.as_tensor(physical),
        "spectral": torch.as_tensor(spectral),
        "vmd": torch.as_tensor(vmd),
        "comp_mask": torch.as_tensor(comp_mask),
        "label": torch.as_tensor(labels),
    }


def eval_auc(model, dataset, bags, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_multidomain([bag], dataset)
        with torch.no_grad():
            logit, _ = model(
                batch["signal"].to(device),
                batch["valid_mask"].to(device),
                batch["ot"].to(device),
                batch["spectral"].to(device),
                batch["vmd"].to(device),
                batch["physical"].to(device),
                batch["comp_mask"].to(device),
            )
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(np.array(y_true), np.array(y_prob)))


def train(model, train_bags, val_bags, dataset, task, seed, device, lr=1e-4):
    model.to(device)
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

    model.train()
    for epoch in range(200):
        total_loss = 0.0
        n = 0
        for step, idx in enumerate(sampler):
            if step >= steps_per_epoch:
                break
            bags_batch = [sampler.bags[i] for i in idx]
            batch = collate_multidomain(bags_batch, dataset)
            sig = batch["signal"].to(device)
            vmask = batch["valid_mask"].to(device)
            ot_b = batch["ot"].to(device)
            spec_b = batch["spectral"].to(device)
            vmd_b = batch["vmd"].to(device)
            phys_b = batch["physical"].to(device)
            cmask = batch["comp_mask"].to(device)
            labels_b = batch["label"].to(device)

            logits, _ = model(sig, vmask, ot_b, spec_b, vmd_b, phys_b, cmask)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            total_loss += float(loss.item()) * len(labels_b)
            n += len(labels_b)

        val_auc = eval_auc(model, dataset, val_bags, device)
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


def predict(model, dataset, bags, device):
    model.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_multidomain([bag], dataset)
        with torch.no_grad():
            logit, _ = model(
                batch["signal"].to(device),
                batch["valid_mask"].to(device),
                batch["ot"].to(device),
                batch["spectral"].to(device),
                batch["vmd"].to(device),
                batch["physical"].to(device),
                batch["comp_mask"].to(device),
            )
        rows.append({
            "unit_id": bag.unit_id,
            "subject_id": bag.subject_id,
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


def load_extra_features(artifact_root):
    """Load spectral and VMD features from cache."""
    from pathway_erg.signal.component_cache import load_cache_manifest, CACHE_SCHEMA_VERSION

    root = Path(artifact_root)
    manifest = load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    extra = manifest["extra"]

    # Spectral features
    spectral_zarr_path = root / "data" / "arrays" / "spectral_features_v4.zarr"
    spectral_z = zarr.open_group(str(spectral_zarr_path), mode="r")
    spectral_vecs = np.asarray(spectral_z["components"]["spectral_vector"][:])
    spectral_names = extra.get("spectral_feature_names", [f"spec_{i}" for i in range(spectral_vecs.shape[1])])

    # VMD features
    from pathway_erg.signal.vmd import VMDConfig
    vmd_cfg = VMDConfig()
    pre_cfg_hash = extra["config_hash"]
    vmd_vecs, vmd_names = load_vmd_cache(root, vmd_cfg, pre_cfg_hash)

    return spectral_vecs, vmd_vecs, spectral_names, vmd_names


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    # Load extra features
    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)
    print(f"Loaded spectral: {spectral_vecs.shape}, VMD: {vmd_vecs.shape}")

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Multi-Domain Fusion")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                pred_file = run_dir / "predictions.parquet"
                if pred_file.exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue

                train_bags, test_bags = outer_partition(bags, outer_fold)
                ds = MultidomainERGDataset(bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)

                model = MultiDomainFusion(seed=seed)
                log = train(model, train_bags, test_bags, ds, task, seed, DEVICE)
                pred = predict(model, ds, test_bags, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(
                    pred["target"].values, pred["probability"].values
                )
                print(
                    f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                    f"n={len(pred)} best_epoch={log['best_epoch']}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(pred_file, index=False)

    # Summary
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
            by_fold = []
            for fold in range(5):
                fold_aucs = [aucs[i] for i in range(fold, len(aucs), 5)]
                by_fold.append(np.mean(fold_aucs))
            print(f"\n  {task} per-fold mean: {np.mean(by_fold):.4f} ± {np.std(by_fold):.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

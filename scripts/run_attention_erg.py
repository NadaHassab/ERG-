"""Direction 3: Attention-based ERG classifier.

1D CNN + self-attention over ERG waveforms. Attention weights show which
parts of the waveform and which components matter for classification.
Same nested CV protocol as the neural baselines.
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine


# ── Model ─────────────────────────────────────────────────────────────────
class AttentionERGClassifier(nn.Module):
    """1D CNN + Transformer attention over ERG component waveforms."""

    def __init__(
        self,
        signal_len: int = 128,
        cnn_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 1001,
    ):
        super().__init__()
        torch.manual_seed(seed)

        # 1D CNN over raw waveform (signal_len=128)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (B*L, cnn_channels, 1)
        )
        self.cnn_proj = nn.Linear(cnn_channels, d_model)

        # Transformer self-attention over components
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Attention scorer (separate from classification head)
        self.attn_scorer = nn.Linear(d_model, 1)

    def forward(self, signal: torch.Tensor, valid_mask: torch.Tensor):
        """
        signal: (B, L, 1, T) — per-component waveforms
        valid_mask: (B, L, T) — time-point validity

        Returns: (B,) logits, (B, L) attention_weights
        """
        B, L, _, T = signal.shape

        # CNN per component
        x = signal.squeeze(2)  # (B, L, T)
        x = x.view(B * L, 1, T)  # (B*L, 1, T)
        x = self.cnn(x).squeeze(-1)  # (B*L, cnn_channels)
        x = self.cnn_proj(x)  # (B*L, d_model)

        # Mask invalid components
        comp_valid = valid_mask.any(dim=-1).float()  # (B, L) — 1 if any time point valid
        x = x.view(B, L, -1)  # (B, L, d_model)

        # Zero out invalid components
        x = x * comp_valid.unsqueeze(-1)

        # Transformer attention
        attn_mask = (comp_valid == 0)  # True = padded (ignored by attention)
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Attention-weighted pooling
        attn_weights = self.attn_scorer(x).squeeze(-1)  # (B, L) raw scores
        attn_weights = attn_weights.masked_fill(comp_valid == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, L)
        attn_weights = attn_weights.masked_fill(comp_valid == 0, 0.0)

        # Weighted sum
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        logits = self.head(pooled).squeeze(-1)  # (B,)

        return logits, attn_weights


# ── Training ──────────────────────────────────────────────────────────────
def train_one_task(model, train_bags, val_bags, task, seed, device="cuda"):
    model.to(device)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

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
            batch = collate_bag_units(bags_batch)
            sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
            vmask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
            labels_b = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)

            logits, _ = model(sig, vmask)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            total_loss += float(loss.item()) * len(labels_b)
            n += len(labels_b)

        # Validate
        val_auc = eval_auc(model, val_bags, task, device)
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


def eval_auc(model, bags, task, device):
    model.eval()
    y_true, y_prob = [], []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vmask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad():
            logit, _ = model(sig, vmask)
        y_true.append(bag.target_binary)
        y_prob.append(float(torch.sigmoid(logit[0]).item()))
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(np.array(y_true), np.array(y_prob)))
    except ValueError:
        return 0.5


def predict_bags(model, bags, task, device="cuda"):
    model.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None:
            continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vmask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad():
            logit, attn = model(sig, vmask)
        rows.append({
            "unit_id": bag.unit_id,
            "subject_id": bag.subject_id,
            "target": int(bag.target_binary),
            "logit": float(logit[0]),
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
            scores.append(float("nan"))
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    scores = scores[np.isfinite(scores)]
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR = Path("artifacts/results/attention_erg_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [1001, 2002, 3003]
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Attention ERG Classifier")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                train_bags, test_bags = outer_partition(bags, outer_fold)

                model = AttentionERGClassifier(seed=seed)
                log = train_one_task(model, train_bags, test_bags, task, seed, DEVICE)

                pred = predict_bags(model, test_bags, task, DEVICE)
                point, ci_lo, ci_hi = bootstrap_auroc(
                    pred["target"].values, pred["probability"].values
                )
                print(
                    f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                    f"n={len(pred)} best_epoch={log['best_epoch']}"
                )

                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(run_dir / "predictions.parquet", index=False)

        # Ensemble
        all_preds = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    df["outer_fold"] = fold
                    df["seed"] = seed
                    all_preds.append(df)

        if all_preds:
            all_df = pd.concat(all_preds, ignore_index=True)
            ensemble = all_df.groupby("unit_id").agg(
                target=("target", "first"),
                probability=("probability", "mean"),
                subject_id=("subject_id", "first"),
            ).reset_index()
            point, ci_lo, ci_hi = bootstrap_auroc(
                ensemble["target"].values, ensemble["probability"].values
            )
            print(f"\n  --- {task} ENSEMBLE ---")
            print(f"  AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={len(ensemble)}")

            ensemble.to_parquet(OUT_DIR / f"{task.lower()}_ensemble.parquet", index=False)

    # Save metrics
    metrics = {}
    for task in ["LEOP", "PERG"]:
        p = OUT_DIR / f"{task.lower()}_ensemble.parquet"
        if p.exists():
            ens = pd.read_parquet(p)
            pt, lo, hi = bootstrap_auroc(ens["target"].values, ens["probability"].values)
            metrics[task] = {"point": pt, "ci_low": lo, "ci_high": hi, "n": len(ens)}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

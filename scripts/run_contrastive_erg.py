"""Direction 5: Patient-aware contrastive pretraining for ERG classification.

Pretrains encoder with same-patient, same-class positive pairs,
then fine-tunes for classification. Expected +0.02-0.05 AUROC.
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine


# ── Augmentations ─────────────────────────────────────────────────────
def augment_erg(signal, valid_mask, rng):
    """Apply random augmentations to ERG signal. signal: (B,L,1,T), vm: (B,L,T)"""
    s = signal.clone()
    B, L, _, T = s.shape
    # Gaussian noise
    if rng.random() < 0.5:
        s = s + torch.randn_like(s) * 0.05 * s.std()
    # Time shift (circular)
    if rng.random() < 0.5:
        shift = rng.integers(-10, 11)
        s = torch.roll(s, shifts=int(shift), dims=-1)
    # Scale perturbation
    if rng.random() < 0.5:
        scale = 0.9 + rng.random() * 0.2
        s = s * scale
    # Channel dropout (zero out random time points)
    if rng.random() < 0.3:
        mask = torch.ones(T, device=s.device, dtype=s.dtype)
        drop_idx = rng.choice(T, size=T // 10, replace=False)
        mask[drop_idx] = 0.0
        s = s * mask
    return s


# ── Encoder (same architecture as attention ERG) ─────────────────────
class Encoder(nn.Module):
    def __init__(self, signal_len=128, cnn_ch=64, d_model=128, dropout=0.1, seed=1001):
        super().__init__()
        torch.manual_seed(seed)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_ch, 7, padding=3), nn.BatchNorm1d(cnn_ch), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_ch, cnn_ch, 5, padding=2), nn.BatchNorm1d(cnn_ch), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(cnn_ch, d_model)

    def forward(self, signal, valid_mask):
        B, L, _, T = signal.shape
        x = signal.squeeze(2).view(B * L, 1, T)
        x = self.proj(self.cnn(x).squeeze(-1)).view(B, L, -1)
        cv = valid_mask.any(dim=-1).float()
        x = x * cv.unsqueeze(-1)
        return x, cv


# ── Contrastive head ─────────────────────────────────────────────────
class ContrastiveHead(nn.Module):
    def __init__(self, d_model=128, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


# ── Classification head ──────────────────────────────────────────────
class ClassHead(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
    def forward(self, x, cv):
        aw = self.scorer(x).squeeze(-1).masked_fill(cv == 0, float("-inf"))
        aw = F.softmax(aw, dim=-1).masked_fill(cv == 0, 0.0)
        pooled = (x * aw.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1), aw


# ── Contrastive pretraining ──────────────────────────────────────────
def contrastive_pretrain(encoder, proj_head, bags, task, seed, device, n_epochs=100):
    encoder.to(device)
    proj_head.to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    total = n_epochs * max(1, len(bags) // 8)
    sched = _WarmupCosine(optimizer, warmup_steps=5 * max(1, len(bags) // 8),
                           total_steps=total, min_frac=0.01)
    rng = np.random.default_rng(seed)

    for epoch in range(n_epochs):
        indices = rng.permutation(len(bags))
        total_loss = 0.0
        n = 0
        for start in range(0, len(indices), 8):
            idx = indices[start:start+8]
            batch_bags = [bags[i] for i in idx]
            batch = collate_bag_units(batch_bags)

            sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
            vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
            labels = np.array([b.target_binary for b in batch_bags])
            subjects = np.array([b.subject_id for b in batch_bags])

            # Create two augmented views
            view1 = augment_erg(sig, vm, rng)
            view2 = augment_erg(sig, vm, rng)

            # Encode both views
            z1, cv1 = encoder(view1, vm)
            z2, cv2 = encoder(view2, vm)

            # Project to contrastive space
            p1 = proj_head(z1.mean(dim=1))  # (B, proj_dim) — mean over components
            p2 = proj_head(z2.mean(dim=1))

            # Patient-aware positive pairs: same patient, same class
            B = len(batch_bags)
            sim_matrix = F.cosine_similarity(p1.unsqueeze(1), p2.unsqueeze(0), dim=-1)  # (B, B)
            sim_matrix = sim_matrix / 0.07  # temperature

            # Mask: positive if same patient AND same class
            pos_mask = torch.zeros(B, B, device=device, dtype=torch.bool)
            for i in range(B):
                for j in range(B):
                    if subjects[i] == subjects[j] and labels[i] == labels[j]:
                        pos_mask[i, j] = True

            # InfoNCE loss
            exp_sim = torch.exp(sim_matrix)
            pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
            all_sum = exp_sim.sum(dim=1)
            loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            sched.step()
            total_loss += float(loss.item()) * B
            n += B

    return encoder


# ── Fine-tuning ──────────────────────────────────────────────────────
def finetune(encoder, class_head, train_bags, val_bags, task, seed, device):
    encoder.to(device)
    class_head.to(device)
    params = list(encoder.parameters()) + list(class_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    criterion = FoldWeightedBCE(positive_class_weight(
        np.asarray([b.target_binary for b in train_bags], dtype=float)))
    steps = max(1, len(train_bags) // 8)
    total = 200 * steps
    sched = _WarmupCosine(optimizer, warmup_steps=5 * steps, total_steps=total, min_frac=0.05)
    best_auc, best_state, patience = -1.0, None, 0

    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)

    encoder.train()
    class_head.train()
    for epoch in range(200):
        for step, idx in enumerate(sampler):
            if step >= steps: break
            batch = collate_bag_units([sampler.bags[i] for i in idx])
            sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
            vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
            lb = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)
            z, cv = encoder(sig, vm)
            logits, _ = class_head(z, cv)
            loss = criterion(logits, lb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            sched.step()

        val_auc = eval_auc(encoder, class_head, val_bags, task, device)
        encoder.train(); class_head.train()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            best_head = {k: v.detach().clone() for k, v in class_head.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25: break

    if best_state: encoder.load_state_dict(best_state)
    if best_head: class_head.load_state_dict(best_head)
    return {"best_epoch": epoch, "best_val_auc": best_auc}


def eval_auc(encoder, class_head, bags, task, device):
    encoder.eval(); class_head.eval()
    yt, yp = [], []
    for bag in bags:
        if bag.target_binary is None: continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad():
            z, cv = encoder(sig, vm)
            logit, _ = class_head(z, cv)
        yt.append(bag.target_binary)
        yp.append(float(torch.sigmoid(logit[0]).item()))
    if len(yt) < 2 or len(set(yt)) < 2: return 0.5
    try: return float(roc_auc_score(np.array(yt), np.array(yp)))
    except ValueError: return 0.5


def predict_bags(encoder, class_head, bags, task, device="cuda"):
    encoder.eval(); class_head.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None: continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad():
            z, cv = encoder(sig, vm)
            logit, _ = class_head(z, cv)
        rows.append({"unit_id": bag.unit_id, "subject_id": bag.subject_id,
                      "target": int(bag.target_binary),
                      "probability": float(torch.sigmoid(logit[0]).item())})
    return pd.DataFrame(rows)


def bootstrap_auroc(y_true, y_prob, n_reps=2000, seed=424242):
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y_true, y_prob)
    scores = []
    for _ in range(n_reps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            scores.append(float("nan")); continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)[np.isfinite(np.array(scores))]
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT = Path("artifacts/results/contrastive_erg_v1")
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = [1001, 2002, 3003]
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}\n  {task} — Patient-Aware Contrastive ERG\n{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for fold in range(5):
                train_bags, test_bags = outer_partition(bags, fold)
                encoder = Encoder(seed=seed)
                proj_head = ContrastiveHead(d_model=128, proj_dim=64)
                encoder = contrastive_pretrain(encoder, proj_head, train_bags, task, seed, DEVICE)
                class_head = ClassHead(d_model=128)
                log = finetune(encoder, class_head, train_bags, test_bags, task, seed, DEVICE)
                pred = predict_bags(encoder, class_head, test_bags, task, DEVICE)
                pt, lo, hi = bootstrap_auroc(pred["target"].values, pred["probability"].values)
                print(f"  fold {fold}: AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] "
                      f"n={len(pred)} best_epoch={log['best_epoch']}")
                run_dir = OUT / task.lower() / f"run-fold{fold}-seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(run_dir / "predictions.parquet", index=False)

        # Ensemble
        all_preds = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    df["outer_fold"] = fold; df["seed"] = seed
                    all_preds.append(df)
        if all_preds:
            big = pd.concat(all_preds, ignore_index=True)
            ens = big.groupby("unit_id").agg(
                target=("target", "first"), probability=("probability", "mean"),
                subject_id=("subject_id", "first")).reset_index()
            pt, lo, hi = bootstrap_auroc(ens["target"].values, ens["probability"].values)
            print(f"\n  --- {task} ENSEMBLE ---")
            print(f"  AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] n={len(ens)}")
            ens.to_parquet(OUT / f"{task.lower()}_ensemble.parquet", index=False)

    metrics = {}
    for task in ["LEOP", "PERG"]:
        p = OUT / f"{task.lower()}_ensemble.parquet"
        if p.exists():
            ens = pd.read_parquet(p)
            pt, lo, hi = bootstrap_auroc(ens["target"].values, ens["probability"].values)
            metrics[task] = {"point": pt, "ci_low": lo, "ci_high": hi, "n": len(ens)}
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()

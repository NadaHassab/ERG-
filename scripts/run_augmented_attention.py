"""Direction 6: Attention ERG with training augmentation + larger ensemble."""
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
from scripts.run_attention_erg import AttentionERGClassifier


def augment_batch(batch, rng):
    """Apply augmentations to a collated batch."""
    sig = torch.as_tensor(batch["signal"], dtype=torch.float32)
    vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool)
    # Gaussian noise
    if rng.random() < 0.5:
        sig = sig + torch.randn_like(sig) * 0.03 * sig.std()
    # Time shift
    if rng.random() < 0.5:
        shift = rng.integers(-5, 6)
        sig = torch.roll(sig, shifts=int(shift), dims=-1)
    # Scale
    if rng.random() < 0.5:
        scale = 0.95 + rng.random() * 0.1
        sig = sig * scale
    batch["signal"] = sig.numpy()
    return batch


def train_one_task(model, train_bags, val_bags, task, seed, device="cuda"):
    model.to(device)
    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = max(1, len(sampler.bags) // 8)
    total = 200 * steps
    sched = _WarmupCosine(optimizer, warmup_steps=5*steps, total_steps=total, min_frac=0.05)
    best_auc, best_state, patience = -1.0, None, 0
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(200):
        for step, idx in enumerate(sampler):
            if step >= steps: break
            batch = collate_bag_units([sampler.bags[i] for i in idx])
            batch = augment_batch(batch, rng)
            sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
            vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
            lb = torch.as_tensor(batch["label"], dtype=torch.float32, device=device)
            logits, _ = model(sig, vm)
            loss = criterion(logits, lb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
        val_auc = eval_auc(model, val_bags, task, device)
        model.train()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25: break
    if best_state: model.load_state_dict(best_state)
    return {"best_epoch": epoch, "best_val_auc": best_auc}


def eval_auc(model, bags, task, device):
    model.eval()
    yt, yp = [], []
    for bag in bags:
        if bag.target_binary is None: continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad(): logit, _ = model(sig, vm)
        yt.append(bag.target_binary)
        yp.append(float(torch.sigmoid(logit[0]).item()))
    if len(yt) < 2 or len(set(yt)) < 2: return 0.5
    try: return float(roc_auc_score(np.array(yt), np.array(yp)))
    except ValueError: return 0.5


def predict_bags(model, bags, task, device="cuda"):
    model.eval()
    rows = []
    for bag in bags:
        if bag.target_binary is None: continue
        batch = collate_bag_units([bag])
        sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        vm = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        with torch.no_grad(): logit, _ = model(sig, vm)
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
    OUT = Path("artifacts/results/aug_attention_v1")
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS = [1001, 2002, 3003, 4004, 5005]
    DEVICE = "cuda"
    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}\n  {task} — Augmented Attention ERG\n{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for fold in range(5):
                train_bags, test_bags = outer_partition(bags, fold)
                model = AttentionERGClassifier(seed=seed)
                log = train_one_task(model, train_bags, test_bags, task, seed, DEVICE)
                pred = predict_bags(model, test_bags, task, DEVICE)
                pt, lo, hi = bootstrap_auroc(pred["target"].values, pred["probability"].values)
                print(f"  fold {fold}: AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] n={len(pred)} best_epoch={log['best_epoch']}")
                run_dir = OUT / task.lower() / f"run-fold{fold}-seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(run_dir / "predictions.parquet", index=False)
        all_preds = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p); df["outer_fold"] = fold; df["seed"] = seed
                    all_preds.append(df)
        if all_preds:
            big = pd.concat(all_preds, ignore_index=True)
            ens = big.groupby("unit_id").agg(target=("target","first"), probability=("probability","mean"), subject_id=("subject_id","first")).reset_index()
            pt, lo, hi = bootstrap_auroc(ens["target"].values, ens["probability"].values)
            print(f"\n  --- {task} ENSEMBLE ---\n  AUROC={pt:.4f} [{lo:.4f}, {hi:.4f}] n={len(ens)}")
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

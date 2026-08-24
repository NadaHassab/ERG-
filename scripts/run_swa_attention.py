from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine

from scripts.run_attention_erg import AttentionERGClassifier, eval_auc, predict_bags, bootstrap_auroc


SWA_START_EPOCH = 120
MAX_EPOCHS = 200
PATIENCE = 25


def train_one_task(model, train_bags, val_bags, task, seed, device="cuda"):
    model.to(device)
    swa_model = AveragedModel(model)

    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    steps_per_epoch = max(1, len(sampler.bags) // 8)
    warm_steps = 5 * steps_per_epoch
    pre_swa_total = SWA_START_EPOCH * steps_per_epoch
    sched = _WarmupCosine(optimizer, warmup_steps=warm_steps, total_steps=pre_swa_total, min_frac=0.05)

    best_auc = -1.0
    best_state = None
    patience = 0
    swa_started = False
    swa_sched = None

    model.train()
    for epoch in range(MAX_EPOCHS):
        if epoch == SWA_START_EPOCH:
            swa_started = True
            swa_sched = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=int(0.1 * steps_per_epoch), anneal_strategy="cos")

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

            if swa_started:
                swa_model.update_parameters(model)
                if swa_sched is not None:
                    swa_sched.step()
            else:
                sched.step()

            total_loss += float(loss.item()) * len(labels_b)
            n += len(labels_b)

        val_auc = eval_auc(model, val_bags, task, device)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    if swa_started:
        def _bn_loader(bags, bs=64):
            for i in range(0, len(bags), bs):
                batch = collate_bag_units(bags[i:i+bs])
                sig = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
                vmask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
                yield sig, vmask
        torch.optim.swa_utils.update_bn(_bn_loader(train_bags), swa_model, device=device)
        if best_state is not None:
            model.load_state_dict(best_state)
        model_dict = model.state_dict()
        swa_dict = swa_model.module.state_dict()
        for k in model_dict:
            if k in swa_dict:
                model_dict[k] = swa_dict[k]
        model.load_state_dict(model_dict)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return {"best_epoch": epoch, "best_val_auc": best_auc}


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR = Path("artifacts/results/swa_attention_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [1001, 2002, 3003]
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — SWA Attention ERG Classifier")
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

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_attention_erg import (
    AttentionERGClassifier,
    train_one_task,
    eval_auc,
    predict_bags,
    bootstrap_auroc,
)
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.training.trainer import _WarmupCosine


def pretrain_on_task(model, bags, task, seed, device="cuda"):
    model.to(device)
    sampler = BagSampler(bags, folds={b.outer_fold for b in bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    steps_per_epoch = max(1, len(sampler.bags) // 8)
    total = 200 * steps_per_epoch
    warm = 5 * steps_per_epoch
    sched = _WarmupCosine(optimizer, warmup_steps=warm, total_steps=total, min_frac=0.05)

    model.train()
    for epoch in range(200):
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

    return model


def extract_encoder_weights(model):
    state = model.state_dict()
    return {k: v.clone() for k, v in state.items()
            if not k.startswith("head.") and not k.startswith("attn_scorer.")}


def load_pretrained_weights(target_model, pretrained_encoder):
    state = target_model.state_dict()
    for k, v in pretrained_encoder.items():
        if k in state:
            state[k] = v
    target_model.load_state_dict(state)


def finetune_with_pretrained(pretrained_encoder, train_bags, val_bags, task, seed, device="cuda"):
    model = AttentionERGClassifier(seed=seed)
    load_pretrained_weights(model, pretrained_encoder)
    model.to(device)

    head_params = list(model.head.parameters()) + list(model.attn_scorer.parameters())
    encoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("head.") and not n.startswith("attn_scorer.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4},
    ])

    sampler = BagSampler(train_bags, folds={b.outer_fold for b in train_bags},
                         batch_size=8, seed=seed)
    labels = np.asarray([b.target_binary for b in sampler.bags], dtype=float)
    criterion = FoldWeightedBCE(positive_class_weight(labels))

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
    return model, {"best_epoch": epoch, "best_val_auc": best_auc}


def run_experiment(source_task, target_task, bags_source, bags_target, seeds, device):
    print(f"\n  Pretraining on {source_task} (all {len(bags_source)} bags, all seeds)...")

    pretrained_encoders = {}
    for seed in seeds:
        src_model = AttentionERGClassifier(seed=seed)
        src_model = pretrain_on_task(src_model, bags_source, source_task, seed, device)
        pretrained_encoders[seed] = extract_encoder_weights(src_model)
        print(f"  Pretraining complete for seed {seed}")

    results = {"transfer": [], "baseline": []}

    for seed in seeds:
        print(f"\n  Seed {seed} \u2014 {source_task} -> {target_task}")

        for fold in range(5):
            train_bags, test_bags = outer_partition(bags_target, fold)

            model_ft, log_ft = finetune_with_pretrained(
                pretrained_encoders[seed], train_bags, test_bags, target_task, seed, device
            )
            pred_ft = predict_bags(model_ft, test_bags, target_task, device)
            pt_ft, lo_ft, hi_ft = bootstrap_auroc(
                pred_ft["target"].values, pred_ft["probability"].values
            )
            results["transfer"].append({
                "seed": seed, "fold": fold, "source": source_task,
                "target": target_task, "auroc": pt_ft, "ci_lo": lo_ft,
                "ci_hi": hi_ft, "n": len(pred_ft),
                "best_epoch": log_ft["best_epoch"],
            })
            print(
                f"    fold {fold}: TRANSFER AUROC={pt_ft:.4f} [{lo_ft:.4f}, {hi_ft:.4f}] "
                f"n={len(pred_ft)} best_epoch={log_ft['best_epoch']}"
            )

            model_bl = AttentionERGClassifier(seed=seed)
            log_bl = train_one_task(model_bl, train_bags, test_bags, target_task, seed, device)
            pred_bl = predict_bags(model_bl, test_bags, target_task, device)
            pt_bl, lo_bl, hi_bl = bootstrap_auroc(
                pred_bl["target"].values, pred_bl["probability"].values
            )
            results["baseline"].append({
                "seed": seed, "fold": fold, "source": None,
                "target": target_task, "auroc": pt_bl, "ci_lo": lo_bl,
                "ci_hi": hi_bl, "n": len(pred_bl),
                "best_epoch": log_bl["best_epoch"],
            })
            print(
                f"    fold {fold}: BASELINE AUROC={pt_bl:.4f} [{lo_bl:.4f}, {hi_bl:.4f}] "
                f"n={len(pred_bl)} best_epoch={log_bl['best_epoch']}"
            )

    return results


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR = Path("artifacts/results/crosstask_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEEDS = [1001, 2002, 3003]
    DEVICE = "cuda"

    bags_perg = build_task_bags(caches, "PERG", "primary_nine_step")
    bags_leop = build_task_bags(caches, "LEOP", "primary_nine_step")

    print(f"PERG bags: {len(bags_perg)}, LEOP bags: {len(bags_leop)}")

    all_results = {}

    print(f"\n{'='*60}")
    print("  Experiment 1: Pretrain PERG -> Fine-tune LEOP")
    print(f"{'='*60}")
    r1 = run_experiment("PERG", "LEOP", bags_perg, bags_leop, SEEDS, DEVICE)
    all_results["PERG_to_LEOP"] = r1

    print(f"\n{'='*60}")
    print("  Experiment 2: Pretrain LEOP -> Fine-tune PERG")
    print(f"{'='*60}")
    r2 = run_experiment("LEOP", "PERG", bags_leop, bags_perg, SEEDS, DEVICE)
    all_results["LEOP_to_PERG"] = r2

    summary = {}
    for direction, res in all_results.items():
        transfer_aurocs = [r["auroc"] for r in res["transfer"]]
        baseline_aurocs = [r["auroc"] for r in res["baseline"]]
        summary[direction] = {
            "transfer": {
                "mean_auroc": float(np.mean(transfer_aurocs)),
                "std_auroc": float(np.std(transfer_aurocs)),
                "all_aurocs": transfer_aurocs,
            },
            "baseline": {
                "mean_auroc": float(np.mean(baseline_aurocs)),
                "std_auroc": float(np.std(baseline_aurocs)),
                "all_aurocs": baseline_aurocs,
            },
            "delta": float(np.mean(transfer_aurocs) - np.mean(baseline_aurocs)),
        }
        print(f"\n  {direction}:")
        print(f"    Transfer:  {summary[direction]['transfer']['mean_auroc']:.4f} "
              f"\u00b1 {summary[direction]['transfer']['std_auroc']:.4f}")
        print(f"    Baseline:  {summary[direction]['baseline']['mean_auroc']:.4f} "
              f"\u00b1 {summary[direction]['baseline']['std_auroc']:.4f}")
        print(f"    Delta:     {summary[direction]['delta']:+.4f}")

    for direction, res in all_results.items():
        df_transfer = pd.DataFrame(res["transfer"])
        df_baseline = pd.DataFrame(res["baseline"])
        df_transfer.to_parquet(OUT_DIR / f"{direction}_transfer.parquet", index=False)
        df_baseline.to_parquet(OUT_DIR / f"{direction}_baseline.parquet", index=False)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    all_frames = []
    for direction, res in all_results.items():
        for r in res["transfer"]:
            r["direction"] = direction
            all_frames.append(r)
        for r in res["baseline"]:
            r["direction"] = direction
            all_frames.append(r)
    pd.DataFrame(all_frames).to_parquet(OUT_DIR / "all_results.parquet", index=False)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

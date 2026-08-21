#!/usr/bin/env python3
"""Embedding classifiers: frozen encoder → classical classifier.

Extracts unit-level embeddings from the frozen encoder (e6 checkpoints),
then fits classical classifiers (logreg, SVM, gradient boosting) using
nested cross-validation. Reports patient-level AUROC with clustered
bootstrap CI, matching the neural evaluation protocol.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathway_erg.data.datasets import LoadedCaches, BagUnit
from pathway_erg.training.separate import build_task_bags, outer_partition as ep_outer_partition
from pathway_erg.config import load_config, DataConfig
from pathway_erg.training.separate import SeparateTrainingConfig, build_stage_model
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.evaluation.probes import load_model_from_checkpoint
from pathway_erg.data.collate import collate_bag_units
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "LEOP": {
        "e6": "artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-leop-fold{fold}-seed{seed}/final.pt",
        "e7c": "artifacts/results/separate_raw_ot_hierarchical_sslinit_unfrozen_v1/runs/separate_raw_ot_hierarchical_sslinit_unfrozen_v1-leop-fold{fold}-seed{seed}/final.pt",
    },
    "PERG": {
        "e6": "artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-perg-fold{fold}-seed{seed}/final.pt",
        "e7c": "artifacts/results/separate_raw_ot_hierarchical_sslinit_unfrozen_v1/runs/separate_raw_ot_hierarchical_sslinit_unfrozen_v1-perg-fold{fold}-seed{seed}/final.pt",
    },
}
SEEDS = [1001, 2002, 3003]
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 424242
DATA_CFG_PATH = "configs/data/local.yaml"

CLASSIFIERS = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ]),
    "histgb": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    ),
}


# ── Embedding extraction ──────────────────────────────────────────────────
def extract_embeddings(model, bags, task, batch_size=8):
    """Extract unit-level embeddings from the model's encode_bag layer."""
    model.eval()
    embeddings = []
    unit_ids = []
    subject_ids = []
    targets = []
    for start in range(0, len(bags), batch_size):
        chunk = bags[start:start + batch_size]
        batch = collate_bag_units(chunk)
        with torch.no_grad():
            enc = model.encode_bag(batch, task)
            tok = enc.token.detach().cpu().numpy()
        for i, bag in enumerate(chunk):
            embeddings.append(tok[i])
            unit_ids.append(bag.unit_id)
            subject_ids.append(bag.subject_id)
            targets.append(int(bag.target_binary))
    return (
        np.array(embeddings),
        np.array(unit_ids, dtype=object),
        np.array(subject_ids, dtype=object),
        np.array(targets),
    )


def extract_component_embeddings(model, bags, task, batch_size=8):
    """Extract per-component embeddings (before aggregation) for mean pooling."""
    model.eval()
    from pathway_erg.data.collate import collate_bag_units
    embeddings = []
    unit_ids = []
    subject_ids = []
    targets = []
    for start in range(0, len(bags), batch_size):
        chunk = bags[start:start + batch_size]
        batch = collate_bag_units(chunk)
        with torch.no_grad():
            enc = model.encode_component(batch)
            tok = enc.token[:, 0, :].detach().cpu().numpy()  # (N_components, dim)
        # Group by unit_id
        batch_unit_ids = np.array([b.unit_id for b in chunk], dtype=object)
        unique_units = np.unique(batch_unit_ids)
        for uid in unique_units:
            mask = batch_unit_ids == uid
            pooled = tok[mask].mean(axis=0)  # mean pooling over components
            embeddings.append(pooled)
            unit_ids.append(uid)
            # find subject and target
            idx = np.where(mask)[0][0]
            subject_ids.append(chunk[idx].subject_id)
            targets.append(int(chunk[idx].target_binary))
    return (
        np.array(embeddings),
        np.array(unit_ids, dtype=object),
        np.array(subject_ids, dtype=object),
        np.array(targets),
    )


# ── Bootstrap CI ──────────────────────────────────────────────────────────
def bootstrap_auroc(y_true, y_prob, n_reps=2000, seed=424242):
    """Clustered bootstrap AUROC CI (resampling clusters, not rows)."""
    rng = np.random.default_rng(seed)
    clusters = np.unique(np.arange(len(y_true)))
    point = roc_auc_score(y_true, y_prob)
    scores = []
    for _ in range(n_reps):
        idx = rng.choice(clusters, size=len(clusters), replace=True)
        # expand cluster indices to row indices
        row_idx = np.repeat(idx, 1)  # each cluster = 1 row (unit level)
        if len(np.unique(y_true[row_idx])) < 2:
            scores.append(float("nan"))
            continue
        scores.append(roc_auc_score(y_true[row_idx], y_prob[row_idx]))
    scores = np.array(scores)
    scores = scores[np.isfinite(scores)]
    ci_low = float(np.percentile(scores, 2.5))
    ci_high = float(np.percentile(scores, 97.5))
    return point, ci_low, ci_high


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    data_cfg = load_config(DataConfig, DATA_CFG_PATH)
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")

    all_results = {}

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task}")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        for ckpt_version in ["e6", "e7c"]:
            print(f"\n  --- {ckpt_version} embeddings ---")
            task_results = {clf_name: [] for clf_name in CLASSIFIERS}

            for outer_fold in range(5):
                outer_train, outer_test = ep_outer_partition(bags, outer_fold)

                # Extract embeddings for train and test
                all_embeddings = []
                all_unit_ids = []
                all_subject_ids = []
                all_targets = []

                # Use all 3 seeds, average embeddings
                for seed in SEEDS:
                    ckpt_path = CHECKPOINTS[task][ckpt_version].format(fold=outer_fold, seed=seed)
                    if not Path(ckpt_path).exists():
                        print(f"  [WARN] checkpoint missing: {ckpt_path}")
                        continue
                    model, cfg, _ = load_model_from_checkpoint(ckpt_path)

                    emb_train, uid_train, sid_train, tgt_train = extract_embeddings(
                        model, outer_train, task
                    )
                    emb_test, uid_test, sid_test, tgt_test = extract_embeddings(
                        model, outer_test, task
                    )

                    all_embeddings.append((emb_train, emb_test))
                    all_unit_ids.append((uid_train, uid_test))
                    all_subject_ids.append((sid_train, sid_test))
                    all_targets.append((tgt_train, tgt_test))

                if not all_embeddings:
                    print(f"  [SKIP] fold {outer_fold}: no checkpoints found")
                    continue

                # Average across seeds
                emb_train_avg = np.mean([e[0] for e in all_embeddings], axis=0)
                emb_test_avg = np.mean([e[1] for e in all_embeddings], axis=0)
                tgt_train = all_targets[0][0]
                tgt_test = all_targets[0][1]

                for clf_name, clf in CLASSIFIERS.items():
                    from sklearn.base import clone
                    clf_fold = clone(clf)
                    clf_fold.fit(emb_train_avg, tgt_train)
                    y_prob = clf_fold.predict_proba(emb_test_avg)[:, 1]
                    point, ci_low, ci_high = bootstrap_auroc(
                        tgt_test, y_prob, n_reps=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
                    )
                    task_results[clf_name].append({
                        "fold": outer_fold,
                        "point": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n": len(tgt_test),
                    })

            # Summary across folds
            print(f"\n  --- {task} {ckpt_version} summary ---")
            summary_key = f"{task}_{ckpt_version}"
            all_results[summary_key] = {}
            for clf_name, fold_results in task_results.items():
                if not fold_results:
                    continue
                # Weighted average across folds
                total_n = sum(r["n"] for r in fold_results)
                weighted_point = sum(r["point"] * r["n"] for r in fold_results) / total_n
                weighted_ci_low = sum(r["ci_low"] * r["n"] for r in fold_results) / total_n
                weighted_ci_high = sum(r["ci_high"] * r["n"] for r in fold_results) / total_n
                all_results[summary_key][clf_name] = {
                    "point": weighted_point,
                    "ci_low": weighted_ci_low,
                    "ci_high": weighted_ci_high,
                    "n": total_n,
                    "folds": fold_results,
                }
                print(
                    f"  {clf_name:>8s}: AUROC={weighted_point:.4f} "
                    f"[{weighted_ci_low:.4f}, {weighted_ci_high:.4f}] "
                    f"n={total_n}"
                )

    # Save results
    out_dir = Path("artifacts/results/embedding_classifiers_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

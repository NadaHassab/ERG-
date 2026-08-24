"""UMAP-based ERG classification pipeline.

Based on Anwar et al. (2026) "UMAP-ERG: A dimensionality reduction pipeline
for electroretinogram data analysis"

Approach: Extract multi-domain features → UMAP reduction → SVM/LogReg classifier.
No data leakage: UMAP fit on training data only, transform on test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches, ComponentRow
from pathway_erg.training.separate import build_task_bags, outer_partition
from scripts.run_multidomain_fusion import load_extra_features

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/umap_v1")
N_SCALES = 16


def extract_component_features(comp, spectral_vecs_dict, vmd_vecs_dict, scal_cache):
    """Extract flat feature vector for a single component."""
    sig = comp.signal.astype(np.float32)
    features = []

    # Signal statistics (4 features)
    features.extend([
        np.mean(sig), np.std(sig),
        np.max(sig) - np.min(sig),
        np.sum(np.abs(np.diff(sig))),
    ])

    # OT (135 features)
    features.extend(comp.ot_vector.astype(np.float32).tolist())

    # Physical (8 features)
    features.extend(comp.physical.astype(np.float32).tolist())

    # Spectral (10 features)
    sid = comp.global_component_id
    if sid in spectral_vecs_dict:
        features.extend(spectral_vecs_dict[sid].tolist())
    else:
        features.extend([0.0] * 10)

    # VMD (80 features)
    if sid in vmd_vecs_dict:
        features.extend(vmd_vecs_dict[sid].tolist())
    else:
        features.extend([0.0] * 80)

    # CWT summary stats (N_SCALES features)
    if sid in scal_cache:
        cwt = scal_cache[sid]
        features.extend([float(np.mean(cwt[s])) for s in range(N_SCALES)])
    else:
        features.extend([0.0] * N_SCALES)

    return np.array(features, dtype=np.float32)


def build_bag_features(bag, spectral_vecs_dict, vmd_vecs_dict, scal_cache):
    """Average component features across bag."""
    comp_feats = []
    for comp in bag.components:
        comp_feats.append(extract_component_features(comp, spectral_vecs_dict, vmd_vecs_dict, scal_cache))
    return np.mean(comp_feats, axis=0)


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)
    spectral_vecs_dict = {comp.global_component_id: spectral_vecs[i]
                          for i, bag in enumerate(caches.leop_bags)
                          for comp in bag.components} if hasattr(caches, 'leop_bags') else {}
    vmd_vecs_dict = {comp.global_component_id: vmd_vecs[i]
                     for i, bag in enumerate(caches.leop_bags)
                     for comp in bag.components} if hasattr(caches, 'leop_bags') else {}

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — UMAP-based Classification")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        # Build spectral/vmd dicts for this task
        task_spec_dict = {}
        task_vmd_dict = {}
        for i, bag in enumerate(bags):
            for comp in bag.components:
                cid = comp.global_component_id
                if cid not in task_spec_dict and i < len(spectral_vecs):
                    task_spec_dict[cid] = spectral_vecs[i]
                if cid not in task_vmd_dict and i < len(vmd_vecs):
                    task_vmd_dict[cid] = vmd_vecs[i]

        # Precompute CWT
        from scripts.run_cwt_erg import precompute_scalograms
        scal_cache = precompute_scalograms(bags, N_SCALES)

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue

                train_bags, test_bags = outer_partition(bags, outer_fold)

                # Extract features
                X_train = np.array([build_bag_features(b, task_spec_dict, task_vmd_dict, scal_cache)
                                    for b in train_bags])
                y_train = np.array([b.target_binary for b in train_bags])
                X_test = np.array([build_bag_features(b, task_spec_dict, task_vmd_dict, scal_cache)
                                   for b in test_bags])
                y_test = np.array([b.target_binary for b in test_bags])

                # UMAP + SVM pipeline
                if HAS_UMAP:
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("umap", umap.UMAP(n_components=min(20, len(X_train) - 2),
                                           random_state=seed, metric="euclidean")),
                        ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
                    ])
                else:
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
                    ])

                pipe.fit(X_train, y_train)
                probs = pipe.predict_proba(X_test)[:, 1]
                pt = float(roc_auc_score(y_test, probs))
                print(f"  fold {outer_fold}: AUROC={pt:.4f} n_test={len(y_test)}")

                run_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame({
                    "bag_id": [b.unit_id for b in test_bags],
                    "target": y_test,
                    "probability": probs,
                })
                df.to_parquet(run_dir / "predictions.parquet", index=False)

    # Summary
    for task in ["LEOP", "PERG"]:
        aucs = []
        for fold in range(5):
            for seed in SEEDS:
                p = OUT_DIR / task.lower() / f"run-fold{fold}-seed{seed}/predictions.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    pt = float(roc_auc_score(df["target"].values, df["probability"].values))
                    aucs.append(pt)
        if aucs:
            by_fold = [np.mean([aucs[i] for i in range(f, len(aucs), 5)]) for f in range(5)]
            print(f"\n  {task} per-fold mean: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

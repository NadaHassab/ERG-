"""Graph Signal Processing features for ERG classification.

Based on "ERG-Graph: a graph signal processing approach" (2025)
Transform ERG waveforms into graph representations and extract
topological features (centrality, clustering, algebraic connectivity).

No data leakage: graph construction from training data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches, ComponentRow
from pathway_erg.training.separate import build_task_bags, outer_partition
from pathway_erg.evaluation.metrics import roc_auc_score

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    HAS_SK = True
except ImportError:
    HAS_SK = False

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/gsp_v1")


def signal_to_graph(sig, k=5, sigma=1.0):
    """Convert 1D signal to adjacency matrix using k-NN graph with Gaussian kernel.

    Each sample point is a node, edges weighted by similarity.
    """
    n = len(sig)
    sig_2d = sig.reshape(-1, 1).astype(np.float64)
    dists = np.abs(sig_2d - sig_2d.T)

    # k-NN graph
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:k+1]
        for j in neighbors:
            w = np.exp(-dists[i, j]**2 / (2 * sigma**2))
            adj[i, j] = w
            adj[j, i] = w
    return adj


def graph_features(adj):
    """Extract graph topological features from adjacency matrix."""
    n = adj.shape[0]
    features = {}

    # Degree and strength
    degree = adj.sum(axis=1)
    features["mean_degree"] = float(np.mean(degree))
    features["std_degree"] = float(np.std(degree))
    features["max_degree"] = float(np.max(degree))

    # Clustering coefficient (average over nodes)
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            continue
        k = len(neighbors)
        sub_adj = adj[np.ix_(neighbors, neighbors)]
        triangles = np.sum(sub_adj > 0) / 2
        clustering[i] = 2 * triangles / (k * (k - 1)) if k > 1 else 0
    features["mean_clustering"] = float(np.mean(clustering))

    # Centrality (degree-based approximation)
    features["mean_centrality"] = float(np.mean(degree / (n - 1))) if n > 1 else 0

    # Algebraic connectivity (Fiedler value)
    try:
        deg_matrix = np.diag(degree)
        laplacian = deg_matrix - adj
        eigenvalues = np.sort(np.real(eigs(laplacian, k=2, which="SM")[0]))
        features["algebraic_connectivity"] = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
        features["spectral_gap"] = float(eigenvalues[-1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0
    except Exception:
        features["algebraic_connectivity"] = 0.0
        features["spectral_gap"] = 0.0

    # Edge density
    max_edges = n * (n - 1) / 2
    features["edge_density"] = float(np.sum(adj > 0) / 2 / max_edges) if max_edges > 0 else 0

    # Average edge weight
    triu = adj[np.triu_indices(n, k=1)]
    features["mean_edge_weight"] = float(np.mean(triu[triu > 0])) if np.any(triu > 0) else 0

    # Entropy of degree distribution
    deg_norm = degree / degree.sum() if degree.sum() > 0 else np.ones(n) / n
    deg_norm = deg_norm[deg_norm > 0]
    features["degree_entropy"] = float(-np.sum(deg_norm * np.log(deg_norm + 1e-10)))

    # Modularity proxy: variance of within-community vs between-community
    features["degree_assortativity"] = float(np.corrcoef(degree, degree)[0, 1]) if n > 2 else 0

    return features


def extract_gsp_features(bag):
    """Extract GSP features for a bag (average over components)."""
    all_features = []
    for comp in bag.components:
        sig = comp.signal.astype(np.float64)
        # Subsample for graph construction (128 points too many for full graph)
        # Use first 32 points for tractable graph construction
        sig_sub = sig[::4][:32]
        adj = signal_to_graph(sig_sub, k=4, sigma=1.0)
        feats = graph_features(adj)
        all_features.append(feats)

    # Average features across components
    keys = all_features[0].keys()
    result = {}
    for k in keys:
        result[k] = float(np.mean([f[k] for f in all_features]))
    return result


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — Graph Signal Processing Features")
        print(f"{'='*60}")

        bags = build_task_bags(caches, task, "primary_nine_step")

        # Precompute GSP features for all bags
        print("  Precomputing GSP features...")
        gsp_feats = {}
        for i, bag in enumerate(bags):
            gsp_feats[bag.unit_id] = extract_gsp_features(bag)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(bags)} bags processed")

        # Convert to matrix
        feat_names = list(list(gsp_feats.values())[0].keys())
        X_all = np.array([[gsp_feats[b.unit_id][k] for k in feat_names]
                          for b in bags])
        y_all = np.array([b.target_binary for b in bags])
        print(f"  Features shape: {X_all.shape}, feature names: {feat_names}")

        for seed in SEEDS:
            print(f"\n  --- seed {seed} ---")
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "predictions.parquet").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue

                train_idx = [i for i, b in enumerate(bags) if b.outer_fold != outer_fold]
                test_idx = [i for i, b in enumerate(bags) if b.outer_fold == outer_fold]

                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_test, y_test = X_all[test_idx], y_all[test_idx]

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
                ])
                pipe.fit(X_train, y_train)
                probs = pipe.predict_proba(X_test)[:, 1]

                if len(set(y_test)) < 2:
                    print(f"  fold {outer_fold}: single class (skip)")
                    continue
                pt = float(roc_auc_score(y_test, probs))
                print(f"  fold {outer_fold}: AUROC={pt:.4f} n_test={len(y_test)}")

                run_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame({
                    "bag_id": [bags[i].unit_id for i in test_idx],
                    "target": y_test, "probability": probs,
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

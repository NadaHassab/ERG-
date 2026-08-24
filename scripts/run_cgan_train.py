"""Train MultiDomain+CWT on CGAN-augmented data.

Loads synthetic signals from run_cgan_erg.py and adds them as additional
training bags. Synthetic bags copy OT/physical from nearest real bag to
avoid zero-feature collapse. No data leakage: synthetic signals generated
from training data only, test data never used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multidomain_cwt import (
    MultiDomainCWT, collate_multidomain_cwt, train_model, predict_model,
    bootstrap_auroc,
)
from scripts.run_multidomain_fusion import load_extra_features, MultidomainERGDataset
from scripts.run_cwt_erg import precompute_scalograms
from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches, BagUnit, ComponentRow
from pathway_erg.training.separate import build_task_bags, outer_partition

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/cgan_aug_trained_v1")
N_SCALES = 16
SIGNAL_LEN = 128
MAX_SYNTH_PER_CLASS = 100  # Keep small to avoid overwhelming real data


def create_synthetic_bags(synth_signals, synth_labels, real_bags, task):
    """Create synthetic BagUnits copying OT/physical from nearest real bag.

    Synthetic bags keep the synthetic waveform but borrow feature vectors
    from real bags so the model sees non-zero domain features.
    """
    from scipy.spatial.distance import cdist

    # Collect real component signals and features
    real_sigs = []
    real_ot = []
    real_physical = []
    for bag in real_bags:
        for comp in bag.components:
            real_sigs.append(comp.signal.astype(np.float32))
            real_ot.append(comp.ot_vector.astype(np.float32))
            real_physical.append(comp.physical.astype(np.float32))
    real_sigs = np.array(real_sigs)
    real_ot = np.array(real_ot)
    real_physical = np.array(real_physical)

    bags = []
    for i, (sig, label) in enumerate(zip(synth_signals, synth_labels)):
        # Find nearest real component by waveform similarity
        dists = np.linalg.norm(real_sigs - sig[np.newaxis, :].astype(np.float32), axis=1)
        nn_idx = int(np.argmin(dists))

        comp = ComponentRow(
            global_component_id=f"synth_{task}_{i}",
            global_recording_id=f"synth_rec_{i}",
            subject_id=f"synth_subj_{i}",
            visit_id=None, dataset=task,
            component_id="L_A_TO_B" if task == "LEOP" else "P_EARLY",
            unit_id=f"synth_unit_{i}", protocol="synthetic",
            eye=None, stimulus_value=0.0, stimulus_unit="",
            landmark_confidence=1.0, outer_fold=0,
            signal=sig.astype(np.float64),
            signal_mask=np.ones(SIGNAL_LEN, dtype=bool),
            ot_vector=real_ot[nn_idx].astype(np.float64),
            physical=real_physical[nn_idx].astype(np.float64),
        )
        bag = BagUnit(
            unit_id=f"synth_{task}_{i}",
            subject_id=f"synth_subj_{i}",
            visit_id=None, dataset=task,
            target_binary=int(label), outer_fold=0,
            components=(comp,),
        )
        bags.append(bag)
    return bags


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    augmented_dir = Path("artifacts/results/cgan_aug_v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    spectral_vecs, vmd_vecs, spectral_names, vmd_names = load_extra_features(data_cfg.artifact_root)

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — MultiDomain+CWT on CGAN-Augmented Data")
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

                synth_dir = augmented_dir / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                synth_signals_path = synth_dir / "synth_signals.npy"
                synth_labels_path = synth_dir / "synth_labels.npy"

                if not synth_signals_path.exists():
                    print(f"  fold {outer_fold}: NO SYNTHETIC DATA (skip)")
                    continue

                synth_signals_all = np.load(synth_signals_path)
                synth_labels_all = np.load(synth_labels_path)

                train_bags, test_bags = outer_partition(bags, outer_fold)

                # Cap synthetic count per class to MAX_SYNTH_PER_CLASS
                keep_idx = []
                for cls in [0, 1]:
                    cls_idx = np.where(synth_labels_all == cls)[0]
                    if len(cls_idx) > MAX_SYNTH_PER_CLASS:
                        cls_idx = np.random.RandomState(seed).choice(cls_idx, MAX_SYNTH_PER_CLASS, replace=False)
                    keep_idx.extend(cls_idx.tolist())
                keep_idx = sorted(keep_idx)
                synth_signals = synth_signals_all[keep_idx]
                synth_labels = synth_labels_all[keep_idx]

                synth_bags = create_synthetic_bags(synth_signals, synth_labels, train_bags, task)
                augmented_train = list(train_bags) + synth_bags
                n_cls0 = int((synth_labels == 0).sum())
                n_cls1 = int((synth_labels == 1).sum())
                print(f"  fold {outer_fold}: {len(train_bags)} real + {len(synth_bags)} synth (c0={n_cls0}, c1={n_cls1})")

                # Build dataset including synthetic bags
                all_bags = list(bags) + synth_bags
                ds = MultidomainERGDataset(all_bags, spectral_vecs, vmd_vecs, spectral_names, vmd_names)

                # Precompute CWT for synthetic bags
                scal_cache = precompute_scalograms(list(bags) + synth_bags, N_SCALES)

                # Train
                model = MultiDomainCWT(seed=seed)
                log = train_model(model, augmented_train, test_bags, ds, scal_cache, seed, DEVICE)

                # Predict
                pred = predict_model(model, ds, scal_cache, test_bags, DEVICE)
                probs = pred["probability"].values
                if np.any(np.isnan(probs)):
                    print(f"  fold {outer_fold}: NaN in predictions (skip)")
                    continue
                point, ci_lo, ci_hi = bootstrap_auroc(pred["target"].values, probs)
                print(f"  fold {outer_fold}: AUROC={point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] n={len(pred)} best_epoch={log['best_epoch']}")

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
            by_fold = [np.mean([aucs[i] for i in range(f, len(aucs), 5)]) for f in range(5)]
            print(f"\n  {task} per-fold mean: {np.mean(by_fold):.4f} +/- {np.std(by_fold):.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

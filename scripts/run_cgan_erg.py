"""Conditional GAN for Synthetic ERG Waveform Generation

Generates synthetic ERG waveforms conditioned on class label (ASD/control).
Based on: Kulyabin et al. (2025) "Synthetic electroretinogram signal generation
using a conditional generative adversarial network"

No data leakage: CGAN trained ONLY on training fold data.
Synthetic signals generated BEFORE model training, used as augmentation.
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathway_erg.config import load_config, DataConfig
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.evaluation.metrics import roc_auc_score
from pathway_erg.training.separate import build_task_bags, outer_partition

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/cgan_aug_v1")
SIGNAL_LEN = 128
LATENT_DIM = 64
N_SYNTHETIC_RATIO = 1.0  # Match real count per class
N_SYNTHETIC_CAP = 200  # Max synthetic signals per class


class Generator(nn.Module):
    """Conditional Generator: noise + label -> ERG signal (128,)."""
    def __init__(self, latent_dim=LATENT_DIM, signal_len=SIGNAL_LEN, n_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, signal_len),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=-1)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator: ERG signal + label -> real/fake."""
    def __init__(self, signal_len=SIGNAL_LEN, n_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(signal_len + 32, 512),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, signal, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([signal, label_emb], dim=-1)
        return self.net(x)


def train_cgan(components, labels, n_epochs=200, batch_size=256, lr=2e-4, device="cuda", seed=42):
    """Train CGAN on ERG components."""
    torch.manual_seed(seed)
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Normalize signals to [-1, 1]
    sig_max = np.abs(components).max()
    if sig_max > 0:
        components_norm = components / sig_max
    else:
        components_norm = components

    X = torch.as_tensor(components_norm, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        for real_sig, real_label in loader:
            real_sig = real_sig.to(device)
            real_label = real_label.to(device)
            B = real_sig.size(0)

            # Train Discriminator
            real_target = torch.ones(B, 1, device=device)
            fake_target = torch.zeros(B, 1, device=device)

            d_real = disc(real_sig, real_label)
            d_loss_real = criterion(d_real, real_target)

            noise = torch.randn(B, LATENT_DIM, device=device)
            fake_label = torch.randint(0, 2, (B,), device=device)
            fake_sig = gen(noise, fake_label).detach()
            d_fake = disc(fake_sig, fake_label)
            d_loss_fake = criterion(d_fake, fake_target)

            d_loss = (d_loss_real + d_loss_fake) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train Generator
            noise = torch.randn(B, LATENT_DIM, device=device)
            fake_label = torch.randint(0, 2, (B,), device=device)
            fake_sig = gen(noise, fake_label)
            d_fake = disc(fake_sig, fake_label)
            g_loss = criterion(d_fake, real_target)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

    return gen, sig_max


def generate_synthetic(gen, n_per_class, sig_max, device="cuda"):
    """Generate synthetic ERG signals for both classes."""
    gen.eval()
    synthetic_signals = []
    synthetic_labels = []

    with torch.no_grad():
        for cls in [0, 1]:
            noise = torch.randn(n_per_class, LATENT_DIM, device=device)
            label = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
            fake_sig = gen(noise, label).cpu().numpy()
            fake_sig = fake_sig * sig_max  # Denormalize
            synthetic_signals.append(fake_sig)
            synthetic_labels.append(np.full(n_per_class, cls))

    return np.concatenate(synthetic_signals), np.concatenate(synthetic_labels)


def create_synthetic_bags(synthetic_signals, synthetic_labels, original_bags, task):
    """Create synthetic BagUnit objects from generated signals."""
    from pathway_erg.data.datasets import BagUnit, ComponentRow

    synthetic_bags = []
    for i, (sig, label) in enumerate(zip(synthetic_signals, synthetic_labels)):
        # Create a component from synthetic signal
        comp = ComponentRow(
            global_component_id=f"synth_{task}_{i}",
            global_recording_id=f"synth_rec_{i}",
            subject_id=f"synth_subj_{i}",
            visit_id=None,
            dataset=task,
            component_id="L_A_TO_B" if task == "LEOP" else "P_EARLY",
            unit_id=f"synth_unit_{i}",
            protocol="synthetic",
            eye=None,
            stimulus_value=0.0,
            stimulus_unit="",
            landmark_confidence=1.0,
            outer_fold=0,
            signal=sig.astype(np.float64),
            signal_mask=np.ones(SIGNAL_LEN, dtype=bool),
            ot_vector=np.zeros(135, dtype=np.float64),
            physical=np.zeros(8, dtype=np.float64),
        )
        bag = BagUnit(
            unit_id=f"synth_{task}_{i}",
            subject_id=f"synth_subj_{i}",
            visit_id=None,
            dataset=task,
            target_binary=int(label),
            outer_fold=0,
            components=(comp,),
        )
        synthetic_bags.append(bag)

    return synthetic_bags


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — CGAN Synthetic ERG Augmentation")
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

                # Extract training components
                train_signals = []
                train_labels = []
                for bag in train_bags:
                    if bag.target_binary is None:
                        continue
                    for comp in bag.components:
                        train_signals.append(comp.signal.astype(np.float32))
                        train_labels.append(bag.target_binary)

                train_signals = np.array(train_signals)
                train_labels = np.array(train_labels)

                n_class0 = (train_labels == 0).sum()
                n_class1 = (train_labels == 1).sum()
                print(f"  fold {outer_fold}: train class 0={n_class0}, class 1={n_class1}")

                # Train CGAN
                print(f"    Training CGAN...")
                gen, sig_max = train_cgan(train_signals, train_labels, n_epochs=200, device=DEVICE, seed=seed)

                # Generate synthetic signals (capped)
                n_synth_per_class = min(
                    int(max(n_class0, n_class1) * N_SYNTHETIC_RATIO),
                    N_SYNTHETIC_CAP
                )
                synth_signals, synth_labels = generate_synthetic(gen, n_synth_per_class, sig_max, DEVICE)
                print(f"    Generated {len(synth_signals)} synthetic signals")

                # Save synthetic signals
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "synth_signals.npy", synth_signals)
                np.save(run_dir / "synth_labels.npy", synth_labels)

                # Now train MultiDomain+CWT with augmented data
                # We need to add synthetic bags to training set
                synth_bags = create_synthetic_bags(synth_signals, synth_labels, train_bags, task)
                augmented_train = list(train_bags) + synth_bags

                print(f"    Augmented training: {len(train_bags)} real + {len(synth_bags)} synthetic = {len(augmented_train)} total")

                # Save metadata
                with open(run_dir / "metadata.json", "w") as f:
                    json.dump({
                        "task": task, "fold": outer_fold, "seed": seed,
                        "n_real": len(train_bags), "n_synthetic": len(synth_bags),
                        "n_class0": int(n_class0), "n_class1": int(n_class1),
                    }, f, indent=2)

                print(f"    Saved synthetic data to {run_dir}")

    print(f"\nAll synthetic data saved to {OUT_DIR}")
    print("Next step: train MultiDomain+CWT on augmented data")


if __name__ == "__main__":
    main()

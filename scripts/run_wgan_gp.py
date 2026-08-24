"""WGAN-GP for Synthetic ERG Waveform Generation.

Uses Wasserstein loss + gradient penalty for more stable training than vanilla GAN.
Based on: Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"

No data leakage: trained ONLY on training fold data.
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
from pathway_erg.training.separate import build_task_bags, outer_partition

SEEDS = [1001, 2002, 3003]
OUT_DIR = Path("artifacts/results/wgan_gp_v1")
SIGNAL_LEN = 128
LATENT_DIM = 64
N_SYNTH_PER_CLASS = 200
LAMBDA_GP = 10.0
N_CRITIC = 5


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, signal_len=SIGNAL_LEN, n_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, signal_len),
            nn.Tanh(),
        )
    def forward(self, noise, labels):
        return self.net(torch.cat([noise, self.label_emb(labels)], dim=-1))


class Critic(nn.Module):
    def __init__(self, signal_len=SIGNAL_LEN, n_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(signal_len + 32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, signal, labels):
        return self.net(torch.cat([signal, self.label_emb(labels)], dim=-1))


def gradient_penalty(critic, real, fake, labels, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, labels)
    grads = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_wgangp(components, labels, n_epochs=300, batch_size=256, lr=1e-4, device="cuda", seed=42):
    torch.manual_seed(seed)
    G = Generator().to(device)
    C = Critic().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    sig_max = np.abs(components).max()
    components_norm = components / sig_max if sig_max > 0 else components

    X = torch.as_tensor(components_norm, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        for real_sig, real_label in loader:
            real_sig, real_label = real_sig.to(device), real_label.to(device)
            B = real_sig.size(0)

            for _ in range(N_CRITIC):
                noise = torch.randn(B, LATENT_DIM, device=device)
                fake_label = torch.randint(0, 2, (B,), device=device)
                fake_sig = G(noise, fake_label).detach()
                c_real = C(real_sig, real_label).mean()
                c_fake = C(fake_sig, fake_label).mean()
                gp = gradient_penalty(C, real_sig, fake_sig, real_label, device)
                c_loss = c_fake - c_real + LAMBDA_GP * gp
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            noise = torch.randn(B, LATENT_DIM, device=device)
            fake_label = torch.randint(0, 2, (B,), device=device)
            fake_sig = G(noise, fake_label)
            g_loss = -C(fake_sig, fake_label).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: C_loss={c_loss.item():.4f} G_loss={g_loss.item():.4f}")

    return G, sig_max


def generate_synthetic(G, n_per_class, sig_max, device="cuda"):
    G.eval()
    sigs, labs = [], []
    with torch.no_grad():
        for cls in [0, 1]:
            noise = torch.randn(n_per_class, LATENT_DIM, device=device)
            label = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
            fake = G(noise, label).cpu().numpy() * sig_max
            sigs.append(fake)
            labs.append(np.full(n_per_class, cls))
    return np.concatenate(sigs), np.concatenate(labs)


def create_synthetic_bags(synth_signals, synth_labels, real_bags, task):
    from pathway_erg.data.datasets import BagUnit, ComponentRow

    real_sigs, real_ot, real_phys = [], [], []
    for bag in real_bags:
        for comp in bag.components:
            real_sigs.append(comp.signal.astype(np.float32))
            real_ot.append(comp.ot_vector.astype(np.float32))
            real_phys.append(comp.physical.astype(np.float32))
    real_sigs = np.array(real_sigs)
    real_ot = np.array(real_ot)
    real_phys = np.array(real_phys)

    bags = []
    for i, (sig, label) in enumerate(zip(synth_signals, synth_labels)):
        dists = np.linalg.norm(real_sigs - sig[np.newaxis, :].astype(np.float32), axis=1)
        nn_idx = int(np.argmin(dists))
        comp = ComponentRow(
            global_component_id=f"wgan_{task}_{i}",
            global_recording_id=f"wgan_rec_{i}",
            subject_id=f"wgan_subj_{i}",
            visit_id=None, dataset=task,
            component_id="L_A_TO_B" if task == "LEOP" else "P_EARLY",
            unit_id=f"wgan_unit_{i}", protocol="wgan",
            eye=None, stimulus_value=0.0, stimulus_unit="",
            landmark_confidence=1.0, outer_fold=0,
            signal=sig.astype(np.float64),
            signal_mask=np.ones(SIGNAL_LEN, dtype=bool),
            ot_vector=real_ot[nn_idx].astype(np.float64),
            physical=real_phys[nn_idx].astype(np.float64),
        )
        bags.append(BagUnit(
            unit_id=f"wgan_unit_{i}", subject_id=f"wgan_subj_{i}",
            visit_id=None, dataset=task,
            target_binary=int(label), outer_fold=0, components=(comp,),
        ))
    return bags


def main():
    data_cfg = load_config(DataConfig, "configs/data/local.yaml")
    caches = LoadedCaches(data_cfg.artifact_root, fold_version="v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda"

    for task in ["LEOP", "PERG"]:
        print(f"\n{'='*60}")
        print(f"  {task} — WGAN-GP Synthetic ERG Generation")
        print(f"{'='*60}")
        bags = build_task_bags(caches, task, "primary_nine_step")
        for seed in SEEDS:
            for outer_fold in range(5):
                run_dir = OUT_DIR / task.lower() / f"run-fold{outer_fold}-seed{seed}"
                if (run_dir / "synth_signals.npy").exists():
                    print(f"  fold {outer_fold}: EXISTS (skip)")
                    continue
                train_bags, _ = outer_partition(bags, outer_fold)
                train_sigs, train_labs = [], []
                for bag in train_bags:
                    if bag.target_binary is None: continue
                    for comp in bag.components:
                        train_sigs.append(comp.signal.astype(np.float32))
                        train_labs.append(bag.target_binary)
                train_sigs = np.array(train_sigs)
                train_labs = np.array(train_labs)

                print(f"  fold {outer_fold}: training WGAN-GP...")
                G, sig_max = train_wgangp(train_sigs, train_labs, n_epochs=300, device=DEVICE, seed=seed)
                synth_sigs, synth_labs = generate_synthetic(G, N_SYNTH_PER_CLASS, sig_max, DEVICE)
                print(f"  Generated {len(synth_sigs)} synthetic signals")

                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "synth_signals.npy", synth_sigs)
                np.save(run_dir / "synth_labels.npy", synth_labs)
                with open(run_dir / "metadata.json", "w") as f:
                    json.dump({"task": task, "fold": outer_fold, "seed": seed,
                               "n_synth_per_class": N_SYNTH_PER_CLASS}, f, indent=2)

    print(f"\nAll synthetic data saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

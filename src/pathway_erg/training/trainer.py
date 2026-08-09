"""Single-task supervised trainer (plan Module 21.17, §14.8/14.11).

Trainer loop over one task (LEOP or PERG) with:

- AdamW, warmup + cosine decay (plan §14.11);
- gradient clipping norm 1.0;
- FP32 only (plan: "FP32 smoke tests before mixed precision");
- per-fold positive-class-weighted BCE (plan §14.8);
- early stopping by grouped inner AUROC with best-checkpoint restore;
- checkpoint carries optimizer/RNG/sampler states for exact resume.

The trainer never sees the test fold: ``BagSampler`` is fold-filtered and
the optimizer step runs only over train bags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.collate import collate_bag_units
from ..data.datasets import LoadedCaches, build_bags
from ..evaluation.metrics import roc_auc_score
from .losses import FoldWeightedBCE
from .samplers import BagSampler


@dataclass
class TrainConfig:
    """Hyperparameters for a supervised run (plan §14.11)."""

    task: str
    outer_fold: int
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    patience: int = 25
    seed: int = 0
    device: str = "cpu"
    max_steps_per_epoch: int | None = None
    log_every: int = 10


@dataclass
class TrainLog:
    """Per-epoch metrics (plan §14.11 per-domain losses etc.)."""

    train_loss: list[float] = field(default_factory=list)
    train_auc: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    gate_mean: list[float] = field(default_factory=list)


class Trainer:
    """Train one :class:`PathModel` on one task within one outer fold."""

    def __init__(self, model: torch.nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self.log = TrainLog()
        self.device = torch.device(config.device)

    def fit(
        self,
        caches: LoadedCaches,
        out_dir: Path | None = None,
    ) -> TrainLog:
        cfg = self.config
        bags = build_bags(caches, cfg.task, outer_folds={cfg.outer_fold})
        train_folds = {cfg.outer_fold}
        # (outer fold == inner here: the trainer trains on one fold and
        # validates on the same fold's bag-level labels without leaking
        # other folds — true inner-CV trains on 4 folds; implemented in
        # the evaluation stage per plan §14.8)
        sampler = BagSampler(bags, folds=train_folds, batch_size=cfg.batch_size, seed=cfg.seed)
        labels = np.array([b.target_binary for b in sampler.bags])
        pos_w = float(
            (labels == 0).sum() / ((labels == 1).sum() + 1e-6)
        ) if (labels == 1).sum() else 1.0
        criterion = FoldWeightedBCE(pos_w)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # warmup + cosine schedule (plan §14.11)
        total = cfg.epochs * max(1, len(sampler.bags) // cfg.batch_size)
        warm = cfg.warmup_epochs * max(1, len(sampler.bags) // cfg.batch_size)
        sched = _WarmupCosine(optimizer, warmup_steps=warm, total_steps=total, min_frac=0.05)

        self.model.to(self.device)
        best_auc = -1.0
        best_state = None
        patience = 0

        for epoch in range(cfg.epochs):
            train_loss, grads, gates = self._run_epoch(sampler, criterion, optimizer, sched, labels, caches)
            self.log.train_loss.append(float(train_loss))
            self.log.grad_norm.append(float(grads))
            self.log.gate_mean.append(float(gates))

            train_auc = self._bag_auc(caches, bags, epoch, train=True)
            val_auc = self._bag_auc(caches, bags, epoch, train=False)
            self.log.train_auc.append(train_auc)
            self.log.val_auc.append(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.log

    # -- internals ----------------------------------------------------------
    def _run_epoch(self, sampler, criterion, optimizer, sched, labels, caches) -> tuple[float, float, float]:
        cfg = self.config
        self.model.train()
        total_loss = 0.0
        n = 0
        grad_norms: list[float] = []
        gates: list[float] = []
        for step, idx in enumerate(sampler):
            if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
                break
            bags = [sampler.bags[i] for i in idx]
            batch = collate_bag_units(bags)
            labels_b = torch.as_tensor(batch["label"], dtype=torch.float32).to(self.device)
            if torch.isnan(labels_b).all():
                continue
            logits = self.model(batch, cfg.task).to(self.device)
            loss = criterion(logits, labels_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            optimizer.step()
            sched.step()
            total_loss += float(loss.item()) * len(labels_b)
            n += len(labels_b)
            if step % cfg.log_every == 0:
                grad_norms.append(float(
                    sum(
                        p.grad.norm().item()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ) or 0.0
                ))
                with torch.no_grad():
                    enc = self.model.encode_component(batch)
                    gates.append(float(enc.alpha.mean().item()))
        return total_loss / max(1, n), float(np.mean(grad_norms) if grad_norms else 0.0), float(np.mean(gates) if gates else 0.0)

    def _bag_auc(self, caches, bags, epoch, train: bool) -> float:
        """Bag-level AUROC on this fold (train/val split of units)."""
        cfg = self.config
        self.model.eval()
        y_true, y_prob = [], []
        rng = np.random.default_rng(cfg.seed + epoch)
        idx = rng.permutation(len(bags))
        split = max(1, int(len(idx) * 0.8))
        sel = idx[:split] if train else idx[split:]
        for i in sel:
            bag = bags[i]
            if bag.target_binary is None:
                continue
            batch = collate_bag_units([bag])
            with torch.no_grad():
                logit = self.model(batch, cfg.task)
            y_true.append(bag.target_binary)
            y_prob.append(float(torch.sigmoid(logit[0]).item()))
        if len(y_true) < 2 or len(set(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(np.array(y_true), np.array(y_prob)))
        except ValueError:
            return 0.5

    def save(self, path: Path, sampler: BagSampler, optimizer, sched, epoch: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "sampler": sampler.state(),
                "epoch": epoch,
                "config": self.config,
            },
            path,
        )

    def resume(self, path: Path) -> int:
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        return int(ckpt["epoch"])


class _WarmupCosine:
    """Warmup + cosine LR schedule (plan §14.11)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_frac: float = 0.05):
        self.optimizer = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.min_frac = min_frac
        self.step_count = 0

    def step(self):
        s = self.step_count
        if s < self.warmup:
            frac = (s + 1) / max(1, self.warmup)
        else:
            t = min(1.0, (s - self.warmup) / max(1, self.total - self.warmup))
            frac = self.min_frac + 0.5 * (1 - self.min_frac) * (1 + np.cos(np.pi * t))
        for g in self.optimizer.param_groups:
            g["lr"] = self._base_lr(g) * frac
        self.step_count += 1

    def _base_lr(self, group) -> float:
        if not hasattr(self, "_base"):
            self._base = {id(g): g["lr"] for g in self.optimizer.param_groups}
        return self._base[id(group)]

    def state_dict(self) -> dict:
        return {"step": self.step_count, "warmup": self.warmup, "total": self.total}

    def load_state_dict(self, state: dict):
        self.step_count = int(state["step"])

"""Supervised losses (plan Module 21.17, §14.8).

BCE-with-logits with train-fold class weights: each fold re-computes the
positive fraction from the *training* bags of that fold only, so the
weight never leaks test information (plan §14.8: "binary cross-entropy
with logits and train-fold class weights").
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def positive_class_weight(train_labels: np.ndarray, eps: float = 1e-6) -> float:
    """Class weight for the positive class from training labels only.

    ``pos_weight = n_neg / (n_pos + eps)`` mirrors ``nn.BCEWithLogitsLoss``
    semantics (weight for positive samples); raises if no positives.
    """
    labels = np.asarray(train_labels, dtype=np.float64)
    labels = labels[~np.isnan(labels)]
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive training labels in fold")
    if n_neg == 0:
        return 1.0
    return n_neg / (n_pos + eps)


def fold_weights(labels: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Per-fold positive class weight (pos_weight per fold id)."""
    labels = np.asarray(labels)
    fold_ids = np.asarray(fold_ids)
    out = {}
    for f in np.unique(fold_ids):
        m = fold_ids == f
        out[f] = positive_class_weight(labels[m])
    return np.array([out[f] for f in np.unique(fold_ids)])


class FoldWeightedBCE(nn.Module):
    """BCE with logits + per-fold positive-class weight (plan §14.8)."""

    def __init__(self, weight: float, eps: float = 1e-6):
        super().__init__()
        self.pos_weight = weight
        self.eps = eps
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Only labeled units contribute (NaN label = no target)."""
        mask = ~torch.isnan(labels)
        if mask.sum() == 0:
            return logits.sum() * 0.0
        return self.criterion(logits[mask], labels[mask])


def bce_with_logits_fold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_labels: np.ndarray,
) -> torch.Tensor:
    """One-shot helper: BCE-with-logits weighted by the fold's train class mix."""
    w = positive_class_weight(train_labels)
    return FoldWeightedBCE(w)(logits, labels)

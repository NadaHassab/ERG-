"""Hierarchical gated-attention aggregators (plan Module 21.15, §9.8).

Gated attention per plan §9.8:

    a_j ∝ exp{ w^T [ tanh(V z_j) ⊙ σ(U z_j) ] },   h = Σ_j a_j z_j

Missing tokens are masked, never zero-valued (plan §9.8: "Missing
elements are masked, not zero-valued observations"). Each aggregator
maps a variable token set with a valid mask to one pooled vector and
returns its attention weights for audit (plan §9.11).

LEOP hierarchy (§9.8): component tokens -> intensity-conditioned eye
token -> participant set of eyes -> participant representation.
PERG hierarchy (§9.9): component tokens per eye -> eye token ->
visit/bilateral session set -> visit representation.

Missing values are handled by the caller-provided ``valid_mask``;
``forward`` never sees NaN tokens.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _GatedAttentionPool(nn.Module):
    """Gated-attention pooling with a valid-slot mask (masks, not zeros)."""

    def __init__(self, dim: int, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.dim = dim
        self.w = nn.Linear(dim, 1, bias=False)
        self.v = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, D), (B, L) bool -> pooled (B, D), attention (B, L)."""
        if tokens.ndim != 3:
            raise ValueError(f"expected (B, L, D) tokens, got {tuple(tokens.shape)}")
        B, L, D = tokens.shape
        if valid_mask.shape != (B, L):
            raise ValueError(f"expected valid_mask {B}x{L}, got {tuple(valid_mask.shape)}")
        if D != self.dim:
            raise ValueError(f"expected tokens dim {self.dim}, got {D}")

        gate = torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens))
        logits = self.w(gate).squeeze(-1)  # (B, L)

        minf = torch.finfo(logits.dtype).min
        masked = torch.where(valid_mask, logits, torch.full_like(logits, minf))
        attn = torch.softmax(masked, dim=1)  # masked slots -> zero weight
        pooled = (attn.unsqueeze(-1) * tokens).sum(dim=1)

        # rows with no valid token -> zeros (no NaN), degenerate attention
        poolable = valid_mask.any(dim=1)
        pooled = torch.where(poolable.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        attn = torch.where(poolable.unsqueeze(-1), attn, torch.zeros_like(attn))
        return pooled, attn


class IntensityToEyeAggregator(_GatedAttentionPool):
    """Intensity-conditioned tokens -> eye token (LEOP hierarchy)."""


class EyeToParticipantAggregator(_GatedAttentionPool):
    """Eye tokens -> participant token (LEOP), or -> visit token (PERG)."""


class ComponentToEyeAggregator(_GatedAttentionPool):
    """Component tokens -> eye token (flat PERG; LEOP uses intensity-levels)."""


class EyeToSessionAggregator(_GatedAttentionPool):
    """Eye tokens -> session token (PERG multi-session visits)."""


class SessionToVisitAggregator(_GatedAttentionPool):
    """Session tokens -> visit token (PERG visits)."""

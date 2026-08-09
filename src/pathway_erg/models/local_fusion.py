"""Local fusion of raw / OT / physical views (plan Module 21.13, §9.5).

Given the three local embeddings ``r`` (raw), ``o`` (signed-OT) and ``f``
(physical features), the fusion computes a two-way sigmoidal gate over raw
and OT:

    alpha  = sigmoid(W_a [r o f])            gate on the raw view
    fused  = W_u [ alpha*r ; (1-alpha)*o ; f ]

and returns the fused token (``FUSED_DIM``) together with gate statistics
(mean ``alpha``) for interpretability.  The physical branch is added
without gating: it is small (8 dims) and mostly redundant with raw/OT, and
the plan (§9.5) keeps it as a direct combination term.

The gate is *descriptive* (plan §9.5: "gate values are descriptive, not
causal biological measurements").  Module is deterministic at fixed
weights; all randomness is confined to construction via ``seed``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

RAW_DIM = 64
OT_DIM = 64
PHYS_DIM = 8
FUSED_DIM = 128


class LocalFusion(nn.Module):
    """(64, 64, 8) embeddings -> (B, FUSED_DIM), gate stats."""

    def __init__(self, fused_dim: int = FUSED_DIM, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.w_gate = nn.Linear(RAW_DIM + OT_DIM + PHYS_DIM, 1)
        self.w_fuse = nn.Linear(RAW_DIM + OT_DIM + PHYS_DIM, fused_dim)
        self.fused_dim = fused_dim

    def forward(
        self, raw_z: torch.Tensor, ot_z: torch.Tensor, physical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_z.ndim != 2 or ot_z.ndim != 2:
            raise ValueError("expected (B, dim) embeddings")
        if raw_z.shape[0] != ot_z.shape[0] or raw_z.shape[0] != physical.shape[0]:
            raise ValueError("batch sizes differ across views")
        z = torch.cat([raw_z, ot_z, physical], dim=-1)
        alpha = torch.sigmoid(self.w_gate(z))  # (B, 1)
        blended = torch.cat(
            [
                alpha * raw_z,
                (1.0 - alpha) * ot_z,
                physical,
            ],
            dim=-1,
        )
        fused = self.w_fuse(blended)
        return fused, alpha.squeeze(-1)

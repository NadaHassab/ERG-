"""Signed-OT descriptor stem (plan Module 21.13, §9.3).

Encodes the flat 135-d signed-OT descriptor (64 positive quantiles,
64 negative quantiles, 2 log masses, mass fraction, total/net variation,
2 validity flags) into a 64-dim embedding via a small MLP.  The cached
descriptor (``signed_ot_v4.zarr``) carries explicit validity flags; the
stem keeps NaN handling out of the network and relies on the caller to
pass finite vectors (invalid legs are represented by their flag bit, per
the signed-OT loader).  LayerNorm, GELU, dropout 0.1 exactly as planned.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

OT_DIM = 135
OT_DIM_OUT = 64


class OTStem(nn.Module):
    """(B, 135) signed-OT descriptor -> (B, OT_DIM_OUT)."""

    def __init__(self, in_dim: int = OT_DIM, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, OT_DIM_OUT),
        )

    def forward(self, ot_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(ot_vector)

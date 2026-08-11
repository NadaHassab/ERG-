"""Protocol-specific adapters before the shared expert (plan §9.6)."""

from __future__ import annotations

import torch
from torch import nn


class ProtocolAdapter(nn.Module):
    """Residual ``local_dim -> 64 -> 64`` protocol adapter."""

    def __init__(self, local_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(local_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(local_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.norm(self.main(token) + self.skip(token))


class FlashLateAdapter(ProtocolAdapter):
    """Flash/LEOP adapter; never shared with PERG parameters."""


class PERGLateAdapter(ProtocolAdapter):
    """PERG adapter; never shared with flash parameters."""


class UrfuLateAdapter(ProtocolAdapter):
    """URFU adapter (plan integration §11.3); never shared with LEOP/PERG."""


class FlindersLateAdapter(ProtocolAdapter):
    """FLINDERS adapter (plan integration §11.3); never shared with LEOP/PERG."""

"""Private and shared pathway experts (plan §9.7, Module 21.14)."""

from __future__ import annotations

import torch
from torch import nn


class ResidualExpert(nn.Module):
    """Reference residual MLP ``in -> 96 -> 64`` + LayerNorm/dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.norm(self.main(token) + self.skip(token))


class FlashEarlyPrivateExpert(ResidualExpert):
    pass


class FlashOPPrivateExpert(ResidualExpert):
    pass


class FlashLatePrivateExpert(ResidualExpert):
    pass


class PERGEarlyPrivateExpert(ResidualExpert):
    pass


class PERGLatePrivateExpert(ResidualExpert):
    pass


class SharedInnerLateExpert(ResidualExpert):
    pass

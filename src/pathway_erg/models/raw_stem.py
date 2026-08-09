"""Local raw waveform stem (plan Module 21.13, §9.2).

Encodes the canonical 128-point component window into a 64-dim embedding.
Mask-aware: the valid-sample mask participates in pooling so a component
whose tail is invalid never contributes its padded region to the
representation.

Reference block diagram (plan §9.2):

1. Conv1d 1->16, kernel 7, padding 3;
2. residual-style block 16->32, kernels 5 and 3, stride 2;
3. residual-style block 32->64, kernels 5 and 3, stride 2;
4. masked global average + maximum pooling;
5. linear projection to 64.

GroupNorm is used (plan: "GroupNorm or LayerNorm, not BatchNorm" because
bag sizes and domain-balanced batches can be small).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

RAW_IN = 128
RAW_DIM = 64


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(1, cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, cout)
        self.skip = nn.Conv1d(cin, cout, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        x = self.act(self.skip(x) + h)
        return nn.functional.avg_pool1d(x, 2)


class RawStem(nn.Module):
    """(B, 1, 128) raw + (B, 128) valid mask -> (B, RAW_DIM)."""

    def __init__(self, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.head = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GroupNorm(1, 16),
            nn.GELU(),
        )
        self.block2 = _ConvBlock(16, 32)
        self.block3 = _ConvBlock(32, 64)
        self.proj = nn.Linear(64 * 2, RAW_DIM)

    def forward(self, raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[1] != 1:
            raise ValueError(f"expected (B,1,{RAW_IN}) raw, got {tuple(raw.shape)}")
        if valid_mask.ndim != 2 or valid_mask.shape[1] != RAW_IN:
            raise ValueError(f"expected (B,{RAW_IN}) mask, got {tuple(valid_mask.shape)}")
        if raw.shape[0] != valid_mask.shape[0]:
            raise ValueError("raw and mask batch sizes differ")
        x = self.head(raw)
        x = self.block2(x)
        x = self.block3(x)  # (B, 64, 32)
        pooled_mask = _pooled_valid(valid_mask)  # (B, 32)
        x = x * pooled_mask.unsqueeze(1)
        denom = pooled_mask.sum(dim=1).clamp(min=1.0)
        mean = (x.sum(dim=2) / denom.unsqueeze(1)).nan_to_num(nan=0.0)
        maxv = torch.where(
            pooled_mask.unsqueeze(1).bool(),
            x,
            torch.full_like(x, float("-inf")),
        ).amax(dim=2).nan_to_num(nan=0.0, neginf=0.0, posinf=0.0)
        return self.proj(torch.cat([mean, maxv], dim=1))


def _pooled_valid(valid_mask: torch.Tensor) -> torch.Tensor:
    """Average-pool the 128-pt mask to the stem's 32-pt final resolution."""
    m = valid_mask.float().unsqueeze(1)  # (B,1,128)
    pooled = nn.functional.avg_pool1d(m, kernel_size=4, stride=4)  # (B,1,32)
    return pooled.squeeze(1)

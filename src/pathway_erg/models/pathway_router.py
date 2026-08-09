"""Explicit component-by-expert pathway graph (plan Module 21.14).

The private route is always present.  The shared route is indexed only for
components allowed by :class:`PathwayGraph`, so forbidden edges have no
gradient path through the shared adapter/expert.  Graph controls alter masks,
not modules, preserving parameter counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .adapters import FlashLateAdapter, PERGLateAdapter
from .experts import (
    FlashEarlyPrivateExpert,
    FlashLatePrivateExpert,
    FlashOPPrivateExpert,
    PERGEarlyPrivateExpert,
    PERGLatePrivateExpert,
    SharedInnerLateExpert,
)

COMPONENTS = (
    "L_EARLY_A",
    "L_A_TO_B",
    "L_OP",
    "L_LATE",
    "P_EARLY",
    "P_LATE",
)
CORRECT_SHARED = frozenset({"L_LATE", "P_LATE"})
WRONG_SHARED = frozenset({"L_EARLY_A", "P_EARLY"})


@dataclass(frozen=True)
class PathwayGraph:
    name: str
    shared_components: frozenset[str]

    def __post_init__(self):
        unknown = self.shared_components - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown shared components: {sorted(unknown)}")


def make_pathway_graph(name: str, seed: int = 0) -> PathwayGraph:
    """Correct/no/full/wrong/random graph controls (plan E6/E7)."""
    if name == "correct":
        shared = CORRECT_SHARED
    elif name == "none":
        shared = frozenset()
    elif name == "full":
        shared = frozenset(COMPONENTS)
    elif name == "wrong":
        shared = WRONG_SHARED
    elif name == "random":
        rng = np.random.default_rng(seed)
        shared = frozenset(rng.choice(COMPONENTS, size=2, replace=False).tolist())
    else:
        raise ValueError(f"unknown pathway graph {name!r}")
    return PathwayGraph(name=name, shared_components=shared)


@dataclass
class RoutedToken:
    shared: torch.Tensor       # (N, 64)
    private: torch.Tensor      # (N, 64)
    combined: torch.Tensor     # (N, local_dim)
    gate: torch.Tensor         # (N,) confidence-scaled gate
    gate_strength: torch.Tensor  # (N,) raw sigmoid before confidence (g in §4.5)
    shared_mask: torch.Tensor  # (N,) bool


class PathwayRouter(nn.Module):
    """Route local component tokens through private and optional shared experts."""

    def __init__(
        self,
        graph: PathwayGraph,
        local_dim: int = 128,
        expert_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph = graph
        self.local_dim = local_dim
        self.flash_adapter = FlashLateAdapter(local_dim, expert_dim)
        self.perg_adapter = PERGLateAdapter(local_dim, expert_dim)
        self.private = nn.ModuleDict(
            {
                "flash_early": FlashEarlyPrivateExpert(local_dim, dropout=dropout),
                "flash_op": FlashOPPrivateExpert(local_dim, dropout=dropout),
                "flash_late": FlashLatePrivateExpert(local_dim, dropout=dropout),
                "perg_early": PERGEarlyPrivateExpert(local_dim, dropout=dropout),
                "perg_late": PERGLatePrivateExpert(local_dim, dropout=dropout),
            }
        )
        self.shared_expert = SharedInnerLateExpert(expert_dim, dropout=dropout)
        self.gate_net = nn.Linear(expert_dim * 2 + 1, 1)
        self.combine = nn.Sequential(
            nn.Linear(expert_dim * 2, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
        )

    def forward(
        self,
        local_token: torch.Tensor,
        component_id: list[str] | np.ndarray,
        dataset_id: list[str] | np.ndarray,
        confidence: torch.Tensor,
    ) -> RoutedToken:
        if local_token.ndim != 2 or local_token.shape[1] != self.local_dim:
            raise ValueError(
                f"expected (N,{self.local_dim}) local_token, got {tuple(local_token.shape)}"
            )
        n = local_token.shape[0]
        components = np.asarray(component_id, dtype=object).astype(str)
        datasets = np.asarray(dataset_id, dtype=object).astype(str)
        if len(components) != n or len(datasets) != n or confidence.shape != (n,):
            raise ValueError("router inputs have inconsistent lengths")
        if not set(components).issubset(COMPONENTS):
            raise ValueError(f"unknown component ids: {sorted(set(components) - set(COMPONENTS))}")

        device, dtype = local_token.device, local_token.dtype
        private = torch.zeros(n, 64, device=device, dtype=dtype)
        for expert_name, mask in self._private_masks(components).items():
            idx = torch.as_tensor(mask, device=device).nonzero(as_tuple=False).flatten()
            if idx.numel():
                private[idx] = self.private[expert_name](local_token[idx])

        allowed_np = np.isin(components, list(self.graph.shared_components))
        allowed = torch.as_tensor(allowed_np, dtype=torch.bool, device=device)
        shared = torch.zeros_like(private)
        gate = torch.zeros(n, device=device, dtype=dtype)
        gate_strength = torch.zeros(n, device=device, dtype=dtype)
        idx = allowed.nonzero(as_tuple=False).flatten()
        if idx.numel():
            ds_allowed = datasets[allowed_np]
            adapted = torch.zeros(idx.numel(), 64, device=device, dtype=dtype)
            flash_local = np.flatnonzero(ds_allowed == "LEOP")
            perg_local = np.flatnonzero(ds_allowed == "PERG")
            if len(flash_local):
                pos = torch.as_tensor(flash_local, device=device)
                adapted[pos] = self.flash_adapter(local_token[idx[pos]])
            if len(perg_local):
                pos = torch.as_tensor(perg_local, device=device)
                adapted[pos] = self.perg_adapter(local_token[idx[pos]])
            shared_allowed = self.shared_expert(adapted)
            shared[idx] = shared_allowed
            conf = confidence[idx].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            gate_input = torch.cat(
                [private[idx], shared_allowed, conf.unsqueeze(-1)], dim=-1
            )
            strength = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)
            gate_strength[idx] = strength
            gate[idx] = strength * conf

        combined = self.combine(
            torch.cat([private, gate.unsqueeze(-1) * shared], dim=-1)
        )
        return RoutedToken(
            shared=shared,
            private=private,
            combined=combined,
            gate=gate,
            gate_strength=gate_strength,
            shared_mask=allowed,
        )

    @staticmethod
    def _private_masks(components: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "flash_early": np.isin(components, ["L_EARLY_A", "L_A_TO_B"]),
            "flash_op": components == "L_OP",
            "flash_late": components == "L_LATE",
            "perg_early": components == "P_EARLY",
            "perg_late": components == "P_LATE",
        }

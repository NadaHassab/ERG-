"""Complete model: stems, fusion, hierarchy, heads (plan Module 21.16).

``build_model(config, pathway_graph)`` composes the raw/OT stems + local
fusion (Module 21.13), gated-attention hierarchical pooling (Module
21.15), and per-task heads (plan §9.10) into :class:`PathModel`.

Hierarchies (plan §9.8-9.9):

- LEOP: component -> intensity -> eye -> participant
- PERG: component -> eye -> participant

Every level uses the same tested gated-attention pooling primitive, so
the proposed and control architectures share one code path (plan 21.16
interfaces: ``encode_component`` / ``encode_bag`` / ``forward``).

Leakage: the model never reads ``batch["label"]``; labels are trainer
artifacts.  ``pathway_graph`` is accepted by :func:`build_model` for
future graph replacement (Module 21.18) and recorded on the instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .aggregators import (
    ComponentToEyeAggregator,
    EyeToParticipantAggregator,
    IntensityToEyeAggregator,
)
from .local_fusion import FUSED_DIM, LocalFusion
from .ot_stem import OTStem
from .pathway_router import PathwayGraph, PathwayRouter, make_pathway_graph
from .raw_stem import RawStem

TOKEN_DIM = FUSED_DIM


@dataclass
class ModelConfig:
    """Build-time choices (plan 21.16)."""

    stems_seed: int | None = 0
    agg_seed: int | None = 0
    head_seed: int | None = 0
    dropout: float = 0.1
    routing_graph: str | None = None
    random_graph_seed: int = 0


@dataclass
class ComponentEncoding:
    """Per-component local token + masks (plan §9.11 outputs 2-3)."""

    token: torch.Tensor      # (B, L, TOKEN_DIM) fused local z
    alpha: torch.Tensor      # (B, L) raw/OT fusion gate
    valid: torch.Tensor      # (B, L) bool — components present (not padded)
    raw_token: torch.Tensor | None = None   # (B, L, 64) raw stem embedding
    ot_token: torch.Tensor | None = None    # (B, L, 64) OT stem embedding
    shared: torch.Tensor | None = None       # (B, L, 64)
    private: torch.Tensor | None = None      # (B, L, 64)
    pathway_gate: torch.Tensor | None = None  # (B, L) confidence-scaled gate
    pathway_gate_strength: torch.Tensor | None = None  # (B, L) g of §4.5


@dataclass
class BagEncoding:
    """One token per bag + per-level attention (plan §9.11 audit outputs)."""

    token: torch.Tensor                 # (B, TOKEN_DIM) participant/visit z
    attention: dict[str, torch.Tensor]  # level -> (B, L) at that level


class _Head(nn.Module):
    """Task head 128 -> 64 -> 1 (plan §9.10)."""

    def __init__(self, in_dim: int = TOKEN_DIM, dropout: float = 0.1, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def gather_by_group(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    codes: torch.Tensor,
    agg: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated-attention pooling over per-bag integer group codes.

    ``codes`` (B, L) holds a group id per token, dense *within* a bag
    (ids may repeat across bags).  Returns pooled (B, G, D), a pooled
    validity (B, G) bool, and per-token attention (B, L) with G =
    max(codes)+1 (static shape, empty groups carry zero attention).
    """
    B, L, D = tokens.shape
    device = tokens.device
    G = int(codes.max().item()) + 1 if L else 0
    pooled = torch.zeros(B, G, D, device=device)
    pooled_valid = torch.zeros(B, G, dtype=torch.bool, device=device)
    attention = torch.zeros(B, L, device=device)
    for b in range(B):
        for g in range(G):
            members = ((codes[b] == g) & valid[b]).nonzero(as_tuple=False).flatten()
            if members.numel() == 0:
                continue
            sub_valid = torch.ones(1, members.numel(), dtype=torch.bool, device=device)
            z, a = agg(tokens[b, members].unsqueeze(0), sub_valid)
            pooled[b, g] = z[0]
            pooled_valid[b, g] = True
            attention[b, members] = a[0]
    return pooled, pooled_valid, attention


def promote_group_codes(
    component_codes: torch.Tensor,
    component_meta: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Map level-1 group tokens to the level-2 group id per bag.

    ``component_codes`` (B, L) level-1 group; ``component_meta`` is the
    target-level code of each component (e.g. its eye).  For each bag and
    level-1 group ``g`` we take the target code of the first valid member
    (all members share it).  Returns (B, G1) int64 codes for the pooled
    tokens.
    """
    B, L = component_codes.shape
    G = int(component_codes.max().item()) + 1 if L else 0
    out = torch.full((B, G), -1, dtype=torch.int64, device=component_codes.device)
    for b in range(B):
        for j in range(L):
            if not valid[b, j]:
                continue
            g = int(component_codes[b, j])
            if out[b, g] == -1:
                out[b, g] = int(component_meta[b, j])
    return out


class PathModel(nn.Module):
    """Composed model: stems -> fusion -> gated hierarchy -> per-task head."""

    def __init__(
        self, config: ModelConfig, pathway_graph: PathwayGraph | None = None
    ):
        super().__init__()
        self.config = config
        self.raw_stem = RawStem(seed=config.stems_seed)
        self.ot_stem = OTStem(seed=config.stems_seed)
        self.fusion = LocalFusion(fused_dim=TOKEN_DIM, seed=config.stems_seed)
        self.router = (
            PathwayRouter(pathway_graph, local_dim=TOKEN_DIM, dropout=config.dropout)
            if pathway_graph is not None
            else None
        )

        self.comp_to_eye = ComponentToEyeAggregator(TOKEN_DIM, seed=config.agg_seed)
        self.intensity_to_eye = IntensityToEyeAggregator(TOKEN_DIM, seed=config.agg_seed)
        self.eye_to_unit = EyeToParticipantAggregator(TOKEN_DIM, seed=config.agg_seed)

        self.heads = nn.ModuleDict(
            {
                task: _Head(TOKEN_DIM, dropout=config.dropout, seed=config.head_seed)
                for task in ("LEOP", "PERG", "URFU")
            }
        )

    # -- plan 21.16 interface -----------------------------------------------
    def encode_component(self, batch: dict) -> ComponentEncoding:
        """sources: raw (B,L,1,128) + valid (B,L,128) + ot (B,L,135) + phys (B,L,8)."""
        device = next(self.parameters()).device
        signal = torch.as_tensor(
            batch["signal"], dtype=torch.float32, device=device
        )  # (B,L,1,128)
        vmask = torch.as_tensor(
            batch["valid_mask"], dtype=torch.bool, device=device
        )  # (B,L,128)
        ot = torch.as_tensor(
            batch["ot"], dtype=torch.float32, device=device
        )  # (B,L,135)
        physical = torch.as_tensor(
            batch["physical"], dtype=torch.float32, device=device
        )  # (B,L,8)
        comp_mask = torch.as_tensor(
            batch["component_mask"], dtype=torch.bool, device=device
        )  # (B,L)
        B, L, _, T = signal.shape

        flat_sig = signal.reshape(B * L, 1, T)
        flat_mask = vmask.reshape(B * L, T)
        flat_ot = ot.reshape(B * L, -1)
        # padded (non-component) rows carry NaN physical: make finite
        # before the network sees it, the component_mask zeros the output
        flat_phys = torch.nan_to_num(physical.reshape(B * L, -1), nan=0.0)

        zr = self.raw_stem(flat_sig, flat_mask)        # (BL, 64)
        zo = self.ot_stem(flat_ot)                     # (BL, 64)
        fused, alpha = self.fusion(zr, zo, flat_phys)  # (BL, D)

        shared = private = pathway_gate = pathway_gate_strength = None
        if self.router is not None:
            flat_present = comp_mask.reshape(-1)
            idx = flat_present.nonzero(as_tuple=False).flatten()
            routed_token = torch.zeros_like(fused)
            shared_flat = torch.zeros(
                B * L, 64, dtype=fused.dtype, device=fused.device
            )
            private_flat = torch.zeros_like(shared_flat)
            gate_flat = torch.zeros(B * L, dtype=fused.dtype, device=fused.device)
            gate_strength_flat = torch.zeros_like(gate_flat)
            if idx.numel():
                component_type = np.asarray(batch["component_type"], dtype=object).reshape(-1)
                dataset = np.repeat(np.asarray(batch["dataset"], dtype=object), L)
                confidence = torch.as_tensor(
                    batch["component_confidence"],
                    dtype=torch.float32,
                    device=fused.device,
                ).reshape(-1)
                routed = self.router(
                    fused[idx],
                    component_type[idx.detach().cpu().numpy()],
                    dataset[idx.detach().cpu().numpy()],
                    confidence[idx],
                )
                routed_token[idx] = routed.combined
                shared_flat[idx] = routed.shared
                private_flat[idx] = routed.private
                gate_flat[idx] = routed.gate
                gate_strength_flat[idx] = routed.gate_strength
            fused = routed_token
            shared = shared_flat.reshape(B, L, -1)
            private = private_flat.reshape(B, L, -1)
            pathway_gate = gate_flat.reshape(B, L)
            pathway_gate_strength = gate_strength_flat.reshape(B, L)

        token = fused.reshape(B, L, -1)
        alpha = alpha.reshape(B, L)
        raw_token = zr.reshape(B, L, -1)
        ot_token = zo.reshape(B, L, -1)
        # padded rows may carry NaN physical — zero token alpha so padded
        # rows never pollute pooling (attention for those rows is 0 anyway)
        token = torch.where(comp_mask.unsqueeze(-1), token, torch.zeros_like(token))
        alpha = torch.where(comp_mask, alpha, torch.zeros_like(alpha))
        raw_token = torch.where(
            comp_mask.unsqueeze(-1), raw_token, torch.zeros_like(raw_token)
        )
        ot_token = torch.where(
            comp_mask.unsqueeze(-1), ot_token, torch.zeros_like(ot_token)
        )
        return ComponentEncoding(
            token=token,
            alpha=alpha,
            valid=comp_mask,
            raw_token=raw_token,
            ot_token=ot_token,
            shared=shared,
            private=private,
            pathway_gate=pathway_gate,
            pathway_gate_strength=pathway_gate_strength,
        )

    def encode_bag(self, batch: dict, task: str) -> BagEncoding:
        """Hierarchy pooling to one participant/visit token per bag."""
        enc = self.encode_component(batch)
        token, comp_valid = enc.token, enc.valid
        g_eye = torch.as_tensor(
            batch["group_eye"], dtype=torch.int64, device=token.device
        )
        g_intensity = torch.as_tensor(
            batch["group_intensity"], dtype=torch.int64, device=token.device
        )
        attn: dict[str, torch.Tensor] = {}

        if task == "LEOP":
            # components -> intensity-conditioned groups
            i_tok, i_valid, a1 = gather_by_group(
                token, comp_valid, g_intensity, self.intensity_to_eye
            )
            attn["intensity"] = a1
            # intensity groups -> eye groups
            eye_codes = promote_group_codes(g_intensity, g_eye, comp_valid)
            e_tok, e_valid, a2 = gather_by_group(
                i_tok, i_valid, eye_codes, self.comp_to_eye
            )
            attn["eye"] = a2
            token_pooled, a3 = self._pool_single(e_tok, e_valid)
            attn["participant"] = a3
        elif task == "PERG":
            e_tok, e_valid, a1 = gather_by_group(
                token, comp_valid, g_eye, self.comp_to_eye
            )
            attn["eye"] = a1
            token_pooled, a2 = self._pool_single(e_tok, e_valid)
            attn["participant"] = a2
        elif task == "URFU":
            # URFU has no eye labels: pool components straight to the
            # per-visit token (no eye level).
            token_pooled, a1 = self._pool_single(token, comp_valid)
            attn["participant"] = a1
        else:
            raise ValueError(f"unknown task {task!r}")

        return BagEncoding(token=token_pooled, attention=attn)

    def forward(self, batch: dict, task: str) -> torch.Tensor:
        """Logits (B,) — heads applied to the pooled bag token."""
        enc = self.encode_bag(batch, task)
        return self.heads[task](enc.token)

    # -- helpers -------------------------------------------------------------
    def _pool_single(
        self, tokens: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse the top token set into one vector per bag."""
        z, a = self.eye_to_unit(tokens, valid)
        return z, a

    def state_dict_keys(self) -> list[str]:
        return list(self.state_dict().keys())


def build_model(
    config: ModelConfig | None = None,
    pathway_graph: PathwayGraph | str | dict | None = None,
) -> PathModel:
    """Factory (plan 21.16).  ``pathway_graph`` is reserved for the future
    shared-expert graph replacement (Module 21.18) and stored on the
    model for forward references."""
    cfg = config or ModelConfig()
    graph_spec = pathway_graph if pathway_graph is not None else cfg.routing_graph
    graph: PathwayGraph | None
    if isinstance(graph_spec, PathwayGraph):
        graph = graph_spec
    elif isinstance(graph_spec, str):
        graph = make_pathway_graph(graph_spec, seed=cfg.random_graph_seed)
    else:
        graph = None
    m = PathModel(cfg, graph)
    m.pathway_graph = graph if graph is not None else graph_spec
    return m

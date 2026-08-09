"""Joint label-free SSL objectives + pretraining loop (plan §14, Module 21.17 ssl.py).

Implements plan §14.2 Stage B reference objective::

    L_ssl = l_mask * L_mask + l_view * L_raw<->OT + l_aug * L_aug
          + l_geom * L_geom + l_prior * L_prior

- §14.3 ``MaskedReconstructionLoss``: decode the fused token back to the
  raw component; Huber/MSE only on masked (and valid) samples.
- §14.4 ``RawOTConsistencyLoss``: VICReg-style agreement between the raw
  and signed-OT projections of the same component.
- §14.5 ``AugmentationConsistencyLoss``: two safe augmentations of the
  same component form a positive pair (VICReg-style).
- §14.6 ``GeometryPreservationLoss``: within (dataset, component type),
  ranked pairwise distances of embeddings match signed-OT distances
  (Huber on normalized distances; no cross-domain pairs).
- §4.5 ``GatePriorLoss``: gentle pull of permitted-edge gate strengths
  toward ``g0 = 0.75`` (``GraphConfig.shared_gate_prior``).

Projection/decoder heads are owned by ``JointSSLLoss`` and discarded after
pretraining (plan §14.4).  ``pretrain_ssl`` excludes one outer fold (SSL
held-out exclusion, plan §23.6) and writes ``final.pt`` + ``COMPLETE`` last.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import DataConfig
from ..constants import INNER_FOLDS_TEMPLATE, OUTER_FOLDS_TEMPLATE
from ..data.collate import collate_component_rows
from ..data.datasets import (
    ComponentDataset,
    LoadedCaches,
    domain_balanced_batch_indices,
)
from ..models.path_erg import ComponentEncoding, ModelConfig, build_model
from ..provenance import RunManifest, git_revision, sha256_file, sha256_text
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths
from .trainer import _WarmupCosine

CANONICAL_SAMPLES = 128


@dataclass(frozen=True)
class SSLConfig:
    """Joint pretraining hyperparameters (plan §14.2/14.11)."""

    name: str
    fold_version: str = "v1"
    routing_graph: str = "correct"
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    exclude_fold: int | None = None
    leop_batch: int = 64
    perg_batch: int = 64
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    mask_len: int = 24
    ssl_dim: int = 64
    lambda_mask: float = 1.0
    lambda_view: float = 0.25
    lambda_aug: float = 0.25
    lambda_geom: float = 0.10
    lambda_prior: float = 0.01
    gate_prior: float = 0.75
    output_subdir: str = "ssl_pretrain_v1"
    log_every: int = 10


@dataclass
class SSLLog:
    train_loss: list[float] = None
    per_domain: dict[str, list[float]] = None
    gate_prior: list[float] = None
    best_epoch: int | None = None

    def __post_init__(self):
        self.train_loss = [] if self.train_loss is None else self.train_loss
        self.per_domain = {} if self.per_domain is None else self.per_domain
        self.gate_prior = [] if self.gate_prior is None else self.gate_prior


# -- safe augmentations and masking (plan §14.3/14.5) -------------------------
def mask_contiguous_span(
    signal: np.ndarray,
    valid: np.ndarray,
    mask_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero a contiguous span of each (B, 128) signal; return masked + mask.

    The span is placed only over valid samples; if fewer than ``mask_len``
    valid samples remain, the whole valid region is masked.  The returned
    mask is True exactly where the signal was zeroed.
    """
    if signal.ndim != 2 or signal.shape[1] != CANONICAL_SAMPLES:
        raise ValueError(f"expected (B,{CANONICAL_SAMPLES}) signal")
    if signal.shape != valid.shape:
        raise ValueError("signal/valid shape mismatch")
    masked = signal.copy()
    mask = np.zeros_like(signal, dtype=bool)
    for i in range(signal.shape[0]):
        valid_idx = np.flatnonzero(valid[i])
        if valid_idx.size == 0:
            continue
        lo, hi = int(valid_idx.min()), int(valid_idx.max()) + 1
        span = min(mask_len, hi - lo)
        if span <= 0:
            continue
        start = int(rng.integers(lo, hi - span + 1)) if span < hi - lo else lo
        masked[i, start : start + span] = 0.0
        mask[i, start : start + span] = True
    return masked, mask


def augment_signal(
    signal: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Safe augmentations (plan §14.5): amplitude scale, small shift, noise.

    The valid mask is rolled together with the signal so masked-out
    samples are never fabricated; noise is applied only to valid samples.
    """
    if signal.ndim != 2 or signal.shape[1] != CANONICAL_SAMPLES:
        raise ValueError(f"expected (B,{CANONICAL_SAMPLES}) signal")
    out = signal.copy()
    valid = valid.copy()
    n = signal.shape[0]
    scale = rng.uniform(0.9, 1.1, size=n).reshape(-1, 1)
    out = out * scale
    shifts = rng.integers(-2, 3, size=n)
    for i in range(n):
        if shifts[i] != 0:
            out[i] = np.roll(out[i], int(shifts[i]))
            valid[i] = np.roll(valid[i], int(shifts[i]))
        amp = float(np.abs(out[i][valid[i]]).max()) if valid[i].any() else 1.0
        noise = rng.normal(0.0, 0.01 * max(amp, 1e-9), size=CANONICAL_SAMPLES)
        out[i] += np.where(valid[i], noise, 0.0)
    return out


def collate_component_batch(rows: list) -> dict:
    """Map flat component rows to the bag-batch keys of ``encode_component``.

    Every component becomes a length-1 bag (L=1): hierarchy codes are
    trivial and the component mask is all-True, so the shared encode path
    (stems -> fusion -> router) runs unchanged.
    """
    flat = collate_component_rows(rows)
    n = len(rows)
    one = np.ones((n, 1), dtype=bool)
    return {
        "signal": flat["signal"][:, None, :, :],
        "valid_mask": flat["valid_mask"][:, None, :],
        "ot": flat["ot"][:, None, :],
        "physical": flat["physical"][:, None, :],
        "component_mask": one,
        "group_eye": np.zeros((n, 1), dtype=np.int64),
        "group_intensity": np.zeros((n, 1), dtype=np.int64),
        "component_type": np.asarray([r.component_id for r in rows], dtype=object)[:, None],
        "component_confidence": np.asarray(
            [float(r.landmark_confidence) for r in rows], dtype=np.float32
        )[:, None],
        "dataset": flat["dataset"],
    }


# -- loss terms ---------------------------------------------------------------
class _VICReg(nn.Module):
    """VICReg (Bardes et al. 2022) on two (N, D) projection tensors.

    ``inv_weight`` controls the invariance term; variance and covariance
    terms always use gamma=1.  Returns a scalar loss.
    """

    def __init__(self, inv_weight: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.inv_weight = inv_weight
        self.gamma = gamma

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim != 2 or a.shape != b.shape:
            raise ValueError(f"expected matching (N, D), got {tuple(a.shape)}/{tuple(b.shape)}")
        inv = torch.nn.functional.mse_loss(a, b)
        var = self._variance(a) + self._variance(b)
        cov = self._covariance(a) + self._covariance(b)
        return self.inv_weight * inv + self.gamma * var + self.gamma * cov

    @staticmethod
    def _variance(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0)
        return torch.mean(torch.nn.functional.relu(1.0 - std))

    @staticmethod
    def _covariance(x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        if n < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (n - 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).sum() / d


class MaskedReconstructionLoss(nn.Module):
    """§14.3: decode the fused token back to the raw component."""

    def __init__(self, token_dim: int, out_dim: int = CANONICAL_SAMPLES):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(
        self,
        enc: ComponentEncoding,
        signal: torch.Tensor,   # (B, 1, 128) masked input
        mask: torch.Tensor,     # (B, 1, 128) bool masked positions
        valid: torch.Tensor,    # (B, 1, 128) original valid samples
    ) -> torch.Tensor:
        target = signal * mask.to(signal.dtype)
        recon = self.decoder(enc.token).reshape_as(mask)
        select = mask & valid
        if not select.any():
            return torch.zeros((), device=signal.device, dtype=signal.dtype)
        loss = torch.nn.functional.huber_loss(
            recon[select], target[select], reduction="mean"
        )
        return loss


class RawOTConsistencyLoss(nn.Module):
    """§14.4: VICReg agreement between raw and OT views."""

    def __init__(self, stem_dim: int = 64, ssl_dim: int = 64, inv_weight: float = 1.0):
        super().__init__()
        self.raw_head = nn.Sequential(
            nn.Linear(stem_dim, ssl_dim), nn.LayerNorm(ssl_dim)
        )
        self.ot_head = nn.Sequential(
            nn.Linear(stem_dim, ssl_dim), nn.LayerNorm(ssl_dim)
        )
        self.vicreg = _VICReg(inv_weight=inv_weight)

    def forward(self, enc: ComponentEncoding) -> torch.Tensor:
        raw = enc.raw_token[enc.valid]
        ot = enc.ot_token[enc.valid]
        if raw.numel() == 0 or raw.shape[0] < 2:
            return torch.zeros((), device=raw.device, dtype=raw.dtype)
        return self.vicreg(self.raw_head(raw), self.ot_head(ot))


class AugmentationConsistencyLoss(nn.Module):
    """§14.5: two safe augmentations of the same component are a positive pair."""

    def __init__(self, token_dim: int, ssl_dim: int = 64, inv_weight: float = 1.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(token_dim, ssl_dim), nn.LayerNorm(ssl_dim)
        )
        self.vicreg = _VICReg(inv_weight=inv_weight)

    def forward(self, enc_a: ComponentEncoding, enc_b: ComponentEncoding) -> torch.Tensor:
        a = enc_a.token[enc_a.valid]
        b = enc_b.token[enc_b.valid]
        if a.numel() == 0 or a.shape[0] < 2:
            return torch.zeros((), device=a.device, dtype=a.dtype)
        return self.vicreg(self.head(a), self.head(b))


class GeometryPreservationLoss(nn.Module):
    """§14.6: within (dataset, component type) match embedding and sOT distances.

    Distances are z-score normalized with batch statistics (training-fold
    statistics in production); only groups with >= 2 members contribute.
    """

    @staticmethod
    def forward(enc: ComponentEncoding, batch: dict) -> torch.Tensor:
        tokens = enc.token[enc.valid]
        types = np.asarray(batch["component_type"], dtype=object)[enc.valid.detach().cpu().numpy()]
        ot = torch.as_tensor(batch["ot"], dtype=torch.float32, device=tokens.device)[enc.valid]
        if tokens.shape[0] < 2:
            return torch.zeros((), device=tokens.device, dtype=tokens.dtype)
        total = torch.zeros((), device=tokens.device, dtype=tokens.dtype)
        n_groups = 0
        for t in np.unique(types):
            members = np.flatnonzero(types == t)
            if members.size < 2:
                continue
            z = torch.nn.functional.normalize(tokens[members], dim=-1)
            dz = torch.cdist(z, z)
            s = ot[members]
            ds = torch.cdist(s, s)
            if float(ds.max()) > 0 and float(dz.max()) > 0:
                dz_n = dz / dz.mean()
                ds_n = ds / ds.mean()
            else:
                continue
            m = torch.triu(torch.ones_like(dz), diagonal=1).bool()
            if not m.any():
                continue
            total = total + torch.nn.functional.huber_loss(
                dz_n[m], ds_n[m], reduction="mean"
            )
            n_groups += 1
        return total / max(1, n_groups)


class GatePriorLoss(nn.Module):
    """§4.5: (g - g0)^2 over permitted shared edges, g0 from GraphConfig."""

    def __init__(self, gate_prior: float = 0.75):
        super().__init__()
        self.gate_prior = float(gate_prior)

    def forward(self, enc: ComponentEncoding) -> tuple[torch.Tensor, float]:
        if enc.pathway_gate_strength is None:
            return torch.zeros((), dtype=torch.float32), 0.0
        g = enc.pathway_gate_strength[enc.valid]
        if g.numel() == 0:
            return torch.zeros((), device=g.device, dtype=g.dtype), 0.0
        loss = ((g - self.gate_prior) ** 2).mean()
        return loss, float(loss.item())


class JointSSLLoss(nn.Module):
    """Stage B reference objective (plan §14.2) with per-domain terms.

    Heads live here and are discarded after pretraining (plan §14.4).
    """

    def __init__(self, config: SSLConfig, token_dim: int, stem_dim: int = 64):
        super().__init__()
        self.lambda_mask = float(config.lambda_mask)
        self.lambda_view = float(config.lambda_view)
        self.lambda_aug = float(config.lambda_aug)
        self.lambda_geom = float(config.lambda_geom)
        self.lambda_prior = float(config.lambda_prior)
        self.mask_len = int(config.mask_len)
        self.masked = MaskedReconstructionLoss(token_dim)
        self.view = RawOTConsistencyLoss(stem_dim, config.ssl_dim)
        self.aug = AugmentationConsistencyLoss(token_dim, config.ssl_dim)
        self.geom = GeometryPreservationLoss()
        self.prior = GatePriorLoss(config.gate_prior)

    def forward(
        self, model, batch: dict, seed_offset: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss for one (single-dataset) component batch, plus per-term log."""
        device = next(model.parameters()).device
        rng = np.random.default_rng(seed_offset)
        signal = torch.as_tensor(batch["signal"], dtype=torch.float32, device=device)
        valid = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=device)
        B, L, _, T = signal.shape
        flat_signal = signal.reshape(B * L, T).detach().cpu().numpy()
        flat_valid = valid.reshape(B * L, T).detach().cpu().numpy()

        masked_np, mask_np = mask_contiguous_span(
            flat_signal, flat_valid, self.mask_len, rng
        )
        batch_m = dict(batch)
        batch_m["signal"] = masked_np.reshape(B, L, 1, T).astype(np.float32)
        enc_m = model.encode_component(batch_m)
        enc = model.encode_component(batch)
        mask = torch.as_tensor(mask_np.reshape(B, L, 1, T), dtype=torch.bool, device=device)
        valid_t = valid.reshape(B, L, 1, T)
        loss_mask = self.lambda_mask * self.masked(enc_m, torch.as_tensor(
            batch_m["signal"], dtype=torch.float32, device=device
        ), mask, valid_t)

        loss_view = self.lambda_view * self.view(enc)

        aug_a_np = augment_signal(flat_signal, flat_valid, rng)
        aug_b_np = augment_signal(flat_signal, flat_valid, rng)
        batch_a = dict(batch)
        batch_a["signal"] = aug_a_np.reshape(B, L, 1, T).astype(np.float32)
        batch_b = dict(batch)
        batch_b["signal"] = aug_b_np.reshape(B, L, 1, T).astype(np.float32)
        enc_a = model.encode_component(batch_a)
        enc_b = model.encode_component(batch_b)
        loss_aug = self.lambda_aug * self.aug(enc_a, enc_b)

        loss_geom = self.lambda_geom * self.geom(enc, batch)
        loss_prior, prior_val = self.prior(enc)

        total = loss_mask + loss_view + loss_aug + loss_geom + self.lambda_prior * loss_prior
        terms = {
            "mask": float(loss_mask.item()),
            "view": float(loss_view.item()),
            "aug": float(loss_aug.item()),
            "geom": float(loss_geom.item()),
            "prior": float(prior_val),
            "total": float(total.item()),
        }
        return total, terms


# -- pretraining loop ---------------------------------------------------------
def pretrain_ssl(
    cfg: SSLConfig,
    data_cfg: DataConfig,
    caches: LoadedCaches | None = None,
) -> tuple[Path, SSLLog]:
    """Run Stage B on all folds except ``exclude_fold``; staged checkpoint.

    Domain-balanced steps (plan 14.2): one LEOP batch + one PERG batch per
    step, losses summed with equal weight so both domains contribute
    equally to every optimizer step after the first.
    """
    if cfg.exclude_fold is None:
        raise ValueError("SSL pretraining must exclude one outer fold (plan §23.6)")
    train_folds = set(cfg.outer_folds) - {cfg.exclude_fold}
    if not train_folds:
        raise ValueError("no training folds left after exclusion")

    caches = caches or LoadedCaches(data_cfg.artifact_root, fold_version=cfg.fold_version)
    leop = ComponentDataset(caches, "LEOP", outer_folds=train_folds)
    perg = ComponentDataset(caches, "PERG", outer_folds=train_folds)
    if len(leop) == 0 or len(perg) == 0:
        raise ValueError("empty SSL component pool for one domain")

    model = build_model(
        ModelConfig(routing_graph=cfg.routing_graph, stems_seed=cfg.seed,
                    agg_seed=cfg.seed, head_seed=cfg.seed)
    )
    model.to(cfg.device)
    model.train()
    loss_fn = JointSSLLoss(cfg, token_dim=128)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(
        int(np.ceil(len(leop) / cfg.leop_batch)),
        int(np.ceil(len(perg) / cfg.perg_batch)),
    )
    total_steps = cfg.epochs * steps_per_epoch
    sched = _WarmupCosine(
        optimizer, warmup_steps=cfg.warmup_epochs * steps_per_epoch,
        total_steps=total_steps,
    )
    plan = domain_balanced_batch_indices(
        len(leop), len(perg), cfg.leop_batch, cfg.perg_batch, seed=cfg.seed
    )
    log = SSLLog()
    step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_terms: dict[str, float] = {}
        n_steps = 0
        for leop_idx, perg_idx in plan:
            if n_steps >= steps_per_epoch:
                break
            step += 1
            terms_both: dict[str, float] = {}
            total = torch.zeros((), device=cfg.device, dtype=torch.float32)
            for idx, ds in ((leop_idx, "LEOP"), (perg_idx, "PERG")):
                if len(idx) == 0:
                    continue
                rows = [leop[i] for i in idx] if ds == "LEOP" else [perg[i] for i in idx]
                batch = collate_component_batch(rows)
                loss, terms = loss_fn(model, batch, seed_offset=step * 1000 + (1 if ds == "PERG" else 0))
                total = total + loss
                for k, v in terms.items():
                    terms_both[f"{ds.lower()}_{k}"] = v
            if n_steps == 0 and float(total.item()) == 0.0:
                continue
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()), cfg.grad_clip
            )
            optimizer.step()
            sched.step()
            epoch_loss += float(total.item())
            for k, v in terms_both.items():
                epoch_terms[k] = epoch_terms.get(k, 0.0) + v
            n_steps += 1
            if step % cfg.log_every == 0:
                log.gate_prior.append(epoch_terms.get("leop_prior", 0.0) / max(1, n_steps))
        log.train_loss.append(epoch_loss / max(1, n_steps))
        for k, v in epoch_terms.items():
            log.per_domain.setdefault(k, []).append(v / max(1, n_steps))
        log.best_epoch = epoch

    out_dir = Path(data_cfg.artifact_root) / "results" / cfg.output_subdir / f"fold{cfg.exclude_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError(f"completed SSL run already exists: {out_dir}")
    payload = {
        "model": model.state_dict(),
        "heads": loss_fn.state_dict(),
        "config": asdict(cfg),
        "log": asdict(log),
        "train_folds": sorted(train_folds),
        "exclude_fold": cfg.exclude_fold,
        "n_components": {"LEOP": len(leop), "PERG": len(perg)},
    }
    tmp = out_dir / "final.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(out_dir / "final.pt")
    _write_ssl_manifest(cfg, data_cfg, out_dir)
    complete.write_text("complete\n")
    return out_dir / "final.pt", log


def _write_ssl_manifest(
    cfg: SSLConfig, data_cfg: DataConfig, out_dir: Path
) -> None:
    root = Path(data_cfg.artifact_root)
    outer = root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=cfg.fold_version)
    inner = root / "data" / "splits" / INNER_FOLDS_TEMPLATE.format(version=cfg.fold_version)
    manifest = RunManifest(kind="ssl_pretrain", name=out_dir.name)
    manifest.config_hash = sha256_text(json.dumps(asdict(cfg), sort_keys=True))
    manifest.data_hash = sha256_file(
        cache_paths(root, CACHE_SCHEMA_VERSION)["manifest"]
    )
    manifest.split_hash = sha256_text(sha256_file(outer) + sha256_file(inner))
    manifest.code_revision = git_revision(Path.cwd())
    manifest.extra = {"exclude_fold": cfg.exclude_fold, "checkpoint": "final.pt"}
    manifest.write_atomic(out_dir / "run_manifest.json")

"""Local stem + fusion tests (plan Module 21.13): shapes, masks, determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.models.local_fusion import FUSED_DIM, LocalFusion
from pathway_erg.models.ot_stem import OT_DIM_OUT, OTStem
from pathway_erg.models.raw_stem import RAW_DIM, RawStem


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(3)
    np.random.seed(3)
    raw = torch.randn(4, 1, 128)
    mask = torch.ones(4, 128, dtype=torch.bool)
    mask[:, 100:] = False  # invalid tail
    ot = torch.randn(4, 135)
    phys = torch.randn(4, 8)
    return raw, mask, ot, phys


def test_raw_stem_shape(batch):
    raw, mask, _, _ = batch
    out = RawStem(seed=4)(raw, mask)
    assert tuple(out.shape) == (4, RAW_DIM)


def test_raw_stem_deterministic(batch):
    raw, mask, _, _ = batch
    a = RawStem(seed=4)(raw, mask)
    b = RawStem(seed=4)(raw, mask)
    assert torch.equal(a, b)


def test_raw_stem_finite_with_empty_mask(batch):
    raw, _, _, _ = batch
    empty = torch.zeros(4, 128, dtype=torch.bool)
    out = RawStem(seed=4)(raw, empty)
    assert torch.isfinite(out).all()


def test_raw_stem_gradients(batch):
    raw, mask, _, _ = batch
    stem = RawStem(seed=4)
    out = stem(raw, mask)
    loss = out.sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in stem.parameters())


def test_raw_stem_rejects_bad_shape(batch):
    raw, mask, _, _ = batch
    stem = RawStem(seed=4)
    with pytest.raises(ValueError):
        stem(raw[:, 0, :50].unsqueeze(0), mask[:1, :50])


def test_raw_stem_masked_pooling_respects_mask(batch):
    raw, mask, _, _ = batch
    stem = RawStem(seed=4)
    full = stem(raw, torch.ones(4, 128, dtype=torch.bool))
    partial = stem(raw, torch.zeros(4, 128, dtype=torch.bool))
    assert not torch.equal(full, partial)


def test_ot_stem_shape_and_determinism(batch):
    _, _, ot, _ = batch
    a = OTStem(seed=4)(ot)
    b = OTStem(seed=4)(ot)
    assert tuple(a.shape) == (4, OT_DIM_OUT)
    assert torch.equal(a, b)


def test_ot_stem_finite(batch):
    _, _, ot, _ = batch
    assert torch.isfinite(OTStem(seed=4)(ot)).all()


def test_local_fusion_shape_and_gate(batch):
    raw, mask, ot, phys = batch
    ez = RawStem(seed=4)(raw, mask)
    oz = OTStem(seed=4)(ot)
    fused, alpha = LocalFusion(seed=4)(ez, oz, phys)
    assert tuple(fused.shape) == (4, FUSED_DIM)
    assert tuple(alpha.shape) == (4,)
    assert ((0.0 <= alpha) & (alpha <= 1.0)).all()


def test_local_fusion_deterministic(batch):
    raw, mask, ot, phys = batch
    ez = RawStem(seed=4)(raw, mask)
    oz = OTStem(seed=4)(ot)
    f1, a1 = LocalFusion(seed=4)(ez, oz, phys)
    f2, a2 = LocalFusion(seed=4)(ez, oz, phys)
    assert torch.equal(f1, f2) and torch.equal(a1, a2)


def test_local_fusion_gradients(batch):
    raw, mask, ot, phys = batch
    ez = RawStem(seed=4)(raw, mask)
    oz = OTStem(seed=4)(ot)
    model = LocalFusion(seed=4)
    fused, alpha = model(ez, oz, phys)
    (fused.sum() + alpha.mean()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_dims_flow_from_datasets():
    # Phase-6 contract: 128 canonical samples, 135-d OT, 8 physical feats
    from pathway_erg.data.collate import collate_component_rows
    from pathway_erg.data.datasets import CANONICAL_SAMPLES, OT_DIM

    assert CANONICAL_SAMPLES == 128
    assert OT_DIM == 135
    assert collate_component_rows is not None

"""Aggregator tests (plan Module 21.15): pooling via gated attention.

Plan-mandated cases: permutation invariance, mask invariance, attention
sums, single-element bags, missing eye/session (empty mas0k), no NaN on
all optional components missing.
"""

from __future__ import annotations

import pytest
import torch

from pathway_erg.models.aggregators import (
    ComponentToEyeAggregator,
    EyeToParticipantAggregator,
    EyeToSessionAggregator,
    IntensityToEyeAggregator,
    SessionToVisitAggregator,
)

DIM = 32


@pytest.fixture(scope="module")
def tokens():
    torch.manual_seed(3)
    return torch.randn(4, 6, DIM)


def test_attention_sums_to_one(tokens):
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    _, attn = EyeToParticipantAggregator(DIM, seed=2)(tokens, mask)
    assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-4)


def test_attention_respects_mask(tokens):
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[0, :2] = True
    pooled, attn = EyeToParticipantAggregator(DIM, seed=4)(tokens, mask)
    assert torch.allclose(attn[0].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.count_nonzero(attn[1:]) == 0
    assert torch.count_nonzero(attn[0, 2:]) == 0
    # pooled is a convex combination of the valid tokens only
    assert torch.allclose(pooled[0], attn[0] @ tokens[0], atol=1e-4)


def test_attention_pooling_is_permutation_invariant(tokens):
    mask = torch.ones(4, 6, dtype=torch.bool)
    agg = EyeToParticipantAggregator(DIM, seed=5)
    pooled, _ = agg(tokens, mask)
    perm = torch.tensor([5, 3, 1, 0, 2, 4])
    pooled_p, _ = agg(tokens[:, perm], mask[:, perm])
    assert torch.allclose(pooled, pooled_p, atol=1e-4)


def test_back_to_back_determinism(tokens):
    agg = EyeToParticipantAggregator(DIM, seed=4)
    mask = torch.ones(4, 6, dtype=torch.bool)
    p1, a1 = agg(tokens, mask)
    p2, a2 = agg(tokens, mask)
    assert torch.equal(p1, p2) and torch.equal(a1, a2)


def test_empty_mask_no_nan(tokens):
    mp = torch.zeros(4, 6, dtype=torch.bool)
    p, a = EyeToParticipantAggregator(DIM, seed=4)(tokens, mp)
    assert torch.isfinite(p).all()
    assert torch.equal(a, torch.zeros_like(a))


def test_single_element_bag(tokens):
    one = torch.zeros(4, 6, dtype=torch.bool)
    one[2, 3] = True
    p, a = EyeToParticipantAggregator(DIM, seed=4)(tokens, one)
    assert torch.allclose(p[2], tokens[2, 3], atol=1e-4)
    assert torch.allclose(a[2, 3], torch.tensor(1.0), atol=1e-4)


def test_all_registry_modules_pool(tokens):
    mask = torch.ones(4, 6, dtype=torch.bool)
    for cls in (
        IntensityToEyeAggregator,
        EyeToParticipantAggregator,
        ComponentToEyeAggregator,
        EyeToSessionAggregator,
        SessionToVisitAggregator,
    ):
        p, a = cls(DIM, seed=1)(tokens, mask)
        assert p.shape == (4, DIM)
        assert a.shape == (4, 6)
        assert torch.isfinite(p).all()


def test_dim_and_mask_validation(tokens):
    agg = EyeToParticipantAggregator(DIM, seed=4)
    with pytest.raises(ValueError):
        agg(tokens[:, :, :DIM - 2], torch.ones(4, 6, dtype=torch.bool))
    with pytest.raises(ValueError):
        agg(tokens, torch.ones(4, 5, dtype=torch.bool))

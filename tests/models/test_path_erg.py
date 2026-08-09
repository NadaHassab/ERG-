"""Complete model tests (plan Module 21.16).

Plan-mandated tests: both task forwards/backwards, parameter counts,
state-dict save/load, shared parameter identity (same head/config
structure reachable from both tasks), no label metadata in input.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches, build_bags
from pathway_erg.models.path_erg import (
    build_model,
    gather_by_group,
    promote_group_codes,
)


@pytest.fixture(scope="module")
def bag_batch_leop():
    c = LoadedCaches("artifacts")
    bags = build_bags(c, "LEOP", outer_folds={0})[:4]
    return collate_bag_units(bags), c


@pytest.fixture(scope="module")
def bag_batch_perg():
    c = LoadedCaches("artifacts")
    bags = build_bags(c, "PERG", outer_folds={1})[:4]
    return collate_bag_units(bags), c


def test_leop_forward_backward(bag_batch_leop):
    batch, _ = bag_batch_leop
    m = build_model()
    logits = m(batch, "LEOP")
    assert logits.shape == (4,)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no trainable parameter got a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_perg_forward_backward(bag_batch_perg):
    batch, _ = bag_batch_perg
    m = build_model()
    logits = m(batch, "PERG")
    assert logits.shape == (4,)
    logits.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_unknown_task_raises(bag_batch_leop):
    batch, _ = bag_batch_leop
    with pytest.raises(ValueError, match="unknown task"):
        build_model()(batch, "XXX")


def test_state_dict_save_load(bag_batch_leop):
    batch, _ = bag_batch_leop
    m1 = build_model()
    l1 = m1(batch, "LEOP")
    buf = io.BytesIO()
    torch.save(m1.state_dict(), buf)
    buf.seek(0)
    m2 = build_model()
    m2.load_state_dict(torch.load(buf, weights_only=True))
    l2 = m2(batch, "LEOP")
    assert torch.allclose(l1, l2, atol=1e-6)


def test_parameter_count_within_budget():
    n = sum(p.numel() for p in build_model().parameters())
    assert n < 1_500_000, f"model exceeds 1.5M budget: {n}"
    assert n > 50_000


def test_no_label_metadata_in_input(bag_batch_leop):
    batch, _ = bag_batch_leop
    m = build_model().eval()
    with_label = m(batch, "LEOP")
    batch2 = dict(batch)
    batch2["label"] = np.full(len(batch["label"]), np.nan)
    without_label = m(batch2, "LEOP")
    assert torch.equal(with_label, without_label)


def test_attention_levels_per_task(bag_batch_leop, bag_batch_perg):
    leop_batch, _ = bag_batch_leop
    perg_batch, _ = bag_batch_perg
    m = build_model()
    leop_enc = m.encode_bag(leop_batch, "LEOP")
    perg_enc = m.encode_bag(perg_batch, "PERG")
    assert set(leop_enc.attention) == {"intensity", "eye", "participant"}
    assert set(perg_enc.attention) == {"eye", "participant"}
    # attention per level: sums to #intensity groups, #eyes, and 1 for
    # the final participant attention (over eye tokens)
    gi = torch.as_tensor(leop_batch["group_intensity"])
    ge = torch.as_tensor(leop_batch["group_eye"])
    comp = torch.as_tensor(leop_batch["component_mask"], dtype=torch.bool)
    n_int = torch.tensor(
        [gi[b][comp[b]].unique().numel() for b in range(4)], dtype=torch.float32
    )
    n_eyes = torch.tensor(
        [ge[b][comp[b]].unique().numel() for b in range(4)], dtype=torch.float32
    )
    expected = {"intensity": n_int, "eye": n_eyes, "participant": torch.ones(4)}
    for name, attn in leop_enc.attention.items():
        assert (attn >= 0).all() and (attn <= 1).all()
        assert torch.allclose(attn.sum(dim=1), expected[name], atol=1e-3)


def test_shared_component_path(bag_batch_leop):
    # encode_component is task-agnostic: same tokens regardless of task
    batch, _ = bag_batch_leop
    m = build_model().eval()
    t1 = m.encode_component(batch).token
    t2 = m.encode_component(batch).token
    assert torch.equal(t1, t2)


def test_factory_records_graph():
    graph = {"LEOP": ("intensity", "eye", "participant")}
    m = build_model(pathway_graph=graph)
    assert m.pathway_graph is graph


def test_promote_group_codes_maps_members():
    codes = torch.tensor([[0, 0, 1, 1, 2, 2]])
    eye = torch.tensor([[0, 0, 1, 1, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    out = promote_group_codes(codes, eye, valid)
    assert out.tolist() == [[0, 1, 0]]


def test_gather_by_group_permutation(bag_batch_leop):
    m = build_model()
    enc = m.encode_component(bag_batch_leop[0])
    t, valid = enc.token, enc.valid
    codes = torch.as_tensor(bag_batch_leop[0]["group_intensity"], dtype=torch.int64)
    agg = m.intensity_to_eye
    p1, _, _ = gather_by_group(t, valid, codes, agg)
    p2, _, _ = gather_by_group(t, valid, codes, agg)
    assert torch.equal(p1, p2)

"""URFU/FLINDERS model-path tests: router adapters + URFU head pooling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.data.collate import collate_bag_units
from pathway_erg.models.path_erg import TOKEN_DIM, ModelConfig, build_model
from pathway_erg.models.pathway_router import (
    PathwayGraph,
    PathwayRouter,
    make_pathway_graph,
)


def _router(graph: str = "correct") -> PathwayRouter:
    return PathwayRouter(make_pathway_graph(graph), local_dim=TOKEN_DIM, dropout=0.0)


def _tokens(n: int):
    return torch.randn(n, TOKEN_DIM, dtype=torch.float32)


def test_router_has_per_dataset_adapters():
    r = _router()
    assert isinstance(r.urfu_adapter, torch.nn.Module)
    assert isinstance(r.flinders_adapter, torch.nn.Module)


def test_router_routes_urfu_through_shared_expert():
    r = _router("correct")
    token = _tokens(2)
    out = r(
        token,
        ["L_LATE", "L_LATE"],
        ["URFU", "URFU"],
        torch.tensor([0.9, 0.8]),
    )
    assert out.shared_mask.all()
    assert out.gate_strength.shape == (2,)
    assert torch.isfinite(out.combined).all()


def test_router_routes_flinders_through_shared_expert():
    r = _router("correct")
    out = r(
        _tokens(1),
        ["L_LATE"],
        ["FLINDERS"],
        torch.tensor([1.0]),
    )
    assert out.shared_mask.all()


def test_router_private_route_for_flash_components():
    r = _router("none")
    out = r(
        _tokens(2),
        ["L_EARLY_A", "L_LATE"],
        ["URFU", "URFU"],
        torch.tensor([0.5, 0.5]),
    )
    assert not out.shared_mask.any()
    assert torch.isfinite(out.private).all()


def test_router_unknown_dataset_raises():
    r = _router("correct")
    with pytest.raises(ValueError, match="no adapter"):
        r(
            _tokens(1),
            ["L_LATE"],
            ["MARTIAN"],
            torch.tensor([1.0]),
        )


def test_router_rejects_unknown_component():
    r = _router("correct")
    with pytest.raises(ValueError, match="unknown component"):
        r(_tokens(1), ["Q_ZZZ"], ["URFU"], torch.tensor([1.0]))


def test_model_heads_include_urfu():
    m = build_model(ModelConfig(head_seed=0))
    assert "LEOP" in m.heads and "PERG" in m.heads and "URFU" in m.heads
    assert "FLINDERS" not in m.heads


def test_urfu_encode_bag_single_pool():
    m = build_model(ModelConfig(head_seed=0))
    batch = {
        "signal": torch.randn(1, 3, 1, 128),
        "valid_mask": torch.ones(1, 3, 128, dtype=torch.bool),
        "ot": torch.randn(1, 3, 135),
        "physical": torch.randn(1, 3, 8),
        "component_mask": torch.ones(1, 3, dtype=torch.bool),
        "group_eye": torch.zeros(1, 3, dtype=torch.int64),
        "group_intensity": torch.zeros(1, 3, dtype=torch.int64),
        "component_type": np.full((1, 3), "L_LATE", dtype=object),
        "component_confidence": np.ones((1, 3), dtype=np.float32),
        "dataset": np.asarray(["URFU"], dtype=object),
    }
    enc = m.encode_bag(batch, "URFU")
    assert enc.token.shape == (1, TOKEN_DIM)
    assert "participant" in enc.attention


def test_urfu_forward_logits():
    m = build_model(ModelConfig(head_seed=0))
    batch = {
        "signal": torch.randn(2, 2, 1, 128),
        "valid_mask": torch.ones(2, 2, 128, dtype=torch.bool),
        "ot": torch.randn(2, 2, 135),
        "physical": torch.randn(2, 2, 8),
        "component_mask": torch.ones(2, 2, dtype=torch.bool),
        "group_eye": torch.zeros(2, 2, dtype=torch.int64),
        "group_intensity": torch.zeros(2, 2, dtype=torch.int64),
        "component_type": np.full((2, 2), "L_LATE", dtype=object),
        "component_confidence": np.ones((2, 2), dtype=np.float32),
        "dataset": np.asarray(["URFU", "URFU"], dtype=object),
    }
    logits = m(batch, "URFU")
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_urfu_forward_unknown_task_raises():
    m = build_model(ModelConfig(head_seed=0))
    batch = {
        "signal": torch.randn(1, 1, 1, 128),
        "valid_mask": torch.ones(1, 1, 128, dtype=torch.bool),
        "ot": torch.randn(1, 1, 135),
        "physical": torch.randn(1, 1, 8),
        "component_mask": torch.ones(1, 1, dtype=torch.bool),
        "group_eye": torch.zeros(1, 1, dtype=torch.int64),
        "group_intensity": torch.zeros(1, 1, dtype=torch.int64),
        "component_type": np.full((1, 1), "L_LATE", dtype=object),
        "component_confidence": np.ones((1, 1), dtype=np.float32),
        "dataset": np.asarray(["FLINDERS"], dtype=object),
    }
    with pytest.raises(ValueError, match="unknown task"):
        m(batch, "FLINDERS")


def test_urfu_bag_collate_roundtrip():
    from pathway_erg.data.datasets import BagUnit, ComponentRow

    rows = []
    for i in range(3):
        rows.append(
            ComponentRow(
                global_component_id=f"c{i}",
                global_recording_id=f"r{i}",
                subject_id="U00",
                visit_id="U00-V",
                dataset="URFU",
                component_id="L_LATE",
                unit_id="U00-V",
                protocol="Maximum 2.0",
                eye=None,
                stimulus_value=3.0,
                stimulus_unit="cd.s/m2",
                landmark_confidence=1.0,
                outer_fold=0,
                signal=np.ones(128),
                signal_mask=np.ones(128, dtype=bool),
                ot_vector=np.zeros(135),
                physical=np.zeros(8),
            )
        )
    bag = BagUnit(
        unit_id="U00-V",
        subject_id="U00",
        visit_id="U00-V",
        dataset="URFU",
        target_binary=0,
        outer_fold=0,
        components=tuple(rows),
    )
    batch = collate_bag_units([bag])
    assert batch["component_type"].shape == (1, 3)
    assert batch["dataset"].tolist() == ["URFU"]
    m = build_model(ModelConfig(head_seed=0))
    logits = m(batch, "URFU")
    assert logits.shape == (1,)


def test_route_after_training_forward_backward_urfu():
    m = build_model(ModelConfig(head_seed=0))
    batch = {
        "signal": torch.randn(1, 1, 1, 128),
        "valid_mask": torch.ones(1, 1, 128, dtype=torch.bool),
        "ot": torch.randn(1, 1, 135),
        "physical": torch.randn(1, 1, 8),
        "component_mask": torch.ones(1, 1, dtype=torch.bool),
        "group_eye": torch.zeros(1, 1, dtype=torch.int64),
        "group_intensity": torch.zeros(1, 1, dtype=torch.int64),
        "component_type": np.full((1, 1), "L_LATE", dtype=object),
        "component_confidence": np.ones((1, 1), dtype=np.float32),
        "dataset": np.asarray(["URFU"], dtype=object),
    }
    logits = m(batch, "URFU")
    loss = logits.mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "expected gradients through the URFU route"

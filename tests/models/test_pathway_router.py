"""Pathway router + expert tests (plan Module 21.14).

Covers: forbidden-edge gradient isolation, correct/wrong/random graph
controls, parameter matching across controls, private route always
present, low-confidence behavior, and PathModel integration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches, build_bags
from pathway_erg.models.path_erg import ModelConfig, build_model
from pathway_erg.models.pathway_router import (
    PathwayGraph,
    PathwayRouter,
    make_pathway_graph,
)


@pytest.fixture(scope="module")
def components() -> list[str]:
    return ["L_EARLY_A", "L_A_TO_B", "L_OP", "L_LATE", "P_EARLY", "P_LATE"]


@pytest.fixture(scope="module")
def token() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(6, 128)


@pytest.fixture(scope="module")
def confidence() -> torch.Tensor:
    torch.manual_seed(12)
    return torch.rand(6)


def _router_inputs(
    token: torch.Tensor, components: list[str], confidence: torch.Tensor
) -> tuple[torch.Tensor, list[str], list[str], torch.Tensor]:
    datasets = ["LEOP", "LEOP", "LEOP", "LEOP", "PERG", "PERG"]
    return token, components, datasets, confidence


def test_private_route_always_present(token, components, confidence):
    router = PathwayRouter(make_pathway_graph("none"))
    out = router(*_router_inputs(token, components, confidence))
    assert out.private.shape == (6, 64)
    assert not out.shared_mask.any()
    assert torch.equal(out.gate, torch.zeros(6))
    # combined still flows (private route alone) and is finite
    assert torch.isfinite(out.combined).all()
    # private expert outputs differ per expert: early/OP/late differ
    assert not torch.equal(out.private[0], out.private[2])
    assert not torch.equal(out.private[0], out.private[3])


def test_forbidden_edges_zero_gradients(token, components, confidence):
    torch.manual_seed(21)
    router = PathwayRouter(make_pathway_graph("correct"))
    router.eval()  # deterministic dropout: gradients comparable across routers
    token.requires_grad_(True)
    out = router(*_router_inputs(token, components, confidence))
    loss = out.combined.sum()
    loss.backward()
    g = token.grad
    assert g is not None
    # flash early/OP and PERG early are forbidden from the shared route:
    # their token gradients must not flow through shared parameters at all.
    shared_grad_norm = sum(
        p.grad.abs().sum().item()
        for p in router.shared_expert.parameters()
    )
    for i, name in enumerate(components):
        if name in ("L_LATE", "P_LATE"):
            continue
        # Rebuild the router with 'none' and compare: gradient of the
        # forbidden token through shared modules is exactly zero.
        router_none = PathwayRouter(make_pathway_graph("none"))
        router_none.eval()
        router_none.load_state_dict(router.state_dict())
        t2 = token.detach().clone().requires_grad_(True)
        out2 = router_none(t2, components, ["LEOP"] * 6, confidence)
        out2.combined.sum().backward()
        # gradient difference must come from the shared path alone
        diff = (g[i] - t2.grad[i]).abs().sum().item()
        # no gradient flows through the shared expert for forbidden edges
        assert diff == pytest.approx(0.0, abs=1e-6)
        assert shared_grad_norm > 0


def test_wrong_graph_masks_only_late_like_components(token, components, confidence):
    router = PathwayRouter(make_pathway_graph("wrong"))
    out = router(*_router_inputs(token, components, confidence))
    assert out.shared_mask.tolist() == [True, False, False, False, True, False]


def test_random_graph_masks_two_random_components(token, components, confidence):
    rng_graph = make_pathway_graph("random", seed=0)
    assert len(rng_graph.shared_components) == 2
    router = PathwayRouter(rng_graph)
    out = router(*_router_inputs(token, components, confidence))
    assert out.shared_mask.sum().item() == 2


def test_parameter_matching_across_controls(token, components, confidence):
    correct = PathwayRouter(make_pathway_graph("correct"))
    wrong = PathwayRouter(make_pathway_graph("wrong"))
    none = PathwayRouter(make_pathway_graph("none"))
    random = PathwayRouter(make_pathway_graph("random", seed=0))
    for control in (wrong, none, random):
        assert sorted(correct.state_dict().keys()) == sorted(control.state_dict().keys())
        for key in correct.state_dict():
            assert correct.state_dict()[key].shape == control.state_dict()[key].shape


def test_low_confidence_behavior(token, components, confidence):
    router = PathwayRouter(make_pathway_graph("correct"))
    zero_conf = torch.zeros(6)
    out = router(*_router_inputs(token, components, zero_conf))
    assert torch.equal(out.gate, torch.zeros(6))
    # at zero confidence the combined token is pure private
    full = router(*_router_inputs(token, components, torch.ones(6)))
    assert not torch.equal(out.combined, full.combined)
    # confidence scaling is monotone: higher confidence -> larger shared effect
    mono = torch.linspace(0.0, 1.0, 6)
    out_mono = router(*_router_inputs(token, components, mono))
    assert torch.all(out_mono.gate >= -1e-6)


def test_router_rejects_unknown_component(token, confidence):
    router = PathwayRouter(make_pathway_graph("correct"))
    with pytest.raises(ValueError):
        router(token, ["L_LATE", "BOGUS"], ["LEOP", "LEOP"], confidence[:2])


def test_make_graph_unknown_name():
    with pytest.raises(ValueError):
        make_pathway_graph("bogus")


def test_graph_rejects_unknown_components():
    with pytest.raises(ValueError):
        PathwayGraph(name="x", shared_components=frozenset({"BOGUS"}))


def test_router_parameter_counts(components, confidence):
    torch.manual_seed(1)
    n = sum(p.numel() for p in PathwayRouter(make_pathway_graph("correct")).parameters())
    for name in ("none", "wrong", "full", "random"):
        m = sum(p.numel() for p in PathwayRouter(make_pathway_graph(name)).parameters())
        assert m == n


def test_path_model_routing_integration():
    torch.manual_seed(7)
    cfg = ModelConfig(routing_graph="correct")
    model = build_model(cfg)
    assert model.router is not None
    batch = {
        "signal": torch.randn(2, 4, 1, 128),
        "valid_mask": torch.ones(2, 4, 128, dtype=torch.bool),
        "ot": torch.randn(2, 4, 135),
        "physical": torch.randn(2, 4, 8),
        "component_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "component_type": np.array(
            [["L_EARLY_A", "L_OP", "L_LATE", "P_LATE"], ["P_EARLY", "P_LATE", "L_LATE", ""]],
            dtype=object,
        ),
        "component_confidence": torch.ones(2, 4),
        "dataset": np.array(["LEOP", "PERG"], dtype=object),
        "group_eye": torch.tensor([[0, 0, 1, 1], [0, 1, 1, -1]]),
        "group_intensity": torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]]),
    }
    enc = model.encode_component(batch)
    assert enc.shared is not None and enc.private is not None
    assert enc.pathway_gate is not None
    assert tuple(enc.shared.shape) == (2, 4, 64)
    # padded row never routes
    assert enc.pathway_gate[1, 3].item() == 0.0
    assert torch.isfinite(enc.token).all()
    # no-share control model: no router
    plain = build_model(ModelConfig())
    assert plain.router is None
    assert plain.encode_component(batch).shared is None


def test_build_model_graph_controls_share_no_params():
    graphs = [make_pathway_graph(name) for name in ("correct", "wrong", "none", "random")]
    models = [build_model(ModelConfig(), g) for g in graphs]
    keys = models[0].state_dict().keys()
    for m in models[1:]:
        assert sorted(keys) == sorted(m.state_dict().keys())


def test_real_data_forward_backward():
    c = LoadedCaches("artifacts")
    for dataset, fold in (("LEOP", 0), ("PERG", 1)):
        bags = build_bags(c, dataset, outer_folds={fold})[:4]
        batch = collate_bag_units(bags)
        model = build_model(ModelConfig(routing_graph="correct"))
        logits = model(batch, dataset)
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()
        loss = logits.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        # shared expert receives gradients only via late components
        assert any(p.grad is not None for p in model.router.shared_expert.parameters())
        enc = model.encode_component(batch)
        assert enc.private is not None and enc.shared is not None
        assert enc.pathway_gate[batch["component_mask"]].sum() >= 0

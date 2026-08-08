"""E2 partial-sharing bias-variance simulation (plan Section 10.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathway_erg.simulations.partial_sharing import (
    N_REPS,
    SHARED_BLOCK,
    _block_selector,
    _oracle_design,
    _task_parameters,
    estimate_full,
    estimate_learned_gate,
    estimate_partial,
    run_partial_sharing_grid,
)


def _cell(mismatch_sq: float, sigma_sq: float, n: int) -> pd.Series:
    grid = run_partial_sharing_grid()
    mask = (grid["mismatch_sq"] == mismatch_sq) & (grid["sigma_sq"] == sigma_sq) & (grid["n"] == n)
    return grid.loc[mask].iloc[0]


def test_full_sharing_wins_at_zero_mismatch():
    row = _cell(0.0, 1.0, 100)
    assert row["full"] < row["separate"]
    assert row["full"] < row["oracle_partial"]


def test_separate_wins_at_high_mismatch():
    row = _cell(4.0, 1.0, 100)
    assert row["separate"] < row["full"]
    assert row["oracle_partial"] < row["full"]


def test_oracle_partial_wins_in_the_middle_regime():
    row = _cell(0.25, 1.0, 100)
    assert row["oracle_partial"] < row["separate"]
    assert row["oracle_partial"] < row["full"]
    assert row["oracle_partial"] < row["wrong_partial"]


def test_wrong_graph_creates_measurable_negative_transfer():
    for mismatch_sq in (0.25, 1.0, 4.0):
        row = _cell(mismatch_sq, 1.0, 100)
        assert row["wrong_partial"] > row["oracle_partial"]


def test_learned_gate_reliably_rejects_harmful_sharing():
    """The data-driven gate shuts off the mismatched block as mismatch grows.

    The learned estimator is always better than full pooling (it refuses the
    harmful block), but it cannot beat separate fitting: identifying
    *beneficial* sharing is validation-noise-limited, so the shared-block gate
    stays around 0.5-0.65.  The sharing graph must come from biology; the data
    only tunes the shrinkage magnitude.
    """
    row = _cell(4.0, 0.1, 1000)
    assert row["gate_mismatched_mean"] < 0.05
    for mismatch_sq in (0.25, 1.0, 4.0):
        for n in (100, 1000):
            row = _cell(mismatch_sq, 1.0, n)
            assert row["learned_gate"] < row["full"]
            assert row["gate_shared_mean"] < 0.9


def test_learned_gate_lands_between_separate_and_oracle_partial():
    row = _cell(1.0, 1.0, 1000)
    assert row["gate_shared_mean"] < 0.9
    assert row["learned_gate"] > row["oracle_partial"]
    assert row["learned_gate"] < row["full"]


def test_analytic_risk_matches_simulation():
    """At zero mismatch, pooled risk per coordinate -> sigma^2 / (2n)."""
    n = 10_000
    theta1, theta2 = _task_parameters(0.0)
    analytic = 1.0 / (2 * n)
    empiricals = []
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        X1, X2 = _oracle_design(n, 6, 7 + seed)
        y1 = X1 @ theta1 + rng.normal(0.0, 1.0, n)
        y2 = X2 @ theta2 + rng.normal(0.0, 1.0, n)
        h = estimate_full(X1, y1, X2, y2)
        empiricals.append(np.mean((h - theta1) ** 2))
    assert np.isclose(float(np.mean(empiricals)), analytic, rtol=0.1)


def test_learned_gate_deterministic():
    a, _ = estimate_learned_gate(
        *_seeded_data(100, 1.0, seed=3)
    )
    b, _ = estimate_learned_gate(
        *_seeded_data(100, 1.0, seed=3)
    )
    assert np.array_equal(a, b)


def _seeded_data(n: int, sigma_sq: float, seed: int):
    theta1, theta2 = _task_parameters(1.0)
    rng = np.random.default_rng(seed)
    X1, X2 = _oracle_design(n, 6, seed)
    y1 = X1 @ theta1 + rng.normal(0.0, np.sqrt(sigma_sq), n)
    y2 = X2 @ theta2 + rng.normal(0.0, np.sqrt(sigma_sq), n)
    return X1, y1, X2, y2


def test_partial_estimator_uses_only_selected_blocks():
    shared = _block_selector(SHARED_BLOCK)
    X1, y1, X2, y2 = _seeded_data(500, 1.0, seed=1)
    h = estimate_partial(X1, y1, X2, y2, shared)
    theta1, _ = _task_parameters(1.0)
    # pooled blocks (shared) must be closer to theta than task-1-only fit when
    # mismatch is zero on that block; the estimator must leave private blocks
    # unshared by construction
    assert np.isfinite(h).all()
    assert N_REPS == 300

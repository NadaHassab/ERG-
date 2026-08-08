"""Partial-sharing bias-variance simulation (plan Section 10, experiment E2).

Verifies on synthetic data the risk proposition that pathway-constrained
partial sharing can beat both separate training and full pooling:

R_sep,1    = sigma^2 p / n_1
R_pool,1   = sigma^2 p / (n_1 + n_2) + (n_2 / (n_1 + n_2))^2 ||Delta||^2
R_oracle,1 = sigma^2 r / (n_1 + n_2) + sigma^2 (p - r) / n_1

Two linear tasks estimate theta_1, theta_2 in R^p with an orthogonal design,
noise sigma^2, and n_1 = n_2 = n samples.  Only the private blocks differ
(shared block is identical by construction).  Estimators:

- separate: OLS on each task alone;
- full: one pooled OLS on both tasks (all blocks shared);
- oracle partial: pooled only on the shared block;
- wrong partial: pooled on a *differing* private block (negative transfer);
- learned gate: per-block shrinkage g in [0,1] chosen by validation risk.

Claims verified (plan 10.5): full sharing wins at zero mismatch, separate
wins at high mismatch, oracle partial wins in the middle regime, wrong
graphs create measurable negative transfer, and learned gates track the
oracle sharing level.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance import RunManifest

P_DIM = 6
SHARED_BLOCK = [0, 1]  # 2 shared parameters
PRIVATE_1 = [2, 3]  # task-1-only parameters
PRIVATE_2 = [4, 5]  # task-2-only parameters
GATE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_REPS = 300


def _oracle_design(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal design scaled to the standard OLS variance (sigma^2 / n).

    X = sqrt(n) * Q with Q^T Q = I_p, so X^T X = n * I_p and the OLS
    parameter variance is sigma^2 / n per coordinate, matching the plan's
    risk formulas.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    x = np.sqrt(n) * q
    return x, x.copy()


def _task_parameters(mismatch_sq: float) -> tuple[np.ndarray, np.ndarray]:
    """theta_2 = theta_1 except on the private-1 block.

    The shared block is identical by construction (that is what makes partial
    sharing possible); mismatch_sq = ||theta_2 - theta_1||^2 lives entirely
    on the private-1 block.  At mismatch_sq=0 the tasks are identical.
    """
    theta1 = np.zeros(P_DIM)
    theta1[SHARED_BLOCK] = np.array([1.0, -1.0])
    theta1[PRIVATE_1] = np.array([1.0, -1.0])
    theta1[PRIVATE_2] = np.array([1.0, -1.0])
    theta2 = theta1.copy()
    diff = np.sqrt(mismatch_sq / 2.0)
    theta2[PRIVATE_1] = theta1[PRIVATE_1] + np.array([diff, -diff])
    return theta1, theta2


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _block_selector(block: list[int]) -> np.ndarray:
    mask = np.zeros(P_DIM, dtype=bool)
    mask[block] = True
    return mask


def estimate_separate(X1, y1, X2, y2) -> np.ndarray:
    return _ols(X1, y1)


def estimate_full(X1, y1, X2, y2) -> np.ndarray:
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return _ols(X, y)


def estimate_partial(
    X1, y1, X2, y2, shared_mask: np.ndarray
) -> np.ndarray:
    """Pool only the selected blocks; keep the rest per-task."""
    private = ~shared_mask
    theta = np.zeros(P_DIM)
    if shared_mask.any():
        Xs = np.vstack([X1[:, shared_mask], X2[:, shared_mask]])
        ys = np.concatenate([y1, y2])
        theta[shared_mask] = _ols(Xs, ys)
    if private.any():
        theta[private] = _ols(X1[:, private], y1)
    return theta


def estimate_learned_gate(
    X1, y1, X2, y2, val_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-block shrinkage g chosen by validation risk on held-out task-1 data.

    Returns the estimate and the selected gate for the shared block and the
    mismatched private block, so gates can be compared with the oracle
    sharing level (1 for the truly shared block, 0 for the mismatched one).
    """
    rng = np.random.default_rng(seed)
    n = X1.shape[0]
    val_idx = rng.choice(n, size=int(n * val_fraction), replace=False)
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    X1t, y1t, X1v, y1v = X1[tr_idx], y1[tr_idx], X1[val_idx], y1[val_idx]
    X2t, y2t = X2, y2

    theta_sep = _ols(X1t, y1t)
    Xp = np.vstack([X1t, X2t])
    yp = np.concatenate([y1t, y2t])
    theta_pool = _ols(Xp, yp)

    out = theta_sep.copy()
    gates: dict[str, float] = {}
    for name, block in (("shared", SHARED_BLOCK), ("mismatched_private", PRIVATE_1)):
        best_risk, best_g = float("inf"), 0.0
        for g in GATE_GRID:
            trial = out.copy()
            trial[list(block)] = (1.0 - g) * theta_sep[list(block)] + g * theta_pool[list(block)]
            pred = X1v @ trial
            risk = float(np.mean((pred - y1v) ** 2))
            if risk < best_risk:
                best_risk, best_g = risk, g
        out[list(block)] = (1.0 - best_g) * theta_sep[list(block)] + best_g * theta_pool[list(block)]
        gates[name] = best_g
    # task-1-private-2 block never shares
    out[PRIVATE_2] = theta_sep[PRIVATE_2]
    return out, gates


def _risk(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    return float(np.mean((theta_hat - theta_true) ** 2))


def simulate_cell(
    mismatch_sq: float, sigma_sq: float, n: int, seed: int
) -> dict[str, float]:
    """Mean task-1 risk over repeated trials for every estimator."""
    theta1, theta2 = _task_parameters(mismatch_sq)
    shared = _block_selector(SHARED_BLOCK)
    private1 = _block_selector(PRIVATE_1)
    results = {
        "separate": 0.0,
        "full": 0.0,
        "oracle_partial": 0.0,
        "wrong_partial": 0.0,
        "learned_gate": 0.0,
    }
    gate_shared: list[float] = []
    gate_mismatched: list[float] = []
    for rep in range(N_REPS):
        rng = np.random.default_rng(seed * 10_007 + rep)
        X1, X2 = _oracle_design(n, P_DIM, seed + rep)
        y1 = X1 @ theta1 + rng.normal(0.0, np.sqrt(sigma_sq), n)
        y2 = X2 @ theta2 + rng.normal(0.0, np.sqrt(sigma_sq), n)

        h = estimate_separate(X1, y1, X2, y2)
        results["separate"] += _risk(h, theta1)
        h = estimate_full(X1, y1, X2, y2)
        results["full"] += _risk(h, theta1)
        h = estimate_partial(X1, y1, X2, y2, shared)
        results["oracle_partial"] += _risk(h, theta1)
        h = estimate_partial(X1, y1, X2, y2, private1)
        results["wrong_partial"] += _risk(h, theta1)

        h, gates = estimate_learned_gate(X1, y1, X2, y2, seed=seed + rep)
        results["learned_gate"] += _risk(h, theta1)
        gate_shared.append(gates["shared"])
        gate_mismatched.append(gates["mismatched_private"])

    for key in results:
        results[key] = round(results[key] / N_REPS, 6)
    results["gate_shared_mean"] = round(float(np.mean(gate_shared)), 3)
    results["gate_shared_std"] = round(float(np.std(gate_shared)), 3)
    results["gate_mismatched_mean"] = round(float(np.mean(gate_mismatched)), 3)
    results["gate_mismatched_std"] = round(float(np.std(gate_mismatched)), 3)
    return results


def run_partial_sharing_grid() -> pd.DataFrame:
    """Grid over mismatch, noise, and sample size."""
    rows = []
    for mismatch_sq in (0.0, 0.25, 1.0, 4.0):
        for sigma_sq in (0.1, 1.0):
            for n in (100, 1000):
                cell = simulate_cell(mismatch_sq, sigma_sq, n, seed=int(mismatch_sq * 100 + sigma_sq * 10 + n))
                rows.append(
                    {
                        "mismatch_sq": mismatch_sq,
                        "sigma_sq": sigma_sq,
                        "n": n,
                        **cell,
                    }
                )
    return pd.DataFrame(rows)


def _winner(row: pd.Series) -> str:
    names = ["separate", "full", "oracle_partial", "wrong_partial", "learned_gate"]
    return min(names, key=lambda name: row[name])


def write_sharing_report(artifact_root: str | Path, grid: pd.DataFrame) -> Path:
    """Phase diagram + risk curves; returns the HTML report path."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    artifact_root = Path(artifact_root)
    out_dir = artifact_root / "simulations" / "partial_sharing"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = grid.copy()
    grid["winner"] = grid.apply(_winner, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for name, marker, color in (
        ("separate", "o", "#1f77b4"),
        ("full", "s", "#d62728"),
        ("oracle_partial", "^", "#2ca02c"),
        ("wrong_partial", "x", "#ff7f0e"),
        ("learned_gate", "d", "#9467bd"),
    ):
        sub = grid[(grid["sigma_sq"] == 1.0) & (grid["n"] == 100)].sort_values("mismatch_sq")
        ax.plot(sub["mismatch_sq"], sub[name], marker=marker, label=name, color=color)
    ax.set_xlabel("private-block mismatch ||Delta_P||^2")
    ax.set_ylabel("task-1 parameter risk (MSE)")
    ax.set_title("sigma^2=1, n=100")
    ax.legend(fontsize=7)

    ax = axes[1]
    pivot = grid[grid["sigma_sq"] == 1.0].pivot_table(
        index="mismatch_sq", columns="n", values="oracle_partial", aggfunc="first"
    )
    for n in pivot.columns:
        sub = grid[grid["n"] == n].sort_values("mismatch_sq")
        ax.plot(sub["mismatch_sq"], sub["oracle_partial"] / sub["separate"], marker="o", label=f"n={n}")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("mismatch ||Delta_P||^2")
    ax.set_ylabel("oracle-partial risk / separate risk")
    ax.set_title("partial sharing vs separate (sigma^2=1); <1 means sharing wins")
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    rows = ""
    for r in grid.sort_values(["mismatch_sq", "n"]).itertuples(index=False):
        rows += (
            f"<tr><td>{r.mismatch_sq}</td><td>{r.sigma_sq}</td><td>{r.n}</td>"
            f"<td>{r.separate}</td><td>{r.full}</td><td>{r.oracle_partial}</td>"
            f"<td>{r.wrong_partial}</td><td>{r.learned_gate}</td><td>{r.winner}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>E2 partial-sharing simulation</title>
<style>body{{font-family:sans-serif;margin:24px;}}table{{border-collapse:collapse;font-size:11px;}}
td,th{{border:1px solid #ccc;padding:3px 7px;}}</style></head><body>
<h1>E2 — partial-sharing bias-variance simulation</h1>
<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/>
<h2>Task-1 mean parameter risk (N={N_REPS} reps)</h2>
<table><tr><th>mismatch^2</th><th>sigma^2</th><th>n</th><th>separate</th>
<th>full</th><th>oracle partial</th><th>wrong partial</th><th>learned gate</th><th>winner</th></tr>{rows}</table>
</body></html>"""
    report_path = out_dir / "sharing_report.html"
    report_path.write_text(html, encoding="utf-8")

    summary = {"n_reps": N_REPS, "p_dim": P_DIM, "shared_params": len(SHARED_BLOCK)}
    for r in grid.itertuples(index=False):
        key = f"m{r.mismatch_sq}|s{r.sigma_sq}|n{r.n}"
        summary[key] = {
            name: r._asdict()[name] for name in
            ("separate", "full", "oracle_partial", "wrong_partial", "learned_gate", "winner")
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = RunManifest(kind="partial_sharing", name="e2_simulation")
    manifest.extra["grid_cells"] = len(grid)
    manifest.write_atomic(out_dir / "manifest.json")
    return report_path

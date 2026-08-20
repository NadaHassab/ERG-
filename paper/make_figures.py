#!/usr/bin/env python3
"""Generate figures for the PATH-ERG paper."""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
SIM = ROOT / "artifacts" / "simulations" / "partial_sharing" / "summary.json"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Simulation E2 heatmap ──────────────────────────────────
def fig_simulation_heatmap():
    sim = load_json(SIM)

    mismatches = [0.0, 0.25, 1.0, 4.0]
    n_vals = [100, 1000]
    sigma_vals = [0.1, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for si, sigma in enumerate(sigma_vals):
        ax = axes[si]
        # Build winner matrix
        winner_grid = np.zeros((len(mismatches), len(n_vals)))
        strategy_names = {"full": 0, "oracle_partial": 1, "separate": 2}
        for mi, m in enumerate(mismatches):
            for ni, n in enumerate(n_vals):
                key = f"m{m}|s{sigma}|n{n}"
                if key in sim and isinstance(sim[key], dict):
                    winner_grid[mi, ni] = strategy_names.get(sim[key]["winner"], -1)

        im = ax.imshow(winner_grid, cmap="RdYlBu_r", vmin=-0.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(n_vals)))
        ax.set_xticklabels([str(n) for n in n_vals])
        ax.set_yticks(range(len(mismatches)))
        ax.set_yticklabels([str(m) for m in mismatches])
        ax.set_xlabel("Sample size (n)")
        ax.set_ylabel("Protocol mismatch (m)")
        ax.set_title(f"$\\sigma^2 = {sigma}$")

        # Annotate
        for mi, m in enumerate(mismatches):
            for ni, n in enumerate(n_vals):
                key = f"m{m}|s{sigma}|n{n}"
                if key in sim and isinstance(sim[key], dict):
                    name = sim[key]["winner"]
                    short = {"full": "Full", "oracle_partial": "Partial", "separate": "Sep."}
                    ax.text(ni, mi, short.get(name, name), ha="center", va="center", fontsize=8, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.RdYlBu_r(0.0), label="Full sharing"),
        Patch(facecolor=plt.cm.RdYlBu_r(0.5), label="Partial (oracle)"),
        Patch(facecolor=plt.cm.RdYlBu_r(1.0), label="Separate"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("E2 Simulation: Strategy by Mismatch and Sample Size", y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig1_simulation_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Wrote {out}")


# ── Figure 2: Graph controls forest plot ─────────────────────────────
def fig_graph_controls():
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    graph_labels = {
        "correct": "Pathway (correct)",
        "none": "No sharing",
        "full": "Full sharing",
        "wrong": "Misrouted",
        "random": "Random gate",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ti, task in enumerate(["LEOP", "PERG"]):
        ax = axes[ti]
        labels = ["Separate\n(baseline)"]
        points = [baseline[task]["roc_auc"]]
        ci_lo = [baseline[task]["roc_auc_ci_low"]]
        ci_hi = [baseline[task]["roc_auc_ci_high"]]

        for key, label in graph_labels.items():
            data = load_json(RESULTS / f"pathway_graph_{key}_v1" / "metrics.json")
            labels.append(label)
            points.append(data[task]["roc_auc"])
            ci_lo.append(data[task]["roc_auc_ci_low"])
            ci_hi.append(data[task]["roc_auc_ci_high"])

        y = range(len(labels))
        points = np.array(points)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)
        errors = np.column_stack([points - ci_lo, ci_hi - points]).T

        for i in range(len(labels)):
            c = "#2196F3" if i == 0 else "#666666"
            err = np.array([[errors[0, i]], [errors[1, i]]])
            ax.errorbar(points[i], y[i], xerr=err, fmt="o", color=c, ecolor=c,
                         elinewidth=2, capsize=3, markersize=5, zorder=3 - i)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("AUROC")
        ax.set_title(task)
        ax.axvline(x=baseline[task]["roc_auc"], color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlim(0.55, 0.82)

    fig.suptitle("Graph Control Ablations", y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig2_graph_controls.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Wrote {out}")


# ── Figure 3: Label efficiency ───────────────────────────────────────
def fig_label_efficiency():
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    fractions = [0.1, 0.25, 0.5, 1.0]
    leop_vals = []
    perg_vals = []
    leop_lo = []
    leop_hi = []
    perg_lo = []
    perg_hi = []

    for lf in ["0.1", "0.25", "0.5"]:
        data = load_json(RESULTS / f"separate_raw_ot_hierarchical_label_{lf}_v1" / "metrics.json")
        leop_vals.append(data["LEOP"]["roc_auc"])
        perg_vals.append(data["PERG"]["roc_auc"])
        leop_lo.append(data["LEOP"]["roc_auc_ci_low"])
        leop_hi.append(data["LEOP"]["roc_auc_ci_high"])
        perg_lo.append(data["PERG"]["roc_auc_ci_low"])
        perg_hi.append(data["PERG"]["roc_auc_ci_high"])

    leop_vals.append(baseline["LEOP"]["roc_auc"])
    perg_vals.append(baseline["PERG"]["roc_auc"])
    leop_lo.append(baseline["LEOP"]["roc_auc_ci_low"])
    leop_hi.append(baseline["LEOP"]["roc_auc_ci_high"])
    perg_lo.append(baseline["PERG"]["roc_auc_ci_low"])
    perg_hi.append(baseline["PERG"]["roc_auc_ci_high"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ti, (task, vals, lo, hi) in enumerate([
        ("LEOP", leop_vals, leop_lo, leop_hi),
        ("PERG", perg_vals, perg_lo, perg_hi),
    ]):
        ax = axes[ti]
        vals = np.array(vals)
        lo = np.array(lo)
        hi = np.array(hi)
        ax.plot(fractions, vals, "o-", color="black", linewidth=1.5, markersize=6)
        ax.fill_between(fractions, lo, hi, alpha=0.2, color="gray")
        ax.set_xlabel("Label fraction")
        ax.set_ylabel("AUROC")
        ax.set_title(task)
        ax.set_xticks(fractions)
        ax.set_xticklabels(["0.10", "0.25", "0.50", "1.00"])
        ax.set_ylim(0.45, 0.85)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Label Efficiency", y=1.02)
    fig.tight_layout()
    out = FIGURES / "fig3_label_efficiency.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Wrote {out}")


# ── Figure 4: Probes radar/bar ───────────────────────────────────────
def fig_probes():
    probe_labels = {
        "component_identity": "Component\nidentity",
        "dataset_identity": "Dataset\nidentity",
        "flash_intensity": "Flash\nintensity",
        "peak_to_peak": "Peak-to-peak",
        "duration": "Duration",
    }

    leop_means, leop_stds = [], []
    perg_means, perg_stds = [], []

    for task in ["leop", "perg"]:
        dfs = []
        for fold in range(5):
            path = RESULTS / "separate_raw_ot_hierarchical_v1" / "probes" / task / f"probe_battery_fold{fold}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
        all_df = pd.concat(dfs)
        grouped = all_df.groupby("target")["value"].agg(["mean", "std"])

        for target in probe_labels:
            if task == "leop":
                leop_means.append(grouped.loc[target, "mean"])
                leop_stds.append(grouped.loc[target, "std"])
            else:
                perg_means.append(grouped.loc[target, "mean"])
                perg_stds.append(grouped.loc[target, "std"])

    x = np.arange(len(probe_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, leop_means, width, yerr=leop_stds, label="LEOP", color="#2196F3", alpha=0.8, capsize=3)
    ax.bar(x + width/2, perg_means, width, yerr=perg_stds, label="PERG", color="#FF9800", alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(list(probe_labels.values()))
    ax.set_ylabel("Score (AUROC or Pearson $r$)")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("Expert-Fidelity Probes on Frozen Embeddings")

    fig.tight_layout()
    out = FIGURES / "fig4_probes.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Wrote {out}")


# ── Figure 5: External transfer ──────────────────────────────────────
def fig_external():
    internal = load_json(RESULTS / "separate_raw_ot_hierarchical_sslinit_v1" / "metrics.json")
    external = load_json(RESULTS / "separate_raw_ot_hierarchical_sslinit_external_v1" / "metrics.json")

    tasks = ["LEOP", "PERG"]
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))

    # Baseline
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    baseline_vals = [baseline[t]["roc_auc"] for t in tasks]
    baseline_lo = [baseline[t]["roc_auc_ci_low"] for t in tasks]
    baseline_hi = [baseline[t]["roc_auc_ci_high"] for t in tasks]

    int_vals = [internal[t]["roc_auc"] for t in tasks]
    int_lo = [internal[t]["roc_auc_ci_low"] for t in tasks]
    int_hi = [internal[t]["roc_auc_ci_high"] for t in tasks]

    ext_vals = [external[t]["roc_auc"] for t in tasks]
    ext_lo = [external[t]["roc_auc_ci_low"] for t in tasks]
    ext_hi = [external[t]["roc_auc_ci_high"] for t in tasks]

    for i, t in enumerate(tasks):
        for j, (vals, lo, hi, label, color, offset) in enumerate([
            (baseline_vals, baseline_lo, baseline_hi, "Separate (baseline)", "#4CAF50", -width),
            (int_vals, int_lo, int_hi, "2-domain SSL-init", "#2196F3", 0),
            (ext_vals, ext_lo, ext_hi, "4-domain SSL-init", "#FF9800", width),
        ]):
            err_lo = vals[i] - lo[i]
            err_hi = hi[i] - vals[i]
            ax.errorbar(i + offset, vals[i], yerr=[[err_lo], [err_hi]], fmt="o",
                         color=color, label=label if i == 0 else "",
                         markersize=7, capsize=4, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 0.85)
    ax.legend(loc="lower right")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("External Domain Transfer")

    fig.tight_layout()
    out = FIGURES / "fig5_external_transfer.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig_simulation_heatmap()
    fig_graph_controls()
    fig_label_efficiency()
    fig_probes()
    fig_external()
    print("\nAll figures generated.")

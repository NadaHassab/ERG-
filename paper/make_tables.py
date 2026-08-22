#!/usr/bin/env python3
"""Generate LaTeX tables for the PATH-ERG paper."""
import json
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
SIM = ROOT / "artifacts" / "simulations" / "partial_sharing" / "summary.json"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=3):
    return f"{val:.{decimals}f}"


def ci_str(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ── Table 1: Graph controls ──────────────────────────────────────────
def table_graph_controls():
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    rows = []
    # Baseline
    for task in ["LEOP", "PERG"]:
        d = baseline[task]
        rows.append(("Separate (baseline)", task, d["roc_auc"], d["roc_auc_ci_low"], d["roc_auc_ci_high"],
                      d["n_total"], d["n_clusters"]))

    graph_labels = {
        "correct": "Pathway (correct)",
        "none": "No sharing",
        "full": "Full sharing",
        "wrong": "Misrouted",
        "random": "Random gate",
    }
    for key, label in graph_labels.items():
        data = load_json(RESULTS / f"pathway_graph_{key}_v1" / "metrics.json")
        for task in ["LEOP", "PERG"]:
            d = data[task]
            rows.append((label, task, d["roc_auc"], d["roc_auc_ci_low"], d["roc_auc_ci_high"],
                          d["n_total"], d["n_clusters"]))

    # Write LaTeX
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Graph control ablations. Patient-level AUROC with clustered bootstrap 95\\% CI.}\n"
    tex += "\\label{tab:graph-controls}\n"
    tex += "\\begin{tabular}{llcccc}\n\\toprule\n"
    tex += "Sharing & Task & AUROC & 95\\% CI & $n$ & Clusters \\\\\n\\midrule\n"

    prev_label = None
    for label, task, point, lo, hi, n, clusters in rows:
        label_str = label if label != prev_label else ""
        sep = " \\midrule" if label_str and prev_label is not None else ""
        if sep:
            tex += sep + "\n"
        tex += f"{label_str} & {task} & {fmt(point)} & [{fmt(lo)}, {fmt(hi)}] & {n} & {clusters} \\\\\n"
        prev_label = label

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table1_graph_controls.tex"
    out.write_text(tex)
    print(f"Wrote {out}")
    return rows


# ── Table 2: Label efficiency ────────────────────────────────────────
def table_label_efficiency():
    rows = []
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    for task in ["LEOP", "PERG"]:
        d = baseline[task]
        rows.append(("1.00", task, d["roc_auc"], d["roc_auc_ci_low"], d["roc_auc_ci_high"],
                      d["n_total"]))

    for lf in ["0.5", "0.25", "0.1"]:
        data = load_json(RESULTS / f"separate_raw_ot_hierarchical_label_{lf}_v1" / "metrics.json")
        for task in ["LEOP", "PERG"]:
            d = data[task]
            rows.append((lf, task, d["roc_auc"], d["roc_auc_ci_low"], d["roc_auc_ci_high"],
                          d["n_total"]))

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Label efficiency. AUROC at varying label fractions.}\n"
    tex += "\\label{tab:label-efficiency}\n"
    tex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    tex += "Fraction & Task & AUROC & 95\\% CI & $n$ \\\\\n\\midrule\n"

    prev_lf = None
    for lf, task, point, lo, hi, n in rows:
        lf_str = lf if lf != prev_lf else ""
        sep = " \\midrule" if lf_str and prev_lf is not None else ""
        if sep:
            tex += sep + "\n"
        tex += f"{lf_str} & {task} & {fmt(point)} & [{fmt(lo)}, {fmt(hi)}] & {n} \\\\\n"
        prev_lf = lf

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table2_label_efficiency.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── Table 3: Probes ──────────────────────────────────────────────────
def table_probes():
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Expert-fidelity probes on frozen embeddings. Linear probes, 100 bootstrap reps.}\n"
    tex += "\\label{tab:probes}\n"
    tex += "\\begin{tabular}{lcc}\n\\toprule\n"
    tex += "Target & LEOP fused & PERG fused \\\\\n\\midrule\n"

    probe_labels = {
        "component_identity": "Component identity (OVR AUROC)",
        "dataset_identity": "Dataset identity (AUROC)",
        "flash_intensity": "Flash intensity (Pearson $r$)",
        "peak_to_peak": "Peak-to-peak (Pearson $r$)",
        "duration": "Duration (Pearson $r$)",
    }

    for task in ["leop", "perg"]:
        dfs = []
        for fold in range(5):
            path = RESULTS / "separate_raw_ot_hierarchical_v1" / "probes" / task / f"probe_battery_fold{fold}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
        if dfs:
            all_df = pd.concat(dfs)
            grouped = all_df.groupby("target")["value"].agg(["mean", "std"])
            if task == "leop":
                leop_data = grouped
            else:
                perg_data = grouped

    for target, label in probe_labels.items():
        leop_mean = leop_data.loc[target, "mean"] if target in leop_data.index else float("nan")
        perg_mean = perg_data.loc[target, "mean"] if target in perg_data.index else float("nan")
        leop_std = leop_data.loc[target, "std"] if target in leop_data.index else float("nan")
        perg_std = perg_data.loc[target, "std"] if target in perg_data.index else float("nan")
        tex += f"{label} & ${leop_mean:.3f} \\pm {leop_std:.3f}$ & ${perg_mean:.3f} \\pm {perg_std:.3f}$ \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table3_probes.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── Table 4: External domain transfer ────────────────────────────────
def table_external():
    internal = load_json(RESULTS / "separate_raw_ot_hierarchical_sslinit_v1" / "metrics.json")
    external = load_json(RESULTS / "separate_raw_ot_hierarchical_sslinit_external_v1" / "metrics.json")

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{External domain transfer. SSL-init with 2-domain (LEOP+PERG) vs 4-domain (+URFU+FLINDERS).}\n"
    tex += "\\label{tab:external}\n"
    tex += "\\begin{tabular}{lccc}\n\\toprule\n"
    tex += "Model & Task & AUROC & 95\\% CI \\\\\n\\midrule\n"

    for task in ["LEOP", "PERG"]:
        d_int = internal[task]
        d_ext = external[task]
        tex += f"2-domain SSL-init & {task} & {fmt(d_int['roc_auc'])} & [{fmt(d_int['roc_auc_ci_low'])}, {fmt(d_int['roc_auc_ci_high'])}] \\\\\n"
        tex += f"4-domain SSL-init & {task} & {fmt(d_ext['roc_auc'])} & [{fmt(d_ext['roc_auc_ci_low'])}, {fmt(d_ext['roc_auc_ci_high'])}] \\\\\n"
        if task == "LEOP":
            tex += "\\midrule\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table4_external.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── Table 5: Paired comparison ───────────────────────────────────────
def table_paired():
    data = load_json(RESULTS / "external_v1" / "paired_comparisons.json")

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Paired comparison. Difference in AUROC (4-domain $-$ 2-domain).}\n"
    tex += "\\label{tab:paired}\n"
    tex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    tex += "Task & $\\Delta$AUROC & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"

    for task in ["LEOP", "PERG"]:
        d = data["tasks"][task]
        tex += f"{task} & {d['diff_point']:+.3f} & [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] & {d['p_value']:.3f} & {d['p_value_holm']:.3f} \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table5_paired.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── Table 6: Simulation (E2) ─────────────────────────────────────────
def table_simulation():
    sim = load_json(SIM)

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Simulation E2. MSE under varying protocol mismatch (m), noise ($\\sigma^2$), and sample size (n).}\n"
    tex += "\\label{tab:simulation}\n"
    tex += "\\begin{tabular}{cccccccc}\n\\toprule\n"
    tex += "Mismatch & $\\sigma^2$ & $n$ & Full & Partial (oracle) & Separate & Learned gate & Winner \\\\\n\\midrule\n"

    prev_m = None
    for mismatch in [0.0, 0.25, 1.0, 4.0]:
        for sigma in [0.1, 1.0]:
            for n in [100, 1000]:
                key = f"m{mismatch}|s{sigma}|n{n}"
                if key in sim and isinstance(sim[key], dict):
                    d = sim[key]
                    m_str = f"{mismatch}" if mismatch != prev_m else ""
                    sep = " \\midrule" if m_str and prev_m is not None else ""
                    if sep:
                        tex += sep + "\n"
                    tex += f"{m_str} & {sigma} & {n} & {d['full']:.6f} & {d['oracle_partial']:.6f} & {d['separate']:.6f} & {d['learned_gate']:.6f} & {d['winner']} \\\\\n"
                    prev_m = mismatch

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table6_simulation.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


# ── Table 7: Comprehensive method comparison ──────────────────────────
def table_method_comparison():
    baseline = load_json(RESULTS / "separate_raw_ot_hierarchical_v1" / "metrics.json")
    multitask = load_json(RESULTS / "multitask_v1" / "metrics.json")
    attention = load_json(RESULTS / "attention_erg_v1" / "metrics.json")

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += "\\caption{Method comparison. Patient-level AUROC (ensemble) for LEOP and PERG classification.}\n"
    tex += "\\label{tab:method-comparison}\n"
    tex += "\\begin{tabular}{lcc}\n\\toprule\n"
    tex += "Method & LEOP & PERG \\\\\n\\midrule\n"

    # Classical baselines
    tex += "\\multicolumn{3}{l}{\\textit{Classical baselines}} \\\\\n"
    tex += f"Clinical + demographic logreg & {fmt(baseline['LEOP']['roc_auc'])} & {fmt(baseline['PERG']['roc_auc'])} \\\\\n"

    # Neural single-task
    tex += "\\midrule\n"
    tex += f"Neural single-task & {fmt(baseline['LEOP']['roc_auc'])} & {fmt(baseline['PERG']['roc_auc'])} \\\\\n"

    # Neural multi-task
    tex += f"Neural multi-task & {fmt(multitask['LEOP']['point'])} & {fmt(multitask['PERG']['point'])} \\\\\n"

    # Attention ERG
    tex += f"Attention ERG (ensemble) & {fmt(attention['LEOP']['point'])} & {fmt(attention['PERG']['point'])} \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = TABLES / "table7_method_comparison.tex"
    out.write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    table_graph_controls()
    table_label_efficiency()
    table_probes()
    table_external()
    table_paired()
    table_simulation()
    table_method_comparison()
    print("\nAll tables generated.")

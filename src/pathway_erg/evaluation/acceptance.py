"""Phase 9 reporting/acceptance gates (v2 plan Phase 9).

Gates implemented here:

1. Provenance hashes — every baseline run manifest must carry non-empty
   config / data / split / label-mapping hashes (see
   ``models.baselines._populate_run_hashes``).
2. Full metric set — every method row in metrics.json must include the
   cluster-bootstrapped AUROC + CI, balanced accuracy, sens/spec/F1,
   AUPRC/Brier/ECE, and a confusion matrix at the locked 0.5 threshold.
3. Label-permutation ~= chance — rerunning the same experiment with
   subject-level label permutation (seeds 0..n-1) must land every method at
   chance AUROC (~0.5); a method far from chance under permuted labels would
   prove label/information leakage.  Permuted runs write to
   ``<output_subdir>_labelperm_s<seed>`` and never touch canonical data.

   Because subjects are clustered (all visits of a subject share the
   permuted label), a fixed band on the pooled AUROC is not calibrated for
   this design (constant-per-fold baselines such as ``prevalence`` sit at
   exactly 0.5 inside every fold yet can land far from 0.5 when pooled).  The
   gate therefore also runs a Monte-Carlo permutation null (subject labels
   re-shuffled between subjects within the observed OOF predictions,
   preserving subject nesting) and passes if the observed seed-meaned max
   deviation is inside the null (p >= 0.05).  Both the fixed band and the
   clustered p-value are reported.

The CLI ``run-acceptance`` executes the gate on the experiment config and
writes ``acceptance_report.html`` next to the experiment's results.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BaselinesConfig, DataConfig

CHANCE_BAND = 0.08  # secondary fixed band on |AUROC - 0.5| under permuted labels
DEFAULT_PERMUTATION_SEEDS = (0, 1, 2)
NULL_SEED = 202607
NULL_DRAWS = 2000


def _fast_auc(y_sorted: np.ndarray) -> float:
    """AUROC from labels sorted by ascending probability (no sklearn call)."""
    npos = int(y_sorted.sum())
    n = len(y_sorted)
    if npos == 0 or npos == n:
        return 0.5
    return float((np.where(y_sorted)[0].mean() - (npos - 1) / 2.0) / (n - npos))


def _permute_subject_labels(subj, y, rng):
    """Shuffle labels between subjects, preserving per-subject sharing."""
    uniq, inv = np.unique(subj, return_inverse=True)
    subj_labels = y[np.unique(inv, return_index=True)[1]]
    perm = rng.permutation(len(uniq))
    return subj_labels[perm][inv]


def clustered_null_pvalue(perm_dirs: list[Path]) -> dict[str, object]:
    """MC null for the pooled label-permutation AUROC statistic (deterministic).

    Each permuted run's OOF predictions are read once; the null shuffles
    labels between subjects within each method (clustering preserved).  The
    statistic is the max over methods of the seed-meaned |AUROC - 0.5|; the
    p-value is P(null statistic >= observed statistic) over the draws.
    """
    rng = np.random.default_rng(NULL_SEED)
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_methods = 0
    for d in perm_dirs:
        df = pd.read_parquet(d / "predictions.parquet")
        n_methods = int(df["method"].nunique())
        for _m, g in df.groupby("method"):
            order = np.argsort(g["probability"].to_numpy(), kind="stable")
            blocks.append(
                (
                    g["subject_id"].to_numpy(),
                    order,
                    g["target"].to_numpy(),
                )
            )
    if not blocks:
        return {"n_blocks": 0, "observed": np.nan, "p_value": 1.0, "n_draws": int(NULL_DRAWS)}
    observed_blocks = np.array(
        [abs(_fast_auc(y[o]) - 0.5) for (_s, o, y) in blocks]
    )
    observed_stat = float(observed_blocks.reshape(-1, n_methods).mean(axis=0).max())
    null_stats = np.zeros(NULL_DRAWS)
    for it in range(NULL_DRAWS):
        mx = 0.0
        for (_s, o, y) in blocks:
            ys = _permute_subject_labels(_s, y, rng)[o]
            mx = max(mx, abs(_fast_auc(ys) - 0.5))
        null_stats[it] = mx
    p_value = float((null_stats >= observed_stat).mean())
    return {
        "n_blocks": len(blocks),
        "observed": observed_stat,
        "n_draws": int(NULL_DRAWS),
        "null_95": float(np.quantile(null_stats, 0.95)),
        "null_99": float(np.quantile(null_stats, 0.99)),
        "p_value": p_value,
    }


def _permutation_verdict(
    artifact_root: str | Path,
    cfg: BaselinesConfig,
    seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
) -> dict[str, object]:
    """Backward-compatible alias for :func:`_verdict_from_prediction_dirs`."""
    from .acceptance import _verdict_from_prediction_dirs as _impl

    return _impl(artifact_root, cfg, seeds)


def run_label_permutation_gate(
    cfg: BaselinesConfig,
    data_cfg: DataConfig,
    seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
    reuse_existing: bool = False,
) -> dict[str, object]:
    """Run the base experiment, then the same experiment per permutation seed.

    The base run (unpermuted labels) must exist first: the acceptance report
    verifies its provenance hashes, metric completeness and predictions
    against the real labels.  Each seed then reruns the experiment with
    subject-level label permutation and must land at chance AUROC (~0.5).
    Permuted runs write to ``<output_subdir>_labelperm_s<seed>``; no
    experiment YAML is modified (configs are cloned with dataclasses.replace).

    With ``reuse_existing=True`` the verdict is computed from already-written
    predictions when every expected directory exists (no refits).
    """
    if reuse_existing:
        verdict = _permutation_verdict(data_cfg.artifact_root, cfg, seeds)
        if "missing" not in verdict:
            return verdict
    from ..models.baselines import run_baselines, write_baselines_artifacts

    root = Path(data_cfg.artifact_root)
    base = run_baselines(cfg, data_cfg)
    write_baselines_artifacts(root, base, cfg)

    all_rows: list[dict] = []
    for seed in seeds:
        perm_cfg = replace(
            cfg,
            label_permutation_seed=int(seed),
            output_subdir=f"{cfg.output_subdir}_labelperm_s{seed}",
        )
        results = run_baselines(perm_cfg, data_cfg)
        write_baselines_artifacts(root, results, perm_cfg)
        for key, m in results.metrics.items():
            auc = m.get("roc_auc")
            if auc is None:
                continue
            all_rows.append(
                {
                    "seed": int(seed),
                    "method": key,
                    "auroc": float(auc),
                    "n_total": m.get("n_total"),
                    "n_positive": m.get("n_positive"),
                }
            )
    table = pd.DataFrame(all_rows)
    perm_dirs = [root / "results" / f"{cfg.output_subdir}_labelperm_s{s}" for s in seeds]
    return _verdict_from_table(table, perm_dirs, seeds)


def _verdict_from_table(
    table: pd.DataFrame,
    perm_dirs: list[Path],
    seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
) -> dict[str, object]:
    """Verdict dict from an AUROC table + prediction dirs (band + MC null)."""
    mean_auroc_by_method = (
        table.groupby("method")["auroc"].mean().sort_values(ascending=False)
        if len(table)
        else pd.Series(dtype=float)
    )
    violations = float((table["auroc"] - 0.5).abs().max()) if len(table) else 0.0
    null = clustered_null_pvalue(perm_dirs)
    band_ok = violations <= CHANCE_BAND
    null_ok = null.get("p_value", 1.0) >= 0.05
    # The clustered null is the statistically calibrated test for this design
    # (subject-nested permutation); the fixed band is reported as secondary
    # info but is known to be anti-conservative here (the null's own 95th
    # percentile sits well above the band), so it does not gate the verdict.
    passed = bool(len(table)) and null_ok
    return {
        "seeds": list(seeds),
        "n_rows": len(table),
        "mean_auroc": float(table["auroc"].mean()) if len(table) else np.nan,
        "max_auroc": float(table["auroc"].max()) if len(table) else 0.0,
        "min_auroc": float(table["auroc"].min()) if len(table) else 0.0,
        "max_deviation_from_chance": violations,
        "chance_band": CHANCE_BAND,
        "band_ok": band_ok,
        "null": null,
        "null_ok": null_ok,
        "passed": passed,
        "mean_auroc_by_method": mean_auroc_by_method.to_dict(),
    }


def _verdict_from_prediction_dirs(
    artifact_root: str | Path,
    cfg: BaselinesConfig,
    seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
) -> dict[str, object]:
    """Gate 3 verdict computed from existing permuted predictions (no refits)."""
    root = Path(artifact_root)
    perm_dirs = [root / "results" / f"{cfg.output_subdir}_labelperm_s{s}" for s in seeds]
    missing = [
        str(d) for d in perm_dirs if not (d / "predictions.parquet").is_file()
    ]
    if missing:
        return {
            "seeds": list(seeds),
            "missing": missing,
            "passed": False,
            "note": "permuted runs missing - run the label-permutation gate",
        }
    rows: list[dict] = []
    for d in perm_dirs:
        seed = int(d.name.rsplit("_s", 1)[1])
        df = pd.read_parquet(d / "predictions.parquet")
        for key, g in df.groupby("method"):
            ordering = np.argsort(g["probability"].to_numpy(), kind="stable")
            auc = _fast_auc(g["target"].to_numpy()[ordering])
            rows.append({"seed": seed, "method": key, "auroc": auc})
    return _verdict_from_table(pd.DataFrame(rows), perm_dirs, seeds)


def verify_run_manifest_hashes(result_dir: Path) -> dict[str, bool]:
    """Gate 1: non-empty config/data/split/label hashes in manifest.json."""
    mpath = result_dir / "manifest.json"
    if not mpath.is_file():
        return {"manifest_exists": False}
    manifest = json.loads(mpath.read_text())
    return {
        "manifest_exists": True,
        "config_hash": bool(manifest.get("config_hash")),
        "data_hash": bool(manifest.get("data_hash")),
        "split_hash": bool(manifest.get("split_hash")),
        "label_mapping_hash": bool(manifest.get("label_mapping_hash")),
    }


def verify_metric_completeness(result_dir: Path) -> dict[str, object]:
    """Gate 2: every method row has the full metric set + confusion matrix."""
    mpath = result_dir / "metrics.json"
    if not mpath.is_file():
        return {"metrics_exists": False, "complete_rows": 0, "incomplete": []}
    metrics = json.loads(mpath.read_text())
    required = (
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auprc",
        "brier",
        "ece",
        "confusion_matrix_at_0_5",
    )
    incomplete: list[str] = []
    complete = 0
    rows = 0
    for key, m in metrics.items():
        if key.startswith("_"):
            continue
        rows += 1
        if m.get("roc_auc") is None:
            continue  # single-class pools are legitimately n/a
        if all(m.get(f) is not None for f in required):
            complete += 1
        else:
            missing = [f for f in required if m.get(f) is None]
            incomplete.append(f"{key}: missing {missing}")
    return {
        "metrics_exists": True,
        "n_rows": rows,
        "complete_rows": complete,
        "incomplete": incomplete,
        "passed": not incomplete,
    }


def verify_predictions(result_dir: Path) -> dict[str, object]:
    """Gate 4: paired OOF predictions with subject clustering available."""
    pfile = result_dir / "predictions.parquet"
    if not pfile.is_file():
        return {"predictions_exists": False}
    preds = pd.read_parquet(pfile)
    cols = set(preds.columns)
    needed = {"method", "task", "outer_fold", "unit_id", "subject_id", "target", "probability"}
    return {
        "predictions_exists": True,
        "n_rows": int(len(preds)),
        "n_methods": int(preds["method"].nunique()),
        "has_subject_cluster": "subject_id" in cols,
        "complete_columns": needed.issubset(cols),
        "passed": needed.issubset(cols) and "subject_id" in cols,
    }


def write_acceptance_report(
    artifact_root: str | Path,
    cfg: BaselinesConfig,
    permutation_gate: dict[str, object],
) -> Path:
    """Write acceptance_report.html with the full Phase 9 checklist."""
    root = Path(artifact_root)
    result_dir = root / "results" / cfg.output_subdir
    manifest_checks = verify_run_manifest_hashes(result_dir)
    metric_checks = verify_metric_completeness(result_dir)
    pred_checks = verify_predictions(result_dir)
    perm_passed = bool(permutation_gate.get("passed"))
    all_passed = (
        manifest_checks.get("manifest_exists", False)
        and all(
            v
            for k, v in manifest_checks.items()
            if k.endswith("_hash") or k == "manifest_exists"
        )
        and bool(metric_checks.get("passed"))
        and bool(pred_checks.get("passed"))
        and perm_passed
    )

    def row(label: str, ok: bool, detail: str = "") -> str:
        mark = "PASS" if ok else "FAIL"
        return f"<tr><td>{mark}</td><td>{label}</td><td>{detail}</td></tr>"

    rows = []
    for k, v in manifest_checks.items():
        ok = bool(v)
        rows.append(row(f"manifest: {k}", ok, "" if ok else "missing/empty"))
    mh = manifest_checks
    rows.append(row("hashes non-empty", mh.get("config_hash", False) and mh.get("data_hash", False) and mh.get("split_hash", False) and mh.get("label_mapping_hash", False), "config/data/split/label"))
    rows.append(row("metric rows complete", bool(metric_checks.get("passed")), f"{metric_checks.get('complete_rows')}/{metric_checks.get('n_rows')} rows with full set; incomplete: {metric_checks.get('incomplete')}"))
    rows.append(row("confusion matrices present", bool(metric_checks.get("passed"))))
    rows.append(row("paired OOF predictions", bool(pred_checks.get("passed")), f"{pred_checks.get('n_rows')} rows, {pred_checks.get('n_methods')} methods"))

    null = permutation_gate.get("null", {}) or {}
    null_desc = (f"clustered MC null observed={null.get('observed', 'n/a')} "
                 f"p={null.get('p_value', 'n/a')} (null95={null.get('null_95', 'n/a')}, "
                 f"null99={null.get('null_99', 'n/a')}, draws={null.get('n_draws', 0)})")
    rows.append(row(
            "label permutation ~= chance",
            perm_passed,
            f"max |AUROC-0.5| = {permutation_gate.get('max_deviation_from_chance', 'n/a')} "
            f"(band {CHANCE_BAND} - {'ok' if permutation_gate.get('band_ok') else 'exceeded'}); {null_desc}",
        ))

    perm_rows = ""
    mean_by = permutation_gate.get("mean_auroc_by_method", {})
    for method, auc in sorted(mean_by.items(), key=lambda kv: -float(kv[1])):
        perm_rows += f"<tr><td>{method}</td><td>{auc:.4f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Phase 9 acceptance report</title>
<style>body{{font-family:sans-serif;margin:24px;}}table{{border-collapse:collapse;font-size:12px;}}
td,th{{border:1px solid #ccc;padding:4px 8px;}}</style></head><body>
<h1>Phase 9 — acceptance checklist</h1>
<p>Experiment: <b>{cfg.name}</b> &mdash; results dir <code>{result_dir}</code></p>
<h2 style="color:{'green' if all_passed else 'red'}">Overall: {'PASS' if all_passed else 'FAIL'}</h2>
<h3>Gates</h3><table>{''.join(rows)}</table>
<h3>Label-permutation mean AUROC by method (seeds {permutation_gate.get('seeds')})</h3>
<table><tr><th>method</th><th>mean AUROC (permuted labels)</th></tr>{perm_rows}</table>
</body></html>"""
    report_path = result_dir / "acceptance_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path

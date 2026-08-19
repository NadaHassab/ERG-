"""Command-line entry point.

One unambiguous command per artifact stage.  Notebooks may explore but cannot
produce authoritative results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import DataConfig, PreprocessingConfig, load_config
from .constants import OUTER_FOLDS_TEMPLATE
from .provenance import RunManifest, git_revision


def _load_data(args) -> DataConfig:
    return load_config(DataConfig, args.data)


def cmd_audit(args) -> int:
    from .data.audit import audit_raw_files, write_audit

    cfg = load_config(DataConfig, args.config)
    result = audit_raw_files(cfg)
    manifest = RunManifest(kind="raw_audit", name="raw_files")
    manifest.code_revision = git_revision(Path.cwd())
    manifest.extra["failures"] = result.failures
    out = write_audit(result, manifest.to_dict(), Path(cfg.artifact_root))
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    print(f"wrote: {out}")
    return 0 if not result.failures else 1


def cmd_build_data(args) -> int:
    from .data.audit import audit_raw_files
    from .data.build import build_dataset

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    audit = audit_raw_files(data_cfg)
    if audit.failures:
        print("raw audit has failures; refusing to build", file=sys.stderr)
        for f in audit.failures:
            print(" -", f, file=sys.stderr)
        return 1
    audit_hash = RunManifest(kind="raw_audit", name="raw_files").hash
    artifacts = build_dataset(data_cfg, pre_cfg, raw_audit_hash=audit_hash)
    print(json.dumps(artifacts.checks, indent=2, sort_keys=True))
    print(f"data_hash: {artifacts.data_hash}")
    return 0


def cmd_make_splits(args) -> int:
    from .data.splits import (
        FoldConfig,
        assert_no_leakage,
        make_inner_folds,
        make_outer_folds,
        summarize_folds,
        write_splits,
    )

    data_cfg = load_config(DataConfig, args.data)
    fold_cfg = load_config(FoldConfig, args.fold_config)
    manifest_path = Path(args.build_manifest)
    RunManifest.load(manifest_path)

    subjects = _read_parquet(data_cfg, "participants")
    visits = _read_parquet(data_cfg, "visits")
    recordings = _read_parquet(data_cfg, "recordings")

    outer = make_outer_folds(subjects, visits, fold_cfg)
    inner_by_fold = {
        k: make_inner_folds(outer, subjects, visits, k, fold_cfg) for k in range(fold_cfg.n_outer)
    }
    report = summarize_folds(outer, subjects, visits, fold_cfg.age_bins)
    assert_no_leakage(outer, subjects, visits, recordings)
    paths = write_splits(outer, inner_by_fold, report, Path(data_cfg.artifact_root), fold_cfg.version)
    summary = json.loads(paths["summary"].read_text())
    print(json.dumps({"split_hash": summary["split_hash"]}, indent=2))
    print(json.dumps(report, indent=2, sort_keys=True)[:4000])
    return 0


def _read_parquet(data_cfg: DataConfig, name: str):
    import pandas as pd

    path = Path(data_cfg.artifact_root) / "data" / "interim" / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical table: {path}; run build-data first")
    return pd.read_parquet(path)


def cmd_run_qc(args) -> int:
    from .data.splits import FoldConfig
    from .signal.qc_report import run_qc

    data_cfg = load_config(DataConfig, args.data)
    fold_cfg = load_config(FoldConfig, args.fold_config)
    summary = run_qc(data_cfg.artifact_root, fold_cfg.version)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def cmd_fit_scalers(args) -> int:
    import numpy as np
    import zarr

    from .data.splits import FoldConfig
    from .signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths, load_cache_manifest
    from .signal.scalers import fit_fold_safe_scalers

    data_cfg = load_config(DataConfig, args.data)
    fold_cfg = load_config(FoldConfig, args.fold_config)
    n_folds = fold_cfg.n_outer
    cache = cache_paths(data_cfg.artifact_root, CACHE_SCHEMA_VERSION)
    load_cache_manifest(data_cfg.artifact_root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    recordings = pd.read_parquet(Path(data_cfg.artifact_root) / "data" / "interim" / "recordings.parquet")
    folds = pd.read_parquet(
        Path(data_cfg.artifact_root) / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=fold_cfg.version)
    )
    canonical = np.asarray(
        zarr.open_group(str(cache["curves_zarr"]), mode="r")[
            "components"
        ]["canonical_signal"][:]
    )
    cache_dir = Path(data_cfg.artifact_root) / "data" / "scalers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for fold in range(n_folds):
        scalers = fit_fold_safe_scalers(
            components, recordings, folds, canonical, fold, cache_dir
        )
        summaries[str(fold)] = {
            "n_strata": len(scalers),
            "fit_strata": sum(1 for s in scalers.values() if s.fit_count > 0),
            "min_fit_count": min((s.fit_count for s in scalers.values()), default=0),
        }
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


def cmd_run_qa(args) -> int:
    from .qa.report import run_qa

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    summary = run_qa(data_cfg.artifact_root, pre_cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_cache_external(args) -> int:
    from .signal.external_cache import cache_external_components

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    summary = cache_external_components(
        data_cfg.artifact_root,
        pre_cfg,
        datasets=tuple(args.datasets),
        binding=args.binding,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def cmd_make_external_splits(args) -> int:
    from .data.external_splits import EXTERNAL_FOLD_VERSION, build_external_splits
    from .data.splits import FoldConfig

    data_cfg = load_config(DataConfig, args.data)
    fold_cfg = load_config(FoldConfig, args.fold_config)
    subjects = _read_parquet(data_cfg, "participants")
    visits = _read_parquet(data_cfg, "visits")
    recordings = _read_parquet(data_cfg, "recordings")
    result = build_external_splits(
        data_cfg.artifact_root,
        subjects,
        visits,
        recordings,
        fold_cfg,
        datasets=tuple(args.datasets),
        version=args.version or EXTERNAL_FOLD_VERSION,
    )
    summary = json.loads(result.paths["summary"].read_text())
    print(json.dumps({"split_hash": summary["split_hash"]}, indent=2))
    print(json.dumps(result.report, indent=2, sort_keys=True)[:4000])
    return 0


def cmd_cache_components(args) -> int:
    from .signal.component_cache import cache_components

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    summary = cache_components(data_cfg.artifact_root, pre_cfg)
    print(summary)
    return 0


def cmd_cache_vmd(args) -> int:
    from .signal.vmd import VMDConfig
    from .signal.vmd_cache import cache_vmd

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    vmd_cfg = VMDConfig(
        K=args.k,
        alpha=args.alpha,
        tol=args.tol,
        mirror_pad_ms=args.pad_ms,
        max_iter=args.max_iter,
        stability_neighbors=tuple(args.neighbor_k),
    )
    summary = cache_vmd(
        data_cfg.artifact_root, pre_cfg, vmd_cfg, jobs=getattr(args, "jobs", 1)
    )
    print(summary)
    return 0


def cmd_vmd_grid(args) -> int:
    """Plan Section 15.2 hyperparameter grid over a recording subsample."""
    from .signal.vmd_grid import sweep_vmd_grid

    data_cfg = load_config(DataConfig, args.data)
    pre_cfg = load_config(PreprocessingConfig, args.preprocessing)
    summary = sweep_vmd_grid(
        data_cfg.artifact_root,
        pre_cfg,
        n_recordings=args.n_recordings,
        jobs=args.jobs,
        tag=args.tag,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def cmd_run_acceptance(args) -> int:
    from .config import BaselinesConfig
    from .evaluation.acceptance import run_label_permutation_gate, write_acceptance_report

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(BaselinesConfig, args.experiment)
    seeds = tuple(args.seeds)
    gate = run_label_permutation_gate(
        cfg, data_cfg, seeds=seeds, reuse_existing=args.reuse_existing
    )
    report = write_acceptance_report(data_cfg.artifact_root, cfg, gate)
    print(json.dumps(gate, indent=2, sort_keys=True, default=str))
    print(f"acceptance report: {report}")
    return 0 if gate.get("passed") else 1


def cmd_validate_transport(args) -> int:
    from .simulations.transport_validation import run_transport_battery, write_transport_report

    data_cfg = load_config(DataConfig, args.data)
    checks = run_transport_battery()
    report = write_transport_report(data_cfg.artifact_root, checks)
    print(json.dumps([c.to_dict() for c in checks], indent=2, sort_keys=True))
    print(f"report: {report}")
    return 0 if all(c.passed for c in checks) else 1


def cmd_simulate_sharing(args) -> int:
    from .simulations.partial_sharing import run_partial_sharing_grid, write_sharing_report

    data_cfg = load_config(DataConfig, args.data)
    grid = run_partial_sharing_grid()
    report = write_sharing_report(data_cfg.artifact_root, grid)
    cols = [
        "mismatch_sq", "sigma_sq", "n", "separate", "full", "oracle_partial",
        "wrong_partial", "learned_gate", "gate_shared_mean", "gate_mismatched_mean",
    ]
    print(grid[cols].round(4).to_string(index=False))
    print(f"report: {report}")
    return 0


def cmd_run_baselines(args) -> int:
    from .config import BaselinesConfig
    from .models.baselines import run_baselines, write_baselines_artifacts

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(BaselinesConfig, args.experiment)
    results = run_baselines(cfg, data_cfg)
    report = write_baselines_artifacts(
        data_cfg.artifact_root,
        results,
        cfg,
        pywt_note=_pywt_note(),
        experiment_path=Path(args.experiment),
    )
    lines = []
    for key in sorted(results.metrics):
        m = results.metrics[key]
        auc = m.get("roc_auc")
        if auc is None:
            lines.append(f"  {key:<42} n/a ({m.get('note')})")
        else:
            lines.append(
                f"  {key:<42} AUROC={auc:.4f} "
                f"[{m.get('roc_auc_ci_low')}, {m.get('roc_auc_ci_high')}] "
                f"bal_acc={m.get('balanced_accuracy', 0):.4f} "
                f"n={m.get('n_total')} pos={m.get('n_positive')}"
            )
    print("\n".join(lines))
    print(f"report: {report}")
    return 0


def cmd_run_confound_review(args) -> int:
    from .config import DataConfig
    from .evaluation.confound_review import run_confound_review

    data_cfg = load_config(DataConfig, args.data)
    r = run_confound_review(data_cfg.artifact_root)
    print(f"verdict: {r['verdict']}")
    for c in r["checks"]:
        print(f"[{c['outcome']:>4}] {c['check']}: {c['measurement']}")
    print(f"report: {Path(data_cfg.artifact_root) / 'results' / 'confounds' / 'confound_review.md'}")
    return 0 if r["verdict"] == "PASS" else 1


def cmd_run_perg_sensitivity(args) -> int:
    from .config import DataConfig
    from .evaluation.perg_sensitivity import PergSensitivityConfig, run_perg_sensitivity

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(PergSensitivityConfig, args.experiment)
    results = run_perg_sensitivity(cfg, data_cfg)
    import json

    out_dir = Path(data_cfg.artifact_root) / "results" / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    for key in sorted(results):
        if key.startswith("_"):
            continue
        for mid in sorted(results[key]):
            m = results[key][mid]
            auc = m.get("roc_auc")
            if auc is None:
                continue
            print(
                f"  {key:<28} {mid:<34} AUROC={auc:.4f} "
                f"[{m.get('roc_auc_ci_low')}, {m.get('roc_auc_ci_high')}] "
                f"bal_acc={m.get('balanced_accuracy', 0):.4f}"
            )
    print(f"metrics: {out_dir / 'metrics.json'}")
    return 0


def cmd_run_external_coverage(args) -> int:
    from .config import DataConfig
    from .evaluation.external_coverage import run_coverage_report

    data_cfg = load_config(DataConfig, args.data)
    report = run_coverage_report(data_cfg.artifact_root)
    print(json.dumps({"delta": report["delta"]}, indent=2, sort_keys=True, default=str))
    print(f"report: {Path(data_cfg.artifact_root) / 'results' / 'external_coverage' / 'coverage_report.md'}")
    return 0


def cmd_run_flinders_calibration(args) -> int:
    from .config import DataConfig
    from .evaluation.flinders_calibration import run_flinders_calibration

    data_cfg = load_config(DataConfig, args.data)
    report = run_flinders_calibration(data_cfg.artifact_root)
    for feat, r in report["features"].items():
        print(
            f"{feat:<8} KS={r['ks_raw']['statistic']:.3f} "
            f"(adj {r['ks_age_adjusted']['statistic']:.3f}) "
            f"within2SD={r['within_2sd_raw']:.3f} "
            f"(adj {r['within_2sd_age_adjusted']:.3f})"
        )
    print(f"report: {Path(data_cfg.artifact_root) / 'results' / 'flinders_calibration' / 'calibration_report.md'}")
    return 0


def cmd_run_urfu_sanity(args) -> int:
    from .config import DataConfig
    from .evaluation.urfu_sanity import run_urfu_sanity

    data_cfg = load_config(DataConfig, args.data)
    rep = run_urfu_sanity(data_cfg.artifact_root, use_gpu=args.gpu, n_folds=args.folds)
    low, high = rep["auroc_ci_95"]["low"], rep["auroc_ci_95"]["high"]
    low_s, high_s = ("—" if low is None else f"{low:.3f}"), ("—" if high is None else f"{high:.3f}")
    print(
        f"AUROC {rep['auroc']:.3f} [{low_s}, {high_s}] "
        f"bal_acc {rep['balanced_accuracy']:.3f} "
        f"(n={rep['n_subjects']}, healthy={rep['n_healthy']}, reduced={rep['n_reduced']})"
    )
    print(f"report: {Path(data_cfg.artifact_root) / 'results' / 'urfu_sanity' / 'sanity_report.md'}")
    return 0


def cmd_run_leop_la3_transfer(args) -> int:
    from .config import DataConfig
    from .evaluation.leop_la3_transfer import run_leop_la3_transfer

    data_cfg = load_config(DataConfig, args.data)
    rep = run_leop_la3_transfer(data_cfg.artifact_root, n_folds=args.folds)
    for scheme, r in rep["schemes"].items():
        low, high = r["ci_95"]["low"], r["ci_95"]["high"]
        low_s, high_s = ("—" if low is None else f"{low:.3f}"), ("—" if high is None else f"{high:.3f}")
        print(f"  {scheme:<9} AUROC {r['auroc']:.3f} [{low_s}, {high_s}]")
    print(f"report: {Path(data_cfg.artifact_root) / 'results' / 'leop_la3_extnorm_transfer' / 'transfer_report.md'}")
    return 0


def cmd_run_ssl_pretrain(args) -> int:
    import dataclasses

    from .training.ssl import SSLConfig, pretrain_ssl

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(SSLConfig, args.experiment)
    if args.exclude_fold is not None:
        cfg = dataclasses.replace(cfg, exclude_fold=args.exclude_fold)
    ckpt_path, log = pretrain_ssl(cfg, data_cfg)
    print(f"SSL checkpoint: {ckpt_path}")
    print(f"train folds: {sorted(set(cfg.outer_folds) - {cfg.exclude_fold})}")
    if log.train_loss:
        print(f"final epoch total loss: {log.train_loss[-1]:.4f}")
    return 0


def cmd_compare_external_sslinit(args) -> int:
    from .evaluation.external_comparison import compare_external_sslinit

    result = compare_external_sslinit(
        args.external_predictions,
        args.internal_predictions,
        args.output,
        n_reps=args.n_reps,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_run_separate_neural(args) -> int:
    import dataclasses

    from .training.separate import SeparateTrainingConfig, run_separate_training

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(SeparateTrainingConfig, args.experiment)
    if args.resume:
        cfg = dataclasses.replace(cfg, resume=True)
    result = run_separate_training(cfg, data_cfg)
    for task, metrics in sorted(result.metrics.items()):
        print(
            f"{task:<5} AUROC={metrics['roc_auc']:.4f} "
            f"[{metrics['roc_auc_ci_low']:.4f}, {metrics['roc_auc_ci_high']:.4f}] "
            f"n={metrics['n_total']} clusters={metrics['n_clusters']}"
        )
    out = Path(data_cfg.artifact_root) / "results" / cfg.output_subdir
    print(f"predictions: {out / 'predictions.parquet'}")
    return 0


def cmd_run_neural_confound_gate(args) -> int:
    from .evaluation.confound_gate import run_confound_gate
    from .training.separate import SeparateTrainingConfig

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(SeparateTrainingConfig, args.experiment)
    r = run_confound_gate(data_cfg.artifact_root, cfg)
    print(f"verdict: {r['verdict']}")
    for c in r["checks"]:
        print(f"[{c['outcome']:>4}] {c['check']}: {c['measurement']}")
    print(
        "report: "
        f"{Path(data_cfg.artifact_root) / 'results' / 'confounds' / 'neural_confound_gate.md'}"
    )
    return 0 if r["verdict"] == "PASS" else 1


def cmd_run_probes(args) -> int:
    from .data.datasets import LoadedCaches
    from .evaluation.probes import (
        load_model_from_checkpoint,
        run_probe_battery,
        save_probe_report,
    )

    data_cfg = load_config(DataConfig, args.data)
    model, cfg, _ = load_model_from_checkpoint(args.checkpoint)
    caches = LoadedCaches(data_cfg.artifact_root, fold_version=cfg.fold_version)
    results = run_probe_battery(
        model,
        caches,
        test_fold=args.fold,
        n_reps=args.n_reps,
        seed=args.seed,
        device=cfg.device,
    )
    out_dir = (
        Path(data_cfg.artifact_root) / "results" / cfg.output_subdir / "probes"
    )
    task = Path(args.checkpoint).parent.name.split("-fold")[0].split("-")[-1]
    out = save_probe_report(
        results, out_dir / task / f"probe_battery_fold{args.fold}.parquet"
    )
    for r in results:
        ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" if r.ci_low is not None else "[—]"
        print(
            f"{r.frame:<4} {r.stream:<7} {r.target:<18} "
            f"{r.metric}={r.value:.3f} {ci} (n={r.n_eval})"
        )
    print(f"report: {out}")
    return 0


def cmd_run_flinders_routed_calibration(args) -> int:
    from .evaluation.flinders_routed import (
        FlindersRoutedConfig,
        run_flinders_routed_calibration,
    )

    data_cfg = load_config(DataConfig, args.data)
    cfg = load_config(FlindersRoutedConfig, args.config)
    report = run_flinders_routed_calibration(
        data_cfg.artifact_root,
        args.checkpoint,
        args.fold,
        cfg,
        device=args.device,
    )
    out_dir = Path(data_cfg.artifact_root) / "results" / cfg.output_subdir
    print(f"report: {out_dir / 'calibration_report.json'}")
    for stream, rows in sorted(report["per_stream"].items()):
        if isinstance(rows, dict):
            print(f"[{stream}] {rows.get('note', '')}")
            continue
        for row in rows:
            print(
                f"[{stream}] {row['protocol']}: KS={row['ks_median_dim']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                f"FL={row['n_flinders_subjects']} LEOP={row['n_leop_controls']}"
            )
    return 0


def _pywt_note() -> str:
    import pywt

    note = (
        f"PyWavelets {pywt.__version__} installed; importlib metadata reports "
        "1.9.0. The pywt.__version__ string is a stale upstream build artifact "
        "shipped inside the official 1.9.0 wheel (pywt/version.py), not an "
        "environment defect."
    )
    return note if pywt.__version__ != "1.9.0" else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pathway_erg")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="immutable raw-file audit")
    p.add_argument("--config", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("build-data", help="canonical data build")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("cache-components", help="component + signed-OT caches")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.set_defaults(func=cmd_cache_components)

    p = sub.add_parser(
        "cache-external-components",
        help="external (URFU/FLINDERS) component cache, separate binding",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.add_argument("--datasets", nargs="+", default=["URFU", "FLINDERS"])
    p.add_argument("--binding", default="external_v1")
    p.set_defaults(func=cmd_cache_external)

    p = sub.add_parser(
        "make-external-splits",
        help="external (URFU/FLINDERS) subject-keyed nested folds",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--fold-config", default="configs/data/folds.yaml")
    p.add_argument("--datasets", nargs="+", default=["URFU", "FLINDERS"])
    p.add_argument("--version", default=None)
    p.set_defaults(func=cmd_make_external_splits)

    p = sub.add_parser("run-qa", help="pipeline QA HTML report")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.set_defaults(func=cmd_run_qa)

    p = sub.add_parser("validate-transport", help="E1 synthetic transport validation")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_validate_transport)

    p = sub.add_parser("simulate-sharing", help="E2 partial-sharing bias-variance simulation")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_simulate_sharing)

    p = sub.add_parser("fit-scalers", help="fold-safe robust scalers")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--fold-config", default="configs/data/folds.yaml")
    p.set_defaults(func=cmd_fit_scalers)

    p = sub.add_parser("run-qc", help="fold-dependent QC + populations")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--fold-config", default="configs/data/folds.yaml")
    p.set_defaults(func=cmd_run_qc)

    p = sub.add_parser("make-splits", help="nested grouped folds")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--build-manifest", default="artifacts/data/manifests/build_manifest.json")
    p.add_argument("--fold-config", default="configs/data/folds.yaml")
    p.set_defaults(func=cmd_make_splits)

    p = sub.add_parser("cache-vmd", help="VMD mode feature cache (baseline-only)")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.add_argument("--k", type=int, default=5, help="mode count K (plan 15.2 grid)")
    p.add_argument("--alpha", type=float, default=2000.0)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--pad-ms", type=float, default=25.0, help="mirror pad in ms")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--neighbor-k", type=int, nargs="+", default=[4, 6])
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_cache_vmd)

    p = sub.add_parser(
        "vmd-grid",
        help="plan 15.2 VMD hyperparameter grid on a recording subsample",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--preprocessing", default="configs/preprocessing/reference.yaml")
    p.add_argument("--n-recordings", type=int, default=500)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tag", default="s15")
    p.set_defaults(func=cmd_vmd_grid)

    p = sub.add_parser("run-baselines", help="E0/E4 classical baseline suite")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--experiment", default="configs/experiments/e4_baselines.yaml")
    p.set_defaults(func=cmd_run_baselines)

    p = sub.add_parser("run-acceptance", help="Phase 9 acceptance gates (hashes, metrics, label permutation)")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--experiment", default="configs/experiments/e4_baselines.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="recompute verdict from existing predictions instead of refitting",
    )
    p.set_defaults(func=cmd_run_acceptance)

    p = sub.add_parser("run-perg-sensitivity", help="Phase 8 PERG sensitivity ablations")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--experiment", default="configs/experiments/perg_sensitivity_v1.yaml")
    p.set_defaults(func=cmd_run_perg_sensitivity)

    p = sub.add_parser(
        "confound-review",
        help="pre-Phase-6 fallback/confounding shortcut review gate",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_run_confound_review)

    p = sub.add_parser("external-coverage", help="gate 7 probe 1: coverage/diversity report")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_run_external_coverage)

    p = sub.add_parser(
        "flinders-calibration", help="gate 7 probe 2: LEOP controls vs Flinders norm"
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.set_defaults(func=cmd_run_flinders_calibration)

    p = sub.add_parser(
        "urfu-sanity", help="gate 7 probe 3: URFU labels carry signal (held-out)"
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.set_defaults(func=cmd_run_urfu_sanity)

    p = sub.add_parser(
        "leop-la3-transfer",
        help="gate 7 probe 4: LEOP LA3 classification under Flinders external norm",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--folds", type=int, default=5)
    p.set_defaults(func=cmd_run_leop_la3_transfer)

    p = sub.add_parser(
        "run-separate-neural",
        help="item 18: fold-safe separate LEOP/PERG neural baselines",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument(
        "--experiment",
        default="configs/experiments/e6_separate_raw_ot_hierarchical_v1.yaml",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip runs whose run dir already has a COMPLETE marker",
    )
    p.set_defaults(func=cmd_run_separate_neural)

    p = sub.add_parser(
        "neural-confound-gate",
        help="post-hoc confound gate on ensembled neural OOF predictions "
             "(plan Section 17 / E0 decision rule)",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument(
        "--experiment",
        default="configs/experiments/e6_separate_raw_ot_hierarchical_v1.yaml",
    )
    p.set_defaults(func=cmd_run_neural_confound_gate)

    p = sub.add_parser(
        "run-ssl-pretrain",
        help="item 19: joint SSL pretraining with one held-out fold (plan §14)",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument(
        "--experiment",
        default="configs/experiments/e7_ssl_pretrain_v1.yaml",
    )
    p.add_argument(
        "--exclude-fold",
        type=int,
        required=True,
        help="outer fold excluded from SSL and its supervised fine-tuning",
    )
    p.set_defaults(func=cmd_run_ssl_pretrain)

    p = sub.add_parser(
        "compare-external-sslinit",
        help="paired LEOP/PERG comparison of four-domain vs two-domain SSL-init",
    )
    p.add_argument(
        "--external-predictions",
        default="artifacts/results/separate_raw_ot_hierarchical_sslinit_external_v1/predictions.parquet",
    )
    p.add_argument(
        "--internal-predictions",
        default="artifacts/results/separate_raw_ot_hierarchical_sslinit_v1/predictions.parquet",
    )
    p.add_argument(
        "--output",
        default="artifacts/results/external_v1/paired_comparisons.json",
    )
    p.add_argument("--n-reps", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=424242)
    p.set_defaults(func=cmd_compare_external_sslinit)

    p = sub.add_parser(
        "run-probes",
        help="item 22 (E12): linear probes on frozen embeddings of one checkpoint",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="path to final.pt (eval fold must be the fold it never saw)",
    )
    p.add_argument(
        "--fold",
        type=int,
        required=True,
        help="outer fold excluded from the model; probes evaluate on it",
    )
    p.add_argument("--n-reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_run_probes)

    p = sub.add_parser(
        "flinders-routed-calibration",
        help="plan §11.4: headless FLINDERS routed-token calibration probe",
    )
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--config", default="configs/experiments/e9_flinders_routed_v1.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_run_flinders_routed_calibration)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

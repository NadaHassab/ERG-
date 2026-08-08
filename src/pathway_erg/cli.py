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
    report = write_baselines_artifacts(data_cfg.artifact_root, results, cfg, pywt_note=_pywt_note())
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
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=2000.0)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--pad-ms", type=float, default=25.0)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--neighbor-k", type=int, nargs="+", default=[4, 6])
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_cache_vmd)

    p = sub.add_parser("run-baselines", help="E0/E4 classical baseline suite")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--experiment", default="configs/experiments/e4_baselines.yaml")
    p.set_defaults(func=cmd_run_baselines)

    p = sub.add_parser("run-perg-sensitivity", help="Phase 8 PERG sensitivity ablations")
    p.add_argument("--data", default="configs/data/local.yaml")
    p.add_argument("--experiment", default="configs/experiments/perg_sensitivity_v1.yaml")
    p.set_defaults(func=cmd_run_perg_sensitivity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Confound audit analyses (plan E0/E11 gates) for the LEOP primary cohort.

Three audits, all consuming the same canonical caches and locked folds as the
recorded baselines (never recomputing them differently):

1. **Slot-count ablation** — does the per-slot ``_n`` (recording count) and
   ``_flagged_rate`` block leak availability into the biological slot model?
   Runs ``slot_logreg`` on the locked outer folds with (a) all columns,
   (b) without ``_n``, (c) without ``_n`` and without ``_flagged_rate``.
2. **Leave-one-site-out (LOSO)** — train on one acquisition site, test on the
   other (both directions), for slot / clinical / derot families. A site or
   recording-count detector fails by construction; surviving performance is
   the strongest biology evidence available at this sample size.
3. **Sex-adjusted reporting** — overall, per-stratum and stratum-weighted
   AUROC for every prediction set produced here.

Outputs (versioned, never touching baseline artifacts):
``artifacts/results/confounds/confound_audit.json`` and ``.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import BaselinesConfig
from ..constants import INNER_FOLDS_TEMPLATE, OUTER_FOLDS_TEMPLATE
from ..models.baselines import (
    FeatureSet,
    _fit_transform_features,
    _load_units,
    _make_estimator,
    _parameter_grid,
    e4_clinical_leops,
    e4_derot_features,
    select_and_fit,
)
from ..models.leop_cohorts import (
    cohort_component_mask,
    cohort_recordings_mask,
    cohort_unit_mask,
)
from ..models.slot_features import e4_slot_features
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths

SEED = 777


# ---------------------------------------------------------------------------
# Loading (mirrors run_baselines for the LEOP primary nine-step cohort)
# ---------------------------------------------------------------------------


def load_leop_primary(root: Path):
    participants = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    cache = cache_paths(root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    folds = pd.read_parquet(root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version="v1"))
    inner = pd.read_parquet(root / "data" / "splits" / INNER_FOLDS_TEMPLATE.format(version="v1"))
    sot = np.asarray(
        zarr.open_group(str(cache["sot_zarr"]), mode="r")["components"]["sot_vector"][:]
    )

    units = _load_units("LEOP", participants, visits, folds)
    unit_mask = cohort_unit_mask(units, recordings, "primary_nine_step")
    units = units[unit_mask].reset_index(drop=True)

    rec_mask = cohort_recordings_mask(recordings, "primary_nine_step")
    comp_mask = cohort_component_mask(components, recordings, "primary_nine_step")
    keep = np.flatnonzero(comp_mask.to_numpy())
    rec = recordings[rec_mask].reset_index(drop=True)
    comp = components[comp_mask].reset_index(drop=True)
    sot_c = sot[keep]

    sx = units["sex_standardized"].astype(str).str.lower()
    units["sex"] = np.where(sx.isin(["1", "male"]), "M", "F")
    return units, comp, rec, sot_c, inner


# ---------------------------------------------------------------------------
# Feature-set variants
# ---------------------------------------------------------------------------


def _drop_columns(fs: FeatureSet, suffixes: tuple[str, ...], tag: str) -> FeatureSet:
    keep = [i for i, n in enumerate(fs.names) if not n.endswith(suffixes)]
    return FeatureSet(
        unit_id=fs.unit_id,
        X=fs.X[:, keep],
        names=[fs.names[i] for i in keep],
        per_unit_n=fs.per_unit_n,
        notes={**fs.notes, "ablation": tag, "n_dropped": len(fs.names) - len(keep)},
    )


def build_feature_sets(
    units: pd.DataFrame, comp: pd.DataFrame, rec: pd.DataFrame, sot: np.ndarray
) -> dict[str, FeatureSet]:
    slot = e4_slot_features(units, comp, rec, "LEOP")
    return {
        "slot_full": slot,
        "slot_no_n": _drop_columns(slot, ("_n",), "no per-slot recording counts"),
        "slot_no_counts": _drop_columns(
            slot, ("_n", "_flagged_rate"), "no counts, no flag rates"
        ),
        "clinical": e4_clinical_leops(units, comp, rec),
        "derot": e4_derot_features(units, comp, rec, "LEOP", sot),
    }


# ---------------------------------------------------------------------------
# Evaluation harnesses
# ---------------------------------------------------------------------------


def _auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def sex_report(y: np.ndarray, p: np.ndarray, sex: np.ndarray) -> dict:
    out = {"auroc": _auroc(y, p), "n": int(len(y)), "n_pos": int(np.sum(y))}
    strata = {}
    weighted, wsum = 0.0, 0
    for s in ("F", "M"):
        m = np.asarray(sex) == s
        a = _auroc(y[m], p[m]) if m.sum() > 15 else None
        strata[s] = {"auroc": a, "n": int(m.sum()), "n_pos": int(np.sum(y[m]))}
        if a is not None:
            weighted += a * m.sum()
            wsum += int(m.sum())
    out["strata"] = strata
    out["sex_adjusted"] = weighted / wsum if wsum else None
    return out


def oof_locked_fold(
    fs: FeatureSet,
    units: pd.DataFrame,
    inner: pd.DataFrame,
    kind: str,
    cfg: BaselinesConfig,
    method_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold predictions on the locked outer folds (pipeline-faithful)."""
    y_all = units["target_binary"].to_numpy(float)
    prob = np.full(len(units), np.nan)
    X = fs.X
    for outer_fold in sorted(units["outer_fold"].unique()):
        test = (units["outer_fold"] == outer_fold).to_numpy()
        train = ~test
        X_tr, X_te = X[train], X[test]
        if X_tr.shape[1] == 0:
            prob[test] = np.mean(y_all[train])
            continue
        X_tr_p, X_te_p, _ = _fit_transform_features(X_tr, X_te, method_name, cfg)
        est, _params, _inner_auc = select_and_fit(
            kind,
            "LEOP",
            X_tr_p,
            y_all[train],
            units["subject_id"].to_numpy()[train],
            inner,
            int(outer_fold),
            cfg.seed,
            use_gpu=cfg.use_gpu,
        )
        prob[test] = est.predict_proba(X_te_p)[:, 1]
    return y_all, prob


def _inner_select_fit(
    kind: str, X: np.ndarray, y: np.ndarray, seed: int, use_gpu: bool, n_splits: int = 4
):
    """Small stratified inner selection for LOSO (one row per participant)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_score, best_params = -np.inf, None
    for params in _parameter_grid(kind):
        scores = []
        for tr, te in skf.split(X, y):
            est = _make_estimator(kind, params, seed, use_gpu)
            try:
                est.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))
            except Exception:
                scores.append(np.nan)
        score = float(np.nanmean(scores))
        if np.isfinite(score) and score > best_score:
            best_score, best_params = score, params
    if best_params is None:
        raise ValueError(f"no valid inner fold for {kind!r} during LOSO selection")
    est = _make_estimator(kind, best_params, seed, use_gpu).fit(X, y)
    return est, best_params, best_score


def loso(
    fs: FeatureSet,
    units: pd.DataFrame,
    kind: str,
    cfg: BaselinesConfig,
    method_name: str,
) -> dict[str, dict]:
    """Leave-one-site-out, both directions."""
    out: dict[str, dict] = {}
    sites = sorted(s for s in units["site"].dropna().unique())
    X = fs.X
    y = units["target_binary"].to_numpy(float)
    sex = units["sex"].to_numpy()
    for train_site in sites:
        for test_site in sites:
            if train_site == test_site:
                continue
            tr = (units["site"] == train_site).to_numpy()
            te = (units["site"] == test_site).to_numpy()
            if te.sum() < 20 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            X_tr_p, X_te_p, _ = _fit_transform_features(X[tr], X[te], method_name, cfg)
            est, params, inner_auc = _inner_select_fit(
                kind, X_tr_p, y[tr], cfg.seed, cfg.use_gpu
            )
            p = est.predict_proba(X_te_p)[:, 1]
            rep = sex_report(y[te], p, sex[te])
            rep["selected_params"] = {k: float(v) for k, v in params.items()}
            rep["inner_auc"] = float(inner_auc)
            out[f"train_site{train_site}_test_site{test_site}"] = rep
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_confound_audit(
    artifact_root: str | Path = "artifacts",
    use_gpu: bool = True,
    out_subdir: str = "confounds",
    run_slot_oof: bool = True,
    run_loso: bool = True,
) -> dict:
    root = Path(artifact_root)
    cfg = BaselinesConfig(name="confound_audit", use_gpu=use_gpu, seed=SEED)
    units, comp, rec, sot, inner = load_leop_primary(root)
    fsets = build_feature_sets(units, comp, rec, sot)

    sex = units["sex"].to_numpy()
    results: dict[str, object] = {
        "cohort": "LEOP_primary_nine_step",
        "n_units": int(len(units)),
        "n_positive": int(units["target_binary"].sum()),
        "site_counts": {str(k): int(v) for k, v in units["site"].value_counts().items()},
        "sex_counts": {str(k): int(v) for k, v in units["sex"].value_counts().items()},
        "use_gpu": bool(use_gpu),
        "feature_dims": {k: int(fs.X.shape[1]) for k, fs in fsets.items()},
    }

    if run_slot_oof:
        slot_cv: dict[str, object] = {}
        for variant in ("slot_full", "slot_no_n", "slot_no_counts"):
            fs = fsets[variant]
            y, p = oof_locked_fold(fs, units, inner, "logreg", cfg, "slot")
            slot_cv[variant] = sex_report(y, p, sex)
        results["slot_count_ablation_locked_cv"] = slot_cv

    if run_loso:
        loso_all: dict[str, object] = {}
        for fam, kind in (("slot_no_counts", "logreg"), ("clinical", "logreg"), ("derot", "logreg")):
            loso_all[fam] = loso(fsets[fam], units, kind, cfg, "slot")
        results["loso"] = loso_all

    out_dir = root / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confound_audit.json").write_text(json.dumps(results, indent=2))
    (out_dir / "confound_audit.md").write_text(_render_markdown(results))
    return results


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _render_markdown(results: dict) -> str:
    lines = [
        "# Confound audit — LEOP primary nine-step cohort",
        "",
        f"- units: {results['n_units']} (ASD={results['n_positive']})",
        f"- sites: {results['site_counts']} | sex: {results['sex_counts']}",
        f"- feature dims: {results['feature_dims']}",
        "",
        "## 1. Slot-count ablation (locked outer-fold CV, slot_logreg)",
        "",
        "| variant | AUROC | sex-adjusted | F stratum | M stratum |",
        "|---|---|---|---|---|",
    ]
    for name, rep in (results.get("slot_count_ablation_locked_cv") or {}).items():
        lines.append(
            f"| {name} | {_fmt(rep['auroc'])} | {_fmt(rep['sex_adjusted'])} | "
            f"{_fmt(rep['strata']['F']['auroc'])} | {_fmt(rep['strata']['M']['auroc'])} |"
        )
    lines += [
        "",
        "## 2. Leave-one-site-out (LOSO)",
        "",
        "| family | split | AUROC | sex-adjusted | n (pos) |",
        "|---|---|---|---|---|",
    ]
    for fam, splits in (results.get("loso") or {}).items():
        for split, rep in splits.items():
            lines.append(
                f"| {fam} | {split} | {_fmt(rep['auroc'])} | {_fmt(rep['sex_adjusted'])} | "
                f"{rep['n']} ({rep['n_pos']}) |"
            )
    lines += [
        "",
        "Interpretation: a slot-count/availability leak shows up as slot_full > "
        "slot_no_counts; a site detector collapses under LOSO. Surviving "
        "sex-adjusted performance is the biology evidence.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LEOP confound audit (E0/E11)")
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--skip-slot-oof", action="store_true")
    ap.add_argument("--skip-loso", action="store_true")
    args = ap.parse_args()
    res = run_confound_audit(
        artifact_root=args.artifact_root,
        use_gpu=not args.no_gpu,
        run_slot_oof=not args.skip_slot_oof,
        run_loso=not args.skip_loso,
    )
    print(json.dumps(res, indent=2)[:3000])

"""LEOP transfer probe (plan gate 7, probe 4).

Does adding Flinders normative controls to the *feature standardization* (not
to the labels, and not to test folds) change LEOP Control-vs-ASD
classification?  The frozen LEOP primary endpoint is the nine-step cohort,
which Flinders cannot contribute to (it has no nine-step traces).  The shared
feature space is the ISCEV photopic **LA3** protocol (both datasets measure
the same physiology with skin electrodes), so the probe classifies LEOP LA3
Control-vs-ASD with per-subject a/b features, under two scaler schemes:

1. ``baseline`` — scaler fitted on LEOP training folds only,
2. ``extnorm`` — scaler fitted on LEOP training folds **plus** all Flinders
   LA3 healthy-control feature rows (external reference, no label mixing;
   Flinders rows never enter a test fold and contribute no labels).

The question is whether a normative healthy reference at standardization time
helps, hurts, or leaves LEOP classification unchanged (with bootstrap CIs).
Results are written to a new output subdir; frozen baseline artifacts are
never touched.

Outputs: ``artifacts/results/leop_la3_extnorm_transfer/transfer_report.{json,md}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..data import flinders
from ..models.baselines import _make_estimator, _parameter_grid
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths

INTERIM = Path("artifacts/data/interim")
FLINDERS_XLSX = Path("14747349 (2)/ISCEV Control ERG Flinders University.xlsx")
SEED = 777
N_FOLDS = 5
FEATURES = ["a_amp", "b_amp", "a_time", "b_time"]
FEATURE_NAMES = {
    "a_amp": "a-wave amplitude (µV)",
    "b_amp": "b-wave amplitude (µV)",
    "a_time": "a-wave latency (ms)",
    "b_time": "b-wave latency (ms)",
}


def _flinders_la3_reference() -> pd.DataFrame:
    rows = list(flinders.iter_feature_rows(FLINDERS_XLSX))
    df = pd.DataFrame(rows)
    df = df[df["Test"] == "LA3"].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES)


def _leop_la3_subjects() -> pd.DataFrame:
    """Per-subject median LA3 a/b features for LEOP Control/ASD subjects."""
    cache = cache_paths("artifacts", CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    recordings = pd.read_parquet(INTERIM / "recordings.parquet")
    participants = pd.read_parquet(INTERIM / "participants.parquet")

    la3 = recordings[
        (recordings["dataset"] == "LEOP")
        & (recordings["protocol"] == "LA3")
        & (recordings["waveform_kind"] == "ERG")
    ]
    la3 = la3.merge(
        participants[["global_subject_id", "group_raw"]], on="global_subject_id", how="left"
    )
    la3 = la3[la3["group_raw"].isin(["Control", "ASD"])]
    ids = set(la3["global_recording_id"])

    amp = components[components["global_recording_id"].isin(ids)][
        ["global_recording_id", "component_id", "physical_features_json"]
    ]
    rows = []
    for _, r in amp.iterrows():
        phys = json.loads(r["physical_features_json"])
        cid = r["component_id"]
        rows.append(
            {
                "global_recording_id": r["global_recording_id"],
                "component_id": cid,
                "peak_uv": float(phys["max_uv"]) if cid == "L_A_TO_B" else None,
                "trough_uv": float(phys["min_uv"]) if cid == "L_EARLY_A" else None,
            }
        )
    wide = pd.DataFrame(rows)
    wide = wide.pivot(
        index="global_recording_id", columns="component_id", values="peak_uv"
    )
    wide = wide.rename(columns={"L_A_TO_B": "b_amp"})

    early = components[components["global_recording_id"].isin(ids)][
        ["global_recording_id", "component_id", "landmark_times_json", "landmark_amplitudes_json"]
    ]
    lrows = []
    for _, r in early.iterrows():
        times = json.loads(r["landmark_times_json"]) if isinstance(r["landmark_times_json"], str) else {}
        amps = json.loads(r["landmark_amplitudes_json"]) if isinstance(r["landmark_amplitudes_json"], str) else {}
        lrows.append(
            {
                "global_recording_id": r["global_recording_id"],
                "a_amp": amps.get("a_trough"),
                "a_time": times.get("a_trough"),
                "b_time": times.get("b_peak"),
            }
        )
    land = pd.DataFrame(lrows)
    land = land.groupby("global_recording_id", as_index=False).agg(
        a_amp=("a_amp", "median"),
        a_time=("a_time", "median"),
        b_time=("b_time", "median"),
    )
    feat = wide.merge(land, on="global_recording_id", how="inner")

    rec_meta = la3[["global_recording_id", "global_subject_id", "group_raw"]].drop_duplicates("global_recording_id")
    out = feat.merge(rec_meta, on="global_recording_id", how="inner")
    out = out.groupby(["global_subject_id", "group_raw"], as_index=False)[FEATURES].median()
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def _scale(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int = SEED, reps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        a = _auroc(y[idx], p[idx])
        if a is not None:
            aucs.append(a)
    if not aucs:
        return {"low": None, "high": None}
    low, high = np.percentile(aucs, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def _run_scheme(
    X: np.ndarray,
    y: np.ndarray,
    ext_mu: np.ndarray | None,
    ext_sd: np.ndarray | None,
    n_folds: int = N_FOLDS,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    prob = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        if ext_mu is None:
            mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=1)
            X_tr_s, X_te_s = _scale(X_tr, mu, sd), _scale(X_te, mu, sd)
        else:
            X_tr_s, X_te_s = _scale(X_tr, ext_mu, ext_sd), _scale(X_te, ext_mu, ext_sd)
        best_score, best_params = -np.inf, None
        for params in _parameter_grid("logreg"):
            inner = StratifiedKFold(n_splits=min(4, tr.size // 2), shuffle=True, random_state=SEED)
            scores = []
            for itr, ite in inner.split(X_tr, y[tr]):
                est = _make_estimator("logreg", params, SEED, False)
                try:
                    est.fit(X_tr_s[itr], y[tr][itr])
                    scores.append(roc_auc_score(y[tr][ite], est.predict_proba(X_tr_s[ite])[:, 1]))
                except Exception:
                    scores.append(np.nan)
            score = float(np.nanmean(scores)) if scores else -np.inf
            if np.isfinite(score) and score > best_score:
                best_score, best_params = score, params
        if best_params is None:
            raise ValueError("no valid inner fold for transfer probe logreg")
        est = _make_estimator("logreg", best_params, SEED, False).fit(X_tr_s, y[tr])
        prob[te] = est.predict_proba(X_te_s)[:, 1]
    auroc = _auroc(y, prob)
    return {"auroc": auroc, "ci_95": _bootstrap_ci(y, prob), "n_pos": int(np.sum(y))}


def run_leop_la3_transfer(
    artifact_root: str | Path = "artifacts", n_folds: int = N_FOLDS
) -> dict:
    root = Path(artifact_root)
    le = _leop_la3_subjects()
    fl = _flinders_la3_reference()

    y = (le["group_raw"] == "ASD").to_numpy(int)
    X = le[FEATURES].to_numpy(float)
    ext_mu = fl[FEATURES].to_numpy(float).mean(axis=0)
    ext_sd = fl[FEATURES].to_numpy(float).std(axis=0, ddof=1)

    baseline = _run_scheme(X, y, None, None, n_folds)
    extnorm = _run_scheme(X, y, ext_mu, ext_sd, n_folds)

    report = {
        "probe": "LEOP LA3 Control-vs-ASD transfer (Flinders external norm)",
        "protocol": "LA3",
        "n_leop_subjects": int(len(le)),
        "n_control": int((le["group_raw"] == "Control").sum()),
        "n_asd": int((le["group_raw"] == "ASD").sum()),
        "n_flinders_reference_rows": int(len(fl)),
        "features": FEATURES,
        "feature_units": FEATURE_NAMES,
        "n_folds": n_folds,
        "schemes": {
            "baseline": {"description": "scaler fitted on LEOP train folds only", **baseline},
            "extnorm": {
                "description": "scaler fitted on LEOP train + Flinders LA3 healthy controls (no label mixing)",
                **extnorm,
            },
        },
        "flinders_reference": {
            col: {"mean": float(fl[col].mean()), "sd": float(fl[col].std(ddof=1))}
            for col in FEATURES
        },
    }

    out_dir = root / "results" / "leop_la3_extnorm_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (out_dir / "transfer_report.md").write_text(_render_markdown(report))
    return report


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _render_markdown(report: dict) -> str:
    b = report["schemes"]["baseline"]
    e = report["schemes"]["extnorm"]
    lines = [
        "# LEOP LA3 transfer probe (gate 7, probe 4)",
        "",
        f"Protocol: **{report['protocol']}** (shared with Flinders), "
        f"{report['n_leop_subjects']} LEOP subjects "
        f"(Control={report['n_control']}, ASD={report['n_asd']}), "
        f"Flinders LA3 reference rows: {report['n_flinders_reference_rows']}.",
        "",
        f"- **baseline** (scaler on LEOP train only): AUROC "
        f"**{_fmt(b['auroc'])}** [{_fmt(b['ci_95']['low'])}, {_fmt(b['ci_95']['high'])}]",
        f"- **extnorm** (scaler on LEOP train + Flinders healthy controls): AUROC "
        f"**{_fmt(e['auroc'])}** [{_fmt(e['ci_95']['low'])}, {_fmt(e['ci_95']['high'])}]",
        "",
        "Flinders LA3 reference (mean±SD):",
    ]
    for col in report["features"]:
        r = report["flinders_reference"][col]
        lines.append(f"- {report['feature_units'][col]}: {r['mean']:.2f}±{r['sd']:.2f}")
    lines += [
        "",
        "Interpretation: if extnorm ≈ baseline, LEOP standardization is robust "
        "to the healthy external reference; a shift indicates sensitivity to "
        "the reference population at scaling time. Flinders contributes only "
        "to scaling — never to labels or test folds.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LEOP LA3 Flinders-norm transfer probe (gate 7 probe 4)")
    ap.add_argument("--artifact-root", default="artifacts")
    args = ap.parse_args()
    rep = run_leop_la3_transfer(artifact_root=args.artifact_root)
    for scheme, r in rep["schemes"].items():
        print(
            f"{scheme:<9} AUROC {_fmt(r['auroc'])} "
            f"[{_fmt(r['ci_95']['low'])}, {_fmt(r['ci_95']['high'])}]"
        )

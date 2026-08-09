"""URFU supervised-sanity probe (plan gate 7, probe 3).

Establishes that the URFU labels carry signal before any transfer claims:
a held-out participant-level diagnostic task, "healthy" vs "reduced", using
the URFU ``features_1`` a/b-wave amplitudes/latencies (the Maximum 2.0 ERG
Response protocol — the canonical diagnostic protocol in the URFU papers).

The diagnosis mapping is explicit and reviewable (``data/urfu_labels.py``,
reviewer PENDING_CLINICAL_REVIEW): every observed free-text diagnosis is
mapped to healthy (0) / reduced (1) / ineligible (None); subjects without a
mapped label are excluded.  Participants are never split across folds; each
subject contributes one median feature vector per protocol.

Outputs (versioned, never touching baseline artifacts):
``artifacts/results/urfu_sanity/sanity_report.{json,md}`` + the diagnosis
mapping table ``diagnosis_mapping_urfu.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..data.urfu_labels import build_urfu_mapping, make_urfu_target
from ..models.baselines import _make_estimator, _parameter_grid

INTERIM = Path("artifacts/data/interim")
SEED = 777
PROBE_PROTOCOL = "Maximum 2.0 ERG Response"
FEATURE_COLUMNS = [
    "a-wave amplitude, µV",
    "a-wave latency, ms",
    "b-wave amplitude, µV",
    "b-wave latency, ms",
]


def _urfu_frame() -> tuple[pd.DataFrame, dict]:
    """Per-subject Median Maximum-2.0 features_1 + mapped target."""
    import json as _json

    visits = pd.read_parquet(INTERIM / "visits.parquet")
    recordings = pd.read_parquet(INTERIM / "recordings.parquet")
    ur = recordings[
        (recordings["dataset"] == "URFU")
        & (recordings["protocol"] == "Maximum 2.0")
        & (recordings["waveform_kind"] == "ERG")
    ].copy()

    rows = []
    for _, r in ur.iterrows():
        j = _json.loads(r["supplied_features_json"])
        f = j.get("features_1") or {}
        rows.append(
            {
                "global_subject_id": r["global_subject_id"],
                **{col: f.get(col) for col in FEATURE_COLUMNS},
            }
        )
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    med = df.groupby("global_subject_id", as_index=False)[FEATURE_COLUMNS].median()
    med = med.dropna(subset=FEATURE_COLUMNS, how="all")

    diag = visits[visits["dataset"] == "URFU"][
        ["global_subject_id", "diagnosis1_raw"]
    ]
    mapping = build_urfu_mapping(diag["diagnosis1_raw"])
    diag["target"] = diag["diagnosis1_raw"].map(
        lambda d: make_urfu_target(d, mapping)
    )
    out = med.merge(diag[["global_subject_id", "target"]], on="global_subject_id", how="inner")
    out = out.dropna(subset=["target"])
    return out.reset_index(drop=True), {
        "mapping": mapping,
        "n_subjects": int(len(out)),
        "n_healthy": int((out["target"] == 0).sum()),
        "n_reduced": int((out["target"] == 1).sum()),
    }


def _estimate_auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int = SEED, reps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    if len(np.unique(y)) < 2:
        return {"low": None, "high": None}
    aucs = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        a = _estimate_auroc(y[idx], p[idx])
        if a is not None:
            aucs.append(a)
    if not aucs:
        return {"low": None, "high": None}
    low, high = np.percentile(aucs, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def run_urfu_sanity(
    artifact_root: str | Path = "artifacts",
    use_gpu: bool = False,
    n_folds: int = 5,
) -> dict:
    root = Path(artifact_root)
    frame, meta = _urfu_frame()
    y = frame["target"].to_numpy(float)
    X = frame[FEATURE_COLUMNS].to_numpy(float)
    subject_ids = frame["global_subject_id"].to_numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    prob = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        best_score, best_params = -np.inf, None
        for params in _parameter_grid("logreg"):
            inner_skf = StratifiedKFold(n_splits=min(4, tr.size // 2), shuffle=True, random_state=SEED)
            scores = []
            for itr, ite in inner_skf.split(X[tr], y[tr]):
                est = _make_estimator("logreg", params, SEED, use_gpu)
                try:
                    est.fit(X[tr][itr], y[tr][itr])
                    scores.append(roc_auc_score(y[tr][ite], est.predict_proba(X[tr][ite])[:, 1]))
                except Exception:
                    scores.append(np.nan)
            score = float(np.nanmean(scores)) if scores else -np.inf
            if np.isfinite(score) and score > best_score:
                best_score, best_params = score, params
        if best_params is None:
            raise ValueError("no valid inner fold for URFU sanity logreg")
        est = _make_estimator("logreg", best_params, SEED, use_gpu).fit(X[tr], y[tr])
        prob[te] = est.predict_proba(X[te])[:, 1]

    pred = pd.DataFrame(
        {"global_subject_id": subject_ids, "y": y, "prob_reduced": prob}
    )
    auroc = _estimate_auroc(y, prob)
    ci = _bootstrap_ci(y, prob)
    bal_acc = float(balanced_accuracy_score(y, prob > 0.5)) if auroc is not None else None

    report = {
        "probe": "URFU supervised sanity (healthy vs reduced)",
        "protocol": PROBE_PROTOCOL,
        "n_folds": n_folds,
        "feature_columns": FEATURE_COLUMNS,
        "n_subjects": meta["n_subjects"],
        "n_healthy": meta["n_healthy"],
        "n_reduced": meta["n_reduced"],
        "mapping_version": meta["mapping"].version,
        "mapping_reviewer": meta["mapping"].reviewer,
        "auroc": auroc,
        "auroc_ci_95": ci,
        "balanced_accuracy": bal_acc,
        "subject_ids": subject_ids.tolist(),
        "use_gpu": bool(use_gpu),
    }

    out_dir = root / "results" / "urfu_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    (out_dir / "sanity_report.md").write_text(_render_markdown(report))
    pred.to_parquet(out_dir / "sanity_predictions.parquet", index=False)
    from ..data.urfu_labels import write_urfu_mapping

    write_urfu_mapping(meta["mapping"], out_dir)
    return report


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# URFU supervised sanity (gate 7, probe 3)",
        "",
        f"Protocol: **{report['protocol']}**, {report['n_folds']}-fold "
        f"participant-level CV (healthy={report['n_healthy']}, "
        f"reduced={report['n_reduced']}).",
        "",
        f"- AUROC: **{_fmt(report['auroc'])}** "
        f"[{_fmt(report['auroc_ci_95']['low'])}, {_fmt(report['auroc_ci_95']['high'])}]",
        f"- balanced accuracy: {_fmt(report['balanced_accuracy'])}",
        f"- features: {', '.join(report['feature_columns'])}",
        f"- mapping: {report['mapping_version']} ({report['mapping_reviewer']})",
        "",
        "Interpretation: AUROC clearly above 0.5 on held-out participants "
        "shows the diagnosis labels carry signal in the waveform-derived "
        "features; near-chance AUROC would mean the labels or features are not "
        "discriminative before any transfer claim is made.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="URFU supervised sanity (gate 7 probe 3)")
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    rep = run_urfu_sanity(artifact_root=args.artifact_root, use_gpu=args.gpu)
    print(
        f"AUROC {_fmt(rep['auroc'])} "
        f"[{_fmt(rep['auroc_ci_95']['low'])}, {_fmt(rep['auroc_ci_95']['high'])}] "
        f"bal_acc {_fmt(rep['balanced_accuracy'])} "
        f"(n={rep['n_subjects']}, healthy={rep['n_healthy']}, reduced={rep['n_reduced']})"
    )

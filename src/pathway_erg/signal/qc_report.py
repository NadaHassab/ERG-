"""Fold-dependent QC run: technical flags, locked populations, shortcut checks.

For each outer fold, thresholds are fitted on outer-train people and applied
to that fold's test people.  Also produces a missingness summary by class, a
missingness-only baseline, a QC-only classifier, and flag-rate tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..constants import (
    LR_MAX_ITER,
    MIN_CLASSIFIER_EVENTS,
    MIN_CV_FOLDS,
    OUTER_FOLDS_TEMPLATE,
    QC_CV_FOLDS,
)
from .component_cache import _load_waveforms
from .qc_flags import (
    compute_all_qc,
    fit_qc_thresholds,
    flag_recording,
    lock_populations,
    write_qc_artifacts,
)


def _cv_auc(X, y) -> tuple[float, float]:
    if y.sum() < MIN_CLASSIFIER_EVENTS or y.sum() == y.size:
        return float("nan"), float("nan")
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=LR_MAX_ITER)
    aucs = cross_val_score(lr, Xs, y, cv=min(QC_CV_FOLDS, max(MIN_CV_FOLDS, int(np.bincount(y).min()))), scoring="roc_auc")
    return float(np.mean(aucs)), float(np.std(aucs))


def run_qc(artifact_root: str | Path, split_version: str) -> dict[str, object]:
    artifact_root = Path(artifact_root)
    interim = artifact_root / "data" / "interim"
    recordings = pd.read_parquet(interim / "recordings.parquet")
    visits = pd.read_parquet(interim / "visits.parquet")
    sessions = pd.read_parquet(interim / "sessions.parquet")
    folds = pd.read_parquet(
        artifact_root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=split_version)
    )
    waveforms = _load_waveforms(artifact_root)

    base_qc = compute_all_qc(recordings, waveforms)
    recordings = recordings.merge(
        visits[["global_visit_id", "target_binary", "diagnosis1_raw"]],
        on="global_visit_id",
        how="left",
    )
    # visits carry their own dataset column; keep recordings' pristine
    recordings = recordings[recordings.columns[~recordings.columns.str.endswith(("_x", "_y"))]]

    # --- per-fold honest flagging ----------------------------------------
    flag_rows: list[dict] = []
    all_thresholds: dict[int, dict] = {}
    for fold in sorted(folds["outer_fold"].unique()):
        train_units = set(folds[folds["outer_fold"] != fold]["unit_id"])
        test_units = set(folds[folds["outer_fold"] == fold]["unit_id"])
        qc_for_fit = base_qc.merge(
            recordings[["global_recording_id", "global_subject_id", "dataset", "protocol"]],
            on="global_recording_id",
        )
        thresholds = fit_qc_thresholds(qc_for_fit, train_units)
        all_thresholds[int(fold)] = thresholds
        qc = base_qc.merge(
            recordings[["global_recording_id", "dataset", "protocol"]], on="global_recording_id"
        )
        for rec in recordings[recordings["global_subject_id"].isin(test_units)].itertuples(index=False):
            row = qc[qc["global_recording_id"] == rec.global_recording_id].iloc[0]
            thr = thresholds.get((row["dataset"], row["protocol"]), {})
            quantities = {
                "noise_derivative_mad_relative": row["noise_derivative_mad_relative"],
                "boundary_jump_relative": row["boundary_jump_relative"],
                "saturation_fraction": row["saturation_fraction"],
                "total_variation_relative": row["total_variation_relative"],
            }
            flags = flag_recording(quantities, thr)
            flag_rows.append(
                {
                    "global_recording_id": rec.global_recording_id,
                    "global_visit_id": rec.global_visit_id,
                    "global_subject_id": rec.global_subject_id,
                    "outer_fold": fold,
                    "n_flags": len(flags),
                    "flags_json": json.dumps(flags),
                }
            )
    flag_df = pd.DataFrame(flag_rows).merge(
        recordings[["global_recording_id", "dataset"]], on="global_recording_id", how="left"
    )

    # --- locked populations ----------------------------------------------
    populations = lock_populations(recordings, visits, sessions, flag_df, folds)
    populations = populations.merge(
        visits[["global_visit_id", "target_binary"]], on="global_visit_id", how="left"
    )

    # --- missingness summary per visit ------------------------------------
    rec_meta = recordings.merge(
        flag_df.groupby("global_recording_id")["n_flags"].max().rename("n_flags"),
        on="global_recording_id",
        how="left",
    )
    miss_rows = []
    for v in visits.itertuples(index=False):
        vr = rec_meta[rec_meta["global_visit_id"] == v.global_visit_id]
        if v.dataset == "LEOP":
            has_op = bool((vr["waveform_kind"] == "OP").any())
            n_erg = int((vr["waveform_kind"] == "ERG").sum())
            n_eyes = int(vr["eye"].dropna().nunique())
            n_sessions = int(vr["global_session_id"].nunique())
            protocols = "|".join(sorted(vr["protocol"].unique()))
            miss_rows.append(
                {
                    "global_visit_id": v.global_visit_id,
                    "dataset": v.dataset,
                    "target_binary": v.target_binary,
                    "n_sessions": n_sessions,
                    "n_erg": n_erg,
                    "has_op": has_op,
                    "n_eyes": n_eyes,
                    "protocols": protocols,
                    "n_flags": int(vr["n_flags"].fillna(0).sum()),
                }
            )
        else:
            n_eyes = int(vr["eye"].dropna().nunique())
            n_sessions = int(vr["global_session_id"].nunique())
            miss_rows.append(
                {
                    "global_visit_id": v.global_visit_id,
                    "dataset": v.dataset,
                    "target_binary": v.target_binary,
                    "n_sessions": n_sessions,
                    "n_erg": int(len(vr)),
                    "has_op": False,
                    "n_eyes": n_eyes,
                    "protocols": "perg",
                    "n_flags": int(vr["n_flags"].fillna(0).sum()),
                }
            )
    missingness = pd.DataFrame(miss_rows)

    # --- baseline checks --------------------------------------------------
    leop = missingness[missingness["dataset"] == "LEOP"].dropna(subset=["target_binary"])
    Xl = leop[["n_sessions", "n_erg", "has_op", "n_eyes"]].astype(float).values
    leop_miss_auc, leop_miss_std = _cv_auc(Xl, leop["target_binary"].astype(int).values)

    perg = missingness[missingness["dataset"] == "PERG"].dropna(subset=["target_binary"])
    Xp = perg[["n_sessions", "n_erg", "n_eyes"]].astype(float).values
    perg_miss_auc, perg_miss_std = _cv_auc(Xp, perg["target_binary"].astype(int).values)

    qc_feat = flag_df.merge(
        recordings[["global_recording_id", "global_visit_id", "target_binary"]],
        on="global_recording_id",
        how="left",
    ).dropna(subset=["target_binary"])
    Xq = qc_feat[["n_flags"]].values
    qc_auc, qc_std = _cv_auc(Xq, qc_feat["target_binary"].astype(int).values)

    # --- flag rate by class/site/age/sex -----------------------------------
    participants = pd.read_parquet(interim / "participants.parquet")
    flag_df = flag_df.merge(
        participants[["global_subject_id", "site", "age_years", "sex_standardized"]],
        on="global_subject_id",
        how="left",
    )
    flag_df = flag_df.rename(columns={"sex_standardized": "sex"})
    flag_df["age_bin"] = pd.cut(
        flag_df["age_years"], bins=[0, 10, 18, 40, 120], labels=["0-9", "10-17", "18-39", "40+"]
    )
    rate_rows = []
    for group_cols in (("dataset", "site"), ("dataset", "age_bin"), ("dataset", "sex")):
        g = flag_df.groupby(list(group_cols), dropna=False, observed=True)["n_flags"].agg(["size", "sum"])
        for keys, r in g.iterrows():
            key_list = keys if isinstance(keys, tuple) else (keys,)
            rate_rows.append(
                {
                    "grouping": "+".join(group_cols),
                    "key": " | ".join(str(k) for k in key_list),
                    "n": int(r["size"]),
                    "flagged": int(r["sum"]),
                    "rate": float(r["sum"]) / r["size"],
                }
            )
    flag_rates = pd.DataFrame(rate_rows)

    write_qc_artifacts(artifact_root, flag_df, all_thresholds, populations, missingness)

    summary = {
        "n_flag_recs": int(flag_df["n_flags"].gt(0).sum()),
        "flag_rate": float(flag_df["n_flags"].gt(0).mean()),
        "flag_rate_by_class": (
            flag_df.merge(recordings[["global_recording_id", "target_binary"]], on="global_recording_id")
            .groupby("target_binary")["n_flags"]
            .apply(lambda s: float(s.gt(0).mean()))
            .to_dict()
        ),
        "population_sizes": {
            pop: int(populations[pop].sum()) for pop in ("all_valid", "high_qc", "complete")
        },
        "population_by_class": {
            pop: {
                str(k): int(v)
                for k, v in populations.groupby("target_binary")[pop].sum().astype(int).to_dict().items()
            }
            for pop in ("all_valid", "high_qc", "complete")
        },
        "missingness_only_auc": {
            "leop": round(leop_miss_auc, 4),
            "leop_std": round(leop_miss_std, 4),
            "perg": round(perg_miss_auc, 4),
            "perg_std": round(perg_miss_std, 4),
        },
        "qc_only_auc": round(qc_auc, 4),
        "qc_only_auc_std": round(qc_std, 4),
        "missingness_by_class": {
            str(k): v.to_dict()
            for k, v in missingness.groupby(["dataset", "target_binary"])[
                ["n_sessions", "has_op", "n_eyes"]
            ]
            .mean()
            .round(2)
            .iterrows()
        },
    }
    flag_rates.to_parquet(artifact_root / "data" / "qc" / "flag_rates.parquet", index=False)
    with (artifact_root / "data" / "qc" / "qc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary

"""Post-hoc confound gate for neural (Phase 6) out-of-fold predictions.

The pre-Phase-6 ``confound-review`` gate (§3.29) proved that no cheap
non-signal channel (fallback mask, protocol-count availability, missingness)
can drive the *classical* baselines up to biology level.  That gate cannot,
however, certify a trained neural model: a deep model could in principle have
memorised a shortcut channel from the very same inputs.  This module re-checks
the trained neural out-of-fold predictions unit-by-unit against the same
channels, on the *identical* subject set, and applies the plan Section 17
(E0) decision rule:

1. **fallback-rate channel** — per-subject fraction of components that used a
   landmark fallback must not predict the label above the biology band.
2. **QC flag-rate channel** — per-subject fraction of flagged components
   (confound columns permanently dropped from E4; same rule here).
3. **OP-missingness channel** — per-subject fraction of visits without
   oscillatory-potential components (LEOP only; PERG has no OPs).
4. **Protocol-count availability** — reported as INFO on ``primary_nine_step``
   (the cohort forbids availability features; ``baselines.py`` raises); the
   locked reference AUC is 0.632 on LEOP.
5. **Sex channel (plan E11)** — INFO: sex-only AUROC plus sex-stratified
   neural AUROC, because the LEOP recruitment imbalance (controls 62 % male
   vs 24 % among ASD) must not carry the result.
6. **Signal-over-shortcut margin** — the neural AUROC must exceed the
    strongest measured shortcut channel by at least ``GATE_MARGIN``.

Gated checks (1, 2, 3, 6) produce a PASS/FAIL verdict; advisory checks
(4, 5) are reported as INFO.  Channel AUC band: a channel that predicts the
label in *either* direction at shortcut level (AUROC > 0.65 or < 0.35) fails
the gate.  Every AUC is computed on the exact per-unit rows of the neural
predictions (strict unit alignment: a channel row without a prediction row,
or vice versa, raises), with subject-clustered bootstrap CIs (plan §18.3).
A FAIL means the neural result cannot yet be read as biology: the confound
channel explains what the model does.

Outputs (versioned; never touching run artifacts):
``artifacts/results/confounds/neural_confound_gate.json`` and ``.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..qa.report import _load_merged
from ..training.separate import SeparateTrainingConfig
from .metrics import cluster_bootstrap_ci

# Decision thresholds (documented in the report; the shortcut cap mirrors
# LABEL_SHORTCUT_AUC_MAX from evaluation.confound_review — the biology band
# of the locked primary cohort spans 0.657-0.689, so 0.65 keeps a margin).
SHORTCUT_AUC_MAX = 0.65
GATE_MARGIN = 0.05        # neural must beat the strongest channel by >= this
MIN_AUC_N = 20           # channel AUC requires >= this many labeled units
MIN_STRATUM_N = 15       # sex-stratified AUC requires this many per stratum

CHANNELS = ("fallback_rate", "qc_flag_rate", "op_missingness")


@dataclass
class GateCheck:
    """One row in the gate report."""

    check: str
    outcome: str  # PASS | FAIL | INFO
    measurement: str
    note: str = ""

    def to_json(self) -> dict:
        return {
            "check": self.check,
            "outcome": self.outcome,
            "measurement": self.measurement,
            "note": self.note,
        }


def _auc_with_ci(
    y: np.ndarray, score: np.ndarray, cluster: np.ndarray, seed: int
) -> dict:
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    unique = np.unique(y)
    if len(unique) < 2:
        return {"auc": None, "ci_low": None, "ci_high": None, "n": int(len(y)),
                "note": "single-class unit set"}
    auc = float(roc_auc_score(y, score))
    ci = cluster_bootstrap_ci(
        y, score, np.asarray(cluster), metric="roc_auc",
        n_reps=2000, seed=seed, confidence=0.95,
    )
    return {
        "auc": auc,
        "ci_low": ci.ci_low,
        "ci_high": ci.ci_high,
        "n": int(len(y)),
        "note": "",
    }


# ---------------------------------------------------------------------------
# Channel features (subject-level, from the canonical caches)
# ---------------------------------------------------------------------------


def subject_channel_features(
    merged: pd.DataFrame,
    participants: pd.DataFrame,
) -> pd.DataFrame:
    """Per-subject confound channels for the merged component table.

    Returns one row per ``global_subject_id`` with columns
    ``fallback_rate``, ``qc_flag_rate``, ``op_missingness``,
    ``protocol_count`` and ``sex``.  ``op_missingness`` is NaN for datasets
    without oscillatory-potential components (PERG).
    """
    erg = merged.copy().assign(
        flagged=(~merged["component_qc_flags"].isna()).astype(int)
    )
    by_subj = (
        erg.groupby("global_subject_id")
        .agg(
            dataset=("dataset", "first"),
            fallback_rate=("fallback_used", "mean"),
            qc_flag_rate=("flagged", "mean"),
            protocol_count=("protocol", "nunique"),
        )
        .reset_index()
    )

    # OP missingness: fraction of the subject's visits without any OP
    # component (OP pairing is a LEOP property; PERG has no OPs, so the
    # channel is NaN there).
    if (erg["waveform_kind"] == "OP").any():
        has_op = (
            erg.groupby(["global_subject_id", "global_visit_id"])["waveform_kind"]
            .apply(lambda s: (s == "OP").any())
            .rename("has_op")
        )
        visits_per_subj = has_op.groupby(level=0).size()
        visits_with_op = has_op.groupby(level=0).sum()
        op_missingness = pd.DataFrame(
            {
                "global_subject_id": np.asarray(visits_per_subj.index),
                "op_missingness": 1.0 - (
                    visits_with_op.to_numpy() / visits_per_subj.to_numpy()
                ),
            }
        )
    else:
        op_missingness = pd.DataFrame(
            {
                "global_subject_id": np.asarray(
                    erg["global_subject_id"].unique()
                ),
                "op_missingness": np.nan,
            }
        )
    out = by_subj.merge(op_missingness, on="global_subject_id", how="left")
    dataset_has_op = erg.groupby("dataset")["waveform_kind"].apply(
        lambda s: (s == "OP").any()
    )
    out.loc[
        ~out["dataset"].map(dataset_has_op).fillna(False), "op_missingness"
    ] = np.nan

    sex_map = participants.set_index("global_subject_id")["sex_standardized"]
    sx = sex_map.reindex(out["global_subject_id"]).astype(str).str.lower()
    out["sex"] = np.where(sx.isin(["1", "male"]), "M", "F")
    return out


def _channel_aucs(
    frame: pd.DataFrame,
    channels: tuple[str, ...],
    y: np.ndarray,
    subject_id: np.ndarray,
    seed: int,
) -> list[GateCheck]:
    """One single-feature AUROC + CI per channel on the prediction units."""
    checks: list[GateCheck] = []
    for ch in channels:
        score = frame[ch].to_numpy(float)
        if np.isnan(score).any():
            checks.append(GateCheck(ch, "INFO", "n/a — channel undefined here", ""))
            continue
        if np.isnan(y).any() or len(y) < MIN_AUC_N:
            checks.append(GateCheck(ch, "INFO", "n/a — too few labeled units", ""))
            continue
        rep = _auc_with_ci(y, score, subject_id, seed)
        if rep["auc"] is None:
            checks.append(GateCheck(ch, "INFO", "n/a — single-class unit set", ""))
            continue
        outcome = (
            "FAIL"
            if rep["auc"] > SHORTCUT_AUC_MAX or rep["auc"] < (1.0 - SHORTCUT_AUC_MAX)
            else "PASS"
        )
        checks.append(
            GateCheck(
                ch,
                outcome,
                f"AUROC {rep['auc']:.3f} [{rep['ci_low']:.3f}, {rep['ci_high']:.3f}] "
                f"(band {1.0 - SHORTCUT_AUC_MAX:.2f}-{SHORTCUT_AUC_MAX})",
                f"n={rep['n']}; single-feature subject-level AUROC",
            )
        )
    return checks


def _sex_report(
    frame: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    subject_id: np.ndarray,
    seed: int,
) -> list[GateCheck]:
    """Plan E11: sex-only AUROC + sex-stratified neural AUROC (INFO)."""
    if "sex" not in frame.columns or frame["sex"].isna().all():
        return [GateCheck("sex", "INFO", "n/a — no standardized sex labels", "")]
    sex = frame["sex"].to_numpy(str)
    checks: list[GateCheck] = []
    if len(np.unique(sex)) < 2:
        checks.append(GateCheck("sex", "INFO", "n/a — single sex in unit set", ""))
        return checks
    prop = pd.get_dummies(sex).to_numpy(float)[:, 0]
    if len(np.unique(y)) < 2:
        checks.append(GateCheck("sex", "INFO", "n/a — single-class unit set", ""))
        return checks
    sex_only = float(roc_auc_score(y, prop))
    if sex_only > SHORTCUT_AUC_MAX or sex_only < (1.0 - SHORTCUT_AUC_MAX):
        checks.append(
            GateCheck(
                "sex-only channel",
                "FAIL",
                f"AUROC {sex_only:.3f} (band {1.0 - SHORTCUT_AUC_MAX:.2f}-{SHORTCUT_AUC_MAX})",
                "sex imbalance alone predicts the label at shortcut level (E11)",
            )
        )
    else:
        checks.append(
            GateCheck(
                "sex-only channel",
                "PASS",
                f"AUROC {sex_only:.3f} (band {1.0 - SHORTCUT_AUC_MAX:.2f}-{SHORTCUT_AUC_MAX})",
                "sex imbalance does not alone reach the shortcut band (E11)",
            )
        )
    for s in ("F", "M"):
        m = sex == s
        rep = _auc_with_ci(y[m], p[m], subject_id[m], seed)
        if rep["auc"] is None or m.sum() < MIN_STRATUM_N:
            checks.append(
                GateCheck(f"neural AUROC within {s}", "INFO",
                          "n/a" if m.sum() < MIN_STRATUM_N else "single-class stratum",
                          f"n={int(m.sum())}")
            )
        else:
            checks.append(
                GateCheck(
                    f"neural AUROC within {s}",
                    "INFO",
                    f"{rep['auc']:.3f} [{rep['ci_low']:.3f}, {rep['ci_high']:.3f}]",
                    f"n={rep['n']}; stratum-weighted reporting per E11",
                )
            )
    return checks


def _margin_check(
    neural_auc: float | None, channel_aucs: list[float]
) -> GateCheck:
    """E0 decision rule: neural must clear the strongest channel by >= margin."""
    if neural_auc is None or not channel_aucs:
        return GateCheck(
            "signal-over-shortcut margin",
            "INFO",
            "n/a — no measurable channel AUC",
            "",
        )
    strongest = max(channel_aucs)
    margin = neural_auc - strongest
    outcome = "PASS" if margin >= GATE_MARGIN else "FAIL"
    return GateCheck(
        "signal-over-shortcut margin",
        outcome,
        f"neural {neural_auc:.3f} - strongest channel {strongest:.3f} "
        f"= {margin:+.3f} (min {GATE_MARGIN})",
        f"margin >= {GATE_MARGIN} required for the E0 decision rule",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_ensemble_predictions(artifact_root: Path, cfg: SeparateTrainingConfig) -> pd.DataFrame:
    """Ensembled per-unit predictions written by run_separate_training."""
    path = Path(artifact_root) / "results" / cfg.output_subdir / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run run-separate-neural first "
            "(the gate consumes only completed ensemble predictions)"
        )
    frame = pd.read_parquet(path)
    required = {"task", "unit_id", "subject_id", "target", "calibrated_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    bad = frame["calibrated_probability"].isna() | (
        (frame["calibrated_probability"] < 0.0) | (frame["calibrated_probability"] > 1.0)
    )
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows outside [0, 1] probabilities")
    dup = frame.groupby(["task", "unit_id"]).size()
    if (dup > 1).any():
        raise ValueError("duplicate unit rows in ensembled predictions")
    return frame


def run_confound_gate(
    artifact_root: str | Path,
    cfg: SeparateTrainingConfig,
    out_subdir: str = "confounds",
) -> dict:
    """Gate the ensembled neural predictions against confound channels."""
    root = Path(artifact_root)
    preds = load_ensemble_predictions(root, cfg)

    merged = _load_merged(root)
    participants = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    channels = subject_channel_features(merged, participants)

    results: dict[str, object] = {
        "experiment": cfg.method,
        "cohort": cfg.leop_cohort,
        "checks": [],
    }
    verdicts: list[str] = []
    for task, frame in preds.groupby("task"):
        y = frame["target"].to_numpy(float)
        p = frame["calibrated_probability"].to_numpy(float)
        subj = frame["subject_id"].to_numpy(str)
        if (frame.groupby("subject_id")["target"].nunique() > 1).any():
            raise ValueError(f"{task} has inconsistent labels within a subject cluster")

        per_subject = channels.copy()
        per_subject["global_subject_id"] = per_subject["global_subject_id"].astype(str)
        per_subject = (
            per_subject.set_index("global_subject_id").reindex(subj).reset_index(drop=True)
        )
        if per_subject["fallback_rate"].isna().any() or len(per_subject) != len(frame):
            raise ValueError(
                f"{task} channel alignment failed — a prediction unit has no "
                "channel row (subject-feature mismatch)"
            )

        task_checks: list[GateCheck] = []
        neural = _auc_with_ci(y, p, subj, seed=cfg.bootstrap_seed)
        if neural["auc"] is None:
            verdicts.append("INFO")
            task_checks.append(GateCheck("neural AUROC", "INFO", neural["note"], ""))
        else:
            task_checks.append(
                GateCheck(
                    "neural AUROC",
                    "INFO",
                    f"{neural['auc']:.3f} [{neural['ci_low']:.3f}, {neural['ci_high']:.3f}]",
                    f"n={neural['n']}; subject-clustered bootstrap CI",
                )
            )
        task_checks += _channel_aucs(
            per_subject, CHANNELS, y, subj, seed=cfg.bootstrap_seed
        )
        if task == "LEOP":
            task_checks.append(
                GateCheck(
                    "protocol-count availability",
                    "INFO",
                    "forbidden channel on primary_nine_step (baselines.py raises); "
                    "locked reference LEOP AUROC 0.632",
                    "not gated here by design",
                )
            )
        else:
            task_checks.append(
                GateCheck(
                    "protocol-count availability",
                    "INFO",
                    "not a neural input; repeated PERG visits stay subject-clustered",
                    "LEOP-only availability reference does not apply",
                )
            )
        task_checks += _sex_report(per_subject, y, p, subj, seed=cfg.bootstrap_seed)

        # signal-over-shortcut margin against the strongest gated channel
        channel_aucs = [
            float(c.measurement.split("AUROC ")[1].split(" ")[0])
            for c in task_checks
            if c.outcome != "INFO"
            and c.check in CHANNELS
            and c.measurement.startswith("AUROC ")
        ]
        task_checks.append(_margin_check(neural["auc"], channel_aucs))

        for c in task_checks:
            c.check = f"{task}: {c.check}"
            if c.outcome == "FAIL":
                verdicts.append("FAIL")
            results["checks"].append(c.to_json())
        results[f"task_{task}"] = {
            "n_units": int(len(frame)),
            "neural_auc": neural["auc"],
            "neural_ci": [neural["ci_low"], neural["ci_high"]],
        }

    results["verdict"] = "FAIL" if "FAIL" in verdicts else "PASS"
    out_dir = root / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "neural_confound_gate.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (out_dir / "neural_confound_gate.md").write_text(_render_markdown(results))
    return results


def _render_markdown(results: dict) -> str:
    lines = [
        "# Neural confound gate (plan Section 17 / E0 decision rule)",
        "",
        f"- experiment: {results['experiment']} | cohort: {results['cohort']}",
        f"- verdict: **{results['verdict']}**",
        "",
        "| check | outcome | measurement | note |",
        "|---|---|---|---|",
    ]
    for c in results["checks"]:
        lines.append(
            f"| {c['check']} | {c['outcome']} | `{c['measurement']}` | {c['note']} |"
        )
    lines += [
        "",
        "Rules: shortcut channels (fallback rate, QC flag rate, OP missingness) "
        "must stay inside the 0.35-0.65 AUROC band (a channel is a shortcut "
        "whether it predicts the label or its exact inverse); the neural AUROC "
        "must clear the strongest measured channel by at least 0.05. Sex and "
        "protocol-count rows are advisory (E11 / cohort forbids availability).",
    ]
    return "\n".join(lines) + "\n"

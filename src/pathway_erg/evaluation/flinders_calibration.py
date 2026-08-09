"""Normative Flinders calibration (plan gate 7, probe 2).

Integrity + sanity measure: LEOP healthy controls (Control group, LA3
protocol) and Flinders healthy controls (LA3) measure the same physiology
(skin-electrode ISCEV photopic flash ERG), so their a/b-wave feature
distributions should overlap, not diverge.  PERG is excluded because it is a
pattern ERG with no shared protocol/feature space with Flinders.

For every shared feature (a_time, a_amp, b_time, b_amp in ms/µV):

- age-adjusted residual distributions (linear age model fitted on Flinders
  controls only, applied to both groups),
- two-sample KS test (scipy),
- overlap statistics: fraction of LEOP values within the Flinders
  mean ± 1.96·SD window and the age-adjusted equivalent.

LEOP is reduced to one median value per participant (never counting repeated
curves as independent people).  Flinders LA3 feature rows come from the
source ``Flinders Normal`` sheet (they are not in the recordings table).

Outputs (versioned, never touching baseline artifacts):
``artifacts/results/flinders_calibration/calibration_report.{json,md}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pathway_erg.data import flinders

FLINDERS_XLSX = Path("14747349 (2)/ISCEV Control ERG Flinders University.xlsx")
INTERIM = Path("artifacts/data/interim")

# feature name on the Flinders Normal sheet -> LEOP supplied-feature key
FEATURE_PAIRS = {
    "a_time": "a_time_ms",
    "a_amp": "a_amp_uv",
    "b_time": "b_time_ms",
    "b_amp": "b_amp_uv",
}


def _flinders_la3_frame() -> pd.DataFrame:
    rows = list(flinders.iter_feature_rows(FLINDERS_XLSX))
    df = pd.DataFrame(rows)
    df = df[df["Test"] == "LA3"].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    for col in ("a_time", "a_amp", "b_time", "b_amp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["age", "a_time", "a_amp", "b_time", "b_amp"])


def _leop_la3_control_frame() -> pd.DataFrame:
    import json as _json

    rec = pd.read_parquet(INTERIM / "recordings.parquet")
    parts = pd.read_parquet(INTERIM / "participants.parquet")
    m = (rec["dataset"] == "LEOP") & (rec["protocol"] == "LA3") & (
        rec["waveform_kind"] == "ERG"
    )
    la3 = rec[m].merge(
        parts[["global_subject_id", "group_raw", "age_years"]],
        on="global_subject_id",
        how="left",
    )
    la3 = la3[la3["group_raw"] == "Control"].copy()

    feat_rows = []
    for _, r in la3.iterrows():
        try:
            feat = _json.loads(r["supplied_features_json"])
        except Exception:
            continue
        if not isinstance(feat, dict):
            continue
        feat_rows.append(
            {
                "global_subject_id": r["global_subject_id"],
                "age": r["age_years"],
                **{k: feat.get(v) for k, v in FEATURE_PAIRS.items()},
            }
        )
    df = pd.DataFrame(feat_rows)
    for col in ("a_time", "a_amp", "b_time", "b_amp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # one median value per participant
    med = df.groupby("global_subject_id", as_index=False).agg(
        age=("age", "first"),
        a_time=("a_time", "median"),
        a_amp=("a_amp", "median"),
        b_time=("b_time", "median"),
        b_amp=("b_amp", "median"),
    )
    return med.dropna(subset=["age", "a_time", "a_amp", "b_time", "b_amp"])


def _age_adjusted(feature: str, fl: pd.DataFrame, le: pd.DataFrame) -> dict:
    """Residuals of ``feature`` against age, model fitted on Flinders only."""
    x, y = fl["age"].to_numpy(float), fl[feature].to_numpy(float)
    slope, intercept, *_ = np.polyfit(x, y, 1)
    fl_res = y - (slope * x + intercept)
    le_res = le[feature].to_numpy(float) - (slope * le["age"].to_numpy(float) + intercept)
    return {
        "slope_per_year": float(slope),
        "intercept": float(intercept),
        "flinders_residual_std": float(np.std(fl_res, ddof=1)),
        "leop_residuals": le_res.tolist(),
        "flinders_residuals": fl_res.tolist(),
    }


def _feature_report(feature: str, fl: pd.DataFrame, le: pd.DataFrame) -> dict:
    fl_v, le_v = fl[feature].to_numpy(float), le[feature].to_numpy(float)
    ks = stats.ks_2samp(fl_v, le_v)
    mu, sd = float(np.mean(fl_v)), float(np.std(fl_v, ddof=1))
    within_raw = float(np.mean(np.abs(le_v - mu) <= 1.96 * sd)) if sd > 0 else np.nan
    adj = _age_adjusted(feature, fl, le)
    fl_r = np.asarray(adj["flinders_residuals"])
    le_r = np.asarray(adj["leop_residuals"])
    ks_adj = stats.ks_2samp(fl_r, le_r)
    adj_sd = float(np.std(fl_r, ddof=1))
    within_adj = (
        float(np.mean(np.abs(le_r - np.mean(fl_r)) <= 1.96 * adj_sd))
        if adj_sd > 0
        else np.nan
    )
    return {
        "n_flinders": int(len(fl_v)),
        "n_leop_subjects": int(len(le_v)),
        "flinders_mean": mu,
        "flinders_sd": sd,
        "leop_mean": float(np.mean(le_v)),
        "leop_sd": float(np.std(le_v, ddof=1)),
        "ks_raw": {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)},
        "within_2sd_raw": within_raw,
        "ks_age_adjusted": {
            "statistic": float(ks_adj.statistic),
            "pvalue": float(ks_adj.pvalue),
        },
        "within_2sd_age_adjusted": within_adj,
        "age_slope_per_year": adj["slope_per_year"],
    }


def run_flinders_calibration(artifact_root: str | Path = "artifacts") -> dict:
    root = Path(artifact_root)
    fl = _flinders_la3_frame()
    le = _leop_la3_control_frame()

    report = {
        "protocol": "LA3",
        "comparison": "LEOP Control (LA3 ERG) vs Flinders Normal (LA3)",
        "n_flinders_rows": int(len(fl)),
        "n_leop_subjects": int(len(le)),
        "flinders_age": {
            "n": int(len(fl)),
            "mean": float(fl["age"].mean()),
            "min": float(fl["age"].min()),
            "max": float(fl["age"].max()),
        },
        "leop_age": {
            "n": int(len(le)),
            "mean": float(le["age"].mean()),
            "min": float(le["age"].min()),
            "max": float(le["age"].max()),
        },
        "features": {f: _feature_report(f, fl, le) for f in FEATURE_PAIRS},
    }

    out_dir = root / "results" / "flinders_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    (out_dir / "calibration_report.md").write_text(_render_markdown(report))
    return report


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# Flinders normative calibration (gate 7, probe 2)",
        "",
        f"Comparison: **{report['comparison']}** (shared ISCEV photopic LA3).",
        "",
        f"- Flinders healthy controls: {report['n_flinders_rows']} LA3 rows, "
        f"age {report['flinders_age']['mean']:.1f} y "
        f"[{report['flinders_age']['min']:.1f}-{report['flinders_age']['max']:.1f}]",
        f"- LEOP Control (per-subject median): {report['n_leop_subjects']} subjects, "
        f"age {report['leop_age']['mean']:.1f} y "
        f"[{report['leop_age']['min']:.1f}-{report['leop_age']['max']:.1f}]",
        "",
        "| feature | Flinders mean±SD | LEOP mean±SD | KS raw | within 2SD raw | KS age-adj | within 2SD adj | age slope |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for feat, r in report["features"].items():
        lines.append(
            f"| {feat} | {r['flinders_mean']:.2f}±{r['flinders_sd']:.2f} | "
            f"{r['leop_mean']:.2f}±{r['leop_sd']:.2f} | "
            f"{r['ks_raw']['statistic']:.3f} ({_fmt_p(r['ks_raw']['pvalue'])}) | "
            f"{r['within_2sd_raw']:.3f} | "
            f"{r['ks_age_adjusted']['statistic']:.3f} ({_fmt_p(r['ks_age_adjusted']['pvalue'])}) | "
            f"{r['within_2sd_age_adjusted']:.3f} | "
            f"{r['age_slope_per_year']:.3f} |"
        )
    lines += [
        "",
        "Interpretation: distributions are age-adjusted against the Flinders "
        "control model (fitted on Flinders only). A sanity-healthy overlap shows "
        "small KS statistics and a high within-2SD fraction; divergence would "
        "flag a protocol/electrode/site artifact rather than biology. PERG is "
        "excluded (pattern ERG, no shared protocol).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Flinders normative calibration (gate 7 probe 2)")
    ap.add_argument("--artifact-root", default="artifacts")
    args = ap.parse_args()
    rep = run_flinders_calibration(artifact_root=args.artifact_root)
    for feat, r in rep["features"].items():
        print(
            f"{feat:<8} KS={r['ks_raw']['statistic']:.3f} (adj {r['ks_age_adjusted']['statistic']:.3f}) "
            f"within2SD={r['within_2sd_raw']:.3f} (adj {r['within_2sd_age_adjusted']:.3f})"
        )

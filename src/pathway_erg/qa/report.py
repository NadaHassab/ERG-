"""Pipeline QA report (plan Step 12.11 checks and Phase 3 gate).

Produces a self-contained HTML audit plus a machine-readable summary:

- stratified random curve sample across protocol x site x class x eye x intensity
  with detected landmarks and segment windows overlaid;
- automatic-versus-supplied a/b scatter for LEOP;
- fallback frequency by group / site / diagnosis;
- fallback-mask-only classifier (gate: strong predictive power blocks use);
- no-extrapolation check on stored segment bounds and canonical grids.
"""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..config import PreprocessingConfig
from ..constants import (
    LR_C,
    LR_MAX_ITER,
    MIN_AUC_N,
    MIN_CLASSIFIER_EVENTS,
    PER_STRATUM_SAMPLE,
    QA_SEED,
    QC_CV_FOLDS,
    SUPPORT_EPSILON_MS,
)
from ..data.schemas import WaveformKind
from ..provenance import RunManifest
from ..signal.component_cache import (
    CACHE_SCHEMA_VERSION,
    _load_waveforms,
    cache_paths,
    load_cache_manifest,
)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _load_merged(artifact_root: Path) -> pd.DataFrame:
    cache = cache_paths(artifact_root, CACHE_SCHEMA_VERSION)
    load_cache_manifest(artifact_root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    participants = pd.read_parquet(artifact_root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(artifact_root / "data" / "interim" / "visits.parquet")

    merged = components.merge(
        recordings[
            [
                "global_recording_id",
                "global_subject_id",
                "global_visit_id",
                "protocol",
                "eye",
                "stimulus_value",
                "stimulus_unit",
                "waveform_kind",
                "supplied_features_json",
            ]
        ],
        on="global_recording_id",
        how="left",
    )
    merged = merged.merge(
        participants[["global_subject_id", "site", "group_raw"]],
        on="global_subject_id",
        how="left",
    )
    merged = merged.merge(
        visits[["global_visit_id", "target_binary"]],
        on="global_visit_id",
        how="left",
    )
    merged["dataset"] = merged["component_id"].str[0].map({"L": "LEOP", "P": "PERG"})
    merged["intensity"] = merged.apply(
        lambda row: (
            f"{row['stimulus_value']:g}{row['stimulus_unit']}"
            if pd.notna(row["stimulus_value"])
            else "-"
        ),
        axis=1,
    )
    merged["class_label"] = merged.apply(
        lambda row: (
            str(row["target_binary"])
            if pd.notna(row["target_binary"])
            else ("Control" if row["group_raw"] == "Control" else "n/a")
        ),
        axis=1,
    )
    return merged


def _extract_physical_features(components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw in components["physical_features_json"]:
        d = json.loads(raw)
        rows.append(
            {
                "log_mass_pos": d["log_mass_pos"],
                "log_mass_neg": d["log_mass_neg"],
                "peak_to_peak_uv": d["peak_to_peak_uv"],
                "max_rising_slope_uv_per_ms": d["max_rising_slope_uv_per_ms"],
                "max_falling_slope_uv_per_ms": d["max_falling_slope_uv_per_ms"],
                "area_above_ref_uv_ms": d["area_above_ref_uv_ms"],
                "area_below_ref_uv_ms": d["area_below_ref_uv_ms"],
                "duration_ms": d["duration_ms"],
            }
        )
    return pd.DataFrame(rows)


def _make_stratified_sample(merged: pd.DataFrame, rng: random.Random, per_stratum: int) -> pd.DataFrame:
    key_cols = ["dataset", "protocol", "site", "class_label", "eye", "intensity"]
    strata = merged.groupby(key_cols, dropna=False).groups
    chosen: list[str] = []
    for _key, idx in strata.items():
        ids = merged.loc[idx, "global_recording_id"].tolist()
        chosen.extend(rng.sample(ids, min(per_stratum, len(ids))))
    return merged[merged["global_recording_id"].isin(set(chosen))].drop_duplicates(
        "global_recording_id"
    )


def _curve_plot(
    row: pd.Series,
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    landmarks: dict,
    segments: list[tuple[float, float, str]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(time_ms, signal_uv, lw=0.9, color="#1f77b4", label="baseline-corrected")
    for t, s, name in segments:
        ax.axvspan(t, s, color="orange", alpha=0.12)
        ax.text(t + (s - t) / 2, ax.get_ylim()[1], name, fontsize=6, ha="center", va="top")
    for name, lm in landmarks.items():
        if lm["t"] is None:
            continue
        ax.axvline(lm["t"], color="red", lw=0.8, ls="--", alpha=0.7)
        ax.text(lm["t"], ax.get_ylim()[0], name, fontsize=6, ha="center", va="bottom")
    ax.set_title(
        f"{row['global_recording_id']} | {row['dataset']} {row['protocol']} "
        f"| site={row['site']} | eye={row['eye']} | intensity={row['intensity']} "
        f"| class={row['class_label']}",
        fontsize=7,
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude (uV)")
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper right")
    return fig


def _supplied_vs_detected(merged: pd.DataFrame) -> plt.Figure | None:
    erg = merged[
        (merged["dataset"] == "LEOP")
        & (merged["waveform_kind"] == WaveformKind.ERG.value)
        & (merged["component_id"] == "L_A_TO_B")
    ].copy()
    if erg.empty:
        return None
    detected_times = erg["landmark_times_json"].map(json.loads)
    supplied = erg["supplied_features_json"].map(lambda x: json.loads(x) if x else None)

    pairs_a = []
    pairs_b = []
    for dt, sp in zip(detected_times, supplied, strict=False):
        if not sp:
            continue
        if "a_time_ms" in sp and "a_trough" in dt and dt["a_trough"] is not None and sp["a_time_ms"] is not None:
            pairs_a.append((sp["a_time_ms"], dt["a_trough"]))
        if "b_time_ms" in sp and "b_peak" in dt and dt["b_peak"] is not None and sp["b_time_ms"] is not None:
            pairs_b.append((sp["b_time_ms"], dt["b_peak"]))
    if not pairs_a and not pairs_b:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, pairs, name in ((axes[0], pairs_a, "a-wave"), (axes[1], pairs_b, "b-wave")):
        if pairs:
            s, d = zip(*pairs, strict=True)
            ax.scatter(s, d, s=4, alpha=0.5)
            lims = (min(min(s), min(d)) - 2, max(max(s), max(d)) + 2)
            ax.plot(lims, lims, "r--", lw=0.8)
            mad = float(np.median(np.abs(np.array(d) - np.array(s))))
            ax.set_title(f"{name}: n={len(s)} MAD={mad:.2f} ms")
        else:
            ax.set_title(f"{name}: no pairs")
        ax.set_xlabel("supplied (ms)")
        ax.set_ylabel("detected (ms)")
        ax.tick_params(labelsize=7)
    fig.suptitle("automatic vs supplied LEOP landmarks", fontsize=10)
    return fig


def _fallback_shortcut_check(merged: pd.DataFrame) -> dict:
    """Does the fallback mask alone predict the class label?"""
    erg = merged[merged["component_id"].str.startswith("L")].copy()
    erg = erg[erg["group_raw"].isin(["Control", "ASD"])]
    per_subject = (
        erg.groupby("global_subject_id")
        .agg(n=("global_component_id", "size"), fb=("fallback_used", "sum"), y=("target_binary", "first"))
        .dropna(subset=["y"])
    )
    leop_auc = None
    if len(per_subject) >= MIN_AUC_N and per_subject["y"].sum() > 0 and per_subject["y"].sum() < len(per_subject):
        rate = (per_subject["fb"] / per_subject["n"]).values
        leop_auc = float(roc_auc_score(per_subject["y"].values, rate))

    perg = merged[merged["component_id"].str.startswith("P")].dropna(subset=["target_binary"])
    per_visit = (
        perg.groupby("global_visit_id")
        .agg(n=("global_component_id", "size"), fb=("fallback_used", "sum"), y=("target_binary", "first"))
        .dropna(subset=["y"])
    )
    perg_auc = None
    if len(per_visit) >= MIN_AUC_N and per_visit["y"].sum() > 0 and per_visit["y"].sum() < len(per_visit):
        rate = (per_visit["fb"] / per_visit["n"]).values
        perg_auc = float(roc_auc_score(per_visit["y"].values, rate))
    return {"leop_fallback_only_auc": leop_auc, "perg_fallback_only_auc": perg_auc}


def _fallback_table(merged: pd.DataFrame) -> str:
    rows = []
    for group_cols in (
        ("dataset", "site"),
        ("dataset", "class_label"),
        ("dataset", "protocol"),
        ("dataset", "component_id"),
    ):
        g = (
            merged.groupby(list(group_cols))
            .agg(n=("global_component_id", "size"), fallback=("fallback_used", "sum"))
            .reset_index()
        )
        g["rate"] = (g["fallback"] / g["n"]).round(4)
        for _, r in g.iterrows():
            rows.append(
                {
                    "grouping": "+".join(group_cols),
                    "key": " | ".join(str(r[c]) for c in group_cols),
                    "n": int(r["n"]),
                    "fallback": int(r["fallback"]),
                    "rate": float(r["rate"]),
                }
            )
    return pd.DataFrame(rows).to_html(index=False, classes="qatable")


def _fallback_classifier(merged: pd.DataFrame) -> dict:
    feat = _extract_physical_features(merged)
    y = merged["fallback_used"].astype(int).values
    if y.sum() < MIN_CLASSIFIER_EVENTS or y.sum() == y.size:
        return {"note": "too few fallback events for a stable classifier", "auc": None}
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(feat.values)
    lr = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C)
    aucs = cross_val_score(lr, X, y, cv=QC_CV_FOLDS, scoring="roc_auc", n_jobs=1)
    lr.fit(X, y)
    coefficients = dict(zip(feat.columns, np.round(lr.coef_[0], 3), strict=True))
    return {
        "cv_auc": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "n_fallback": int(y.sum()),
        "standardized_coefficients": coefficients,
    }


def _extrapolation_check(merged: pd.DataFrame, recordings: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    rng = recordings[["global_recording_id", "start_ms", "end_ms"]].drop_duplicates(
        "global_recording_id"
    )
    check = merged[["global_component_id", "global_recording_id", "segment_start_ms", "segment_end_ms"]]
    for row in check.itertuples(index=False):
        rec = rng[rng["global_recording_id"] == row.global_recording_id]
        if rec.empty:
            problems.append(f"{row.global_component_id}: no recording")
            continue
        if row.segment_start_ms + SUPPORT_EPSILON_MS < rec.iloc[0]["start_ms"] or row.segment_end_ms - SUPPORT_EPSILON_MS > rec.iloc[0]["end_ms"]:
            problems.append(f"{row.global_component_id}: segment outside recording support")
    return problems


def run_qa(artifact_root: str | Path, pre_cfg: PreprocessingConfig) -> dict[str, object]:
    """Generate the HTML + summary QA report.  Returns summary dict."""
    artifact_root = Path(artifact_root)
    qa_dir = artifact_root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(QA_SEED)

    merged = _load_merged(artifact_root)
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    waveforms = _load_waveforms(artifact_root)

    # --- stratified visual sample --------------------------------------
    sample = _make_stratified_sample(merged, rng, PER_STRATUM_SAMPLE)
    sections: list[str] = []
    n_images = 0
    for row in sample.itertuples(index=False):
        key = row.global_recording_id
        if key not in waveforms:
            continue
        tm, sig = waveforms[key]
        lm_raw = json.loads(row.landmark_times_json)
        landmarks = {k: {"t": v} for k, v in lm_raw.items()}
        segs = [(row.segment_start_ms, row.segment_end_ms, row.component_id)]
        fig = _curve_plot(pd.Series(row._asdict()), tm, sig, landmarks, segs)
        img = _fig_to_base64(fig)
        n_images += 1
        meta = pd.Series(row._asdict())
        sections.append(
            f'<div class="card"><img src="data:image/png;base64,{img}"/>'
            f'<div class="meta">{meta["global_component_id"]} | {meta["canonicalization_type"]}'
            f' | fallback={meta["fallback_used"]} | conf={meta["landmark_confidence"]:.2f}</div></div>'
        )
    sections_html = "".join(sections)

    # --- supplied vs detected ------------------------------------------
    fig = _supplied_vs_detected(merged)
    svd_html = (
        f'<img src="data:image/png;base64,{_fig_to_base64(fig)}"/>' if fig else "<p>no LEOP a/b pairs</p>"
    )

    fallback_table = _fallback_table(merged)
    fc = _fallback_classifier(merged)
    shortcut = _fallback_shortcut_check(merged)
    extrapolation = _extrapolation_check(merged, recordings)

    qa_summary = {
        "n_components": int(len(merged)),
        "n_strata_cells": int(merged.groupby(["dataset", "protocol", "site", "class_label", "eye", "intensity"], dropna=False).ngroups),
        "n_visual_samples": n_images,
        "fallback": {"total": int(merged["fallback_used"].sum()), "rate": float(merged["fallback_used"].mean())},
        "supplied_vs_detected": None,
        "fallback_classifier": fc,
        "fallback_shortcut": shortcut,
        "extrapolation_problems": extrapolation,
    }
    if fig is not None:
        qa_summary["supplied_vs_detected"] = "computed"
    qa_summary["extrapolation_pass"] = len(extrapolation) == 0

    gate_html = (
        '<span class="pass">PASS</span>' if qa_summary["extrapolation_pass"] else '<span class="fail">FAIL</span>'
    )
    fc_html = (
        f"AUC {fc['cv_auc']:.3f} +/- {fc['cv_auc_std']:.3f} (n_fallback={fc['n_fallback']})"
        if fc.get("cv_auc") is not None
        else fc.get("note", "n/a")
    )
    shortcut_html = (
        f"LEOP {shortcut['leop_fallback_only_auc']:.3f} | "
        f"PERG {shortcut['perg_fallback_only_auc']:.3f} (chance = 0.5)"
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>PATH-ERG pipeline QA</title>
<style>
 body {{ font-family: sans-serif; margin: 24px; color: #222; }}
 h1 {{ font-size: 18px; }} h2 {{ font-size: 14px; margin-top: 28px; }}
 .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; margin: 8px 0; display: inline-block; }}
 .meta {{ font-size: 11px; color: #555; margin-top: 4px; }}
 img {{ max-width: 720px; }}
 table {{ border-collapse: collapse; font-size: 11px; }}
 td, th {{ border: 1px solid #ccc; padding: 3px 7px; }}
 .pass {{ color: #0a7d0a; font-weight: bold; }} .fail {{ color: #b00; font-weight: bold; }}
</style></head><body>
<h1>PATH-ERG pipeline QA report</h1>
<p>generated by pathway_erg qa v1 &mdash; preprocessing version {pre_cfg.version}</p>
<h2>Summary</h2>
<table>
<tr><th>components</th><td>{qa_summary['n_components']}</td></tr>
<tr><th>strata cells</th><td>{qa_summary['n_strata_cells']}</td></tr>
<tr><th>visual samples</th><td>{qa_summary['n_visual_samples']}</td></tr>
<tr><th>fallback rate</th><td>{qa_summary['fallback']['rate']:.4f} ({qa_summary['fallback']['total']})</td></tr>
<tr><th>fallback-mask classifier</th><td>{fc_html}</td></tr>
<tr><th>fallback-only label shortcut</th><td>{shortcut_html}</td></tr>
<tr><th>no extrapolation</th><td>{gate_html}</td></tr>
</table>
<h2>Automatic vs supplied a/b</h2>{svd_html}
<h2>Fallback frequency by group / site / diagnosis / protocol</h2>{fallback_table}
<h2>Stratified visual sample</h2>
<p>one card per sampled recording, landmarks dashed red, shaded segment windows</p>
{sections_html}
</body></html>"""
    (qa_dir / "qa_report.html").write_text(html, encoding="utf-8")

    manifest = RunManifest(kind="qa", name="pipeline_qa")
    manifest.extra["summary"] = qa_summary
    manifest.write_atomic(qa_dir / "qa_manifest.json")
    with (qa_dir / "qa_summary.json").open("w", encoding="utf-8") as f:
        json.dump(qa_summary, f, indent=2, default=str)
    return qa_summary

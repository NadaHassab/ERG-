"""External dataset coverage/diversity report (plan gate 7, probe 1).

Quantifies what the FLINDERS + URFU integration adds to the canonical build:
subjects, visits, sessions, recordings, protocols, ages, sites, sampling
rates, eyes, waveform kinds, and label availability — for the original
LEOP+PERG scope vs the extended LEOP+PERG+FLINDERS+URFU scope.

Outputs (versioned, never touching baseline artifacts):
``artifacts/results/external_coverage/coverage_report.{json,md}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ORIGINAL = ("LEOP", "PERG")
EXTERNAL = ("FLINDERS", "URFU")
ALL_DATASETS = ORIGINAL + EXTERNAL


def _stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _counts(x: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in x.value_counts(dropna=False).items()}


def _summarize(scope: tuple[str, ...], tables: dict[str, pd.DataFrame]) -> dict:
    participants = tables["participants"]
    visits = tables["visits"]
    sessions = tables["sessions"]
    recordings = tables["recordings"]

    m = participants["dataset"].isin(scope)
    p = participants[m]
    v = visits[visits["dataset"].isin(scope)]
    s = sessions[sessions["dataset"].isin(scope)]
    r = recordings[recordings["dataset"].isin(scope)]

    label_eligible = v["target_binary"].notna()
    label_pos = v["target_binary"].eq(1)
    return {
        "scope": list(scope),
        "subjects": int(p.shape[0]),
        "visits": int(v.shape[0]),
        "sessions": int(s.shape[0]),
        "recordings": int(r.shape[0]),
        "subjects_by_dataset": _counts(p["dataset"]),
        "recordings_by_dataset": _counts(r["dataset"]),
        "protocols_by_dataset": {
            str(ds): _counts(g["protocol"]) for ds, g in r.groupby("dataset")
        },
        "waveform_kinds_by_dataset": {
            str(ds): _counts(g["waveform_kind"]) for ds, g in r.groupby("dataset")
        },
        "eyes_by_dataset": {
            str(ds): _counts(g["eye"]) for ds, g in r.groupby("dataset")
        },
        "sites_by_dataset": {
            str(ds): _counts(g["site"]) for ds, g in p.groupby("dataset")
        },
        "age": _stats(p["age_years"]),
        "age_by_dataset": {
            str(ds): _stats(g["age_years"]) for ds, g in p.groupby("dataset")
        },
        "sampling_rates_by_dataset": {
            str(ds): _counts(g["sampling_rate_hz"]) for ds, g in r.groupby("dataset")
        },
        "label_eligible_visits": int(label_eligible.sum()),
        "label_positive_visits": int(label_pos.sum()),
        "label_negative_visits": int((label_eligible & ~label_pos).sum()),
        "label_ineligible_visits": int((~label_eligible).sum()),
    }


def run_coverage_report(artifact_root: str | Path = "artifacts") -> dict:
    root = Path(artifact_root)
    interim = root / "data" / "interim"
    tables = {
        name: pd.read_parquet(interim / f"{name}.parquet")
        for name in ("participants", "visits", "sessions", "recordings")
    }
    for name, df in tables.items():
        assert "dataset" in df.columns, f"{name}.parquet missing 'dataset' column"

    report = {
        "original": _summarize(ORIGINAL, tables),
        "extended": _summarize(ALL_DATASETS, tables),
    }
    before, after = report["original"], report["extended"]
    report["delta"] = {
        "subjects": after["subjects"] - before["subjects"],
        "visits": after["visits"] - before["visits"],
        "sessions": after["sessions"] - before["sessions"],
        "recordings": after["recordings"] - before["recordings"],
        "subjects_by_dataset": after["subjects_by_dataset"],
    }
    report["datasets"] = {k: int(v) for k, v in tables["participants"]["dataset"].value_counts().items()}

    out_dir = root / "results" / "external_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (out_dir / "coverage_report.md").write_text(_render_markdown(report))
    return report


def _fmt_age(stats: dict) -> str:
    if stats.get("n", 0) == 0:
        return "n/a"
    return f"n={stats['n']}, mean {stats['mean']:.1f} y [{stats['min']:.1f}-{stats['max']:.1f}]"


def _render_markdown(report: dict) -> str:
    before, after, delta = report["original"], report["extended"], report["delta"]
    lines = [
        "# External dataset coverage report (gate 7, probe 1)",
        "",
        "## 1. Scope totals",
        "",
        "| metric | LEOP+PERG | + FLINDERS+URFU | Δ |",
        "|---|---|---|---|",
        f"| subjects | {before['subjects']} | {after['subjects']} | +{delta['subjects']} |",
        f"| visits | {before['visits']} | {after['visits']} | +{delta['visits']} |",
        f"| sessions | {before['sessions']} | {after['sessions']} | +{delta['sessions']} |",
        f"| recordings | {before['recordings']} | {after['recordings']} | +{delta['recordings']} |",
        "",
        "## 2. Subjects by dataset",
        "",
        "| dataset | subjects | recordings |",
        "|---|---|---|",
    ]
    for ds in ("LEOP", "PERG", "FLINDERS", "URFU"):
        lines.append(
            f"| {ds} | {after['subjects_by_dataset'].get(ds, 0)} | "
            f"{after['recordings_by_dataset'].get(ds, 0)} |"
        )
    lines += ["", "## 3. Protocols by dataset", ""]
    for ds in ("LEOP", "PERG", "FLINDERS", "URFU"):
        prot = after["protocols_by_dataset"].get(ds, {})
        if prot:
            lines.append(f"- **{ds}**: " + ", ".join(f"{k}={v}" for k, v in sorted(prot.items())))
    lines += ["", "## 4. Age ranges by dataset", ""]
    for ds in ("LEOP", "PERG", "FLINDERS", "URFU"):
        lines.append(f"- **{ds}**: {_fmt_age(after['age_by_dataset'].get(ds, {}))}")
    lines += [
        "",
        "## 5. Labels (visits)",
        "",
        f"- eligible: {after['label_eligible_visits']} "
        f"(pos {after['label_positive_visits']} / neg {after['label_negative_visits']})",
        f"- ineligible (no target): {after['label_ineligible_visits']}",
        "",
        "Interpretation: FLINDERS adds a normative healthy-control cohort "
        "(all target_binary=0); URFU adds labeled-possible waveforms "
        "(target ineligible until diagnosis mapping). Both expand protocol, "
        "site and age coverage beyond the original LEOP+PERG scope.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="External coverage report (gate 7 probe 1)")
    ap.add_argument("--artifact-root", default="artifacts")
    args = ap.parse_args()
    rep = run_coverage_report(artifact_root=args.artifact_root)
    print(json.dumps({"delta": rep["delta"]}, indent=2))

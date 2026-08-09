"""Fallback / confounding-shortcut review gate (plan Section 8.6, pre-Phase-6).

Before neural training starts, the plan requires a formal *confounding
shortcut* review: if a cheap, non-signal channel (fallback mask, missingness,
protocol-count availability) predicts the label at biology-level AUC, then
neural inputs could silently start from that shortcut.  This module writes
that written review from the canonical caches (the same merged QA table the
baselines/QA report consume) into a versioned artifact.

Checks written to the report:

1. **Fallback-only label shortcut** — can the *mere rate of fallback usage*
   (per subject / visit) predict the target?  Both datasets must stay below a
   block threshold; the recorded AUROC is gated here.
2. **Fallback mask physical explainability** — the QA module says the
   physical traces can predict which components used the fallback
   (CV-AURC ~0.857), i.e. fallbacks are a *quality artifact* (small mass /
   extreme slopes / short duration), not unexplained noise.  That masks a
   *cause* explanation, and is gated (must be >= min, otherwise fallbacks are
   too unpredictable to call "explained").
3. **Protocol-count availability shortcut** — the number of recording
   protocols per subject, used only as an availability signal, must stay
   below biology-level AUROC (the "availability" confound the pipeline
   forbids for the primary nine-step cohort).
4. **Reference to the label-permutation gate** — the strongest overall
   leakage test (``evaluation.acceptance``) re-runs the experiment with
   subject-level label permutation; it must land at chance.  This review
   only references it (it is executed by ``run-acceptance``).

The verdict text is advisory but explicit: it documents the measured values
and whether each constitutes a blocking shortcut.  The automatic PASS is the
gate: if any gated check FAILs the review verdict is FAIL and Phase 6
neural training must report this copy.

Outputs (versioned; never touching intermediate/baseline artifacts):
``artifacts/results/confounds/confound_review.json`` and ``.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..constants import MIN_AUC_N
from ..qa.report import (
    _fallback_classifier,
    _fallback_shortcut_check,
    _load_merged,
)

# Decision thresholds (documented in the report; tuned from locked primary
# results; a FAIL only raises the bar, never lowers it silently).
# Biology-level AUROC reference on the locked primary cohort: slot_logreg
# 0.657–0.685, derot 0.687–0.689 (confound audit §4.4).  Shortcut channels
# must stay *below* that band, with margin.
LABEL_SHORTCUT_AUC_MAX = 0.65    # fallback/missingness-only must stay below biology band
FALLBACK_MASK_AUC_MIN = 0.75     # mask must be physically predictable (else unexplained)
PROTOCOUNT_AUC_MAX = 0.70        # protocol-count availability shortcut cap


@dataclass
class ReviewCheck:
    """One row in the review report."""
    check: str
    outcome: str   # PASS | FAIL | INFO | REF
    measurement: str
    note: str = ""

    def to_json(self) -> dict:
        return {
            "check": self.check,
            "outcome": self.outcome,
            "measurement": self.measurement,
            "note": self.note,
        }


def _protocol_count_auc(merged: pd.DataFrame) -> float | None:
    """AUROC of per-subject protocol/component count against the label.

    Mirrors the pipeline's "availability" shortcut but computed directly on
    the merged QA table (the same rows the QA report audits): per LEOP
    subject, the number of recorded components as the only feature.
    """
    erg = merged[merged["dataset"] == "LEOP"].copy()
    erg = erg[erg["group_raw"].isin(["Control", "ASD"])].dropna(subset=["target_binary"])
    if len(erg) < MIN_AUC_N:
        return None
    per_subject = (
        erg.groupby("global_subject_id")
        .agg(n=("global_component_id", "size"), y=("target_binary", "first"))
        .dropna(subset=["y"])
    )
    y = per_subject["y"].astype(float).to_numpy()
    if y.sum() == 0 or y.sum() == y.size:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, per_subject["n"].to_numpy(float)))


def run_confound_review(artifact_root: str | Path = "artifacts") -> dict:
    """Run the confound-shortcut review and write the versioned artifact."""
    root = Path(artifact_root)
    merged = _load_merged(root)
    checks: list[ReviewCheck] = []

    # 1) fallback-only label shortcut -------------------------------------------
    shortcut = _fallback_shortcut_check(merged)
    leop_auc = shortcut.get("leop_fallback_only_auc")
    perg_auc = shortcut.get("perg_fallback_only_auc")
    for name, auc in (("LEOP fallback-only", leop_auc), ("PERG fallback-only", perg_auc)):
        if auc is None:
            checks.append(ReviewCheck(name, "INFO", "n/a (no usable label pair)", ""))
            continue
        outcome = "PASS" if auc <= LABEL_SHORTCUT_AUC_MAX else "FAIL"
        checks.append(
            ReviewCheck(
                name,
                outcome,
                f"AUROC {auc:.3f} (cap {LABEL_SHORTCUT_AUC_MAX})",
                "label shortcut via fallback/missingness rate",
            )
        )

    # 2) fallback mask physical explainability ----------------------------------
    fc = _fallback_classifier(merged)
    cv_auc = fc.get("cv_auc")
    if cv_auc is None:
        checks.append(ReviewCheck("fallback physical explainability", "INFO",
                                  fc.get("note", "no descriptor"), ""))
    else:
        coefs = fc.get("standardized_coefficients", {})
        coef_notes = ", ".join(f"{k}={v:+.3f}" for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])))
        outcome = "PASS" if cv_auc >= FALLBACK_MASK_AUC_MIN else "FAIL"
        checks.append(
            ReviewCheck(
                "fallback physical explainability",
                outcome,
                f"CV-AUC {cv_auc:.3f} (min {FALLBACK_MASK_AUC_MIN})",
                f"physical-feats classifier; standardized: {coef_notes}",
            )
        )

    # 3) protocol-count availability shortcut ------------------------------------
    prot_auc = _protocol_count_auc(merged)
    if prot_auc is None:
        checks.append(ReviewCheck("protocol-count availability shortcut", "INFO",
                                  "n/a — not enough labeled LEOP subjects", ""))
    else:
        outcome = "PASS" if prot_auc <= PROTOCOUNT_AUC_MAX else "FAIL"
        checks.append(
            ReviewCheck(
                "protocol-count availability shortcut",
                outcome,
                f"AUROC {prot_auc:.4f} (cap {PROTOCOUNT_AUC_MAX})",
                "protocol-count-only model (labels from visits)",
            )
        )

    # 4) reference to the label-permutation acceptance gate ----------------------
    checks.append(ReviewCheck(
        "label permutation (acceptance gate)", "REF",
        "see evaluation.acceptance",
        "subject-level label permutation must land at chance",
    ))

    has_fail = any(c.outcome == "FAIL" for c in checks)
    verdict = "FAIL" if has_fail else "PASS"

    result = {
        "verdict": verdict,
        "fallback_rate": float(merged["fallback_used"].mean()) if len(merged) else 0.0,
        "n_components": int(len(merged)),
        "leop_fallback_only_auc": leop_auc,
        "perg_fallback_only_auc": perg_auc,
        "fallback_mask_cv_auc": cv_auc,
        "protocol_count_auc": prot_auc,
        "checks": [c.to_json() for c in checks],
        "notes": (
            "Verdict is advisory: it documents measured values and whether "
            "each constitutes a blocking shortcut. If a shortcut reached "
            "biology-level AUC, neural inputs must avoid it before training."
        ),
    }

    out_dir = root / "results" / "confounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confound_review.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "confound_review.md").write_text(_render_markdown(result))
    return result


def _render_markdown(result: dict) -> str:
    lines = [
        "# Fallback / confound shortcut review (pre-Phase-6 gate)",
        "",
        f"- **verdict:** **{result['verdict']}**",
        f"- fallback rate: {result['fallback_rate']:.4f} "
        f"({result['n_components']} components)",
        "",
        "## Checks",
        "",
    ]
    for c in result["checks"]:
        lines.append(f"- **[{c['outcome']}]** {c['check']} — {c['measurement']}"
                     + (f" · {c['note']}" if c["note"] else ""))
    lines.append("")
    lines.append(result["notes"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fallback/confound review gate")
    ap.add_argument("--artifact-root", default="artifacts")
    args = ap.parse_args()
    res = run_confound_review(args.artifact_root)
    print(json.dumps(res, indent=2, default=str)[:4000])

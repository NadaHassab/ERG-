"""Fallback/confound review gate tests (pre-Phase-6) on the canonical tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathway_erg.evaluation.confound_review import (
    FALLBACK_MASK_AUC_MIN,
    LABEL_SHORTCUT_AUC_MAX,
    PROTOCOUNT_AUC_MAX,
    run_confound_review,
)
from pathway_erg.qa.report import _load_merged

OUT = Path("artifacts/results/confounds")


@pytest.fixture(scope="module")
def report():
    return run_confound_review("artifacts")


def test_artifact_files_written(report):
    assert (OUT / "confound_review.json").is_file()
    json.loads((OUT / "confound_review.json").read_text())
    assert (OUT / "confound_review.md").is_file()


def test_report_has_all_checks(report):
    checks = {c["check"] for c in report["checks"]}
    assert checks == {
        "LEOP fallback-only",
        "PERG fallback-only",
        "fallback physical explainability",
        "protocol-count availability shortcut",
        "label permutation (acceptance gate)",
    }


def test_no_failure_gates(report):
    for c in report["checks"]:
        assert c["outcome"] in ("PASS", "INFO", "REF"), c


def test_verdict_matches_gates(report):
    any_fail = any(c["outcome"] == "FAIL" for c in report["checks"])
    assert report["verdict"] == ("FAIL" if any_fail else "PASS")


def test_thresholds_are_conservative():
    assert LABEL_SHORTCUT_AUC_MAX <= 0.65
    assert FALLBACK_MASK_AUC_MIN >= 0.75
    assert PROTOCOUNT_AUC_MAX <= 0.70


def test_leop_shortcut_below_biology(report):
    # 0.605 must stay below the biology band (slot ~0.657-0.685 / derot 0.687)
    auc = report["leop_fallback_only_auc"]
    assert auc is not None
    assert auc <= LABEL_SHORTCUT_AUC_MAX


def test_perg_shortcut_below_biology(report):
    auc = report["perg_fallback_only_auc"]
    assert auc is not None
    assert auc <= LABEL_SHORTCUT_AUC_MAX


def test_fallback_mask_physically_explained(report):
    assert report["fallback_mask_cv_auc"] is not None
    assert report["fallback_mask_cv_auc"] >= FALLBACK_MASK_AUC_MIN


def test_protocol_count_shortcut_gated(report):
    auc = report["protocol_count_auc"]
    if auc is not None:
        assert auc <= PROTOCOUNT_AUC_MAX


def test_source_has_fallback_data():
    merged = _load_merged(Path("artifacts"))
    assert merged["fallback_used"].sum() > 0

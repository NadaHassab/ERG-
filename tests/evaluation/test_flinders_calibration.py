"""Flinders normative calibration tests (gate 7, probe 2) on real data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathway_erg.evaluation.flinders_calibration import (
    FEATURE_PAIRS,
    run_flinders_calibration,
)

XLSX = Path("14747349 (2)/ISCEV Control ERG Flinders University.xlsx")


@pytest.fixture(scope="module")
def report():
    return run_flinders_calibration("artifacts")


def test_source_exists():
    assert XLSX.is_file()


def test_protocol_is_la3(report):
    assert report["protocol"] == "LA3"


def test_sample_sizes(report):
    assert report["n_flinders_rows"] == 292
    assert report["n_leop_subjects"] > 100


def test_feature_coverage(report):
    assert set(report["features"]) == set(FEATURE_PAIRS)


def test_per_feature_fields(report):
    for r in report["features"].values():
        assert r["n_flinders"] == 292
        assert r["n_leop_subjects"] == report["n_leop_subjects"]
        assert 0.0 <= r["ks_raw"]["statistic"] <= 1.0
        assert 0.0 <= r["within_2sd_raw"] <= 1.0
        assert 0.0 <= r["within_2sd_age_adjusted"] <= 1.0
        assert "age_slope_per_year" in r


def test_healthy_overlap_sanity(report):
    # healthy-control distributions must largely overlap the Flinders window
    for r in report["features"].values():
        assert r["within_2sd_raw"] >= 0.90
        assert r["within_2sd_age_adjusted"] >= 0.90


def test_artifacts_written(report):
    out = Path("artifacts/results/flinders_calibration")
    assert (out / "calibration_report.json").is_file()
    json.loads((out / "calibration_report.json").read_text())
    assert (out / "calibration_report.md").is_file()


def test_age_adjusted_consistent(report):
    # age-adjusted within-2SD fraction must be defined and near raw
    for r in report["features"].values():
        assert abs(r["within_2sd_raw"] - r["within_2sd_age_adjusted"]) < 0.1

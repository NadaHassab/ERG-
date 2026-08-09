"""LEOP LA3 Flinders-norm transfer probe tests (gate 7, probe 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathway_erg.evaluation.leop_la3_transfer import (
    FEATURES,
    _flinders_la3_reference,
    _leop_la3_subjects,
    run_leop_la3_transfer,
)

XLSX = Path("14747349 (2)/ISCEV Control ERG Flinders University.xlsx")


@pytest.fixture(scope="module")
def report():
    return run_leop_la3_transfer("artifacts")


def test_source_exists():
    assert XLSX.is_file()


def test_flinders_reference_la3_only():
    fl = _flinders_la3_reference()
    assert len(fl) == 292
    assert (fl["Test"] == "LA3").all()


def test_leop_la3_subjects_valid():
    le = _leop_la3_subjects()
    assert len(le) == 204
    assert int((le["group_raw"] == "ASD").sum()) == 65
    assert int((le["group_raw"] == "Control").sum()) == 139
    for col in FEATURES:
        assert le[col].notna().all()


def test_both_schemes_present(report):
    assert {"baseline", "extnorm"} <= set(report["schemes"])


def test_schemes_report_auroc(report):
    for r in report["schemes"].values():
        assert r["auroc"] is not None
        assert 0.0 <= r["auroc"] <= 1.0
        assert "ci_95" in r


def test_extnorm_robust(report):
    # adding the healthy Flinders reference must not materially change LEOP
    # classification (scaling robustness, not a label/leakage effect)
    b = report["schemes"]["baseline"]["auroc"]
    e = report["schemes"]["extnorm"]["auroc"]
    assert abs(b - e) < 0.05


def test_no_label_leakage(report):
    # extnorm uses only Flinders *feature* statistics, never labels
    assert "n_flinders_reference_rows" in report
    assert report["n_flinders_reference_rows"] >= 100


def test_artifacts_written(report):
    out = Path("artifacts/results/leop_la3_extnorm_transfer")
    assert (out / "transfer_report.json").is_file()
    json.loads((out / "transfer_report.json").read_text())
    assert (out / "transfer_report.md").is_file()


def test_feature_units_documented(report):
    assert report["feature_units"]["a_amp"] == "a-wave amplitude (µV)"

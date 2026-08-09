"""External coverage report tests (gate 7, probe 1) against the canonical tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pathway_erg.evaluation.external_coverage import (
    EXTERNAL,
    ORIGINAL,
    run_coverage_report,
)

INTERIM = Path("artifacts/data/interim")


@pytest.fixture(scope="module")
def report():
    return run_coverage_report("artifacts")


@pytest.fixture(scope="module")
def recordings():
    return pd.read_parquet(INTERIM / "recordings.parquet")


def test_tables_exist():
    for name in ("participants", "visits", "sessions", "recordings"):
        assert (INTERIM / f"{name}.parquet").is_file(), f"missing {name}.parquet"


def test_original_scope_excludes_external(report):
    for ds in EXTERNAL:
        assert ds not in report["original"]["subjects_by_dataset"]


def test_delta_matches_counts(report, recordings):
    assert report["delta"]["recordings"] == 431
    assert report["delta"]["subjects"] == 187
    by_dataset = recordings["dataset"].value_counts()
    assert report["extended"]["recordings_by_dataset"]["URFU"] == int(by_dataset["URFU"])
    assert report["extended"]["recordings_by_dataset"]["FLINDERS"] == int(
        by_dataset["FLINDERS"]
    )


def test_flinders_all_healthy(report):
    # FLINDERS visits are all eligible healthy controls (target_binary == 0).
    visits = pd.read_parquet(INTERIM / "visits.parquet")
    fl = visits[visits["dataset"] == "FLINDERS"]
    assert fl["target_binary"].notna().all()
    assert (fl["target_binary"] == 0).all()


def test_urfu_ineligible_until_mapped(report):
    visits = pd.read_parquet(INTERIM / "visits.parquet")
    ur = visits[visits["dataset"] == "URFU"]
    assert ur["target_binary"].isna().all()


def test_artifacts_written(report):
    out = Path("artifacts/results/external_coverage")
    assert (out / "coverage_report.json").is_file()
    json.loads((out / "coverage_report.json").read_text())
    assert (out / "coverage_report.md").is_file()


def test_scope_definitions():
    assert set(ORIGINAL) == {"LEOP", "PERG"}
    assert set(EXTERNAL) == {"FLINDERS", "URFU"}

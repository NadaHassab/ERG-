"""URFU supervised-sanity probe tests (gate 7, probe 3) on real data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pathway_erg.data.labels import audit_label_coverage
from pathway_erg.data.urfu_labels import build_urfu_mapping, make_urfu_target
from pathway_erg.evaluation.urfu_sanity import (
    FEATURE_COLUMNS,
    PROBE_PROTOCOL,
    run_urfu_sanity,
)

INTERIM = Path("artifacts/data/interim")


@pytest.fixture(scope="module")
def report():
    return run_urfu_sanity("artifacts")


@pytest.fixture(scope="module")
def urfu_visits():
    visits = pd.read_parquet(INTERIM / "visits.parquet")
    return visits[visits["dataset"] == "URFU"]


def test_mapping_covers_all_labels(urfu_visits):
    mapping = build_urfu_mapping(urfu_visits["diagnosis1_raw"])
    cov = audit_label_coverage(urfu_visits["diagnosis1_raw"], mapping)
    assert cov["complete"] is True
    assert cov["unmapped"] == []


def test_mapping_counts(urfu_visits):
    mapping = build_urfu_mapping(urfu_visits["diagnosis1_raw"])
    ur = urfu_visits.copy()
    ur["target"] = ur["diagnosis1_raw"].map(lambda d: make_urfu_target(d, mapping))
    assert int((ur["target"] == 0).sum()) == 54
    assert int((ur["target"] == 1).sum()) == 27
    assert int(ur["target"].isna().sum()) == 23


def test_mapping_pending_review(urfu_visits):
    mapping = build_urfu_mapping(urfu_visits["diagnosis1_raw"])
    assert mapping.reviewer == "PENDING_CLINICAL_REVIEW"


def test_unmapped_label_raises(urfu_visits):
    mapping = build_urfu_mapping(urfu_visits["diagnosis1_raw"])
    with pytest.raises(ValueError):
        make_urfu_target("Some unmapped diagnosis text.", mapping)


def test_probe_protocol():
    assert PROBE_PROTOCOL == "Maximum 2.0 ERG Response"
    assert len(FEATURE_COLUMNS) == 4


def test_probe_carries_signal(report):
    # held-out participants: the labels must be discriminative
    assert report["auroc"] is not None
    assert report["auroc"] > 0.6
    assert report["n_healthy"] >= 2 * report["n_reduced"]


def test_probe_participant_level(report):
    ids = report["subject_ids"]
    assert len(ids) == report["n_subjects"]
    assert len(set(ids)) == len(ids)  # no participant appears twice


def test_artifacts_written(report):
    out = Path("artifacts/results/urfu_sanity")
    assert (out / "sanity_report.json").is_file()
    json.loads((out / "sanity_report.json").read_text())
    assert (out / "sanity_report.md").is_file()
    assert (out / "sanity_predictions.parquet").is_file()
    assert (out / "diagnosis_mapping_urfu.csv").is_file()


def test_predictions_valid(report):
    pred = pd.read_parquet("artifacts/results/urfu_sanity/sanity_predictions.parquet")
    assert set(pred["y"]) == {0.0, 1.0}
    assert pred["prob_reduced"].between(0, 1).all()

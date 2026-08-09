"""Flinders ISCEV Control ERG parser tests against the bundled xlsx."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathway_erg.data import flinders

XLSX = Path("14747349 (2)/ISCEV Control ERG Flinders University.xlsx")


@pytest.fixture(scope="module")
def counts():
    return flinders.summarize_counts(XLSX)


def test_xlsx_exists():
    assert XLSX.is_file()


def test_feature_rows(counts):
    assert counts["feature_rows"] == 666


def test_feature_subjects(counts):
    assert counts["feature_subjects"] == 82


def test_feature_rows_by_protocol(counts):
    assert counts["feature_rows_by_protocol"] == {
        "LA3": 292,
        "30Hz": 111,
        "DA001": 106,
        "DA3": 90,
        "DA10": 67,
    }


def test_waveform_traces(counts):
    assert counts["waveform_traces"] == 8


def test_empty_and_missing_metadata_traces(counts):
    assert counts["empty_traces"] == 4
    assert counts["metadata_missing_traces"] == 2


def test_traces_by_protocol(counts):
    assert counts["waveform_traces_by_protocol"] == {
        "30Hz": 1,
        "LA3": 1,
        "DA0.01": 2,
        "DA10": 1,
        "DA3": 2,
        "OPS": 1,
    }


def test_waveform_subjects(counts):
    assert counts["waveform_subjects"] == 5


def test_integrity_flags(counts):
    assert counts["near_duplicate_feature_rows"] == 62
    assert counts["missing_feature_cells"] == 434


def test_iter_all_rows_have_ids_and_waveforms_valid():
    n_features = n_traces = 0
    for subject, visit, session, wf, kind in flinders.iter_flinders(XLSX):
        assert subject.global_subject_id.startswith("FLINDERS_")
        assert visit.global_subject_id == subject.global_subject_id
        assert session.global_visit_id == visit.global_visit_id
        if kind == "trace":
            assert wf is not None
            assert wf.record.n_samples == wf.time_ms.size > 0
            assert wf.time_ms[1] - wf.time_ms[0] == pytest.approx(0.512, abs=1e-9)
            n_traces += 1
        else:
            assert kind == "features"
            n_features += 1
    assert n_features == 666
    assert n_traces == 8


def test_recording_ids_unique():
    seen = set()
    for _s, _v, _sess, wf, kind in flinders.iter_flinders(XLSX):
        if kind == "trace":
            assert wf.record.global_recording_id not in seen
            seen.add(wf.record.global_recording_id)
    assert len(seen) == 8

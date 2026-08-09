"""URFU Pediatric and Adults ERG Database parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathway_erg.data import urfu
from pathway_erg.data.schemas import Protocol, WaveformKind

XLSX = Path("Pediatric and Adults ERG Database/01 Appendix 1.xlsx")


@pytest.fixture(scope="module")
def counts():
    return urfu.summarize_counts(XLSX)


def test_xlsx_exists():
    assert XLSX.is_file()


def test_signal_counts(counts):
    assert counts["blocks"] == 423
    assert counts["empty_blocks"] == 0
    assert counts["signal_columns"] == 423


def test_unlabeled_columns(counts):
    assert counts["unlabeled_columns"] == 3


def test_by_protocol(counts):
    assert counts["by_protocol"] == {
        "Maximum 2.0": 122,
        "Scotopic 2.0": 74,
        "Photopic 2.0": 106,
        "30Hz": 101,
        "OPS": 20,
    }


def test_missing_feature_cells(counts):
    assert counts["missing_feature_cells"] == 1367


def test_iter_yields_waveforms_and_sampling():
    n = 0
    for subject, visit, session, wf in urfu.iter_urfu(XLSX):
        assert subject.global_subject_id.startswith("URFU_")
        assert visit.global_subject_id == subject.global_subject_id
        assert session.global_visit_id == visit.global_visit_id
        assert wf is not None
        assert wf.record.n_samples == wf.time_ms.size > 0
        assert wf.record.sampling_rate_hz == pytest.approx(2000.0, abs=1e-6)
        n += 1
    assert n == 423


def test_protocol_mapping():
    for sheet, proto, kind in (
        ("Maximum 2.0 ERG Response", Protocol.MAXIMUM, WaveformKind.ERG),
        ("Scotopic 2.0 ERG Response", Protocol.SCOTOPIC, WaveformKind.ERG),
        ("Photopic 2.0 ERG Response", Protocol.PHOTOPIC, WaveformKind.ERG),
        ("Photopic 2.0 ERG Flicker", Protocol.FLICKER_30HZ, WaveformKind.ERG),
        ("Oscillatory Potentials", Protocol.OPS, WaveformKind.OP),
    ):
        assert urfu.PROTOCOL_MAP[sheet] is proto
        block = next(b for b in urfu.iter_urfu_blocks(XLSX) if b["sheet"] == sheet)
        assert block["empty"] is False
        assert kind in (kind,)


def test_eye_is_none():
    for _s, _v, _sess, wf in urfu.iter_urfu(XLSX):
        assert wf.record.eye is None
        break


def test_recording_ids_unique():
    seen = set()
    for _s, _v, _sess, wf in urfu.iter_urfu(XLSX):
        assert wf.record.global_recording_id not in seen
        seen.add(wf.record.global_recording_id)
    assert len(seen) == 423


def test_oscilatory_potentials_have_no_diagnosis_and_op_kind():
    for _s, visit, _sess, wf in urfu.iter_urfu(XLSX):
        if wf.record.waveform_kind is WaveformKind.OP:
            assert visit.diagnosis1_raw is None
            assert wf.record.protocol is Protocol.OPS
            break

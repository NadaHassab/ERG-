"""LEOPs source parser.

Parses participant JSONs into typed SubjectRecord / VisitRecord / SessionRecord
/ WaveformRecord objects plus Waveform arrays.  The XLSX file is only used for
cross-checking aggregates and is never ingested as duplicate observations.

Source semantics observed in v1.0.0 data:

- demographics.group in {Control, ASD, ASD+ADHD}; category 0 = control.
- demographics.site in {0, 1, 2}; demographics.sex in {0, 1} (raw coding kept).
- protocol in {9_step, 2_step, LA3}.
- op_waveform may be absent or null; that is recorded as missing, not as an
  empty observation.
- wave_id is not guaranteed unique across the dataset (5,309 recordings vs
  5,243 unique wave_id), so it is never used as the sole key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..provenance import sha256_file
from .schemas import (
    Dataset,
    Eye,
    Protocol,
    SessionRecord,
    SubjectRecord,
    VisitRecord,
    Waveform,
    WaveformKind,
    WaveformRecord,
)

DATASET = Dataset.LEOP

EYE_MAP = {"RightEye": Eye.RIGHT, "LeftEye": Eye.LEFT}


class LeopsParseError(ValueError):
    pass


def _opt_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_leops_subject_id(participant_id: str) -> str:
    return f"LEOP_{participant_id}"


def make_leops_visit_id(subject_id: str) -> str:
    return f"{subject_id}_V0"


def make_leops_session_id(visit_id: str) -> str:
    return f"{visit_id}_S0"


def _recording_key(participant_id: str, rec: dict) -> str:
    """Deterministic identity-relevant source key for a recording."""
    raw = json.dumps(
        {
            "pid": participant_id,
            "wave_id": rec.get("wave_id"),
            "protocol": rec.get("protocol"),
            "test_date": rec.get("test_date"),
            "test_time": rec.get("test_time"),
            "test_eye": rec.get("test_eye"),
            "stimulus": rec.get("stimulus"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def iter_leops_subjects(json_root: str | Path) -> Iterator[dict]:
    """Yield raw participant dicts (filename order, deterministic)."""
    root = Path(json_root)
    for path in sorted(root.glob("*.json")):
        with path.open() as fh:
            yield json.load(fh)


def parse_subject(raw: dict, source_path: Path) -> SubjectRecord:
    demo = raw.get("demographics") or {}
    pid = raw.get("participant_id")
    if not pid:
        raise LeopsParseError(f"missing participant_id in {source_path}")
    return SubjectRecord(
        global_subject_id=make_leops_subject_id(pid),
        dataset=DATASET,
        source_subject_id=str(pid),
        repeat_component_id=None,
        age_years=None,
        sex_raw=str(demo.get("sex")) if demo.get("sex") is not None else None,
        sex_standardized=str(demo.get("sex")) if demo.get("sex") is not None else None,
        site=str(demo.get("site")) if demo.get("site") is not None else None,
        group_raw=demo.get("group"),
        participant_qc_flags=(),
        source_checksum=sha256_file(source_path),
    )


def parse_recording(
    participant: SubjectRecord, raw: dict, rec: dict, index: int
) -> tuple[VisitRecord, SessionRecord, list[Waveform]]:
    """Parse one recording into a visit, session, and ERG/OP waveforms."""
    protocol_raw = rec.get("protocol")
    if protocol_raw not in {p.value for p in Protocol if p is not Protocol.PERG}:
        raise LeopsParseError(f"unknown protocol {protocol_raw!r} for {raw.get('participant_id')}")
    protocol = Protocol(protocol_raw)
    visit_id = make_leops_visit_id(participant.global_subject_id)
    session_id = make_leops_session_id(visit_id)

    visit = VisitRecord(
        global_visit_id=visit_id,
        global_subject_id=participant.global_subject_id,
        dataset=DATASET,
        source_record_id=str(raw.get("participant_id")),
        visit_date=rec.get("test_date"),
        diagnosis1_raw=participant.group_raw,
        diagnosis2_raw=None,
        diagnosis3_raw=None,
        target_binary=None,
        target_multiclass=None,
    )

    session = SessionRecord(
        global_session_id=session_id,
        global_visit_id=visit_id,
        dataset=DATASET,
        source_session_index=0,
        session_type=protocol.value,
        acquisition_timestamp_start=f"{rec.get('test_date')}T{rec.get('test_time')}"
        if rec.get("test_date")
        else None,
        eyes_available=(),
    )

    eye_raw = rec.get("test_eye")
    eye = EYE_MAP.get(eye_raw)
    stimulus = rec.get("stimulus") or {}
    flash_tds = _opt_float(stimulus.get("flash_tds"))

    features = rec.get("features") or {}
    common = dict(
        global_subject_id=participant.global_subject_id,
        global_visit_id=visit_id,
        global_session_id=session_id,
        dataset=DATASET,
        protocol=protocol,
        eye=eye,
        stimulus_value=flash_tds,
        stimulus_unit="Td_s",
        source_wave_id=str(rec.get("wave_id")),
        source_file=str(Path(participant.source_subject_id + ".json")),
        source_row_or_column=str(index),
    )

    def _wave(kind: WaveformKind, obj, erg_pair_id: str | None) -> Waveform | None:
        if obj is None:
            return None
        time_ms = np.asarray(obj.get("time_ms"), dtype=float)
        amp_uv = np.asarray(obj.get("amplitude_uv"), dtype=float)
        if time_ms.size == 0 or amp_uv.size != time_ms.size:
            raise LeopsParseError(
                f"bad {kind.value} waveform lengths in {common['source_wave_id']}"
            )
        n = time_ms.size
        dt = float(np.median(np.diff(time_ms))) if n > 1 else float("nan")
        rate = 1000.0 / dt if dt and np.isfinite(dt) else float("nan")
        rec_id = (
            f"{participant.global_subject_id}_{protocol.value}_{index:03d}_{kind.value}"
            f"_{_recording_key(raw.get('participant_id'), rec)}"
        )
        record = WaveformRecord(
            global_recording_id=rec_id,
            waveform_kind=kind,
            array_key=f"{participant.global_subject_id}/{index:03d}/{kind.value}",
            n_samples=n,
            start_ms=float(time_ms[0]),
            end_ms=float(time_ms[-1]),
            median_dt_ms=dt,
            sampling_rate_hz=rate,
            erg_pair_id=erg_pair_id,
            **common,
        )
        return Waveform(time_ms=time_ms, signal_uv=amp_uv, record=record)

    erg = _wave(WaveformKind.ERG, rec.get("erg_waveform"), None)
    if erg is None:
        raise LeopsParseError(f"missing ERG waveform in {common['source_wave_id']}")
    op = _wave(WaveformKind.OP, rec.get("op_waveform"), erg.record.global_recording_id)
    if features:
        erg = Waveform(
            erg.time_ms,
            erg.signal_uv,
            WaveformRecord(
                **{
                    **erg.record.__dict__,
                    "supplied_features_json": json.dumps(features, sort_keys=True),
                }
            ),
        )

    waveforms = [erg]
    if op is not None:
        waveforms.append(op)
    return visit, session, waveforms


def iter_leops(
    json_root: str | Path,
) -> Iterator[tuple[SubjectRecord, VisitRecord, SessionRecord, list[Waveform], list[float | None]]]:
    for raw in iter_leops_subjects(json_root):
        source_path = Path(json_root) / f"{raw.get('participant_id')}.json"
        participant = parse_subject(raw, source_path)
        for index, rec in enumerate(raw.get("recordings") or []):
            visit, session, waveforms = parse_recording(participant, raw, rec, index)
            age = _opt_float(rec.get("age"))
            yield participant, visit, session, waveforms, [age]


def summarize_counts(json_root: str | Path) -> dict:
    """Counts used for cross-checking (mirrors plan Section 3.1)."""
    subjects = 0
    ergs = ops = 0
    by_protocol: dict[str, int] = {}
    by_group: dict[str, int] = {}
    nine_ops: dict[str, tuple[int, int]] = {}
    seen_subjects: set[str] = set()
    for participant, _visit, _session, waveforms, _ages in iter_leops(json_root):
        if participant.global_subject_id not in seen_subjects:
            seen_subjects.add(participant.global_subject_id)
            subjects += 1
        group = participant.group_raw or "unknown"
        by_group[group] = by_group.get(group, 0) + 1
        for wf in waveforms:
            if wf.record.waveform_kind is WaveformKind.ERG:
                ergs += 1
                by_protocol[wf.record.protocol.value] = by_protocol.get(wf.record.protocol.value, 0) + 1
                if wf.record.protocol is Protocol.NINE_STEP:
                    total, present = nine_ops.get(group, (0, 0))
                    nine_ops[group] = (total + 1, present)
            else:
                ops += 1
                if wf.record.protocol is Protocol.NINE_STEP:
                    total, present = nine_ops.get(group, (0, 0))
                    nine_ops[group] = (total, present + 1)
    n9 = {g: (t, p, t - p) for g, (t, p) in nine_ops.items()}
    return {
        "participants": subjects,
        "erg_waveforms": ergs,
        "op_waveforms": ops,
        "op_missing_nine_step": sum(v[2] for v in n9.values()),
        "protocols": by_protocol,
        "groups": by_group,
        "nine_step_by_group": {
            g: {"waveforms": t, "op_present": p, "op_missing": m} for g, (t, p, m) in n9.items()
        },
    }

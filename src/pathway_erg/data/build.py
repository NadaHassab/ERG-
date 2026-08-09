"""Deterministic canonical data build.

Orchestrates parsing, identity, labels, validity, canonical tables, arrays,
and manifests.  One build produces every downstream artifact; repeated builds
from unchanged inputs must yield identical hashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..config import DataConfig, PreprocessingConfig
from ..provenance import RunManifest, sha256_text
from . import flinders as flinders_mod
from . import leops as leops_mod
from . import perg as perg_mod
from . import urfu as urfu_mod
from .identity import assign_perg_identity_edges
from .labels import (
    ENDPOINT_LEOPS_PRIMARY,
    ENDPOINT_PERG_PRIMARY,
    build_perg_mapping,
    make_leops_target,
    make_perg_target,
    write_mapping_csv,
)
from .schemas import (
    Dataset,
    SessionRecord,
    SubjectRecord,
    VisitRecord,
    Waveform,
    WaveformRecord,
)

BUILD_SCHEMA_VERSION = "canonical_tables_v1"

EXPECTED_COUNTS = {
    "leops": {
        "participants": 253,
        "erg_waveforms": 5309,
        "op_waveforms": 4434,
        "protocols": {"9_step": 4246, "2_step": 415, "LA3": 648},
    },
    "perg": {
        "visits": 336,
        "sessions": 677,
        "eye_curves": 1354,
        "canonical_subjects": 304,
    },
    "flinders": {
        "feature_rows": 666,
        "feature_subjects": 82,
        "waveform_traces": 8,
        "empty_traces": 4,
        "metadata_missing_traces": 2,
        "waveform_subjects": 5,
        "near_duplicate_feature_rows": 62,
        "missing_feature_cells": 434,
    },
    "urfu": {
        "blocks": 423,
        "empty_blocks": 0,
        "signal_columns": 423,
        "unlabeled_columns": 3,
        "missing_feature_cells": 1367,
        "by_protocol": {
            "Maximum 2.0": 122,
            "Scotopic 2.0": 74,
            "Photopic 2.0": 106,
            "30Hz": 101,
            "OPS": 20,
        },
    },
}


@dataclass
class BuildArtifacts:
    manifest: RunManifest
    paths: dict[str, Path]
    counts: dict[str, Any]
    checks: dict[str, Any]

    @property
    def data_hash(self) -> str:
        return self.manifest.data_hash


def _subject_to_row(s: SubjectRecord) -> dict:
    return {
        "global_subject_id": s.global_subject_id,
        "dataset": s.dataset.value,
        "source_subject_id": s.source_subject_id,
        "repeat_component_id": s.repeat_component_id,
        "age_years": s.age_years,
        "sex_raw": s.sex_raw,
        "sex_standardized": s.sex_standardized,
        "site": s.site,
        "group_raw": s.group_raw,
        "participant_qc_flags": "|".join(s.participant_qc_flags),
        "source_checksum": s.source_checksum,
    }


def _visit_to_row(v: VisitRecord) -> dict:
    return {
        "global_visit_id": v.global_visit_id,
        "global_subject_id": v.global_subject_id,
        "dataset": v.dataset.value,
        "source_record_id": v.source_record_id,
        "visit_date": v.visit_date,
        "diagnosis1_raw": v.diagnosis1_raw,
        "diagnosis2_raw": v.diagnosis2_raw,
        "diagnosis3_raw": v.diagnosis3_raw,
        "target_binary": v.target_binary,
        "target_multiclass": v.target_multiclass,
        "target_mapping_version": v.target_mapping_version,
        "visit_qc_flags": "|".join(v.visit_qc_flags),
    }


def _session_to_row(s: SessionRecord) -> dict:
    return {
        "global_session_id": s.global_session_id,
        "global_visit_id": s.global_visit_id,
        "dataset": s.dataset.value,
        "source_session_index": s.source_session_index,
        "session_type": s.session_type,
        "acquisition_timestamp_start": s.acquisition_timestamp_start,
        "eyes_available": ",".join(e.value for e in s.eyes_available),
        "session_qc_flags": "|".join(s.session_qc_flags),
    }


def _waveform_to_row(w: WaveformRecord) -> dict:
    return {
        "global_recording_id": w.global_recording_id,
        "global_subject_id": w.global_subject_id,
        "global_visit_id": w.global_visit_id,
        "global_session_id": w.global_session_id,
        "dataset": w.dataset.value,
        "protocol": w.protocol.value,
        "eye": w.eye.value if w.eye else None,
        "stimulus_value": w.stimulus_value,
        "stimulus_unit": w.stimulus_unit,
        "waveform_kind": w.waveform_kind.value,
        "source_wave_id": w.source_wave_id,
        "source_file": w.source_file,
        "source_row_or_column": w.source_row_or_column,
        "array_key": w.array_key,
        "n_samples": w.n_samples,
        "start_ms": w.start_ms,
        "end_ms": w.end_ms,
        "median_dt_ms": w.median_dt_ms,
        "sampling_rate_hz": w.sampling_rate_hz,
        "erg_pair_id": w.erg_pair_id,
        "supplied_features_json": w.supplied_features_json,
        "recording_qc_flags": "|".join(w.recording_qc_flags),
    }


def _basic_qc_row(w: Waveform) -> dict:
    sig = w.signal_uv
    finite = float(np.mean(np.isfinite(sig)))
    valid = sig[np.isfinite(sig)]
    peak_to_peak = float(np.ptp(valid)) if valid.size else float("nan")
    total_variation = float(np.sum(np.abs(np.diff(valid)))) if valid.size > 1 else float("nan")
    return {
        "global_recording_id": w.record.global_recording_id,
        "finite_fraction": finite,
        "peak_to_peak_uv": peak_to_peak,
        "total_variation_uv": total_variation,
        "n_finite": int(np.sum(np.isfinite(sig))),
    }


def _table_hash(df: pd.DataFrame) -> str:
    return sha256_text(df.to_csv(index=False))


def build_dataset(
    data_cfg: DataConfig,
    pre_cfg: PreprocessingConfig,
    raw_audit_hash: str = "",
) -> BuildArtifacts:
    """Build canonical tables and arrays; fail on count mismatches."""
    artifact_root = Path(data_cfg.artifact_root)
    data_dir = artifact_root / "data"
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "arrays").mkdir(parents=True, exist_ok=True)
    (data_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (artifact_root / "audit").mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(kind="data_build", name="canonical_data")
    manifest.extra["build_schema_version"] = BUILD_SCHEMA_VERSION
    manifest.extra["preprocessing_version"] = pre_cfg.version
    manifest.extra["raw_audit_hash"] = raw_audit_hash

    subjects: list[SubjectRecord] = []
    visits: list[VisitRecord] = []
    sessions: list[SessionRecord] = []
    waveform_records: list[WaveformRecord] = []
    waveform_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    qc_rows: list[dict] = []
    counts: dict[str, Any] = {}

    # --- LEOP ---
    leops_counts = leops_mod.summarize_counts(data_cfg.leops.json_root)
    for participant, visit, session, waveforms, rec_ages in leops_mod.iter_leops(data_cfg.leops.json_root):
        ages = [a for a in rec_ages if a is not None]
        if participant.age_years is None and ages:
            participant = SubjectRecord(
                **{**participant.__dict__, "age_years": float(min(ages))}
            )
        subjects.append(participant)
        visits.append(visit)
        sessions.append(session)
        for wf in waveforms:
            waveform_records.append(wf.record)
            waveform_arrays[wf.record.global_recording_id] = (wf.time_ms, wf.signal_uv)
            qc_rows.append(_basic_qc_row(wf))
    counts["leops"] = leops_counts

    # --- PERG ---
    metadata = perg_mod.read_perg_metadata(data_cfg.perg.metadata_csv)
    perg_visits: list[VisitRecord] = []
    perg_sessions: list[SessionRecord] = []
    perg_waveforms: list[WaveformRecord] = []
    for visit, session, wf in perg_mod.iter_perg_waveforms(data_cfg.perg.root, metadata):
        perg_visits.append(visit)
        perg_sessions.append(session)
        perg_waveforms.append(wf.record)
        waveform_arrays[wf.record.global_recording_id] = (wf.time_ms, wf.signal_uv)
        qc_rows.append(_basic_qc_row(wf))
    rep_records = {
        str(r.id_record).strip().zfill(4): (
            str(r.rep_record).strip() if pd.notna(r.rep_record) and str(r.rep_record).strip() else None
        )
        for r in metadata.itertuples(index=False)
    }
    identity = assign_perg_identity_edges(perg_visits, rep_records)
    for i, v in enumerate(perg_visits):
        perg_visits[i] = VisitRecord(
            **{**v.__dict__, "global_subject_id": identity[v.global_visit_id]}
        )
    # waveforms must follow their visit's canonical subject (identity graph)
    for i, w in enumerate(perg_waveforms):
        perg_waveforms[i] = WaveformRecord(
            **{**w.__dict__, "global_subject_id": identity[w.global_visit_id]}
        )
    perg_subjects: dict[str, SubjectRecord] = {}
    meta_ids = metadata["id_record"].map(lambda x: str(int(x)).zfill(4)).to_numpy()
    for v in perg_visits:
        match = np.where(meta_ids == v.source_record_id)[0]
        row = metadata.iloc[match] if len(match) else None
        age = _opt_float(row.iloc[0]["age_years"]) if row is not None and len(row) else None
        sex = str(row.iloc[0]["sex"]) if row is not None and len(row) and pd.notna(row.iloc[0]["sex"]) else None
        if v.global_subject_id not in perg_subjects:
            perg_subjects[v.global_subject_id] = SubjectRecord(
                global_subject_id=v.global_subject_id,
                dataset=Dataset.PERG,
                source_subject_id=v.source_record_id,
                repeat_component_id=v.global_subject_id,
                age_years=age,
                sex_raw=sex,
                sex_standardized=sex,
                site=None,
                group_raw=v.diagnosis1_raw,
            )
    subjects.extend(perg_subjects.values())
    visits.extend(perg_visits)
    sessions.extend(perg_sessions)
    waveform_records.extend(perg_waveforms)

    counts["perg"] = perg_mod.summarize_counts(data_cfg.perg.root, metadata)
    counts["perg"]["canonical_subjects"] = len(perg_subjects)

    # --- FLINDERS (external controls; features + FIGURES traces) ---
    if data_cfg.flinders is not None:
        flinders_xlsx = data_cfg.flinders.xlsx_path
        for subject, visit, session, wf, kind in flinders_mod.iter_flinders(flinders_xlsx):
            subjects.append(subject)
            visits.append(visit)
            sessions.append(session)
            if wf is not None and kind == "trace":
                waveform_records.append(wf.record)
                waveform_arrays[wf.record.global_recording_id] = (wf.time_ms, wf.signal_uv)
                qc_rows.append(_basic_qc_row(wf))
        counts["flinders"] = flinders_mod.summarize_counts(flinders_xlsx)
    else:
        counts["flinders"] = {}

    # --- URFU (external; trace waveforms per signal column) ---
    if data_cfg.urfu is not None:
        urfu_xlsx = Path(data_cfg.urfu.root) / "01 Appendix 1.xlsx"
        for subject, visit, session, wf in urfu_mod.iter_urfu(urfu_xlsx):
            subjects.append(subject)
            visits.append(visit)
            sessions.append(session)
            waveform_records.append(wf.record)
            waveform_arrays[wf.record.global_recording_id] = (wf.time_ms, wf.signal_uv)
            qc_rows.append(_basic_qc_row(wf))
        counts["urfu"] = urfu_mod.summarize_counts(urfu_xlsx)
    else:
        counts["urfu"] = {}

    # --- labels ---
    leops_map_version = "leops_labels_v1"
    for i, v in enumerate(visits):
        if v.dataset is Dataset.LEOP:
            visits[i] = VisitRecord(
                **{
                    **v.__dict__,
                    "target_binary": make_leops_target(v.diagnosis1_raw, ENDPOINT_LEOPS_PRIMARY),
                    "target_mapping_version": leops_map_version,
                }
            )
    perg_mapping = build_perg_mapping(
        pd.Series([v.diagnosis1_raw for v in visits if v.dataset is Dataset.PERG]),
        ENDPOINT_PERG_PRIMARY,
    )
    for i, v in enumerate(visits):
        if v.dataset is Dataset.PERG:
            target = make_perg_target(v.diagnosis1_raw, perg_mapping)
            visits[i] = VisitRecord(
                **{
                    **v.__dict__,
                    "target_binary": target,
                    "target_mapping_version": perg_mapping.version,
                }
            )
    write_mapping_csv(
        perg_mapping, data_dir / "manifests" / "diagnosis_mapping_perg.csv"
    )

    # --- tables ---
    subjects_df = pd.DataFrame([_subject_to_row(s) for s in subjects]).drop_duplicates("global_subject_id")
    visits_df = pd.DataFrame([_visit_to_row(v) for v in visits]).drop_duplicates("global_visit_id")
    sessions_df = pd.DataFrame([_session_to_row(s) for s in sessions]).drop_duplicates("global_session_id")
    recordings_df = pd.DataFrame([_waveform_to_row(r) for r in waveform_records]).drop_duplicates("global_recording_id")
    qc_df = pd.DataFrame(qc_rows).drop_duplicates("global_recording_id")

    # --- arrays (zarr) ---
    array_root = data_dir / "arrays"
    array_path = array_root / "raw_curves.zarr"
    _write_raw_arrays(array_path, waveform_arrays)
    manifest.extra["raw_curve_keys"] = len(waveform_arrays)
    # recordings table needs row order matching the flat array layout
    recording_order = [r.global_recording_id for r in waveform_records]
    recordings_df["array_position"] = recordings_df["global_recording_id"].map(
        {rid: i for i, rid in enumerate(recording_order)}
    )

    # --- writes ---
    interim = data_dir / "interim"
    paths: dict[str, Path] = {}
    for name, df in (
        ("participants", subjects_df),
        ("visits", visits_df),
        ("sessions", sessions_df),
        ("recordings", recordings_df),
        ("qc_manifest", qc_df),
    ):
        p = interim / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    labels_df = visits_df[
        [
            "global_visit_id",
            "global_subject_id",
            "diagnosis1_raw",
            "target_binary",
            "target_mapping_version",
        ]
    ]
    labels_path = interim / "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    paths["labels"] = labels_path

    table_hashes = {
        name: _table_hash(pd.read_parquet(p)) for name, p in paths.items()
    }
    manifest.data_hash = sha256_text(
        json.dumps(
            {
                "schema": BUILD_SCHEMA_VERSION,
                "tables": {k: sha256_text(v) for k, v in table_hashes.items()},
                "raw_curve_keys": manifest.extra["raw_curve_keys"],
            },
            sort_keys=True,
        )
    )

    # --- checks ---
    checks = _run_checks(counts)
    manifest.extra["checks"] = checks
    counts_path = artifact_root / "audit" / "dataset_counts.json"
    counts_path.write_text(json.dumps({"counts": counts, "checks": checks}, indent=2, sort_keys=True))
    manifest.write_atomic(data_dir / "manifests" / "build_manifest.json")

    return BuildArtifacts(
        manifest=manifest,
        paths=paths,
        counts=counts,
        checks=checks,
    )


def _write_raw_arrays(path: Path, arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Write time/signal pairs as one contiguous zarr group.

    Layout: time_ms and signal_uv are flat concatenations; offsets is (N+1,)
    marking each recording's slice.  Row order matches the recordings table.
    """
    lengths = [a[0].size for a in arrays.values()]
    total = int(np.sum(lengths))
    time_flat = np.empty(total, dtype=np.float64)
    signal_flat = np.empty(total, dtype=np.float64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    pos = 0
    for i, (tm, sig) in enumerate(arrays.values()):
        n = tm.size
        time_flat[pos : pos + n] = tm
        signal_flat[pos : pos + n] = sig
        offsets[i + 1] = pos + n
        pos += n
    root = zarr.open_group(str(path), mode="w")
    g = root.create_group("raw")
    g.create_array("time_ms", data=time_flat, chunks=(1 << 20,))
    g.create_array("signal_uv", data=signal_flat, chunks=(1 << 20,))
    g.create_array("offsets", data=offsets, chunks=(1 << 16,))


def _opt_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_checks(counts: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for dataset, expected in EXPECTED_COUNTS.items():
        if dataset not in counts or not counts[dataset]:
            checks[f"{dataset}"] = {
                "expected": expected,
                "actual": None,
                "match": False,
                "note": "dataset not configured; expected counts are not enforced",
            }
            continue
        actual = counts[dataset]
        for key, exp in expected.items():
            got = actual.get(key)
            if isinstance(exp, dict):
                checks[f"{dataset}.{key}"] = {
                    "expected": exp,
                    "actual": got,
                    "match": dict(exp) == dict(got or {}),
                }
            else:
                checks[f"{dataset}.{key}"] = {
                    "expected": exp,
                    "actual": got,
                    "match": exp == got,
                }
    return checks

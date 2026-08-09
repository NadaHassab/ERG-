"""PERG-IOBA source parser.

Parses 336 visit records and every bilateral session without averaging.
Each record CSV contains one or more session column triplets TIME_k, RE_k,
LE_k (49 files with 1, 240 with 2, 41 with 3, 5 with 4, 1 with 5 sessions).

Semantics handled explicitly:

- `NA` visual-acuity values parse as null, never as numbers.
- Sessions are stored individually; never averaged before modeling.
- Labels belong to visits; split groups belong to canonical subjects
  (repeat-link resolution is handled in identity.py).
- `rep_record` may contain multiple referenced IDs (`Id:XXXX - Id:YYYY`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import (
    Dataset,
    Eye,
    Protocol,
    SessionRecord,
    VisitRecord,
    Waveform,
    WaveformKind,
    WaveformRecord,
)

DATASET = Dataset.PERG
MISSING_VALUES = ["NA", "N/A", "", "nan", "NaN"]


class PergParseError(ValueError):
    pass


def _parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in MISSING_VALUES:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_perg_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """Read participants_info.csv with explicit NA handling."""
    df = pd.read_csv(metadata_csv, na_values=MISSING_VALUES, keep_default_na=True)
    if df["id_record"].isna().any() or (df["id_record"].astype(str).str.strip() == "").any():
        raise PergParseError("participants_info.csv contains empty id_record")
    return df


def parse_perg_acuity(metadata_csv: str | Path) -> pd.DataFrame:
    """Per-visit logMAR acuity from participants_info.csv.

    Returns a DataFrame indexed by the 4-digit ``source_record_id`` with one
    row per PERG record (visit): ``va_re_logmar``, ``va_le_logmar`` (null where
    the raw value was NA), ``acuity_missing`` (either eye missing), and
    ``acuity_n_eyes`` (0/1/2).  No modelling happens here and missing values
    stay null; the caller decides how to treat them.
    """
    df = read_perg_metadata(metadata_csv)
    rec = df["id_record"].map(lambda x: str(int(x)).zfill(4)).to_numpy()
    va_re = pd.to_numeric(df["va_re_logMar"], errors="coerce").to_numpy(dtype=float)
    va_le = pd.to_numeric(df["va_le_logMar"], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "source_record_id": rec,
            "va_re_logmar": va_re,
            "va_le_logmar": va_le,
            "acuity_missing": np.isnan(va_re) | np.isnan(va_le),
            "acuity_n_eyes": np.isnan(va_re).astype(int) + np.isnan(va_le).astype(int),
        }
    ).set_index("source_record_id")


def _parse_session_blocks(header: list[str]) -> list[int]:
    """Return the list of session indices present in a CSV header."""
    indices: list[int] = []
    for col in header:
        parts = col.split("_")
        if len(parts) == 2 and parts[0] in {"TIME", "RE", "LE"} and parts[1].isdigit():
            idx = int(parts[1])
            if idx not in indices:
                indices.append(idx)
    return indices


def iter_perg_waveforms(
    root: str | Path, metadata: pd.DataFrame
) -> Iterator[tuple[VisitRecord, SessionRecord, Waveform]]:
    """Yield one Waveform per eye curve (RE and LE per session)."""
    root = Path(root)
    for row in metadata.itertuples(index=False):
        record_id = str(row.id_record).strip().zfill(4)
        visit_id = f"PERG_{record_id}"
        subject_id = f"PERG_SRC_{record_id}"
        csv_path = root / "csv" / f"{record_id}.csv"
        if not csv_path.is_file():
            raise PergParseError(f"missing waveform file for record {record_id}: {csv_path}")
        df = pd.read_csv(csv_path, na_values=MISSING_VALUES)
        header = list(df.columns)
        session_indices = _parse_session_blocks(header)
        if not session_indices:
            raise PergParseError(f"no TIME/RE/LE triplets in {record_id}.csv")
        diagnosis1 = row.diagnosis1
        visit = VisitRecord(
            global_visit_id=visit_id,
            global_subject_id=subject_id,
            dataset=DATASET,
            source_record_id=record_id,
            visit_date=str(row.date) if pd.notna(row.date) else None,
            diagnosis1_raw=str(diagnosis1) if pd.notna(diagnosis1) else None,
            diagnosis2_raw=str(row.diagnosis2) if pd.notna(row.diagnosis2) else None,
            diagnosis3_raw=str(row.diagnosis3) if pd.notna(row.diagnosis3) else None,
            target_binary=None,
            target_multiclass=None,
        )
        for session_index in session_indices:
            time_col = f"TIME_{session_index}"
            if time_col not in df:
                raise PergParseError(f"missing column {time_col} in {record_id}.csv")
            timestamps = df[time_col]
            first_ts = timestamps.iloc[0]
            elapsed_ms = (
                pd.to_timedelta(pd.to_datetime(timestamps) - pd.to_datetime(first_ts)).dt.total_seconds() * 1000
            ).to_numpy(dtype=float)
            session_id = f"{visit_id}_S{session_index}"
            session = SessionRecord(
                global_session_id=session_id,
                global_visit_id=visit_id,
                dataset=DATASET,
                source_session_index=session_index,
                session_type="perg",
                acquisition_timestamp_start=str(first_ts) if pd.notna(first_ts) else None,
                eyes_available=(),
            )
            for eye_key, eye in (("RE", Eye.RIGHT), ("LE", Eye.LEFT)):
                col = f"{eye_key}_{session_index}"
                if col not in df:
                    raise PergParseError(f"missing column {col} in {record_id}.csv")
                values = df[col].to_numpy(dtype=float)
                if values.size != elapsed_ms.size:
                    raise PergParseError(
                        f"length mismatch in {record_id}.csv session {session_index}"
                    )
                n = elapsed_ms.size
                dt = float(np.median(np.diff(elapsed_ms))) if n > 1 else float("nan")
                rate = 1000.0 / dt if dt and np.isfinite(dt) else float("nan")
                rec_id = f"{visit_id}_{session_index}_{eye_key}"
                record = WaveformRecord(
                    global_recording_id=rec_id,
                    global_subject_id=subject_id,
                    global_visit_id=visit_id,
                    global_session_id=session_id,
                    dataset=DATASET,
                    protocol=Protocol.PERG,
                    eye=eye,
                    stimulus_value=None,
                    stimulus_unit="",
                    waveform_kind=WaveformKind.PERG_EYE,
                    source_wave_id=f"{record_id}-{session_index}-{eye_key}",
                    source_file=f"{record_id}.csv",
                    source_row_or_column=str(session_index),
                    array_key=f"{record_id}/{session_index}/{eye_key}",
                    n_samples=n,
                    start_ms=float(elapsed_ms[0]),
                    end_ms=float(elapsed_ms[-1]),
                    median_dt_ms=dt,
                    sampling_rate_hz=rate,
                    erg_pair_id=None,
                )
                yield visit, session, Waveform(
                    time_ms=elapsed_ms, signal_uv=values, record=record
                )


def summarize_counts(root: str | Path, metadata: pd.DataFrame) -> dict:
    """Expected-value checks mirroring plan Section 3.2."""
    visits = len(metadata)
    sessions = set()
    eyes = 0
    for _visit, session, _wf in iter_perg_waveforms(root, metadata):
        sessions.add(session.global_session_id)
        eyes += 1
    samples: list[int] = []
    for _v, _s, wf in iter_perg_waveforms(root, metadata):
        samples.append(wf.record.n_samples)
    return {
        "visit_records": visits,
        "visits": visits,
        "sessions": len(sessions),
        "eye_curves": eyes,
        "n_samples": sorted(set(samples)),
    }

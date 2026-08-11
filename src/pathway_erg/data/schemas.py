"""Typed immutable records for subjects, visits, sessions, recordings,
landmarks, components, bags, and split assignments.

Objects validate IDs, units, time ordering, shapes, masks, and enum values at
construction so dataset-specific bugs surface early instead of silently
propagating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Dataset(str, Enum):  # noqa: UP042
    LEOP = "LEOP"
    PERG = "PERG"
    FLINDERS = "FLINDERS"
    URFU = "URFU"


# Full-field flash ERG datasets share the flash landmark/segment/component
# family (L_* component ids); PERG uses the pattern family (P_* ids).
# External datasets (URFU, FLINDERS) enter the combined path through these
# sets (plan integration §11.3); LEOP/PERG behavior is unchanged.
FLASH_DATASETS: frozenset[Dataset] = frozenset(
    {Dataset.LEOP, Dataset.URFU, Dataset.FLINDERS}
)
PATTERN_DATASETS: frozenset[Dataset] = frozenset({Dataset.PERG})
SUPPORTED_DATASETS: frozenset[Dataset] = frozenset(
    {Dataset.LEOP, Dataset.PERG, Dataset.URFU, Dataset.FLINDERS}
)


class Eye(str, Enum):  # noqa: UP042
    RIGHT = "RE"
    LEFT = "LE"


class WaveformKind(str, Enum):  # noqa: UP042
    ERG = "ERG"
    OP = "OP"
    PERG_EYE = "PERG_EYE"


class Protocol(str, Enum):  # noqa: UP042
    NINE_STEP = "9_step"
    TWO_STEP = "2_step"
    LA3 = "LA3"
    PERG = "PERG"
    FLICKER_30HZ = "30Hz"
    DA001 = "DA001"
    DA3 = "DA3"
    DA10 = "DA10"
    OPS = "OPS"
    SCOTOPIC = "Scotopic 2.0"
    PHOTOPIC = "Photopic 2.0"
    MAXIMUM = "Maximum 2.0"


class ComponentID(str, Enum):  # noqa: UP042
    L_EARLY_A = "L_EARLY_A"
    L_A_TO_B = "L_A_TO_B"
    L_OP = "L_OP"
    L_LATE = "L_LATE"
    P_EARLY = "P_EARLY"
    P_LATE = "P_LATE"


class Partition(str, Enum):  # noqa: UP042
    OUTER_TRAIN = "outer_train"
    OUTER_TEST = "outer_test"
    INNER_TRAIN = "inner_train"
    INNER_VAL = "inner_val"


# ---------------------------------------------------------------------------
# Metadata records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRecord:
    global_subject_id: str
    dataset: Dataset
    source_subject_id: str
    repeat_component_id: str | None
    age_years: float | None
    sex_raw: str | None
    sex_standardized: str | None
    site: str | None
    group_raw: str | None
    participant_qc_flags: tuple[str, ...] = ()
    source_checksum: str = ""

    def __post_init__(self) -> None:
        if not self.global_subject_id:
            raise ValueError("global_subject_id must be non-empty")


@dataclass(frozen=True)
class VisitRecord:
    global_visit_id: str
    global_subject_id: str
    dataset: Dataset
    source_record_id: str
    visit_date: str | None
    diagnosis1_raw: str | None
    diagnosis2_raw: str | None
    diagnosis3_raw: str | None
    target_binary: int | None
    target_multiclass: int | None
    target_mapping_version: str = ""
    visit_qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionRecord:
    global_session_id: str
    global_visit_id: str
    dataset: Dataset
    source_session_index: int
    session_type: str
    acquisition_timestamp_start: str | None
    eyes_available: tuple[Eye, ...] = ()
    session_qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WaveformRecord:
    global_recording_id: str
    global_subject_id: str
    global_visit_id: str
    global_session_id: str
    dataset: Dataset
    protocol: Protocol
    eye: Eye | None
    stimulus_value: float | None
    stimulus_unit: str
    waveform_kind: WaveformKind
    source_wave_id: str
    source_file: str
    source_row_or_column: str
    array_key: str
    n_samples: int
    start_ms: float
    end_ms: float
    median_dt_ms: float
    sampling_rate_hz: float
    erg_pair_id: str | None = None
    supplied_features_json: str = ""
    recording_qc_flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Signal-level objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Waveform:
    time_ms: np.ndarray
    signal_uv: np.ndarray
    record: WaveformRecord

    def __post_init__(self) -> None:
        if self.time_ms.ndim != 1 or self.signal_uv.ndim != 1:
            raise ValueError("time_ms and signal_uv must be 1-D arrays")
        if self.time_ms.shape != self.signal_uv.shape:
            raise ValueError(
                f"time/signal length mismatch: {self.time_ms.shape} vs {self.signal_uv.shape}"
            )
        if self.time_ms.size == 0:
            raise ValueError("empty waveform")
        if not np.all(np.isfinite(self.time_ms)):
            raise ValueError("non-finite timestamps")
        if np.any(np.diff(self.time_ms) <= 0):
            raise ValueError("timestamps must be strictly increasing")


@dataclass(frozen=True)
class Landmark:
    name: str
    time_ms: float | None
    amplitude_uv: float | None
    confidence: float
    source: str  # metadata | automatic | fallback
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")


@dataclass(frozen=True)
class ComponentRecord:
    global_component_id: str
    global_recording_id: str
    component_id: ComponentID
    segment_start_ms: float
    segment_end_ms: float
    canonicalization_type: str  # relative_phase | absolute | none
    canonical_array_key: str
    raw_array_key: str
    landmark_times_ms: tuple[float | None, ...]
    landmark_amplitudes_uv: tuple[float | None, ...]
    landmark_confidence: float
    fallback_used: bool
    physical_features: dict[str, float]
    signed_ot_array_key: str | None = None
    component_qc_flags: tuple[str, ...] = ()
    transform_version: str = ""


@dataclass(frozen=True)
class ComponentWaveform:
    time_ms: np.ndarray
    signal_uv: np.ndarray
    canonical_time: np.ndarray
    canonical_signal: np.ndarray
    valid_mask: np.ndarray
    record: ComponentRecord


@dataclass(frozen=True)
class Segment:
    component_id: ComponentID
    time_ms: np.ndarray
    signal_uv: np.ndarray
    canonical_time: np.ndarray
    canonical_signal: np.ndarray
    physical_features: dict[str, float]
    confidence: float
    canonicalization_type: str = "absolute"
    flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Bags
# ---------------------------------------------------------------------------


@dataclass
class LEOPParticipantBag:
    global_subject_id: str
    target: int | None
    eyes: dict[str, LEOPEye] = field(default_factory=dict)

    def all_waveforms(self):
        for eye in self.eyes.values():
            for intensity in eye.intensities.values():
                yield from intensity.waveforms


@dataclass
class LEOPEye:
    eye: Eye
    intensities: dict[float, LEOPIntensity] = field(default_factory=dict)


@dataclass
class LEOPIntensity:
    flash_tds: float
    waveforms: list[Waveform] = field(default_factory=list)


@dataclass
class PERGVisitBag:
    global_visit_id: str
    global_subject_id: str
    target: int | None
    sessions: list[PERGSession] = field(default_factory=list)


@dataclass
class PERGSession:
    session_id: str
    eye_waveforms: dict[Eye, list[Waveform]] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitAssignment:
    unit_id: str  # global_subject_id (LEOP) or repeat component / subject (PERG)
    dataset: Dataset
    outer_fold: int
    partition: Partition
    inner_fold: int | None = None

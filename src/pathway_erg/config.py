"""Typed, validated configuration objects.

Configurations are loaded from YAML into frozen dataclasses.  Unknown keys are
rejected, defaults are explicit, and every config can be canonical-JSON
serialized and hashed so runs can record their exact configuration hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .constants import (
    BASELINE_MAX_PCA_COMPONENTS,
    BASELINE_PCA_VARIANCE,
    BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP_REPS,
)

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for invalid or unknown configuration content."""


def _canonicalize(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonicalize(asdict(obj))
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _from_dict(schema: type[T], data: dict[str, Any], where: str) -> T:
    if not is_dataclass(schema):
        raise ConfigError(f"{where}: target schema is not a dataclass")
    hints = get_type_hints(schema)
    allowed = {f.name for f in fields(schema)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown configuration keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(schema):
        if f.name not in data:
            if f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                raise ConfigError(f"{where}: missing required key {f.name!r}")
            continue
        value = data[f.name]
        field_type = hints.get(f.name)
        origin = get_origin(field_type)
        if field_type is None:
            kwargs[f.name] = value
            continue
        union_args = get_args(field_type)
        inner = None
        if origin in (list, set, tuple) and union_args:
            inner = get_origin(union_args[0]) or union_args[0]
        elif is_dataclass(field_type):
            inner = field_type
        elif origin is not None and getattr(origin, "__name__", "") == "UnionType":
            for a in union_args:
                if a is type(None):
                    continue
                inner = a
                break
        if inner is not None and is_dataclass(inner):
            if origin in (list, set, tuple):
                if not isinstance(value, list):
                    raise ConfigError(f"{where}.{f.name}: expected list")
                kwargs[f.name] = tuple(_from_dict(inner, item, f"{where}.{f.name}") for item in value)
            else:
                if not isinstance(value, dict):
                    raise ConfigError(f"{where}.{f.name}: expected mapping")
                kwargs[f.name] = _from_dict(inner, value, f"{where}.{f.name}")
        elif origin in (list, set, tuple):
            kwargs[f.name] = value
        else:
            kwargs[f.name] = value
    try:
        return schema(**kwargs)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def load_config(schema: type[T], path: str | Path) -> T:
    """Load a YAML file into a typed config, rejecting unknown keys."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config root must be a mapping")
    return _from_dict(schema, data, str(path))


def config_hash(config: Any) -> str:
    """Stable SHA-256 over the canonical serialization of a config."""
    canonical = json.dumps(_canonicalize(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_to_json(config: Any) -> str:
    """Canonical JSON representation used for storage and hashing."""
    return json.dumps(_canonicalize(config), sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeopsDataConfig:
    json_root: str
    xlsx_path: str


@dataclass(frozen=True)
class PergDataConfig:
    root: str
    metadata_csv: str


@dataclass(frozen=True)
class FlindersDataConfig:
    xlsx_path: str


@dataclass(frozen=True)
class UrfuDataConfig:
    root: str


@dataclass(frozen=True)
class DataConfig:
    leops: LeopsDataConfig
    perg: PergDataConfig
    artifact_root: str = "artifacts"
    flinders: FlindersDataConfig | None = None
    urfu: UrfuDataConfig | None = None


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeopsPreprocessingConfig:
    baseline: str = "prestimulus_median"
    primary_protocol: str = "9_step"
    smoothing_ms: float = 3.0
    stimulus_onset_ms: float = 0.0


@dataclass(frozen=True)
class PergPreprocessingConfig:
    baseline: str = "none"
    offset_sensitivities: tuple[str, ...] = ("none", "whole_trace_median", "robust_trend")
    smoothing_ms: float = 3.0


@dataclass(frozen=True)
class SmoothingConfig:
    method: str = "savitzky_golay"
    window_ms: float = 3.0
    polyorder: int = 3


@dataclass(frozen=True)
class LandmarkDetectionConfig:
    """Landmark search windows per dataset (plan Section 8.1).

    No defaults: every window must be stated in the preprocessing config.
    """

    leops_a_range: tuple[float, float]
    leops_b_range: tuple[float, float]
    leops_late_range: tuple[float, float]
    leops_min_prominence_frac: float
    leops_max_disagreement_ms: float
    leops_min_separation_ms: float
    leops_late_separation_ms: float
    perg_n35_range: tuple[float, float]
    perg_p50_range: tuple[float, float]
    perg_n95_range: tuple[float, float]
    perg_min_prominence_frac: float
    perg_min_separation_ms: float
    perg_late_separation_ms: float


@dataclass(frozen=True)
class LeopsSegmentationConfig:
    early_a_pad_ms: float
    early_a_bound_ms: float
    early_a_fallback_ms: tuple[float, float]
    a_to_b_pad_ms: tuple[float, float]
    a_to_b_fallback_ms: tuple[float, float]
    late_pad_ms: float
    late_fallback_lo_ms: float


@dataclass(frozen=True)
class PergSegmentationConfig:
    early_pad_ms: tuple[float, float]
    early_fallback_ms: tuple[float, float]
    late_pad_ms: tuple[float, float]
    late_fallback_ms: tuple[float, float]


@dataclass(frozen=True)
class SegmentationConfig:
    """Component window geometry (plan Sections 8.2-8.4)."""

    relative_phase_range: tuple[float, float]
    late_end_ms: float
    op_default_confidence: float
    leops: LeopsSegmentationConfig
    perg: PergSegmentationConfig


@dataclass(frozen=True)
class SpectralConfig:
    """Physiological spectral bands for multiscale spectral features.

    Bands are (name, lo_hz, hi_hz) tuples covering the a/b-wave slow
    components, the mid band, the explicit OP band (80-300 Hz), and fast
    content; the dominant-frequency search window and entropy band are
    separate.  Band edges must stay below the Nyquist rate of every dataset
    (LEOP 976 Hz, PERG 833 Hz).
    """

    bands: tuple[tuple[str, float, float], ...] = (
        ("slow", 0.5, 20.0),
        ("mid", 20.0, 80.0),
        ("op", 80.0, 300.0),
        ("fast", 300.0, 500.0),
    )
    dominant_range: tuple[float, float] = (0.5, 250.0)


@dataclass(frozen=True)
class PreprocessingConfig:
    landmarks: LandmarkDetectionConfig
    segmentation: SegmentationConfig
    version: str = "preprocessing_v1"
    segment_length: int = 128
    ot_quantiles: int = 64
    spectral: SpectralConfig = SpectralConfig()
    leops: LeopsPreprocessingConfig = LeopsPreprocessingConfig()
    perg: PergPreprocessingConfig = PergPreprocessingConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    interpolation: str = "pchip"
    hard_finite_fraction: float = 0.95
    max_isolated_nan_gaps: int = 1
    prohibit_extrapolation: bool = True
    mass_tolerance: float = 1e-8
    robust_trend_inlier_mad_multiple: float = 3.0


# ---------------------------------------------------------------------------
# Model / graph configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertConfig:
    shared: tuple[str, ...] = ("INNER_LATE",)
    private: tuple[str, ...] = (
        "FLASH_EARLY",
        "FLASH_OP",
        "FLASH_LATE",
        "PERG_EARLY",
        "PERG_LATE",
    )


@dataclass(frozen=True)
class GraphConfig:
    routes: dict[str, tuple[str, ...]]
    shared_gate_prior: float = 0.75


@dataclass(frozen=True)
class ModelConfig:
    name: str
    experts: ExpertConfig = ExpertConfig()
    graph: GraphConfig | None = None
    token_dim: int = 96
    expert_dim: int = 64
    raw_channels: tuple[int, ...] = (16, 32, 64)
    ot_mlp: tuple[int, ...] = (128, 64)
    physical_mlp: tuple[int, ...] = (32, 16)
    dropout: float = 0.1
    bag_attention_dim: int = 64
    use_geometry_loss: bool = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    kind: str  # ssl | finetune
    task: str = "both"  # leops | perg | both
    epochs: int = 200
    early_stopping_patience: int = 25
    batch_size: int = 16
    encoder_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    max_grad_norm: float = 1.0
    freeze_encoders_epochs: int = 5
    class_weighted: bool = True
    seed: int = 1001
    num_workers: int = 4
    lambda_mask: float = 1.0
    lambda_view: float = 0.25
    lambda_aug: float = 0.25
    lambda_geom: float = 0.10
    lambda_prior: float = 0.01
    label_fraction: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str = "pathway_ot"
    methods: tuple[str, ...] = ()
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (1001, 2002, 3003)


@dataclass(frozen=True)
class BaselinesConfig:
    """E0/E4 classical baseline experiment (plan Sections 16 and 17 E0/E4)."""

    name: str
    fold_version: str = "v1"
    datasets: tuple[str, ...] = ("LEOP", "PERG")
    # LEOP cohort experiment (plan Section 16.6): None | "primary_nine_step" |
    # "secondary_all_protocols". Metrics are namespaced "{dataset}_{cohort}/..."
    # when set; PERG is never affected.
    leop_cohort: str | None = None
    e0_methods: tuple[str, ...] = ("prevalence", "metadata", "availability", "quality")
    e4_methods: tuple[str, ...] = (
        "clinical",
        "pca_fpca",
        "raw_rbf",
        "scdt",
        "derot_lr",
        "derot_rbf",
        "scattering",
    )
    # E4 methods that additionally receive age/sex/(LEOP) site columns as
    # "{method}_demog" variants (plan Section 17 E11 demographic robustness).
    demographic_methods: tuple[str, ...] = ()
    # Output directory under <artifact_root>/results/ (e.g. "baselines" for the
    # legacy pipeline, "baselines_v2" for the corrected pipeline). Versioned
    # outputs keep experiments from overwriting each other.
    output_subdir: str = "baselines"
    # True: run the frozen Phase-0 snapshot (models/legacy_baselines.py) exactly
    # as before, writing to output_subdir. False: run the corrected pipeline.
    legacy: bool = False
    # True: fit logreg/SVM estimators with cuML on the GPU when cuML is
    # installed (legacy snapshot is unaffected). Falls back to scikit-learn
    # silently per estimator kind with a methodology note if unavailable.
    use_gpu: bool = False
    models: tuple[str, ...] = ("logreg", "svm_rbf", "histgb")
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_bootstrap_reps: int = DEFAULT_N_BOOTSTRAP_REPS
    bootstrap_seed: int = BOOTSTRAP_SEED
    confidence: float = DEFAULT_CONFIDENCE
    pca_variance: float = BASELINE_PCA_VARIANCE
    max_pca_components: int = BASELINE_MAX_PCA_COMPONENTS
    seed: int = 777
    # Label-permutation gate (v2 plan Phase 9): when set, target labels are
    # permuted at subject level with this deterministic seed BEFORE any
    # modeling (prevalence preserved, subject clustering preserved), and the
    # seed is recorded in the run notes. Expected outcome: every method's
    # AUROC lands at chance (~0.5) — proving no label/information leakage.
    label_permutation_seed: int | None = None
    # VMD comparator hyperparameters (plan Section 15.2 grid). Empty dict uses
    # the calibrated defaults (K=5, alpha=2000, tol=1e-7, pad=25ms). When set,
    # the matching VMD cache must exist (cache-vmd), and the exact key is
    # recorded in the run manifest/notes so a grid point never silently reuses
    # another configuration's features.
    vmd: dict[str, float | int] | None = None

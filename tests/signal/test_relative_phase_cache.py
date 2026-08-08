"""Phase 3: relative-phase cache correctness tests.

The component cache must store the *canonical* arrays for relative-phase
segments (L_LATE, P_LATE) — i.e. the relative-phase grid produced by
``canonicalize_relative_phase`` — not a re-resample of the physical
``time_ms/signal_uv`` trace (that destroyed phase alignment). Absolute
segments keep the physical-time domain. Caches are schema-versioned: the v2
pipeline reads only its own versioned files and rejects stale ones, while the
legacy v1 files stay untouched for the frozen snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathway_erg.config import PreprocessingConfig, load_config
from pathway_erg.signal.component_cache import (
    CACHE_SCHEMA_VERSION,
    cache_paths,
    process_recording,
)


def _preprocessing() -> PreprocessingConfig:
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def _synthetic_leop_erg():
    """Synthetic LEOP ERG with clear a-wave (12), b-wave (28), late trough (60)."""
    t = np.arange(235) * 1000.0 / 1953.125
    sig = (
        -3.0 * np.exp(-0.5 * ((t - 12.0) / 2.0) ** 2)
        + 12.0 * np.exp(-0.5 * ((t - 28.0) / 3.0) ** 2)
        - 4.0 * np.exp(-0.5 * ((t - 60.0) / 6.0) ** 2)
    )
    return t, sig


def _record() -> pd.Series:
    return pd.Series(
        {
            "dataset": "LEOP",
            "waveform_kind": "ERG",
            "median_dt_ms": 0.512,
            "global_recording_id": "SYNTH_A",
            "array_key": "SYNTH_A",
            "supplied_features_json": "",
        }
    )


def _process(pre_cfg: PreprocessingConfig) -> dict:
    t, sig = _synthetic_leop_erg()
    return process_recording(_record(), t, sig, pre_cfg)


def test_relative_phase_cached_in_phase_units():
    """L_LATE cached canonical_time must live on the relative-phase grid."""
    out = _process(_preprocessing())
    assert out["valid"]
    rows = {r["component_id"]: r for r in out["rows"]}
    assert rows["L_LATE"]["canonicalization_type"] == "relative_phase"
    arrays = out["arrays"][rows["L_LATE"]["canonical_array_key"]]
    ctime = arrays["canonical_time"]
    assert ctime.size == 128
    lo, hi = _preprocessing().segmentation.relative_phase_range
    assert ctime.min() >= lo - 1e-6 and ctime.max() <= hi + 1e-6
    # not the physical-time resample (18..110 ms): the bug this phase fixes
    assert ctime.max() < 50.0


def test_relative_phase_signal_differs_from_physical_resample():
    """The cached signal must not equal resampling the physical trace."""
    out = _process(_preprocessing())
    rows = {r["component_id"]: r for r in out["rows"]}
    arrays = out["arrays"][rows["L_LATE"]["canonical_array_key"]]
    row = rows["L_LATE"]
    phys_grid = np.linspace(row["segment_start_ms"], row["segment_end_ms"], arrays["canonical_time"].size)
    assert not np.allclose(arrays["canonical_time"], phys_grid)


def test_absolute_component_keeps_physical_domain():
    """L_EARLY_A (absolute) cached time stays in physical milliseconds."""
    out = _process(_preprocessing())
    rows = {r["component_id"]: r for r in out["rows"]}
    assert rows["L_EARLY_A"]["canonicalization_type"] == "absolute"
    arrays = out["arrays"][rows["L_EARLY_A"]["canonical_array_key"]]
    ctime = arrays["canonical_time"]
    assert ctime.min() >= 0.0 and ctime.max() <= 30.0  # ~0..17 ms window
    assert ctime.size == 128


def test_alignment_change_sensitivity():
    """Changing the relative-phase range must change the cached L_LATE arrays."""
    pre = _preprocessing()
    default = _process(pre)
    rows = {r["component_id"]: r for r in default["rows"]}
    default_ctime = default["arrays"][rows["L_LATE"]["canonical_array_key"]]["canonical_time"]
    new_range = (0.0, 1.0)
    pre2 = PreprocessingConfig(
        landmarks=pre.landmarks,
        segmentation=pre.segmentation.__class__(
            relative_phase_range=new_range,
            late_end_ms=pre.segmentation.late_end_ms,
            op_default_confidence=pre.segmentation.op_default_confidence,
            leops=pre.segmentation.leops,
            perg=pre.segmentation.perg,
        ),
        **{k: getattr(pre, k) for k in pre.__dataclass_fields__ if k not in ("landmarks", "segmentation")},
    )
    changed = _process(pre2)
    rows2 = {r["component_id"]: r for r in changed["rows"]}
    changed_ctime = changed["arrays"][rows2["L_LATE"]["canonical_array_key"]]["canonical_time"]
    assert changed_ctime.min() >= -1e-6 and changed_ctime.max() <= 1.0 + 1e-6
    assert not np.allclose(default_ctime, changed_ctime)


def test_cache_schema_version_in_manifest(tmp_path):
    """cache_paths versioning: current-schema files are distinct from v1 names."""
    root = Path(tmp_path)
    v2 = cache_paths(root, CACHE_SCHEMA_VERSION)
    v1 = cache_paths(root, 1)
    for key in v2:
        assert v2[key].name != v1[key].name
        assert f"_v{CACHE_SCHEMA_VERSION}" in v2[key].name
    assert v1["curves_zarr"].name == "component_curves.zarr"  # legacy name


def test_v2_loader_rejects_stale_schema(tmp_path):
    from pathway_erg.signal.component_cache import load_cache_manifest

    root = Path(tmp_path)
    v2 = cache_paths(root, CACHE_SCHEMA_VERSION)
    v2["manifest"].parent.mkdir(parents=True, exist_ok=True)
    v2["manifest"].write_text(json.dumps({"extra": {"schema_version": CACHE_SCHEMA_VERSION - 1}}))
    with pytest.raises(ValueError, match="cache schema"):
        load_cache_manifest(root, CACHE_SCHEMA_VERSION)


def test_v2_loader_accepts_current_schema(tmp_path):
    from pathway_erg.signal.component_cache import load_cache_manifest

    root = Path(tmp_path)
    v2 = cache_paths(root, CACHE_SCHEMA_VERSION)
    v2["manifest"].parent.mkdir(parents=True, exist_ok=True)
    v2["manifest"].write_text(
        json.dumps({"extra": {"schema_version": CACHE_SCHEMA_VERSION, "n_valid": 42}})
    )
    m = load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    assert m["extra"]["n_valid"] == 42

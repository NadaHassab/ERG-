"""Component and signed-OT cache pipeline.

Runs the full signal pipeline over every technically valid recording and
writes versioned, indexed arrays:

- component_curves_v2.zarr: canonical 128-point component arrays + valid masks.
- signed_ot_v2.zarr: flat signed-OT descriptor vectors per component.
- components_v2.parquet: per-component metadata (plan Section 11.3).

Everything is deterministic; caches store the transform version and config
hash for provenance.  No silent fallbacks: missing arrays, malformed supplied
features, and degenerate metadata raise instead of being skipped quietly.

Relative-phase segments (L_LATE, P_LATE) are cached on their canonical
relative-phase grid (``canonical_time``/``canonical_signal`` from
segments.py); absolute segments are resampled from physical time.  Caches are
schema-versioned: schema 1 is the legacy/frozen layout read by
``legacy_baselines``; schema 2 fixes the relative-phase cache; schema 3 adds
per-component spectral feature vectors computed on the physical windows;
schema 4 upgrades the signed-OT descriptor (declared reference + normalized
mass fraction) to its final v2 layout.  The v2 pipeline reads only its own
versioned files and rejects stale ones.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..config import PreprocessingConfig, config_hash
from ..data.schemas import Dataset, FLASH_DATASETS, WaveformKind
from ..provenance import RunManifest
from .baseline import apply_offset
from .landmarks import detect_landmarks
from .physical_features import physical_features
from .resample import resample_pchip
from .segments import make_op_segment, make_segments
from .signed_ot import signed_derivative_ot
from .smoothing import smooth_for_analysis
from .spectral import spectral_feature_names, spectral_features
from .validity import check_hard_validity, interpolate_isolated_nan
from .vmd import (
    FrequencyConvention,
    VMDConfig,
    calibrate_vmd_frequency,
    decompose_vmd,
    extract_vmd_features,
)

CACHE_SCHEMA_VERSION = 4


def cache_paths(artifact_root: str | Path, schema_version: int = CACHE_SCHEMA_VERSION) -> dict[str, Path]:
    """Versioned cache file locations.

    Schema 1 keeps the legacy names (component_curves.zarr / signed_ot.zarr /
    components.parquet) used by the frozen baseline snapshot; later schemas
    append a ``_v<N>`` suffix so stale caches are never silently reused.
    """
    artifact_root = Path(artifact_root)
    suffix = "" if schema_version <= 1 else f"_v{schema_version}"
    return {
        "curves_zarr": artifact_root / "data" / "arrays" / f"component_curves{suffix}.zarr",
        "sot_zarr": artifact_root / "data" / "arrays" / f"signed_ot{suffix}.zarr",
        "spectral_zarr": artifact_root / "data" / "arrays" / f"spectral_features{suffix}.zarr",
        "components_parquet": artifact_root / "data" / "interim" / f"components{suffix}.parquet",
        "manifest": artifact_root / "data" / "manifests" / f"component_cache_manifest{suffix}.json",
    }


def load_cache_manifest(artifact_root: str | Path, schema_version: int = CACHE_SCHEMA_VERSION) -> dict:
    """Load and validate a component-cache manifest for the given schema.

    Raises ``ValueError`` when the cache is missing or was written by an
    older schema: stale caches must be rebuilt, never reused.
    """
    manifest_path = cache_paths(artifact_root, schema_version)["manifest"]
    if not manifest_path.is_file():
        raise ValueError(
            f"component cache manifest not found at {manifest_path}; "
            "rebuild with the cache-components command"
        )
    manifest = json.loads(manifest_path.read_text())
    found = int(manifest.get("extra", {}).get("schema_version", 0))
    if found != schema_version:
        raise ValueError(
            f"component cache schema mismatch at {manifest_path}: found "
            f"schema {found!r}, need {schema_version}; "
            "rebuild with the cache-components command"
        )
    return manifest


def _load_waveforms(artifact_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all raw arrays from the flat build cache (fast)."""
    root = zarr.open_group(str(artifact_root / "data" / "arrays" / "raw_curves.zarr"), mode="r")
    g = root["raw"]
    time_flat = np.asarray(g["time_ms"][:], dtype=float)
    signal_flat = np.asarray(g["signal_uv"][:], dtype=float)
    offsets = np.asarray(g["offsets"][:], dtype=np.int64)
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    recordings = recordings.sort_values("array_position")
    out = {}
    for i, row in enumerate(recordings.itertuples(index=False)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        out[row.global_recording_id] = (time_flat[lo:hi], signal_flat[lo:hi])
    return out


def process_recording(
    record: pd.Series,
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    pre_cfg: PreprocessingConfig,
    vmd_cfg: VMDConfig | None = None,
) -> dict:
    """Process one recording into component rows + arrays.

    ``vmd_cfg`` (optional) additionally computes the fixed-length VMD feature
    vector per segment (plan Section 15); VMD uses the physical window with
    mirror padding and the globally calibrated omega->Hz convention.  The main
    schema-4 cache never computes VMD; the separate ``vmd_modes.zarr`` cache
    is built by ``cache-vmd``.
    """
    dataset = Dataset(record["dataset"])
    validity = check_hard_validity(
        time_ms,
        signal_uv,
        pre_cfg.hard_finite_fraction,
        pre_cfg.max_isolated_nan_gaps,
    )
    if not validity.valid:
        return {"valid": False, "reasons": list(validity.reasons)}

    signal = interpolate_isolated_nan(time_ms, signal_uv, validity.fixed_mask)

    # Flash-family datasets (LEOP, URFU, FLINDERS) share the full-field flash
    # baseline policy; PERG is pattern.  LEOP/PERG behavior is unchanged:
    # the external datasets were never processed into the v4 cache.
    if dataset in FLASH_DATASETS and record["waveform_kind"] == WaveformKind.ERG.value:
        offset_policy = pre_cfg.leops.baseline
    elif dataset is Dataset.PERG:
        offset_policy = pre_cfg.perg.baseline
    else:
        offset_policy = "none"
    signal, _stats = apply_offset(
        time_ms,
        signal,
        pre_cfg.leops.stimulus_onset_ms,
        offset_policy,
        pre_cfg.robust_trend_inlier_mad_multiple,
    )

    median_dt = float(record["median_dt_ms"])
    if not np.isfinite(median_dt):
        raise ValueError(
            f"recording {record['global_recording_id']}: non-finite median_dt_ms metadata"
        )
    smoothed = _smooth(time_ms, signal, median_dt, pre_cfg.smoothing)
    supplied = _leops_supplied(record) if dataset is Dataset.LEOP else None
    landmarks = detect_landmarks(dataset, time_ms, smoothed, pre_cfg.landmarks, supplied)

    segments = make_segments(
        dataset,
        time_ms,
        signal,
        landmarks,
        pre_cfg.segmentation,
        pre_cfg.segment_length,
        op_time_ms=None,
        op_signal_uv=None,
    )
    if dataset in FLASH_DATASETS and record["waveform_kind"] == WaveformKind.OP.value:
        segments = [
            make_op_segment(time_ms, signal, pre_cfg.segmentation.op_default_confidence)
        ]
    rows: list[dict] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for seg in segments:
        if seg.canonicalization_type == "relative_phase":
            canonical_time = np.asarray(seg.canonical_time, dtype=float)
            canonical_signal = np.asarray(seg.canonical_signal, dtype=float)
            if canonical_time.size != pre_cfg.segment_length:
                raise ValueError(
                    f"segment {seg.component_id}: relative-phase canonical grid "
                    f"has {canonical_time.size} points, expected {pre_cfg.segment_length}"
                )
        else:
            canonical_time, canonical_signal = resample_pchip(
                seg.time_ms, seg.signal_uv, pre_cfg.segment_length
            )
        features = physical_features(
            seg.time_ms,
            seg.signal_uv,
            confidence=seg.confidence,
            truncation_flags=seg.flags,
        )
        ot = signed_derivative_ot(
            seg.time_ms,
            seg.signal_uv,
            median_dt,
            smoothing=pre_cfg.smoothing,
            n_quantiles=pre_cfg.ot_quantiles,
            mass_tolerance=pre_cfg.mass_tolerance,
        )
        spectral = spectral_features(
            seg.time_ms,
            seg.signal_uv,
            1000.0 / median_dt,
            pre_cfg.spectral.bands,
            pre_cfg.spectral.dominant_range,
        )
        vmd_vector: np.ndarray | None = None
        vmd_diag: dict[str, float] | None = None
        if vmd_cfg is not None:
            vmd_vector, vmd_diag = _vmd_vector(seg, median_dt, vmd_cfg)
        component_id = seg.component_id.value
        array_key = f"{record['global_recording_id']}/{component_id}"
        rows.append(
            {
                "global_component_id": f"{record['global_recording_id']}_{component_id}",
                "global_recording_id": record["global_recording_id"],
                "component_id": component_id,
                "segment_start_ms": float(seg.time_ms[0]),
                "segment_end_ms": float(seg.time_ms[-1]),
                "canonicalization_type": seg.canonicalization_type,
                "canonical_array_key": array_key,
                "raw_array_key": record["array_key"],
                "landmark_times_json": json.dumps({k: _n(lm.time_ms) for k, lm in landmarks.items()}),
                "landmark_amplitudes_json": json.dumps(
                    {k: _n(lm.amplitude_uv) for k, lm in landmarks.items()}
                ),
                "landmark_confidence": float(seg.confidence),
                "fallback_used": any(f in ("fallback-window",) for f in seg.flags)
                or seg.confidence == 0.0,
                "physical_features_json": json.dumps(features, sort_keys=True),
                "signed_ot_array_key": f"{array_key}/sot",
                "component_qc_flags": "|".join(seg.flags),
                "transform_version": ot.transform_version,
            }
        )
        arrays[array_key] = {
            "canonical_time": canonical_time,
            "canonical_signal": canonical_signal,
            "valid_mask": np.isfinite(canonical_signal) & np.isfinite(canonical_time),
            "sot_vector": ot.to_vector(),
            "spectral_vector": spectral,
            "vmd_vector": vmd_vector,
        }
        if vmd_diag is not None:
            diagnostics[array_key] = vmd_diag
    return {"valid": True, "rows": rows, "arrays": arrays, "diagnostics": diagnostics}


def _smooth(time_ms, signal, median_dt, smoothing):

    return smooth_for_analysis(time_ms, signal, median_dt, smoothing)


def _leops_supplied(record: pd.Series) -> dict | None:
    """Supplied LEOP landmark features; None means genuinely absent metadata.

    Malformed JSON raises: corrupt metadata must fail loudly, not vanish.
    """
    raw = record.get("supplied_features_json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"recording {record['global_recording_id']}: malformed supplied_features_json"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"recording {record['global_recording_id']}: supplied_features_json is not a mapping"
        )
    return payload


def _n(value) -> float | None:
    return None if value is None else float(value)


@lru_cache(maxsize=1)
def _calibrated_vmd_convention() -> FrequencyConvention:
    """Globally calibrated omega->Hz convention (plan 15.2 step 6)."""
    return calibrate_vmd_frequency()


def _vmd_vector(
    seg,
    median_dt: float,
    cfg: VMDConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """VMD feature vector + diagnostics for one physical segment window.

    Runs the primary decomposition plus the ``stability_neighbors``
    decompositions (same alpha/tol/padding, neighbouring K) so the per-mode
    stability feature is well defined.  Deterministic (init=1).  The returned
    diagnostics power the plan Section 15.2 hyperparameter grid (reconstruction
    error, residual energy, convergence, mode energies, frequency spread).
    """
    fs = 1000.0 / median_dt
    convention = _calibrated_vmd_convention()
    primary = decompose_vmd(seg.time_ms, seg.signal_uv, cfg, convention)
    neighbor_cfgs = [replace(cfg, K=k) for k in cfg.stability_neighbors]
    neighbors = [
        decompose_vmd(seg.time_ms, seg.signal_uv, nc, convention) for nc in neighbor_cfgs
    ]
    features = extract_vmd_features(primary, cfg, fs, neighbors)
    energy = np.asarray(primary.mode_energy, dtype=float)
    freqs = np.asarray(primary.center_freqs_hz, dtype=float)
    diag = {
        "recon_rms_rel": float(primary.recon_rms_rel),
        "residual_energy_rel": float(primary.residual_energy_rel),
        "converged": bool(primary.converged),
        "n_iterations": float(primary.n_iterations),
        "n_modes_above_1pct_energy": int(
            (energy / (energy.sum() + 1e-12) > 0.01).sum()
        ),
        "center_freq_spread_hz": float(np.nanmax(freqs) - np.nanmin(freqs)),
    }
    return features, diag


def cache_components(
    artifact_root: str | Path,
    pre_cfg: PreprocessingConfig,
) -> dict[str, object]:
    """Run the pipeline over all recordings and write caches."""
    artifact_root = Path(artifact_root)
    out = cache_paths(artifact_root, CACHE_SCHEMA_VERSION)
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    # The frozen v4 cache is LEOP/PERG only: external datasets are processed
    # into their own binding by cache_external_components, never here, so a
    # rebuild against the full recordings table stays byte-identical.
    recordings = recordings[recordings["dataset"].isin(("LEOP", "PERG"))]
    waveforms = _load_waveforms(artifact_root)

    all_rows: list[dict] = []
    array_data: dict[str, np.ndarray] = {}
    n_valid = 0
    reasons: dict[str, int] = {}
    for rec in recordings.itertuples(index=False):
        key = rec.global_recording_id
        if key not in waveforms:
            reasons["missing-array"] = reasons.get("missing-array", 0) + 1
            continue
        time_ms, signal_uv = waveforms[key]
        result = process_recording(
            pd.Series(rec._asdict()), time_ms, signal_uv, pre_cfg
        )
        if not result["valid"]:
            for r in result["reasons"]:
                reasons[r] = reasons.get(r, 0) + 1
            continue
        n_valid += 1
        all_rows.extend(result["rows"])
        for key2, payload in result["arrays"].items():
            array_data[key2] = payload

    components_df = pd.DataFrame(all_rows)
    components_df.to_parquet(out["components_parquet"], index=False)

    comp_root = zarr.open_group(str(out["curves_zarr"]), mode="w")
    sot_root = zarr.open_group(str(out["sot_zarr"]), mode="w")
    spec_root = zarr.open_group(str(out["spectral_zarr"]), mode="w")
    cg = comp_root.create_group("components")
    sg = sot_root.create_group("components")
    spg = spec_root.create_group("components")
    n_spec = 2 * len(pre_cfg.spectral.bands) + 2
    if components_df.empty:
        cg.create_array("canonical_time", shape=(0, pre_cfg.segment_length), chunks=(128, 128), dtype="f8")
        cg.create_array("canonical_signal", shape=(0, pre_cfg.segment_length), chunks=(128, 128), dtype="f8")
        cg.create_array("valid_mask", shape=(0, pre_cfg.segment_length), chunks=(128, 128), dtype=bool)
        sg.create_array("sot_vector", shape=(0, 0), chunks=(128, 128), dtype="f8")
        spg.create_array("spectral_vector", shape=(0, n_spec), chunks=(128, 128), dtype="f8")
    else:
        order = list(components_df["global_component_id"])
        # strict per-row mapping: components.parquet row order == array order
        key_by_component = {
            row.global_component_id: row.canonical_array_key
            for row in components_df.itertuples(index=False)
        }
        canonical_time = np.stack([array_data[key_by_component[c]]["canonical_time"] for c in order])
        canonical_signal = np.stack([array_data[key_by_component[c]]["canonical_signal"] for c in order])
        valid_mask = np.stack([array_data[key_by_component[c]]["valid_mask"] for c in order])
        sot = np.stack([array_data[key_by_component[c]]["sot_vector"] for c in order])
        spectral = np.stack([array_data[key_by_component[c]]["spectral_vector"] for c in order])
        cg.create_array("canonical_time", data=canonical_time, chunks=(256, 128))
        cg.create_array("canonical_signal", data=canonical_signal, chunks=(256, 128))
        cg.create_array("valid_mask", data=valid_mask, chunks=(256, 128))
        sg.create_array("sot_vector", data=sot, chunks=(256, 128))
        spg.create_array("spectral_vector", data=spectral, chunks=(256, 128))

    manifest = RunManifest(kind="component_cache", name="components")
    manifest.extra["schema_version"] = CACHE_SCHEMA_VERSION
    manifest.extra["n_valid"] = n_valid
    manifest.extra["reasons"] = reasons
    manifest.extra["preprocessing_version"] = pre_cfg.version
    manifest.extra["n_components"] = len(all_rows)
    manifest.extra["config_hash"] = config_hash(pre_cfg)
    manifest.extra["spectral_feature_names"] = spectral_feature_names(pre_cfg.spectral.bands)
    manifest.write_atomic(out["manifest"])
    return {
        "n_valid": n_valid,
        "n_components": len(all_rows),
        "reasons": reasons,
    }

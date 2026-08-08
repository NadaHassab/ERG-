"""Separate VMD feature cache (plan line 977: ``vmd_modes.zarr # baseline cache only``).

VMD features live in their own cache rather than in the schema-4 component
cache so that adding the VMD comparator never invalidates the main caches or
existing results.  Keyed by source/config hash (plan 15.2 step 10):

- rows are aligned 1:1 with ``components_v4.parquet`` row order
  (``global_component_id`` order), same as the spectral cache;
- the manifest pins the preprocessing config hash, the VMD config key,
  the calibrated frequency convention and the vmdpy version;
- ``load_vmd_cache`` refuses stale or mismatched caches (rebuild, never reuse).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..config import PreprocessingConfig, config_hash
from ..provenance import RunManifest
from .component_cache import _load_waveforms, cache_paths
from .vmd import FrequencyConvention, VMDConfig, calibrate_vmd_frequency, vmd_feature_names

VMD_CACHE_SCHEMA_VERSION = 1


def vmd_cache_paths(artifact_root: str | Path, schema_version: int = VMD_CACHE_SCHEMA_VERSION) -> dict[str, Path]:
    artifact_root = Path(artifact_root)
    suffix = "" if schema_version <= 1 else f"_v{schema_version}"
    return {
        "vmd_zarr": artifact_root / "data" / "arrays" / f"vmd_modes{suffix}.zarr",
        "manifest": artifact_root / "data" / "manifests" / f"vmd_cache_manifest{suffix}.json",
    }


def _vmd_work(args):
    """Pool worker: VMD feature vectors for one recording."""
    from .component_cache import process_recording

    rec, (time_ms, signal_uv), pre_cfg, vmd_cfg = args
    result = process_recording(rec, time_ms, signal_uv, pre_cfg, vmd_cfg=vmd_cfg)
    if not result["valid"]:
        return []
    return [
        (row["global_component_id"], result["arrays"][row["canonical_array_key"]]["vmd_vector"])
        for row in result["rows"]
    ]


def cache_vmd(
    artifact_root: str | Path,
    pre_cfg: PreprocessingConfig,
    vmd_cfg: VMDConfig,
    convention: FrequencyConvention | None = None,
    jobs: int = 1,
) -> dict[str, object]:
    """Build the VMD feature cache over all components (baseline-only cache).

    Re-runs ``process_recording`` per recording (the only deterministic source
    of the physical segment windows) with the optional VMD computation
    enabled, then stacks the per-component vectors in components.parquet row
    order.  Recordings are independent, so ``jobs > 1`` processes them with a
    multiprocessing pool (results are collected in recording order).
    """
    import multiprocessing as mp

    from .component_cache import process_recording

    artifact_root = Path(artifact_root)
    convention = convention or calibrate_vmd_frequency()
    out = vmd_cache_paths(artifact_root, VMD_CACHE_SCHEMA_VERSION)
    main = cache_paths(artifact_root)
    components_df = pd.read_parquet(main["components_parquet"])
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    waveforms = _load_waveforms(artifact_root)

    jobs = int(jobs)
    if jobs > 1:
        mp.set_start_method("fork", force=True)

    items = [
        (pd.Series(rec._asdict()), waveforms[rec.global_recording_id], pre_cfg, vmd_cfg)
        for rec in recordings.itertuples(index=False)
        if rec.global_recording_id in waveforms
    ]
    vectors: dict[str, np.ndarray] = {}
    n_segments = 0
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            for batch in pool.imap(_vmd_work, items, chunksize=8):
                for cid, vec in batch:
                    vectors[cid] = vec
                n_segments += len(batch)
    else:
        for item in items:
            batch = _vmd_work(item)
            for cid, vec in batch:
                vectors[cid] = vec
            n_segments += len(batch)

    order = list(components_df["global_component_id"])
    missing = [c for c in order if c not in vectors]
    if missing:
        raise ValueError(
            f"VMD cache: {len(missing)} components without a VMD vector "
            f"(first: {missing[0]}); rebuild cache-components first"
        )
    stacked = np.stack([vectors[c] for c in order])

    names = vmd_feature_names(vmd_cfg)
    if stacked.shape[1] != len(names):
        raise ValueError(f"VMD vector width {stacked.shape[1]} != names {len(names)}")

    vmd_root = zarr.open_group(str(out["vmd_zarr"]), mode="w")
    g = vmd_root.create_group("components")
    g.create_array("vmd_vector", data=stacked, chunks=(256, len(names)))

    manifest = RunManifest(kind="component_cache", name="vmd_modes")
    manifest.extra["schema_version"] = VMD_CACHE_SCHEMA_VERSION
    manifest.extra["n_components"] = int(stacked.shape[0])
    manifest.extra["n_segments"] = n_segments
    manifest.extra["preprocessing_version"] = pre_cfg.version
    manifest.extra["config_hash"] = config_hash(pre_cfg)
    manifest.extra["vmd_config_key"] = vmd_cfg.key
    manifest.extra["vmd_feature_names"] = names
    manifest.extra["frequency_convention"] = {
        "hz_per_omega_unit": convention.hz_per_omega_unit,
        "sampling_rates_hz": list(convention.sampling_rates_hz),
        "max_relative_error": convention.max_relative_error,
        "verified": convention.verified,
    }
    manifest.write_atomic(out["manifest"])
    return {
        "n_components": int(stacked.shape[0]),
        "n_segments": n_segments,
        "n_features": len(names),
        "vmd_config_key": vmd_cfg.key,
        "calibration_max_relative_error": convention.max_relative_error,
    }


def load_vmd_cache(
    artifact_root: str | Path,
    vmd_cfg: VMDConfig,
    pre_cfg_hash: str,
    schema_version: int = VMD_CACHE_SCHEMA_VERSION,
) -> tuple[np.ndarray, list[str]]:
    """Load the VMD feature matrix + names for the requested config.

    ``pre_cfg_hash`` is the preprocessing config hash pinned by the main
    component cache (the VMD cache must be built from the same preprocessing).
    Raises ``ValueError`` when the cache is missing or its pinned config hash
    / VMD config key do not match (stale caches are never reused).
    """
    artifact_root = Path(artifact_root)
    out = vmd_cache_paths(artifact_root, schema_version)
    manifest_path = out["manifest"]
    if not manifest_path.is_file():
        raise ValueError(
            f"VMD cache manifest not found at {manifest_path}; "
            "rebuild with the cache-vmd command"
        )
    manifest = json.loads(manifest_path.read_text())
    found = int(manifest.get("extra", {}).get("schema_version", 0))
    if found != schema_version:
        raise ValueError(
            f"VMD cache schema mismatch: found {found}, want {schema_version}; rebuild"
        )
    found_hash = manifest.get("extra", {}).get("config_hash")
    if found_hash != pre_cfg_hash:
        raise ValueError(
            f"VMD cache config hash mismatch ({found_hash} != {pre_cfg_hash}); "
            "rebuild with the cache-vmd command"
        )
    found_key = manifest.get("extra", {}).get("vmd_config_key")
    if found_key != vmd_cfg.key:
        raise ValueError(
            f"VMD cache config key mismatch ({found_key} != {vmd_cfg.key}); rebuild"
        )
    names = list(manifest["extra"].get("vmd_feature_names", []))
    z = zarr.open_group(str(out["vmd_zarr"]), mode="r")
    vectors = np.asarray(z["components"]["vmd_vector"][:])
    if vectors.ndim != 2 or vectors.shape[1] != len(names):
        raise ValueError(f"VMD cache array shape {vectors.shape} inconsistent with names")
    return vectors, names

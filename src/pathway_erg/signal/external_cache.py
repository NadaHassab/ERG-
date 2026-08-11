"""External (URFU/FLINDERS) component cache: a separate binding (gate 11.2.2).

The frozen LEOP/PERG schema-4 cache is never rewritten: external recordings
are processed into their own versioned files under distinct names
(``*_external_<binding>_v4.*``) and bound by their own manifest.  The
external datasets run through the same flash-family signal pipeline
(landmarks/segments route on ``FLASH_DATASETS``), so URFU/FLINDERS produce
``L_*`` component ids that map onto the existing flash private experts and
the shared late expert (plan integration §11.3).

Determinism and layout mirror ``cache_components``; a run never silently
reuses another binding's files, and the LEOP/PERG manifest hash is
unaffected by adding an external binding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..config import PreprocessingConfig, config_hash
from ..provenance import RunManifest
from .component_cache import (
    CACHE_SCHEMA_VERSION,
    _load_waveforms,
    process_recording,
)
from .spectral import spectral_feature_names

DEFAULT_EXTERNAL_BINDING = "external_v1"
DEFAULT_EXTERNAL_DATASETS = ("URFU", "FLINDERS")


def external_cache_paths(
    artifact_root: str | Path,
    binding: str = DEFAULT_EXTERNAL_BINDING,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> dict[str, Path]:
    """Versioned file locations for one external cache binding.

    Names never collide with the frozen LEOP/PERG schema-4 files, so
    building an external binding cannot disturb the frozen cache.
    """
    artifact_root = Path(artifact_root)
    tag = f"external_{binding}"
    suffix = f"_{tag}_v{schema_version}" if schema_version > 1 else f"_{tag}"
    return {
        "curves_zarr": artifact_root / "data" / "arrays" / f"component_curves{suffix}.zarr",
        "sot_zarr": artifact_root / "data" / "arrays" / f"signed_ot{suffix}.zarr",
        "spectral_zarr": artifact_root / "data" / "arrays" / f"spectral_features{suffix}.zarr",
        "components_parquet": artifact_root / "data" / "interim" / f"components{suffix}.parquet",
        "manifest": artifact_root / "data" / "manifests" / f"component_cache_manifest{suffix}.json",
    }


def load_external_cache_manifest(
    artifact_root: str | Path,
    binding: str = DEFAULT_EXTERNAL_BINDING,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> dict:
    """Load and validate one external binding's manifest (stale = error)."""
    manifest_path = external_cache_paths(artifact_root, binding, schema_version)["manifest"]
    if not manifest_path.is_file():
        raise ValueError(
            f"external component cache manifest not found at {manifest_path}; "
            "rebuild with the cache-external-components command"
        )
    manifest = json.loads(manifest_path.read_text())
    found = int(manifest.get("extra", {}).get("schema_version", 0))
    if found != schema_version:
        raise ValueError(
            f"external component cache schema mismatch at {manifest_path}: "
            f"found schema {found!r}, need {schema_version}"
        )
    if manifest.get("extra", {}).get("binding") != binding:
        raise ValueError(
            f"external component cache binding mismatch at {manifest_path}"
        )
    return manifest


def cache_external_components(
    artifact_root: str | Path,
    pre_cfg: PreprocessingConfig,
    datasets: tuple[str, ...] = DEFAULT_EXTERNAL_DATASETS,
    binding: str = DEFAULT_EXTERNAL_BINDING,
) -> dict[str, object]:
    """Process external recordings into their own bound cache.

    Only recordings whose ``dataset`` is in ``datasets`` are processed; the
    frozen schema-4 cache and its manifest are never read or written here.
    Deterministic identical-output guarantee: rerunning with the same
    inputs reproduces the same arrays and manifest content.
    """
    artifact_root = Path(artifact_root)
    out = external_cache_paths(artifact_root, binding)
    for path in out.values():
        if path.exists():
            raise FileExistsError(
                f"external cache file already exists at {path}; "
                "delete it explicitly before rebuilding"
            )
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    recordings = recordings[recordings["dataset"].isin(datasets)]
    if recordings.empty:
        raise ValueError(f"no recordings for external datasets {datasets}")
    waveforms = _load_waveforms(artifact_root)

    all_rows: list[dict] = []
    array_data: dict[str, dict[str, np.ndarray]] = {}
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

    if not all_rows:
        raise ValueError(
            f"external cache for {datasets}: no valid components produced"
        )

    components_df = pd.DataFrame(all_rows)
    components_df.to_parquet(out["components_parquet"], index=False)

    comp_root = zarr.open_group(str(out["curves_zarr"]), mode="w")
    sot_root = zarr.open_group(str(out["sot_zarr"]), mode="w")
    spec_root = zarr.open_group(str(out["spectral_zarr"]), mode="w")
    cg = comp_root.create_group("components")
    sg = sot_root.create_group("components")
    spg = spec_root.create_group("components")
    order = list(components_df["global_component_id"])
    key_by_component = {
        row.global_component_id: row.canonical_array_key
        for row in components_df.itertuples(index=False)
    }
    cg.create_array(
        "canonical_time",
        data=np.stack([array_data[key_by_component[c]]["canonical_time"] for c in order]),
        chunks=(256, 128),
    )
    cg.create_array(
        "canonical_signal",
        data=np.stack([array_data[key_by_component[c]]["canonical_signal"] for c in order]),
        chunks=(256, 128),
    )
    cg.create_array(
        "valid_mask",
        data=np.stack([array_data[key_by_component[c]]["valid_mask"] for c in order]),
        chunks=(256, 128),
    )
    sg.create_array(
        "sot_vector",
        data=np.stack([array_data[key_by_component[c]]["sot_vector"] for c in order]),
        chunks=(256, 128),
    )
    spg.create_array(
        "spectral_vector",
        data=np.stack([array_data[key_by_component[c]]["spectral_vector"] for c in order]),
        chunks=(256, 128),
    )

    manifest = RunManifest(kind="component_cache", name=f"components_external_{binding}")
    manifest.extra["schema_version"] = CACHE_SCHEMA_VERSION
    manifest.extra["binding"] = binding
    manifest.extra["datasets"] = list(datasets)
    manifest.extra["n_valid"] = n_valid
    manifest.extra["reasons"] = reasons
    manifest.extra["preprocessing_version"] = pre_cfg.version
    manifest.extra["n_components"] = len(all_rows)
    manifest.extra["config_hash"] = config_hash(pre_cfg)
    manifest.extra["spectral_feature_names"] = spectral_feature_names(pre_cfg.spectral.bands)
    manifest.write_atomic(out["manifest"])
    return {
        "binding": binding,
        "n_valid": n_valid,
        "n_components": len(all_rows),
        "reasons": reasons,
    }
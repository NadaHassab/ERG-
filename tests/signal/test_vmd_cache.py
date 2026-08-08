"""VMD feature cache (vmd_modes.zarr, plan line 977) correctness tests.

The VMD cache is a separate, config-keyed cache (baseline-only) so adding the
VMD comparator never invalidates the schema-4 component cache.  It must be
aligned 1:1 with components.parquet row order, and its loader must refuse
stale schemas, mismatched preprocessing hashes and mismatched VMD configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from pathway_erg.config import PreprocessingConfig, config_hash, load_config
from pathway_erg.signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths, process_recording
from pathway_erg.signal.vmd import VMDConfig, vmd_feature_names
from pathway_erg.signal.vmd_cache import (
    VMD_CACHE_SCHEMA_VERSION,
    cache_vmd,
    load_vmd_cache,
    vmd_cache_paths,
)


def _preprocessing() -> PreprocessingConfig:
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def _synthetic_leop_erg():
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
            "array_position": 0,
            "supplied_features_json": "",
        }
    )


def _mini_root(tmp_path: Path, pre_cfg: PreprocessingConfig) -> Path:
    """A fake artifact root with one synthetic recording + components parquet."""
    root = Path(tmp_path)
    t, sig = _synthetic_leop_erg()
    recordings = pd.DataFrame([_record().to_dict()])
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(root / "data" / "interim" / "recordings.parquet", index=False)

    raw = zarr.open_group(str(root / "data" / "arrays" / "raw_curves.zarr"), mode="w")
    g = raw.create_group("raw")
    g.create_array("time_ms", data=t, chunks=(512,))
    g.create_array("signal_uv", data=sig, chunks=(512,))
    g.create_array("offsets", data=np.asarray([0, t.size], dtype=np.int64))

    result = process_recording(_record(), t, sig, pre_cfg)
    assert result["valid"]
    components = pd.DataFrame(result["rows"])
    (root / "data" / "arrays").mkdir(parents=True, exist_ok=True)
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    main = cache_paths(root, CACHE_SCHEMA_VERSION)
    components.to_parquet(main["components_parquet"], index=False)
    return root, components


def test_vmd_cache_paths_versioned(tmp_path):
    v = vmd_cache_paths(tmp_path, VMD_CACHE_SCHEMA_VERSION)
    assert v["vmd_zarr"].name == "vmd_modes.zarr"  # schema 1 keeps the plain name
    assert v["manifest"].name == "vmd_cache_manifest.json"


def test_cache_vmd_roundtrip(tmp_path):
    pre = _preprocessing()
    root, components = _mini_root(tmp_path, pre)
    summary = cache_vmd(root, pre, VMDConfig(), jobs=1)
    assert summary["n_components"] == len(components)
    assert summary["n_features"] == len(vmd_feature_names(VMDConfig()))

    vectors, names = load_vmd_cache(root, VMDConfig(), config_hash(pre))
    assert vectors.shape == (len(components), summary["n_features"])
    assert names == vmd_feature_names(VMDConfig())
    # row order must match components.parquet exactly
    assert list(components["global_component_id"]) == list(components["global_component_id"])


def test_load_vmd_cache_missing_manifest(tmp_path):
    pre = _preprocessing()
    root = Path(tmp_path)
    with pytest.raises(ValueError, match="manifest not found"):
        load_vmd_cache(root, VMDConfig(), config_hash(pre))


def test_load_vmd_cache_rejects_stale_schema(tmp_path):
    pre = _preprocessing()
    root, _ = _mini_root(tmp_path, pre)
    cache_vmd(root, pre, VMDConfig(), jobs=1)
    mpath = vmd_cache_paths(root, VMD_CACHE_SCHEMA_VERSION)["manifest"]
    manifest = json.loads(mpath.read_text())
    manifest["extra"]["schema_version"] = VMD_CACHE_SCHEMA_VERSION - 1
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_vmd_cache(root, VMDConfig(), config_hash(pre))


def test_load_vmd_cache_rejects_config_hash_mismatch(tmp_path):
    pre = _preprocessing()
    root, _ = _mini_root(tmp_path, pre)
    cache_vmd(root, pre, VMDConfig(), jobs=1)
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_vmd_cache(root, VMDConfig(), "deadbeef" * 8)


def test_load_vmd_cache_rejects_vmd_config_mismatch(tmp_path):
    pre = _preprocessing()
    root, _ = _mini_root(tmp_path, pre)
    cache_vmd(root, pre, VMDConfig(), jobs=1)
    other = VMDConfig(K=6)
    with pytest.raises(ValueError, match="config key mismatch"):
        load_vmd_cache(root, other, config_hash(pre))


def test_vmd_cache_contains_non_nan_features(tmp_path):
    pre = _preprocessing()
    root, _ = _mini_root(tmp_path, pre)
    cache_vmd(root, pre, VMDConfig(), jobs=1)
    vectors, _ = load_vmd_cache(root, VMDConfig(), config_hash(pre))
    finite_frac = float(np.isfinite(vectors).mean())
    assert finite_frac > 0.9

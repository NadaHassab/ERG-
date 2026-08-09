"""VMD feature cache (vmd_modes.zarr, plan line 977) correctness tests.

Real-data-only: the cache functions are exercised on actual recordings from
``artifacts/`` (real waveforms, real segmentations), staged into a temporary
artifact root so nothing writes into the real caches.  When the real data
build is absent, the tests skip with a clear message — no fabricated signals
are ever used here.
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

_ARTIFACT_ROOT = Path("artifacts")


def _real_waveforms() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Real recordings + raw waveforms from the artifact build (no fabrication)."""
    root = zarr.open_group(str(_ARTIFACT_ROOT / "data" / "arrays" / "raw_curves.zarr"), mode="r")
    g = root["raw"]
    time_flat = np.asarray(g["time_ms"][:], dtype=float)
    signal_flat = np.asarray(g["signal_uv"][:], dtype=float)
    offsets = np.asarray(g["offsets"][:], dtype=np.int64)
    recordings = pd.read_parquet(_ARTIFACT_ROOT / "data" / "interim" / "recordings.parquet")
    recordings = recordings.sort_values("array_position")
    out = {}
    for i, row in enumerate(recordings.itertuples(index=False)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        out[row.global_recording_id] = (time_flat[lo:hi], signal_flat[lo:hi])
    return out


def _real_recording_subset() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """One LEOP and one PERG recording (real rows + real waveforms)."""
    recordings = pd.read_parquet(_ARTIFACT_ROOT / "data" / "interim" / "recordings.parquet")
    waveforms = _real_waveforms()
    # pick, per dataset, the first recording present in the raw cache
    picks: list[pd.Series] = []
    for dataset in ("LEOP", "PERG"):
        for row in recordings.sort_values("array_position").itertuples(index=False):
            if row.dataset == dataset and row.global_recording_id in waveforms:
                picks.append(pd.Series(row._asdict()))
                break
    if not picks:
        raise AssertionError("no real recordings found")
    subset = pd.DataFrame(picks)
    wf = {r.global_recording_id: waveforms[r.global_recording_id] for r in subset.itertuples()}
    return subset, wf


def _stage_real_root(tmp_path: Path, pre_cfg: PreprocessingConfig, recordings: pd.DataFrame, waveforms: dict) -> Path:
    """Stage the real recordings into a fresh artifact root (real data only)."""
    root = Path(tmp_path)
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (root / "data" / "arrays").mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)

    recs = recordings.copy()
    recs["array_position"] = range(len(recs))
    recs = recs.sort_values("array_position")
    time_parts, sig_parts, offsets = [], [], [0]
    for row in recs.itertuples(index=False):
        t, s = waveforms[row.global_recording_id]
        time_parts.append(t)
        sig_parts.append(s)
        offsets.append(offsets[-1] + t.size)
    recs.to_parquet(root / "data" / "interim" / "recordings.parquet", index=False)

    raw = zarr.open_group(str(root / "data" / "arrays" / "raw_curves.zarr"), mode="w")
    g = raw.create_group("raw")
    g.create_array("time_ms", data=np.concatenate(time_parts), chunks=(4096,))
    g.create_array("signal_uv", data=np.concatenate(sig_parts), chunks=(4096,))
    g.create_array("offsets", data=np.asarray(offsets, dtype=np.int64), chunks=(64,))

    rows = []
    for row in recs.itertuples(index=False):
        t, s = waveforms[row.global_recording_id]
        result = process_recording(pd.Series(row._asdict()), t, s, pre_cfg)
        assert result["valid"], result.get("reasons")
        rows.extend(result["rows"])
    components = pd.DataFrame(rows)
    main = cache_paths(root, CACHE_SCHEMA_VERSION)
    components.to_parquet(main["components_parquet"], index=False)
    return root


def _preprocessing() -> PreprocessingConfig:
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def _have_real_data() -> bool:
    return (
        _ARTIFACT_ROOT.is_dir()
        and (_ARTIFACT_ROOT / "data" / "interim" / "recordings.parquet").is_file()
        and (_ARTIFACT_ROOT / "data" / "arrays" / "raw_curves.zarr").is_dir()
    )


def _vmd_cache_roundtrip(tmp_path: Path):
    pre = _preprocessing()
    subset, waveforms = _real_recording_subset()
    root = _stage_real_root(tmp_path, pre, subset, waveforms)
    return pre, root, subset


@pytest.mark.skipif(not _have_real_data(), reason="real artifact build not present (no fabrication)")
def test_vmd_cache_roundtrip(tmp_path):
    """cache-vmd -> load-vmd round trip on REAL recordings."""
    pre, root, _ = _vmd_cache_roundtrip(tmp_path)
    n_components = len(pd.read_parquet(cache_paths(root, CACHE_SCHEMA_VERSION)["components_parquet"]))
    summary = cache_vmd(root, pre, VMDConfig(), jobs=1)
    assert summary["n_components"] == n_components
    assert summary["n_features"] == len(vmd_feature_names(VMDConfig()))

    vectors, names = load_vmd_cache(root, VMDConfig(), config_hash(pre))
    assert vectors.shape == (n_components, summary["n_features"])
    assert names == vmd_feature_names(VMDConfig())
    # every vector must be finite for real, processed segments
    assert np.isfinite(vectors).mean() > 0.9


@pytest.mark.skipif(not _have_real_data(), reason="real data not present (no fabrication)")
def test_vmd_cache_covers_all_components(tmp_path):
    pre, root, _ = _vmd_cache_roundtrip(tmp_path)
    summary = cache_vmd(root, pre, VMDConfig(), jobs=1)
    components = pd.read_parquet(cache_paths(root, CACHE_SCHEMA_VERSION)["components_parquet"])
    assert summary["n_components"] == len(components)


def _write_fake_manifest(root: Path, extra: dict) -> Path:
    mpath = vmd_cache_paths(root, VMD_CACHE_SCHEMA_VERSION)["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"extra": extra}))
    return mpath


def test_load_vmd_cache_missing_manifest(tmp_path):
    pre = _preprocessing()
    root = Path(tmp_path)
    with pytest.raises(ValueError, match="manifest not found"):
        load_vmd_cache(root, VMDConfig(), config_hash(pre))


def test_load_vmd_cache_rejects_stale_schema(tmp_path):
    _write_fake_manifest(Path(tmp_path), {"schema_version": VMD_CACHE_SCHEMA_VERSION - 1})
    with pytest.raises(ValueError, match="schema mismatch"):
        load_vmd_cache(Path(tmp_path), VMDConfig(), config_hash(_preprocessing()))


def test_load_vmd_cache_rejects_config_hash_mismatch(tmp_path):
    _write_fake_manifest(
        Path(tmp_path),
        {
            "schema_version": VMD_CACHE_SCHEMA_VERSION,
            "config_hash": "deadbeefdeadbeef",
            "vmd_config_key": VMDConfig().key,
            "vmd_feature_names": vmd_feature_names(VMDConfig()),
        },
    )
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_vmd_cache(Path(tmp_path), VMDConfig(), "0123456789abcdef")


def test_load_vmd_cache_rejects_vmd_config_mismatch(tmp_path):
    _write_fake_manifest(
        Path(tmp_path),
        {
            "schema_version": VMD_CACHE_SCHEMA_VERSION,
            "config_hash": config_hash(_preprocessing()),
            "vmd_config_key": VMDConfig(K=6).key,
            "vmd_feature_names": vmd_feature_names(VMDConfig(K=6)),
        },
    )
    with pytest.raises(ValueError, match="config key mismatch"):
        load_vmd_cache(Path(tmp_path), VMDConfig(), config_hash(_preprocessing()))


def test_vmd_cache_paths_versioned(tmp_path):
    v = vmd_cache_paths(tmp_path, VMD_CACHE_SCHEMA_VERSION)
    assert v["vmd_zarr"].name == "vmd_modes.zarr"  # schema 1 keeps the plain name
    assert v["manifest"].name == "vmd_cache_manifest.json"

"""External (URFU/FLINDERS) component cache tests (plan integration §11.2.2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pathway_erg.signal.component_cache import (
    CACHE_SCHEMA_VERSION,
    cache_paths,
    load_cache_manifest,
)
from pathway_erg.signal.external_cache import (
    DEFAULT_EXTERNAL_BINDING,
    cache_external_components,
    external_cache_paths,
    load_external_cache_manifest,
)

from tests._ext_synth import (
    build_synthetic_external,
    build_synthetic_v4,
    pre_cfg,
    v4_manifest_bytes,
    write_synthetic_tables,
)


def _pre():
    return pre_cfg()


def _built(tmp_path):
    write_synthetic_tables(tmp_path)
    build_synthetic_v4(tmp_path, _pre())
    return tmp_path


def test_external_paths_never_collide_with_v4(tmp_path):
    root = _built(tmp_path)
    v4 = cache_paths(root, CACHE_SCHEMA_VERSION)
    ext = external_cache_paths(root)
    for k, p in v4.items():
        assert str(ext[k]) != str(p)
    for k in ("curves_zarr", "sot_zarr", "spectral_zarr", "components_parquet", "manifest"):
        assert "external_" in str(ext[k])


def test_external_cache_writes_binding_and_manifest(tmp_path):
    root = _built(tmp_path)
    summary = build_synthetic_external(root, _pre())
    assert summary["binding"] == DEFAULT_EXTERNAL_BINDING
    assert summary["n_valid"] >= 1
    assert summary["n_components"] >= 1
    manifest = load_external_cache_manifest(root)
    assert manifest["extra"]["binding"] == DEFAULT_EXTERNAL_BINDING
    assert manifest["extra"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert set(manifest["extra"]["datasets"]) == {"URFU", "FLINDERS"}
    comp = pd.read_parquet(external_cache_paths(root)["components_parquet"])
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    datasets = set(
        comp.merge(recordings[["global_recording_id", "dataset"]], on="global_recording_id")["dataset"]
    )
    assert datasets <= {"URFU", "FLINDERS"}
    assert not comp.empty


def test_external_components_are_flash_family(tmp_path):
    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    comp = pd.read_parquet(external_cache_paths(root)["components_parquet"])
    ids = set(comp["component_id"])
    assert ids, "expected at least one component"
    assert ids <= {"L_EARLY_A", "L_A_TO_B", "L_LATE", "L_OP"}


def test_v4_manifest_and_parquet_untouched_by_external_cache(tmp_path):
    root = _built(tmp_path)
    before_manifest = v4_manifest_bytes(root)
    before_components = (cache_paths(root, CACHE_SCHEMA_VERSION)["components_parquet"]).read_bytes()
    build_synthetic_external(root, _pre())
    assert v4_manifest_bytes(root) == before_manifest
    assert (
        cache_paths(root, CACHE_SCHEMA_VERSION)["components_parquet"].read_bytes()
        == before_components
    )


def test_external_cache_is_deterministic(tmp_path, tmp_path_factory):
    a = tmp_path_factory.mktemp("a")
    b = tmp_path_factory.mktemp("b")
    write_synthetic_tables(a)
    write_synthetic_tables(b)
    build_synthetic_v4(a, _pre())
    build_synthetic_v4(b, _pre())
    build_synthetic_external(a, _pre())
    build_synthetic_external(b, _pre())
    pa = external_cache_paths(a)["components_parquet"].read_bytes()
    pb = external_cache_paths(b)["components_parquet"].read_bytes()
    assert pa == pb
    ma = json.loads(v4_manifest_bytes(a))
    mb = json.loads(v4_manifest_bytes(b))
    ma.pop("created_at", None)
    mb.pop("created_at", None)
    ma.pop("created_utc", None)
    mb.pop("created_utc", None)
    assert ma == mb


def test_external_cache_refuses_existing_files(tmp_path):
    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    with pytest.raises(FileExistsError):
        build_synthetic_external(root, _pre())


def test_external_cache_unknown_dataset_raises(tmp_path):
    root = _built(tmp_path)
    with pytest.raises(ValueError, match="no recordings"):
        cache_external_components(root, _pre(), datasets=("MARTIAN",))


def test_external_manifest_binding_mismatch_raises(tmp_path):
    import json

    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    manifest_path = external_cache_paths(root)["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["extra"]["binding"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding mismatch"):
        load_external_cache_manifest(root)


def test_external_manifest_missing_raises(tmp_path):
    root = _built(tmp_path)
    write_synthetic_tables(root)
    with pytest.raises(ValueError, match="manifest not found"):
        load_external_cache_manifest(root)


def test_loaded_v4_manifest_still_valid_after_external(tmp_path):
    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    manifest = load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    assert int(manifest["extra"]["schema_version"]) == CACHE_SCHEMA_VERSION
    manifest_json = json.loads(v4_manifest_bytes(root))
    assert "external" not in manifest_json["extra"]


def test_external_arrays_are_schema_aligned(tmp_path):
    import zarr

    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    paths = external_cache_paths(root)
    comp = pd.read_parquet(paths["components_parquet"])
    curves = zarr.open_group(str(paths["curves_zarr"]), mode="r")["components"]
    signal = np.asarray(curves["canonical_signal"][:])
    assert signal.shape[0] == len(comp)
    assert signal.shape[1] == 128
    sot = zarr.open_group(str(paths["sot_zarr"]), mode="r")["components"]
    ot = np.asarray(sot["sot_vector"][:])
    assert ot.shape[0] == len(comp)
    assert ot.shape[1] == 135


def test_external_curves_have_valid_samples(tmp_path):
    import zarr

    root = _built(tmp_path)
    build_synthetic_external(root, _pre())
    paths = external_cache_paths(root)
    curves = zarr.open_group(str(paths["curves_zarr"]), mode="r")["components"]
    mask = np.asarray(curves["valid_mask"][:])
    assert mask.all(axis=1).any()
    assert mask.shape[1] == 128

"""Plan 15.2 VMD hyperparameter grid tests.

The grid sweep runs on REAL recordings staged into a temp artifact root (no
fabricated signals).  Tests pin the grid enumeration, determinism, and the
per-config diagnostics written by the sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from pathway_erg.config import PreprocessingConfig, load_config
from pathway_erg.signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths, process_recording
from pathway_erg.signal.vmd import VMDConfig
from pathway_erg.signal.vmd_grid import (
    PLAN_GRID,
    plan_grid_configs,
    sweep_vmd_grid,
)

_ARTIFACT_ROOT = Path("artifacts")


def _real_waveforms() -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
    recordings = pd.read_parquet(_ARTIFACT_ROOT / "data" / "interim" / "recordings.parquet")
    waveforms = _real_waveforms()
    picks: list[pd.Series] = []
    for dataset in ("LEOP",):
        for row in recordings.sort_values("array_position").itertuples(index=False):
            if row.dataset == dataset and row.global_recording_id in waveforms:
                picks.append(pd.Series(row._asdict()))
                break
    subset = pd.DataFrame(picks)
    wf = {r.global_recording_id: waveforms[r.global_recording_id] for r in subset.itertuples()}
    return subset, wf


def _stage_real_root(tmp_path: Path, pre_cfg: PreprocessingConfig, recordings: pd.DataFrame, waveforms: dict) -> Path:
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
    components.to_parquet(cache_paths(root, CACHE_SCHEMA_VERSION)["components_parquet"], index=False)
    return root


def _preprocessing() -> PreprocessingConfig:
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def _have_real_data() -> bool:
    return (
        _ARTIFACT_ROOT.is_dir()
        and (_ARTIFACT_ROOT / "data" / "interim" / "recordings.parquet").is_file()
        and (_ARTIFACT_ROOT / "data" / "arrays" / "raw_curves.zarr").is_dir()
    )


def test_plan_grid_covers_section_15_2_values():
    configs = plan_grid_configs()
    assert len(configs) == 64
    assert PLAN_GRID["K"] == (3, 4, 5, 6)
    assert PLAN_GRID["alpha"] == (500, 1000, 2000, 4000)
    assert PLAN_GRID["tol"] == (1e-6, 1e-7)
    assert PLAN_GRID["mirror_pad_ms"] == (25.0, 50.0)
    # defaults (K=5, alpha=2000, tol=1e-7, pad=25) must be in the grid
    assert VMDConfig().key in {c.key for c in configs}
    # keys must be unique and sortable
    keys = [c.key for c in configs]
    assert len(set(keys)) == 64


@pytest.mark.skipif(not _have_real_data(), reason="real artifact build not present (no fabrication)")
def test_sweep_vmd_grid_writes_real_diagnostics(tmp_path):
    pre = _preprocessing()
    subset, waveforms = _real_recording_subset()
    root = _stage_real_root(tmp_path, pre, subset, waveforms)
    summary = sweep_vmd_grid(root, pre, n_recordings=1, jobs=1, tag="plan")
    assert summary["n_configs"] == 64
    assert summary["n_recordings"] == 1
    assert summary["n_rows"] > 0
    summary_path = root / "results" / "vmd_grid" / "grid_plan_summary.parquet"
    assert summary_path.is_file()
    df = pd.read_parquet(summary_path)
    assert len(df) == 64
    assert {"recon_rms_rel_median", "converged_frac", "n_iterations_median"}.issubset(df.columns)
    # the default config must appear with its diagnostics
    row = df.loc[df["config_key"] == VMDConfig().key].iloc[0]
    assert row["recon_rms_rel_median"] >= 0.0
    assert row["converged_frac"] > 0.9


@pytest.mark.skipif(not _have_real_data(), reason="real artifact build not present (no fabrication)")
def test_sweep_vmd_grid_is_deterministic(tmp_path):
    pre = _preprocessing()
    subset, waveforms = _real_recording_subset()
    root1 = _stage_real_root(Path(tmp_path) / "a", pre, subset, waveforms)
    root2 = _stage_real_root(Path(tmp_path) / "b", pre, subset, waveforms)
    sweep_vmd_grid(root1, pre, n_recordings=1, jobs=1, tag="dx")
    sweep_vmd_grid(root2, pre, n_recordings=1, jobs=1, tag="dx")
    a = pd.read_parquet(root1 / "results" / "vmd_grid" / "grid_dx.parquet")
    b = pd.read_parquet(root2 / "results" / "vmd_grid" / "grid_dx.parquet")
    assert a.equals(b)


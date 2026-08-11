"""Synthetic external-path fixtures for URFU/FLINDERS tests.

Builds a tiny but structurally faithful artifact tree in ``tmp_path``:
participants/visits/recordings parquet tables, a ``raw_curves.zarr`` wave
store, the frozen v4 component cache, the external (URFU/FLINDERS) bound
cache, and the external subject-keyed splits — enough to exercise
``LoadedCaches``, ``build_bags``, ``pretrain_ssl`` and the routers without
the real artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from pathway_erg.config import PreprocessingConfig, load_config
from pathway_erg.data.external_splits import build_external_splits
from pathway_erg.data.splits import FoldConfig
from pathway_erg.signal.component_cache import CACHE_SCHEMA_VERSION, cache_components, cache_paths
from pathway_erg.signal.external_cache import cache_external_components

T0 = 0.0
DT = 0.5  # 2000 Hz


def pre_cfg() -> PreprocessingConfig:
    """Reference preprocessing config (landmarks/segmentation from YAML)."""
    return load_config(PreprocessingConfig, "configs/preprocessing/reference.yaml")


def flash_waveform(seed: int = 0, a_center: float = 12.0, b_center: float = 28.0) -> np.ndarray:
    """Clean full-field flash ERG: a-wave trough + b-wave peak."""
    t = np.arange(201) * DT
    rng = np.random.default_rng(seed)
    sig = (
        -3.0 * np.exp(-0.5 * ((t - a_center) / 2.0) ** 2)
        + 12.0 * np.exp(-0.5 * ((t - b_center) / 3.0) ** 2)
        + rng.normal(0, 0.05, t.size)
    )
    return sig


def perg_waveform(seed: int = 1) -> np.ndarray:
    """Clean pattern ERG: n35/p50/n95 morphology."""
    t = np.arange(201) * DT
    rng = np.random.default_rng(seed)
    sig = (
        -1.5 * np.exp(-0.5 * ((t - 30.0) / 3.0) ** 2)
        + 3.0 * np.exp(-0.5 * ((t - 55.0) / 4.0) ** 2)
        - 2.0 * np.exp(-0.5 * ((t - 95.0) / 5.0) ** 2)
        + rng.normal(0, 0.05, t.size)
    )
    return sig


def write_synthetic_tables(
    root: Path,
    n_seed_offset: int = 0,
) -> None:
    """Participants/visits/recordings + raw_curves.zarr for a fixed layout.

    Layout: 2 LEOP subjects, 3 PERG subjects/visits, 4 URFU subjects
    (one with two visits => 5 visits, 6 recordings), 2 FLINDERS subjects.
    """
    root = Path(root)
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (root / "data" / "arrays").mkdir(parents=True, exist_ok=True)

    rows = []
    arrays = []
    k = n_seed_offset

    def add(ds, sid, vid, protocol, eye, kind, wave, target=None, age=30.0, sex="M"):
        nonlocal k
        rid = f"{ds}_R{k}"
        find = f"{ds}_F{k}"
        pairs = [f"{ds}_P{k}"]
        rows.append(
            dict(
                global_recording_id=rid,
                global_subject_id=sid,
                global_visit_id=vid,
                dataset=ds,
                protocol=protocol,
                eye=eye,
                stimulus_value=3.0 if ds != "PERG" else 1.0,
                stimulus_unit="cd.s/m2" if ds != "PERG" else "degree",
                waveform_kind=kind,
                median_dt_ms=DT,
                array_position=k,
                array_key=rid,
                source_wave_id=find,
                erg_pair_id=pairs[0],
                target_binary=target,
                supplied_features_json=None,
            )
        )
        arrays.append((np.arange(201) * DT, wave(k)))
        k += 1

    for i in range(2):
        add("LEOP", f"L{i:02d}", f"L{i:02d}-V", "9_step", "RE", "ERG", flash_waveform, target=0)
    for i in range(3):
        add("PERG", f"P{i:02d}", f"P{i:02d}-V{i}", "PERG", "RE", "PERG_EYE", perg_waveform, target=i % 2)
    for i in range(4):
        add("URFU", f"U{i:02d}", f"U{i:02d}-V", "Maximum 2.0", None, "ERG", flash_waveform, target=0)
    add("URFU", "U03", "U03-V2", "Scotopic 2.0", None, "ERG", flash_waveform, target=0)
    for i in range(2):
        add("FLINDERS", f"F{i:02d}", f"F{i:02d}-V", "LA3", None, "ERG", flash_waveform, target=0)

    recordings = pd.DataFrame(rows)
    recordings.to_parquet(root / "data" / "interim" / "recordings.parquet", index=False)

    participants = pd.DataFrame(
        {
            "global_subject_id": sorted(set(recordings["global_subject_id"])),
            "dataset": (
                recordings[["global_subject_id", "dataset"]]
                .drop_duplicates()
                .set_index("global_subject_id")
                .reindex(sorted(set(recordings["global_subject_id"])))["dataset"]
                .tolist()
            ),
        }
    )
    participants["group_raw"] = ["Control" if d == "LEOP" else None for d in participants["dataset"]]
    participants["age_years"] = 30.0
    participants["sex_standardized"] = "M"
    participants["site"] = "synthetic"
    participants.to_parquet(root / "data" / "interim" / "participants.parquet", index=False)

    visits = recordings[["global_visit_id", "global_subject_id", "dataset", "target_binary"]].drop_duplicates()
    visits.to_parquet(root / "data" / "interim" / "visits.parquet", index=False)

    z = zarr.open_group(str(root / "data" / "arrays" / "raw_curves.zarr"), mode="w")
    g = z.create_group("raw")
    time_flat = np.concatenate([t for t, _ in arrays])
    signal_flat = np.concatenate([s for _, s in arrays])
    offsets = np.cumsum([0] + [len(t) for t, _ in arrays]).astype(np.int64)
    g.create_array("time_ms", data=time_flat, chunks=(4096,))
    g.create_array("signal_uv", data=signal_flat, chunks=(4096,))
    g.create_array("offsets", data=offsets, chunks=(4096,))


def build_synthetic_v4(root: Path, pre_cfg: PreprocessingConfig) -> dict:
    return cache_components(root, pre_cfg)


def build_synthetic_external(
    root: Path,
    pre_cfg: PreprocessingConfig,
    datasets: tuple[str, ...] = ("URFU", "FLINDERS"),
    binding: str = "external_v1",
) -> dict:
    return cache_external_components(root, pre_cfg, datasets=datasets, binding=binding)


def build_synthetic_external_splits(
    root: Path, fold_cfg: FoldConfig, datasets: tuple[str, ...] = ("URFU", "FLINDERS")
):
    subjects = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    return build_external_splits(root, subjects, visits, recordings, fold_cfg, datasets=datasets)


def build_synthetic_v1_splits(root: Path, fold_cfg: FoldConfig | None = None) -> dict[str, Path]:
    """LEOP/PERG subject-keyed folds under the v1 template (frozen tables)."""
    from pathway_erg.data.splits import (
        make_inner_folds,
        make_outer_folds,
        summarize_folds,
        write_splits,
    )

    if fold_cfg is None:
        fold_cfg = FoldConfig(
            n_outer=3,
            n_inner=2,
            outer_seed=2026,
            inner_seed=2027,
            version="v1",
            age_bins=(10.0, 18.0, 35.0, 55.0),
            constraints=(),
        )
    subjects = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    outer = make_outer_folds(subjects, visits, fold_cfg)
    inner_by_fold = {
        k: make_inner_folds(outer, subjects, visits, k, fold_cfg)
        for k in range(fold_cfg.n_outer)
    }
    report = summarize_folds(outer, subjects, visits, fold_cfg.age_bins)
    from pathway_erg.data.splits import assert_no_leakage

    assert_no_leakage(outer, subjects, visits, recordings)
    return write_splits(outer, inner_by_fold, report, root, fold_cfg.version)


def v4_manifest_bytes(root: Path) -> bytes:
    return cache_paths(root, CACHE_SCHEMA_VERSION)["manifest"].read_bytes()


def external_fold_config() -> FoldConfig:
    return FoldConfig(
        n_outer=3,
        n_inner=2,
        outer_seed=2026,
        inner_seed=2027,
        version="external_v1",
        age_bins=(10.0, 18.0, 35.0, 55.0),
        constraints=(),
    )


def make_external_fold_constraints() -> FoldConfig:
    from pathway_erg.data.splits import FoldConstraint

    base = external_fold_config()
    from dataclasses import replace

    return replace(
        base,
        constraints=(
            FoldConstraint(column="class", weight=2.0),
            FoldConstraint(column="sex", weight=1.0),
            FoldConstraint(column="age_bin", weight=1.0),
        ),
    )
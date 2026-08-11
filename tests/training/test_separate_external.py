"""Supervised-endpoint gates for the external datasets (plan integration §11.2)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from pathway_erg.config import DataConfig, load_config
from pathway_erg.data.external_splits import build_external_splits
from pathway_erg.data.urfu_labels import (
    URFU_MAPPING_VERSION,
    build_urfu_mapping,
    require_urfu_labels_signed_off,
)

from tests._ext_synth import (
    build_synthetic_external,
    build_synthetic_v1_splits,
    build_synthetic_v4,
    external_fold_config,
    pre_cfg,
    write_synthetic_tables,
)


@pytest.fixture()
def ext_root(tmp_path):
    write_synthetic_tables(tmp_path)
    pre = pre_cfg()
    build_synthetic_v4(tmp_path, pre)
    build_synthetic_external(tmp_path, pre)
    build_synthetic_v1_splits(tmp_path)
    subjects = pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet")
    build_external_splits(tmp_path, subjects, visits, recordings, external_fold_config())
    return tmp_path


def _data_cfg(root) -> DataConfig:
    cfg = load_config(DataConfig, "configs/data/local.yaml")
    return dataclasses.replace(cfg, artifact_root=str(root))


def _caches(root):
    from pathway_erg.data.datasets import LoadedCaches

    return LoadedCaches(
        root,
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )


def test_urfu_labels_gate_blocks_pending_review():
    with pytest.raises(ValueError, match="PENDING_CLINICAL_REVIEW"):
        require_urfu_labels_signed_off()


def test_urfu_labels_gate_wrong_version_blocks():
    with pytest.raises(ValueError, match="does not match"):
        require_urfu_labels_signed_off(version="not_a_version")


def test_build_task_bags_flinders_forbidden(ext_root):
    from pathway_erg.training.separate import build_task_bags

    with pytest.raises(ValueError, match="FLINDERS supervised head is forbidden"):
        build_task_bags(_caches(ext_root), "FLINDERS")


def test_build_task_bags_unknown_task(ext_root):
    from pathway_erg.training.separate import build_task_bags

    with pytest.raises(ValueError, match="unknown task"):
        build_task_bags(_caches(ext_root), "MARTIAN")


def test_build_task_bags_urfu_blocked_until_signoff(ext_root):
    from pathway_erg.training.separate import build_task_bags

    with pytest.raises(ValueError, match="PENDING_CLINICAL_REVIEW"):
        build_task_bags(_caches(ext_root), "URFU")


def test_separate_config_urfu_task_raises_loudly(ext_root):
    from pathway_erg.training.separate import SeparateTrainingConfig, run_separate_training

    cfg = SeparateTrainingConfig(
        name="x",
        tasks=("URFU",),
        outer_folds=(0,),
        seeds=(1,),
        device="cpu",
        output_subdir="never_written",
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )
    with pytest.raises(ValueError, match="PENDING_CLINICAL_REVIEW"):
        run_separate_training(cfg, _data_cfg(ext_root))


def test_separate_config_flinders_task_raises_loudly(ext_root):
    from pathway_erg.training.separate import SeparateTrainingConfig, run_separate_training

    cfg = SeparateTrainingConfig(
        name="x",
        tasks=("FLINDERS",),
        outer_folds=(0,),
        seeds=(1,),
        device="cpu",
        output_subdir="never_written",
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )
    with pytest.raises(ValueError, match="FLINDERS"):
        run_separate_training(cfg, _data_cfg(ext_root))


def test_urfu_sanity_endpoint_mapping_reviewer_pending():
    mapping = build_urfu_mapping(pd.Series(dtype=str))
    assert mapping.version == URFU_MAPPING_VERSION
    assert mapping.reviewer == "PENDING_CLINICAL_REVIEW"

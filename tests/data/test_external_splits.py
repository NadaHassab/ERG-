"""External split construction + leakage tests (plan integration §11.2.3)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pathway_erg.data.external_splits import (
    EXTERNAL_FOLD_VERSION,
    build_external_splits,
    external_unit_frame,
    make_external_outer_folds,
)
from pathway_erg.data.splits import assert_no_leakage

from tests._ext_synth import (
    external_fold_config,
    make_external_fold_constraints,
    write_synthetic_tables,
)


@pytest.fixture()
def frames(tmp_path):
    write_synthetic_tables(tmp_path)
    return (
        pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet"),
        pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet"),
        pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet"),
    )


def test_external_unit_frame_urfu_class_from_visits(frames):
    subjects, visits, _ = frames
    units = external_unit_frame(subjects, visits, "URFU", (10.0, 18.0))
    assert set(units["unit_id"]) == {f"U{i:02d}" for i in range(4)}
    assert units["unit_id"].duplicated().sum() == 0
    assert units["class"].notna().all()


def test_external_unit_frame_flinders_healthy(frames):
    subjects, visits, _ = frames
    units = external_unit_frame(subjects, visits, "FLINDERS", (10.0, 18.0))
    assert (units["class"] == 0).all()
    assert units["n_visits"].tolist() == [1, 1]


def test_external_unit_frame_rejects_other_datasets(frames):
    subjects, visits, _ = frames
    with pytest.raises(ValueError, match="external datasets"):
        external_unit_frame(subjects, visits, "LEOP", (10.0, 18.0))


def test_external_outer_folds_subject_keyed_and_unique(frames):
    subjects, visits, _ = frames
    outer = make_external_outer_folds(subjects, visits, make_external_fold_constraints())
    assert set(outer["dataset"]) == {"URFU", "FLINDERS"}
    for dataset in ("URFU", "FLINDERS"):
        sub = outer[outer["dataset"] == dataset]
        assert sub["unit_id"].duplicated().sum() == 0
        assert sub["unit_id"].isin(subjects["global_subject_id"]).all()


def test_urfu_visits_stay_with_subject(frames):
    subjects, visits, _ = frames
    outer = make_external_outer_folds(subjects, visits, external_fold_config())
    assert len(outer) == 6  # 4 URFU + 2 FLINDERS subjects
    u03 = outer[(outer["dataset"] == "URFU") & (outer["unit_id"] == "U03")]
    assert len(u03) == 1
    subject_fold = int(u03["outer_fold"].iloc[0])
    v = visits[(visits["dataset"] == "URFU") & (visits["global_subject_id"] == "U03")]
    assert len(v) == 2
    assert_no_leakage(outer, subjects, visits, _)  # no cross-fold pair for U03


def test_external_leakage_detects_cross_fold_subject(frames):
    subjects, visits, recordings = frames
    outer = make_external_outer_folds(subjects, visits, external_fold_config())
    broken = outer.copy()
    row = broken[(broken["dataset"] == "URFU") & (broken["unit_id"] == "U03")]
    fold = int(row["outer_fold"].iloc[0])
    dup = row.copy()
    dup["outer_fold"] = (fold + 1) % 3
    broken = pd.concat([broken, dup], ignore_index=True)
    with pytest.raises(AssertionError, match="in multiple outer folds"):
        assert_no_leakage(broken, subjects, visits, recordings)


def test_external_leakage_detects_missing_visit_assignment(frames):
    subjects, visits, recordings = frames
    outer = make_external_outer_folds(subjects, visits, external_fold_config())
    dropped = outer[outer["unit_id"] != "U03"].copy()
    with pytest.raises(AssertionError, match="lack an outer-fold assignment"):
        assert_no_leakage(dropped, subjects, visits, recordings)


def test_external_leakage_passes_for_clean_split(frames):
    subjects, visits, recordings = frames
    outer = make_external_outer_folds(subjects, visits, make_external_fold_constraints())
    assert_no_leakage(outer, subjects, visits, recordings)


def test_build_external_splits_writes_locked_tables(tmp_path):
    write_synthetic_tables(tmp_path)
    subjects = pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet")
    result = build_external_splits(
        tmp_path, subjects, visits, recordings, make_external_fold_constraints()
    )
    assert result.version == EXTERNAL_FOLD_VERSION
    assert result.paths["outer"].is_file()
    assert result.paths["summary"].is_file()
    outer = pd.read_parquet(result.paths["outer"])
    assert set(outer["dataset"]) == {"URFU", "FLINDERS"}
    assert "URFU" in result.report and "FLINDERS" in result.report
    summary = json.loads(result.paths["summary"].read_text())
    assert summary["version"] == EXTERNAL_FOLD_VERSION


def test_build_external_splits_deterministic_hash(tmp_path, tmp_path_factory):
    from dataclasses import replace

    cfg = make_external_fold_constraints()
    hashes = []
    for d in (tmp_path, tmp_path_factory.mktemp("b")):
        write_synthetic_tables(d)
        subjects = pd.read_parquet(d / "data" / "interim" / "participants.parquet")
        visits = pd.read_parquet(d / "data" / "interim" / "visits.parquet")
        recordings = pd.read_parquet(d / "data" / "interim" / "recordings.parquet")
        r = build_external_splits(d, subjects, visits, recordings, cfg)
        hashes.append(json.loads(r.paths["summary"].read_text())["split_hash"])
    assert hashes[0] == hashes[1]
    _ = replace


def test_build_external_splits_empty_subject_error(tmp_path):
    subjects = pd.DataFrame(
        columns=["global_subject_id", "dataset", "age_years", "sex_standardized"]
    )
    visits = pd.DataFrame(columns=["global_visit_id", "global_subject_id", "dataset", "target_binary"])
    recordings = pd.DataFrame(columns=["global_recording_id", "global_subject_id", "global_visit_id", "dataset"])
    with pytest.raises(ValueError, match="no external subjects"):
        build_external_splits(tmp_path, subjects, visits, recordings, external_fold_config())


def test_external_inner_folds_nested_in_outer_train(tmp_path):
    write_synthetic_tables(tmp_path)
    subjects = pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet")
    cfg = make_external_fold_constraints()
    result = build_external_splits(tmp_path, subjects, visits, recordings, cfg)
    inner_all = pd.read_parquet(result.paths["inner"])
    for outer_fold in sorted(result.outer["outer_fold"].unique()):
        inner = inner_all[inner_all["outer_fold_sel"] == outer_fold]
        outer_train_units = set(
            result.outer[
                (result.outer["outer_fold"] != outer_fold) & (result.outer["dataset"] == "URFU")
            ]["unit_id"]
        )
        assert set(inner[inner["dataset"] == "URFU"]["unit_id"]) == outer_train_units

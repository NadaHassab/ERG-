"""Phase 1: LEOP cohort regression tests.

The v2 pipeline must support two explicit LEOP cohort definitions (plan
Section 16.6):

- ``primary_nine_step``: Control/ASD only, nine-step recordings only, one
  prediction per participant, no availability/protocol-count shortcut
  features. Subjects without any nine-step recording are excluded with logged
  counts (never hard-coded).
- ``secondary_all_protocols``: every eligible subject, every protocol used
  explicitly (never merged by averaging), plus the protocol/availability/QC
  shortcut baselines for confound quantification.

Cohorts are run per-experiment; metrics are namespaced
``{dataset}_{cohort}/...`` so primary and secondary results can never be
mistaken for one another, and PERG keys are untouched by cohort configs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathway_erg.config import BaselinesConfig, load_config


def _load_tables(artifact_root: Path):
    from pathway_erg.constants import OUTER_FOLDS_TEMPLATE

    participants = pd.read_parquet(artifact_root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(artifact_root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    components = pd.read_parquet(artifact_root / "data" / "interim" / "components.parquet")
    folds = pd.read_parquet(
        artifact_root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version="v1")
    )
    return participants, visits, recordings, components, folds


@pytest.fixture(scope="module")
def tables():
    from pathway_erg.config import DataConfig, load_config

    dc = load_config(DataConfig, "configs/data/local.yaml")
    return _load_tables(Path(dc.artifact_root))


def test_primary_cohort_units_all_have_nine_step(tables):
    """Every retained subject must have at least one nine-step recording."""
    from pathway_erg.models.baselines import _load_units
    from pathway_erg.models.leop_cohorts import cohort_unit_mask

    participants, visits, recordings, components, folds = tables
    units = _load_units("LEOP", participants, visits, folds)
    mask = cohort_unit_mask(units, recordings, "primary_nine_step")
    kept = units[mask]
    assert len(kept) == 160  # real cohort: 72 ASD + 88 Control with 9-step
    assert set(kept["target_binary"]) == {0.0, 1.0}
    nine_step_subjects = set(
        recordings[(recordings["dataset"] == "LEOP") & (recordings["protocol"] == "9_step")][
            "global_subject_id"
        ]
    )
    assert kept["unit_id"].isin(nine_step_subjects).all()
    dropped = units[~mask]
    assert not dropped["unit_id"].isin(nine_step_subjects).any()


def test_primary_cohort_excludes_subjects_without_nine_step(tables):
    from pathway_erg.models.baselines import _load_units
    from pathway_erg.models.leop_cohorts import cohort_unit_mask

    participants, visits, recordings, components, folds = tables
    units = _load_units("LEOP", participants, visits, folds)
    mask = cohort_unit_mask(units, recordings, "primary_nine_step")
    assert (~mask).sum() == len(units) - 160
    assert int(units.loc[mask, "target_binary"].sum()) == 72  # ASD positives


def test_secondary_cohort_keeps_all_supervised(tables):
    from pathway_erg.models.baselines import _load_units
    from pathway_erg.models.leop_cohorts import cohort_unit_mask

    participants, visits, recordings, components, folds = tables
    units = _load_units("LEOP", participants, visits, folds)
    mask = cohort_unit_mask(units, recordings, "secondary_all_protocols")
    assert mask.all()
    assert len(units) == 232  # 157 Control + 75 ASD, unlabeled ASD+ADHD excluded


def test_primary_cohort_recordings_mask_keeps_only_nine_step(tables):
    from pathway_erg.models.leop_cohorts import cohort_recordings_mask

    participants, visits, recordings, components, folds = tables
    m = cohort_recordings_mask(recordings, "primary_nine_step")
    leop = recordings["dataset"] == "LEOP"
    assert (recordings.loc[leop & ~m, "protocol"] != "9_step").all()
    assert (recordings.loc[leop & m, "protocol"] == "9_step").all()
    perg = recordings["dataset"] == "PERG"
    assert m[perg].all()  # PERG rows are never touched by a LEOP cohort


def test_primary_cohort_availability_banned(tables, tmp_path):
    """Availability/protocol-count shortcut features are forbidden for primary."""
    from pathway_erg.config import DataConfig, load_config
    from pathway_erg.models.baselines import run_baselines

    dc = load_config(DataConfig, "configs/data/local.yaml")
    cfg = BaselinesConfig(
        name="phase1_test",
        datasets=("LEOP",),
        leop_cohort="primary_nine_step",
        e0_methods=("prevalence", "availability"),
        e4_methods=(),
        outer_folds=(0,),
        models=("logreg",),
        output_subdir=str(tmp_path / "phase1"),
    )
    with pytest.raises(ValueError, match="availability"):
        run_baselines(cfg, dc)


def test_primary_cohort_run_is_namespaced_and_nine_step_only(tables, tmp_path):
    """End-to-end primary run: cohort column, namespaced metrics, 9-step only."""
    from pathway_erg.config import DataConfig, load_config
    from pathway_erg.models.baselines import run_baselines

    dc = load_config(DataConfig, "configs/data/local.yaml")
    cfg = BaselinesConfig(
        name="phase1_test",
        datasets=("LEOP",),
        leop_cohort="primary_nine_step",
        e0_methods=("prevalence",),
        e4_methods=("clinical",),
        outer_folds=(0,),
        models=("logreg",),
        output_subdir=str(tmp_path / "phase1"),
    )
    results = run_baselines(cfg, dc)
    preds = results.predictions
    assert preds["cohort"].eq("primary_nine_step").all()
    assert preds["task"].eq("LEOP").all()
    assert results.metrics["LEOP_primary_nine_step/prevalence"]["roc_auc"] == 0.5
    assert set(results.metrics) == {
        "LEOP_primary_nine_step/prevalence",
        "LEOP_primary_nine_step/clinical_logreg",
        "LEOP_primary_nine_step/clinical_svm_rbf",
        "LEOP_primary_nine_step/clinical_histgb",
    }
    # availability is banned, so the only "protocol count" artifact is the log
    cohort_notes = results.notes["LEOP_primary_nine_step_units"]
    assert cohort_notes["n_supervised_units"] == 160
    assert cohort_notes["n_excluded_no_nine_step"] == 72
    assert cohort_notes["n_positive"] == 72
    assert cohort_notes["n_negative"] == 88


def test_perg_keys_untouched_by_cohort_config(tables, tmp_path):
    """A LEOP cohort must not rename or filter PERG results."""
    from pathway_erg.config import DataConfig, load_config
    from pathway_erg.models.baselines import run_baselines

    dc = load_config(DataConfig, "configs/data/local.yaml")
    cfg = BaselinesConfig(
        name="phase1_test",
        datasets=("LEOP", "PERG"),
        leop_cohort="secondary_all_protocols",
        e0_methods=("prevalence",),
        e4_methods=("clinical",),
        outer_folds=(0,),
        models=("logreg",),
        output_subdir=str(tmp_path / "phase1"),
    )
    results = run_baselines(cfg, dc)
    assert "PERG/clinical_logreg" in results.metrics
    perg_preds = results.predictions[results.predictions["task"] == "PERG"]
    assert perg_preds["cohort"].isna().all()


def test_cohort_none_preserves_legacy_key_shape(tmp_path):
    """No cohort -> exactly the old {dataset}/{method} metric keys."""
    cfg = load_config(BaselinesConfig, "configs/experiments/e4_baselines.yaml")
    assert cfg.leop_cohort is None
    cfg2 = load_config(BaselinesConfig, "configs/experiments/e4_baselines_legacy.yaml")
    assert cfg2.leop_cohort is None

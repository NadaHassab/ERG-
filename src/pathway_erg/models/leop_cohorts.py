"""LEOP cohort definitions for the v2 pipeline (plan Section 16.6).

Two explicit cohort experiments:

- ``primary_nine_step``: Control/ASD only, nine-step recordings only, one
  prediction per participant, no availability/protocol-count shortcut
  features. Subjects without any nine-step recording are excluded and the
  exclusion count is logged (never hard-coded).
- ``secondary_all_protocols``: every eligible subject, every protocol used
  explicitly (never merged by averaging), plus the protocol/availability/QC
  shortcut baselines for confound quantification.

Cohorts never touch PERG rows: masks are always LEOP-scoped.
"""

from __future__ import annotations

import pandas as pd

LEOP_COHORTS: tuple[str, ...] = ("primary_nine_step", "secondary_all_protocols")


def _validate_cohort(cohort: str | None) -> None:
    if cohort is not None and cohort not in LEOP_COHORTS:
        raise ValueError(f"unknown LEOP cohort {cohort!r}; expected one of {LEOP_COHORTS}")


def cohort_unit_mask(
    units: pd.DataFrame, recordings: pd.DataFrame, cohort: str | None
) -> pd.Series:
    """Boolean mask over `units` rows retained by the cohort.

    `units` is the supervised-unit table from ``_load_units``; the mask is
    derived from the recordings table so no hard-coded counts exist.
    """
    _validate_cohort(cohort)
    if cohort is None or cohort == "secondary_all_protocols":
        return pd.Series(True, index=units.index)
    if cohort == "primary_nine_step":
        nine_step_subjects = set(
            recordings[
                (recordings["dataset"] == "LEOP") & (recordings["protocol"] == "9_step")
            ]["global_subject_id"]
        )
        return units["unit_id"].isin(nine_step_subjects)
    raise AssertionError("unreachable")


def cohort_recordings_mask(
    recordings: pd.DataFrame, cohort: str | None
) -> pd.Series:
    """Boolean mask over recordings rows usable by the cohort (LEOP-scoped)."""
    _validate_cohort(cohort)
    if cohort is None or cohort == "secondary_all_protocols":
        return pd.Series(True, index=recordings.index)
    if cohort == "primary_nine_step":
        is_leop = recordings["dataset"] == "LEOP"
        return (~is_leop) | (recordings["protocol"] == "9_step")
    raise AssertionError("unreachable")


def cohort_component_mask(
    components: pd.DataFrame, recordings: pd.DataFrame, cohort: str | None
) -> pd.Series:
    """Boolean mask over components rows belonging to cohort recordings."""
    rec_mask = cohort_recordings_mask(recordings, cohort)
    kept = set(recordings.loc[rec_mask, "global_recording_id"])
    return components["global_recording_id"].isin(kept)


def cohort_protocol_counts(recordings: pd.DataFrame, cohort: str | None) -> dict[str, int]:
    """Per-protocol recording counts inside the cohort (for the notes)."""
    rec_mask = cohort_recordings_mask(recordings, cohort)
    leop = recordings["dataset"] == "LEOP"
    table = recordings[rec_mask & leop]
    return {str(k): int(v) for k, v in table["protocol"].value_counts().items()}

"""Categorized QC flag semantics (v2 plan Phase 4).

``component_qc_flags`` is a ``|``-joined list of flag names produced by the
signal pipeline (landmarks, segmentation, resampling).  Feature builders must
never apply "any flag = drop": flags fall into four categories and only one
of them excludes a component from feature computation:

- ``hard_invalid``: component geometry is not landmark-driven (empty window,
  fallback geometry, canonicalization failure) — excluded from every feature
  family (clinical, curve, spectral, transport).
- ``low_confidence``: measurement exists but the landmark was weak or came
  from metadata — kept.
- ``truncated_or_limited_support``: the observed support does not cover the
  full nominal window (relative-phase clipping, boundary extremes) — kept;
  feature values are computed on the observed support.
- ``informational_qc``: record-only flags (e.g. disagreement with supplied
  metadata) — kept and only counted by QC-rate features.

Unknown flags degrade to ``informational_qc`` so new pipeline flags can never
silently start dropping data.  The raw ``any flag`` counts used by the E0
availability/quality shortcut features are intentionally unchanged: they
measure raw QC burden, not exclusion.
"""

from __future__ import annotations

import pandas as pd

CATEGORY_HARD_INVALID = "hard_invalid"
CATEGORY_LOW_CONFIDENCE = "low_confidence"
CATEGORY_TRUNCATED = "truncated_or_limited_support"
CATEGORY_INFORMATIONAL = "informational_qc"

FLAG_CATEGORIES: dict[str, tuple[str, ...]] = {
    CATEGORY_HARD_INVALID: (
        "no-samples-in-window",
        "late-support-too-short",
        "fallback-window",
        "late-landmark-invalid",
    ),
    CATEGORY_LOW_CONFIDENCE: (
        "no-prominence-peak",
        "supplied-only",
    ),
    CATEGORY_TRUNCATED: (
        "truncated-low",
        "truncated-high",
        "boundary-extreme",
    ),
    CATEGORY_INFORMATIONAL: (
        "disagrees-with-supplied",
    ),
}


def flag_categories(flags: str) -> tuple[str, ...]:
    """Category names for a ``|``-joined flag string (empty -> ())."""
    if not flags:
        return ()
    return tuple(
        _category_of(f) for f in flags.split("|") if f
    )


def _category_of(flag: str) -> str:
    for category, members in FLAG_CATEGORIES.items():
        if flag in members:
            return category
    return CATEGORY_INFORMATIONAL


def is_hard_invalid(components: pd.DataFrame) -> pd.Series:
    """True for components that must be excluded from feature computation."""
    return components["component_qc_flags"].fillna("").astype(str).apply(
        lambda s: any(c == CATEGORY_HARD_INVALID for c in flag_categories(s))
    )

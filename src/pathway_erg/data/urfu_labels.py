"""URFU diagnosis mapping for the supervised-sanity endpoint.

The URFU database ships free-text clinical diagnoses in Russian (translated).
Per plan constraint (no silent default), every observed label is mapped
explicitly to one of:

- ``0`` — healthy / within normal limits,
- ``1`` — reduced / dystrophy / organic pathology,
- ``None`` — ineligible: technical recording notes (``Registration of the
  ERG of a narrow pupil...``) or absent diagnosis (OP-sheet-only subjects).

The mapping is a candidate table (reviewer ``PENDING_CLINICAL_REVIEW``), kept
explicit so a clinician can confirm or correct each row before any transfer
claim is made.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labels import DiagnosisMapping

ENDPOINT_URFU_SANITY = "urfu_no_diagnosis_vs_reduced"
URFU_MAPPING_VERSION = "urfu_labels_v1"

# explicit vocabulary: raw label -> (mapped_value, rationale)
_URFU_VOCABULARY: dict[str, tuple[int | None, str]] = {
    "No diagnosis.": (0, "Explicit no-diagnosis healthy reference"),
    "The signal is within normal limits.": (0, "Normal signal"),
    "The functional activity of the retina in both eyes is preserved.": (
        0,
        "Retinal function preserved",
    ),
    "The amplitude-time characteristics of the EP in a homogeneous field correspond to the norm.": (
        0,
        "Amplitude/time within norm",
    ),
    "The functional activity of the retina is preserved.": (0, "Retinal function preserved"),
    "The functional activity of the retina in both eyes is preserved. High level of interference due to increased motor activity of the child.": (
        0,
        "Preserved; interference is a recording-quality note",
    ),
    "The functional activity of the retina at the OU is preserved. No macular pathology was revealed.": (
        0,
        "Preserved; no macular pathology",
    ),
    "The functional activity of the central and peripheral parts of the retina in both eyes is preserved.": (
        0,
        "Preserved",
    ),
    "The functional activity of the retina is preserved, corresponds to the age norm symmetrically on the OU.": (
        0,
        "Preserved, age-normative",
    ),
    "Pronounced organic changes in the outer and inner layers in the center and periphery of the retina.": (
        1,
        "Organic retinal pathology",
    ),
    "Retinal cone-rod dystrophy.": (1, "Cone-rod dystrophy"),
    "Hereditary cone dystrophy.": (1, "Cone dystrophy"),
    "The prognosis for the restoration of OS visual functions is extremely dubious.": (
        1,
        "Reduced function, poor prognosis",
    ),
    "The functional activity of the retina OD is preserved on the OS and is moderately reduced (changes at the level of the outer and middle layers of the retina in the central and peripheral regions).": (
        1,
        "Partial moderate reduction",
    ),
    "The functional activity of the central parts of the retina is moderately reduced.": (
        1,
        "Moderate reduction",
    ),
    "Signs of moderate disturbances in the electrogenesis of the central parts of the retina in the left eye.": (
        1,
        "Electrogenesis disturbances",
    ),
    "OU - maximum ERG b-wave reduction. Central dystrophy with cone-rod dysfunction is possible.": (
        1,
        "B-wave reduction, possible dystrophy",
    ),
    "OU - a pronounced decrease in the b-wave of the maximum response is reduced.": (
        1,
        "B-wave reduction",
    ),
    "Hereditary rod dystrophy of the retina is not excluded.": (
        1,
        "Possible rod dystrophy",
    ),
    "Decrease in electrogenesis and functional activity of the retina.": (
        1,
        "Reduced electrogenesis",
    ),
    "Retinal rod dystrophy with a favorable prognosis is possible.": (
        1,
        "Possible rod dystrophy",
    ),
    "Electrogenesis of the central parts of the retina normally does not exclude dystrophy of the rod apparatus of the retina with a favorable prognosis of the course.": (
        1,
        "Dystrophy not excluded",
    ),
    "Moderate pronounced change in the outer and middle layers of the central and peripheral parts of the retina of the right eye.": (
        1,
        "Outer/middle-layer changes",
    ),
    "The functional activity of the retina of the left eye is protected by a moderate decrease in the functional activity of the retina of the right eye.": (
        1,
        "Moderate decrease",
    ),
    "Decreased the amplitude of the b-wave of the maximum and rod responses.": (
        1,
        "B-wave amplitude decrease",
    ),
    "Reduced functional activity of the retina changes at the level of the outer and middle layers in the central and peripheral parts.": (
        1,
        "Reduced function with layer changes",
    ),
    "Registration of the ERG of a narrow pupil from the skin of the eyelid.": (
        None,
        "Technical recording note, not a clinical diagnosis",
    ),
    "Registration of ERG with a narrow pupil.": (
        None,
        "Technical recording note, not a clinical diagnosis",
    ),
}


def build_urfu_mapping(observed: pd.Series) -> DiagnosisMapping:
    """Construct the explicit URFU sanity mapping for an endpoint.

    Raises if any observed diagnosis string is not in the explicit vocabulary
    (no silent default).  Returns a candidate table for clinical review.
    """
    counts = observed.dropna().astype(str).str.strip().value_counts().sort_index()
    rows = []
    for label, count in counts.items():
        if label not in _URFU_VOCABULARY:
            raise ValueError(
                f"unmapped URFU diagnosis {label!r}; add it to _URFU_VOCABULARY"
            )
        value, rationale = _URFU_VOCABULARY[label]
        rows.append(
            {
                "raw_label": label,
                "mapped_value": value,
                "count": int(count),
                "rationale": rationale,
            }
        )
    # include vocabulary entries not observed (documented completeness)
    observed_set = set(counts.index)
    for label, (value, rationale) in sorted(_URFU_VOCABULARY.items()):
        if label not in observed_set:
            rows.append(
                {
                    "raw_label": label,
                    "mapped_value": value,
                    "count": 0,
                    "rationale": rationale,
                }
            )
    table = pd.DataFrame(rows).sort_values("raw_label").reset_index(drop=True)
    return DiagnosisMapping(
        endpoint=ENDPOINT_URFU_SANITY,
        version=URFU_MAPPING_VERSION,
        table=table,
        reviewer="PENDING_CLINICAL_REVIEW",
        review_date="",
    )


def make_urfu_target(diagnosis_raw: str | None, mapping: DiagnosisMapping) -> int | None:
    return mapping.map(diagnosis_raw)


def require_urfu_labels_signed_off(version: str = URFU_MAPPING_VERSION) -> None:
    """Hard gate (plan integration §11.2): no URFU supervised endpoint may
    run until the diagnosis mapping is clinician-signed (reviewer no longer
    ``PENDING_CLINICAL_REVIEW``).  Raises with the blocking reason; refuses
    to run silently against an unreviewed mapping."""
    mapping = build_urfu_mapping(pd.Series(dtype=str))
    if mapping.version != version:
        raise ValueError(
            f"URFU mapping version {mapping.version!r} does not match "
            f"required {version!r}"
        )
    if mapping.reviewer == "PENDING_CLINICAL_REVIEW" or not mapping.reviewer:
        raise ValueError(
            "URFU supervised endpoint blocked: diagnosis mapping "
            f"{version!r} is PENDING_CLINICAL_REVIEW (reviewer "
            f"{mapping.reviewer!r}); SSL-only use is not blocked"
        )


def write_urfu_mapping(mapping: DiagnosisMapping, out_dir: Path) -> Path:
    from .labels import write_mapping_csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_mapping_csv(mapping, out_dir / "diagnosis_mapping_urfu.csv")

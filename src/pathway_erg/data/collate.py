"""Variable-size batch collators with explicit masks (plan Module 21.12).

One collator function per dataset class:

- ``collate_component_rows`` — stacks flat component samples into
  (B, 1, 128) / (B, 128) / (B, 135) / (B, 8) NumPy tensors plus metadata.
- ``collate_bag_units`` — pads bag components to the longest bag in the
  batch and emits a component-mask so the aggregator can treat missing
  components as absent, never as observed zeros.

No zero-as-missing ambiguity: every padded sample carries a boolean mask;
label NaN means "no target for this unit", never "negative".  Everything
is deterministic and NumPy-based; the training loop converts outputs to
torch tensors (the data layer has no framework dependency).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .datasets import (
    CANONICAL_SAMPLES,
    OT_DIM,
    PHYSICAL_FEATURE_NAMES,
    BagUnit,
    ComponentRow,
)


def collate_component_rows(rows: Iterable[ComponentRow]) -> dict[str, np.ndarray]:
    """Pack component rows into flat batched arrays (deterministic order).

    Returns arrays under fixed keys: ``signal`` (B,1,128), ``valid_mask``
    (B,128), ``ot`` (B,135), ``physical`` (B,8), ``label`` (B,) float32
    (NaN = unlabeled), ``outer_fold`` (B,) int64, plus object metadata
    arrays ``component_ids``, ``dataset``, ``unit_id``, ``protocol`` and
    float ``stimulus_value`` (NaN = absent).
    """
    rows = list(rows)
    n = len(rows)
    signal = np.zeros((n, 1, CANONICAL_SAMPLES), dtype=np.float32)
    valid = np.zeros((n, CANONICAL_SAMPLES), dtype=bool)
    ot = np.zeros((n, OT_DIM), dtype=np.float32)
    physical = np.full((n, len(PHYSICAL_FEATURE_NAMES)), np.nan, dtype=np.float32)
    labels = np.full(n, np.nan, dtype=np.float32)
    folds = np.zeros(n, dtype=np.int64)
    stimulus = np.full(n, np.nan, dtype=np.float32)
    comp_ids: list[str] = []
    datasets: list[str] = []
    unit_ids: list[str] = []
    protocols: list[str] = []

    for i, r in enumerate(rows):
        if r.signal.size != CANONICAL_SAMPLES or r.signal_mask.size != CANONICAL_SAMPLES:
            raise ValueError(f"bad canonical size for {r.global_component_id}")
        if int(r.signal_mask.sum()) == 0:
            raise ValueError(f"empty valid mask for {r.global_component_id}")
        signal[i, 0] = r.signal.astype(np.float32)
        valid[i] = r.signal_mask
        ot[i] = r.ot_vector.astype(np.float32)
        physical[i] = r.physical.astype(np.float32)
        folds[i] = int(r.outer_fold)
        if not (isinstance(r.stimulus_value, float) and np.isnan(r.stimulus_value)):
            stimulus[i] = float(r.stimulus_value)
        comp_ids.append(r.global_component_id)
        datasets.append(r.dataset)
        unit_ids.append(r.unit_id)
        protocols.append(r.protocol)

    return {
        "signal": signal,
        "valid_mask": valid,
        "ot": ot,
        "physical": physical,
        "label": labels,
        "outer_fold": folds,
        "component_ids": np.asarray(comp_ids, dtype=object),
        "dataset": np.asarray(datasets, dtype=object),
        "unit_id": np.asarray(unit_ids, dtype=object),
        "protocol": np.asarray(protocols, dtype=object),
        "stimulus_value": stimulus,
    }


def collate_bag_units(bags: list[BagUnit]) -> dict[str, np.ndarray]:
    """Pack one or more bags into padded tensors with explicit masks.

    Keys: ``signal`` (B,L,1,128) float32, ``valid_mask`` (B,L,128) bool,
    ``ot`` (B,L,135) float32, ``physical`` (B,L,8) float32,
    ``component_mask`` (B,L) bool, ``unit_ids`` (B,) object, ``label``
    (B,) float64 (NaN = no target), ``outer_fold`` (B,) int64, ``dataset``
    (B,) object.

    Hierarchy group codes (per batch position, -1 = padding):
    ``group_eye`` (B,L) int64 — within-bag eye index;
    ``group_intensity`` (B,L) int64 — within-bag (eye, stimulus) index;
    ``group_recording`` (B,L) int64 — within-bag recording index.
    Codes are assigned in first-appearance order so repeated calls are
    stable across batches.
    """
    B = len(bags)
    if B == 0:
        raise ValueError("cannot collate an empty bag list")
    L = max(len(bag.components) for bag in bags)
    signal = np.zeros((B, L, 1, CANONICAL_SAMPLES), dtype=np.float32)
    valid = np.zeros((B, L, CANONICAL_SAMPLES), dtype=bool)
    ot = np.zeros((B, L, OT_DIM), dtype=np.float32)
    physical = np.full((B, L, len(PHYSICAL_FEATURE_NAMES)), np.nan, dtype=np.float32)
    comp_mask = np.zeros((B, L), dtype=bool)
    group_eye = np.full((B, L), -1, dtype=np.int64)
    group_intensity = np.full((B, L), -1, dtype=np.int64)
    group_recording = np.full((B, L), -1, dtype=np.int64)
    labels = np.full(B, np.nan, dtype=np.float64)
    folds = np.zeros(B, dtype=np.int64)
    unit_ids: list[str] = []
    subject_ids: list[str] = []
    visit_ids: list[str | None] = []
    for i, bag in enumerate(bags):
        unit_ids.append(bag.unit_id)
        subject_ids.append(bag.subject_id)
        visit_ids.append(bag.visit_id)
        labels[i] = bag.target_binary if bag.target_binary is not None else np.nan
        folds[i] = int(bag.outer_fold)
        eye_code: dict[str, int] = {}
        inten_code: dict[tuple[str, float], int] = {}
        rec_code: dict[str, int] = {}
        for j, comp in enumerate(bag.components):
            comp_mask[i, j] = True
            signal[i, j, 0] = comp.signal.astype(np.float32)
            valid[i, j] = comp.signal_mask
            ot[i, j] = comp.ot_vector.astype(np.float32)
            physical[i, j] = comp.physical.astype(np.float32)
            eye = comp.eye or "?"
            group_eye[i, j] = eye_code.setdefault(eye, len(eye_code))
            key_inten = (eye, comp.stimulus_value)
            group_intensity[i, j] = inten_code.setdefault(key_inten, len(inten_code))
            group_recording[i, j] = rec_code.setdefault(
                str(comp.global_recording_id), len(rec_code)
            )
    return {
        "signal": signal,
        "valid_mask": valid,
        "ot": ot,
        "physical": physical,
        "component_mask": comp_mask,
        "group_eye": group_eye,
        "group_intensity": group_intensity,
        "group_recording": group_recording,
        "unit_ids": np.asarray(unit_ids, dtype=object),
        "subject_ids": np.asarray(subject_ids, dtype=object),
        "visit_ids": np.asarray(visit_ids, dtype=object),
        "label": labels,
        "outer_fold": folds,
        "dataset": np.asarray([b.dataset for b in bags], dtype=object),
    }

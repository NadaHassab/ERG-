"""Nested bag construction for neural training (plan Module 21.12).

Default PyTorch batches assume rectangular independent samples.  ERG data is
nested: components live inside recordings inside eyes inside subjects
(LEOP) or visits (PERG).  These datasets return the *structure* needed to
train without destroying it:

- ``ComponentDataset`` — one row per cached component; used by the
  component-level self-supervised/representation stages (plan 14.1).
- ``build_bags`` — one :class:`BagUnit` per participant (LEOP) or per visit
  (PERG): all of that unit's cached components with masks and metadata.
- ``domain_balanced_batch_indices`` — deterministic alternating-domain
  batch plan for the joint label-free stage (plan 14.2).

Leakage discipline: fold selection is by the same frozen unit keys the
classical baselines use (LEOP = participant; PERG = canonical subject via
``outer_folds_v1``), and the datasets refuse rows whose unit does not belong
to the requested partition.

Everything is deterministic: iteration order follows the locked parquet
row order; the only RNG is the explicit ``seed`` argument of
``domain_balanced_batch_indices``.  No zero-as-missing trick: invalid
values are carried by explicit boolean masks; pad values are only used by
the collators inside one batch.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..constants import OUTER_FOLDS_TEMPLATE
from ..signal.component_cache import (
    CACHE_SCHEMA_VERSION,
    cache_paths,
    load_cache_manifest,
)
from ..signal.external_cache import (
    external_cache_paths,
    load_external_cache_manifest,
)
from .schemas import SUPPORTED_DATASETS

CANONICAL_SAMPLES = 128
OT_DIM = 135  # 2x64 quantiles + 2 log masses + mass fraction + 2 variations + 2 flags


@dataclass(frozen=True)
class ComponentRow:
    """One component-level sample."""

    global_component_id: str
    global_recording_id: str
    subject_id: str
    visit_id: str
    dataset: str
    component_id: str
    unit_id: str
    protocol: str
    eye: str | None
    stimulus_value: float
    stimulus_unit: str
    landmark_confidence: float
    outer_fold: int
    signal: np.ndarray        # canonical 128-point signal (float64)
    signal_mask: np.ndarray   # valid-sample mask (bool)
    ot_vector: np.ndarray     # signed-OT flat descriptor (135,) (float64)
    physical: np.ndarray      # physical feature vector (8,) (float64)

    def __len__(self) -> int:
        return len(self.signal)


@dataclass(frozen=True)
class BagUnit:
    """One unit (participant or visit) and all of its components."""

    unit_id: str
    subject_id: str
    visit_id: str | None
    dataset: str
    target_binary: int | None
    outer_fold: int
    components: tuple[ComponentRow, ...]

    def __len__(self) -> int:
        return len(self.components)


PHYSICAL_FEATURE_NAMES = (
    "log_mass_pos",
    "log_mass_neg",
    "peak_to_peak_uv",
    "max_rising_slope_uv_per_ms",
    "max_falling_slope_uv_per_ms",
    "area_above_ref_uv_ms",
    "area_below_ref_uv_ms",
    "duration_ms",
)


def _physical_matrix(comp: pd.DataFrame) -> np.ndarray:
    """Parse physical_features_json into a (n_components, 8) float matrix."""
    out = np.full((len(comp), len(PHYSICAL_FEATURE_NAMES)), np.nan, dtype=np.float64)
    for i, raw in enumerate(comp["physical_features_json"]):
        if not raw:
            continue
        d = json.loads(raw)
        out[i] = [float(d.get(k, np.nan)) for k in PHYSICAL_FEATURE_NAMES]
    return out


class LoadedCaches:
    """Versioned component caches + interim tables, force-validated at load.

    Raises ``ValueError`` on cache misalignment so stale data can never be
    silently reused by a training run; every dataset asserts components map
    to a locked outer fold.
    """

    def __init__(
        self,
        artifact_root: str | Path = "artifacts",
        fold_version: str = "v1",
        external_bindings: tuple[str, ...] = (),
        external_fold_version: str | None = None,
    ):
        self.root = Path(artifact_root)
        self.fold_version = fold_version
        self.external_bindings = external_bindings
        if external_fold_version is not None and external_fold_version == fold_version:
            raise ValueError(
                "external_fold_version must differ from fold_version "
                "(external folds are a separate table)"
            )
        cache = cache_paths(self.root, CACHE_SCHEMA_VERSION)
        self.cache_manifest = load_cache_manifest(self.root, CACHE_SCHEMA_VERSION)
        self.components = pd.read_parquet(cache["components_parquet"])
        self.recordings = pd.read_parquet(self.root / "data" / "interim" / "recordings.parquet")
        self.visits = pd.read_parquet(self.root / "data" / "interim" / "visits.parquet")
        curves = zarr.open_group(str(cache["curves_zarr"]), mode="r")["components"]
        self.signal = np.asarray(curves["canonical_signal"][:])
        self.mask = np.asarray(curves["valid_mask"][:])
        sot = zarr.open_group(str(cache["sot_zarr"]), mode="r")["components"]
        self.ot = np.asarray(sot["sot_vector"][:])
        if external_bindings:
            if external_fold_version is None:
                raise ValueError(
                    "external_bindings require external_fold_version "
                    "(external rows must map to locked folds)"
                )
            for binding in external_bindings:
                ext = external_cache_paths(self.root, binding)
                load_external_cache_manifest(self.root, binding)
                ext_components = pd.read_parquet(ext["components_parquet"])
                if ext_components.empty:
                    raise ValueError(f"external binding {binding} has no components")
                self.components = pd.concat(
                    [self.components, ext_components], ignore_index=True
                )
                ext_curves = (
                    zarr.open_group(str(ext["curves_zarr"]), mode="r")["components"]
                )
                ext_sot = zarr.open_group(str(ext["sot_zarr"]), mode="r")["components"]
                ext_signal = np.asarray(ext_curves["canonical_signal"][:])
                ext_mask = np.asarray(ext_curves["valid_mask"][:])
                ext_ot = np.asarray(ext_sot["sot_vector"][:])
                if not (
                    ext_signal.shape[0] == ext_mask.shape[0] == ext_ot.shape[0]
                    == len(ext_components)
                ):
                    raise ValueError(
                        f"external binding {binding}: cache misalignment — rebuild"
                    )
                self.signal = np.concatenate([self.signal, ext_signal], axis=0)
                self.mask = np.concatenate([self.mask, ext_mask], axis=0)
                self.ot = np.concatenate([self.ot, ext_ot], axis=0)
        self.folds = pd.read_parquet(
            self.root
            / "data"
            / "splits"
            / OUTER_FOLDS_TEMPLATE.format(version=fold_version)
        )
        if external_fold_version is not None:
            external_outer = (
                self.root
                / "data"
                / "splits"
                / OUTER_FOLDS_TEMPLATE.format(version=external_fold_version)
            )
            if not external_outer.is_file():
                raise ValueError(
                    f"external split table not found at {external_outer}; "
                    "build it with the make-external-splits command"
                )
            self.folds = pd.concat(
                [self.folds, pd.read_parquet(external_outer)], ignore_index=True
            )
            self.external_fold_version = external_fold_version
        self.physical = _physical_matrix(self.components)
        if not (
            self.signal.shape[0] == self.mask.shape[0] == self.ot.shape[0]
            == len(self.components) == self.physical.shape[0]
        ):
            raise ValueError(
                f"cache misalignment: {len(self.components)} components, "
                f"{self.signal.shape[0]} curves, {self.ot.shape[0]} sot — rebuild"
            )
        if self.signal.shape[1] != CANONICAL_SAMPLES:
            raise ValueError(f"unexpected canonical length {self.signal.shape[1]}")
        if not np.all(np.isfinite(self.mask)):
            raise ValueError("valid masks must be boolean")
        self._table: pd.DataFrame | None = None

    def _base_table(self) -> pd.DataFrame:
        """Components x recordings x labels, one row per component."""
        comp = self.components.merge(
            self.recordings[
                [
                    "global_recording_id",
                    "global_subject_id",
                    "global_visit_id",
                    "dataset",
                    "protocol",
                    "eye",
                    "stimulus_value",
                    "stimulus_unit",
                ]
            ],
            on="global_recording_id",
            how="left",
        ).drop_duplicates("global_component_id")
        target = (
            self.visits[["global_visit_id", "target_binary"]]
            .drop_duplicates("global_visit_id")
            .set_index("global_visit_id")["target_binary"]
        )
        comp["target_binary"] = comp["global_visit_id"].map(target)
        return comp

    def table(self) -> pd.DataFrame:
        """Full component table with locked folds and labels (once)."""
        if self._table is None:
            comp = self._base_table()
            # LEOP/FLINDERS unit = participant, PERG/URFU unit = visit
            # (fold units are always subjects; external folds are subject-keyed)
            comp["unit_id"] = np.where(
                comp["dataset"].isin(("PERG", "URFU")),
                comp["global_visit_id"].astype(str),
                comp["global_subject_id"].astype(str),
            )
            if (
                (comp["dataset"] == "FLINDERS")
                & (comp["target_binary"] == 1)
            ).any():
                raise ValueError(
                    "FLINDERS rows carry a positive supervised label; the "
                    "FLINDERS supervised head is forbidden (plan §11.2)"
                )
            fold_map = self.folds.set_index(["dataset", "unit_id"])["outer_fold"]
            comp["outer_fold"] = [
                fold_map.get((ds, uid), np.nan)
                for ds, uid in zip(comp["dataset"], comp["global_subject_id"].astype(str), strict=True)
            ]
            if comp["outer_fold"].isna().any():
                raise ValueError(
                    "components without a locked fold — rebuild the split table"
                )
            comp["outer_fold"] = comp["outer_fold"].astype(int)
            comp["cache_idx"] = np.arange(len(comp))
            self._table = comp
        return self._table

    def _row(self, r: pd.Series) -> ComponentRow:
        i = int(r["cache_idx"])
        return ComponentRow(
            global_component_id=r["global_component_id"],
            global_recording_id=r["global_recording_id"],
            subject_id=str(r["global_subject_id"]),
            visit_id=str(r["global_visit_id"]),
            dataset=r["dataset"],
            component_id=r["component_id"],
            unit_id=str(r["unit_id"]),
            protocol=r.get("protocol", ""),
            eye=str(r["eye"]) if pd.notna(r.get("eye")) else None,
            stimulus_value=float(r["stimulus_value"]) if pd.notna(r["stimulus_value"]) else np.nan,
            stimulus_unit=str(r.get("stimulus_unit", "")),
            landmark_confidence=float(r.get("landmark_confidence", 1.0)),
            outer_fold=int(r["outer_fold"]),
            signal=self.signal[i],
            signal_mask=self.mask[i],
            ot_vector=self.ot[i],
            physical=self.physical[i],
        )

    def iter_rows(self, dataset: str | None = None) -> Iterator[ComponentRow]:
        """All rows in canonical order (``ComponentDataset`` backend)."""
        tbl = self.table()
        if dataset is not None:
            tbl = tbl[tbl["dataset"] == dataset]
        for _, r in tbl.iterrows():
            yield self._row(r)


class ComponentDataset:
    """One sample per component, aligned to the cache row order.

    Parameters
    ----------
    caches : LoadedCaches
    dataset : str | None
        Restrict to LEOP/PERG; None = both.
    outer_folds : set[int] | None
        If given, only components whose unit belongs to these locked folds
        are reachable (leakage guard).
    """

    def __init__(self, caches: LoadedCaches, dataset: str | None = None,
                 outer_folds: set[int] | None = None):
        tbl = caches.table()
        if dataset is not None:
            tbl = tbl[tbl["dataset"] == dataset]
        if outer_folds is not None:
            tbl = tbl[tbl["outer_fold"].isin(outer_folds)]
        self._tbl = tbl.reset_index(drop=True)
        self._caches = caches

    def __len__(self) -> int:
        return len(self._tbl)

    def __getitem__(self, idx: int) -> ComponentRow:
        return self._caches._row(self._tbl.iloc[idx])


def build_bags(
    caches: LoadedCaches,
    dataset: str,
    outer_folds: set[int] | None = None,
    allowed_recording_ids: set[str] | None = None,
) -> list[BagUnit]:
    """One :class:`BagUnit` per participant (LEOP/FLINDERS) or visit (PERG/URFU).

    Bags appear in first-appearance order of the unit in the canonical
    table (deterministic).  ``outer_folds`` filters which units may appear;
    a unit is dropped entirely if any of its components is outside the
    allowed folds (a partial bag would corrupt the hierarchy).
    """
    if not any(dataset == d.value for d in SUPPORTED_DATASETS):
        raise ValueError(
            f"dataset must be one of "
            f"{sorted(d.value for d in SUPPORTED_DATASETS)}, got {dataset!r}"
        )
    visit_unit = dataset in ("PERG", "URFU")
    tbl = caches.table()
    tbl = tbl[tbl["dataset"] == dataset]
    if allowed_recording_ids is not None:
        tbl = tbl[tbl["global_recording_id"].isin(allowed_recording_ids)]
    if outer_folds is not None:
        tbl = tbl[tbl["outer_fold"].isin(outer_folds)]

    bags: dict[str, list[ComponentRow]] = {}
    for _, r in tbl.iterrows():
        uid = str(r["unit_id"])
        bags.setdefault(uid, []).append(caches._row(r))

    out = []
    for uid, rows in bags.items():
        folders = {row.outer_fold for row in rows}
        if len(folders) != 1:
            raise ValueError(f"unit {uid} spans folds {folders}")
        subject_ids = {row.subject_id for row in rows}
        visit_ids = {row.visit_id for row in rows}
        if len(subject_ids) != 1:
            raise ValueError(f"unit {uid} spans subjects {subject_ids}")
        if visit_unit and len(visit_ids) != 1:
            raise ValueError(f"{dataset} unit {uid} spans visits {visit_ids}")
        target = tbl.loc[tbl["unit_id"].astype(str) == uid, "target_binary"]
        target_v = int(target.iloc[0]) if len(target) and pd.notna(target.iloc[0]) else None
        out.append(
            BagUnit(
                unit_id=uid,
                subject_id=next(iter(subject_ids)),
                visit_id=next(iter(visit_ids)) if visit_unit else None,
                dataset=dataset,
                target_binary=target_v,
                outer_fold=next(iter(folders)),
                components=tuple(rows),
            )
        )
    order = {c.unit_id: i for i, c in enumerate(out)}
    out.sort(key=lambda b: order[b.unit_id])
    return out


def domain_balanced_batch_indices(
    n_leop: int, n_perg: int,
    leop_batch: int, perg_batch: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Deterministic alternating LEOP/PERG batch plan (plan 14.2).

    Each contribution yields fresh (possibly ragged) permutations of
    unit indices; last batches may be shorter.  One LEOP batch and one
    PERG batch are emitted per step regardless of dataset size, so the
    shared expert sees both domains every optimizer step after the first.
    """
    sizes = {"LEOP": n_leop, "PERG": n_perg}
    batches = {"LEOP": leop_batch, "PERG": perg_batch}
    for idx in domain_balanced_epoch_indices(sizes, batches, seed):
        yield (idx.get("LEOP", np.array([], dtype=np.int64)),
               idx.get("PERG", np.array([], dtype=np.int64)))


def domain_balanced_epoch_indices(
    sizes: dict[str, int],
    batches: dict[str, int],
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Deterministic multi-domain step plan (plan integration §11.3).

    Each yielded step maps domain name -> indices for one batch from a
    fresh permutation of that domain (last batch may be short); every
    domain is drawn once per step, so the shared expert sees all domains
    at every optimizer step.  The single plan is consumed once, exactly
    like the two-domain ``domain_balanced_batch_indices``.
    """
    if not sizes:
        raise ValueError("no domains given")
    unknown = set(batches) - set(sizes)
    if unknown:
        raise ValueError(f"batch sizes for unknown domains: {sorted(unknown)}")
    for name, size in sizes.items():
        if size < 0:
            raise ValueError(f"domain {name} has negative size {size}")
    rng = np.random.default_rng(seed)
    perms = {name: rng.permutation(size) for name, size in sizes.items()}
    pos = {name: 0 for name in sizes}
    while any(pos[name] < sizes[name] for name in sizes):
        step_idx: dict[str, np.ndarray] = {}
        for name in sizes:
            batch = batches.get(name, 64)
            if batch < 1:
                raise ValueError(f"batch size for {name} must be >= 1")
            lo = pos[name]
            step_idx[name] = perms[name][lo : lo + batch]
            pos[name] += batch
        yield step_idx

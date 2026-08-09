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
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths

CANONICAL_SAMPLES = 128
OT_DIM = 135  # 2x64 quantiles + 2 log masses + mass fraction + 2 variations + 2 flags


@dataclass(frozen=True)
class ComponentRow:
    """One component-level sample."""

    global_component_id: str
    global_recording_id: str
    dataset: str
    component_id: str
    unit_id: str
    protocol: str
    eye: str | None
    stimulus_value: float
    stimulus_unit: str
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

    def __init__(self, artifact_root: str | Path = "artifacts"):
        self.root = Path(artifact_root)
        cache = cache_paths(self.root, CACHE_SCHEMA_VERSION)
        self.components = pd.read_parquet(cache["components_parquet"])
        self.recordings = pd.read_parquet(self.root / "data" / "interim" / "recordings.parquet")
        self.visits = pd.read_parquet(self.root / "data" / "interim" / "visits.parquet")
        self.folds = pd.read_parquet(
            self.root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version="v1")
        )
        curves = zarr.open_group(str(cache["curves_zarr"]), mode="r")["components"]
        self.signal = np.asarray(curves["canonical_signal"][:])
        self.mask = np.asarray(curves["valid_mask"][:])
        sot = zarr.open_group(str(cache["sot_zarr"]), mode="r")["components"]
        self.ot = np.asarray(sot["sot_vector"][:])
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
            # LEOP unit = participant, PERG unit = visit (fold key is subject)
            comp["unit_id"] = np.where(
                comp["dataset"] == "LEOP",
                comp["global_subject_id"].astype(str),
                comp["global_visit_id"].astype(str),
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
            dataset=r["dataset"],
            component_id=r["component_id"],
            unit_id=str(r["unit_id"]),
            protocol=r.get("protocol", ""),
            eye=str(r["eye"]) if pd.notna(r.get("eye")) else None,
            stimulus_value=float(r["stimulus_value"]) if pd.notna(r["stimulus_value"]) else np.nan,
            stimulus_unit=str(r.get("stimulus_unit", "")),
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


def build_bags(caches: LoadedCaches, dataset: str,
               outer_folds: set[int] | None = None) -> list[BagUnit]:
    """One :class:`BagUnit` per participant (LEOP) or visit (PERG).

    Bags appear in first-appearance order of the unit in the canonical
    table (deterministic).  ``outer_folds`` filters which units may appear;
    a unit is dropped entirely if any of its components is outside the
    allowed folds (a partial bag would corrupt the hierarchy).
    """
    if dataset not in ("LEOP", "PERG"):
        raise ValueError(f"dataset must be LEOP or PERG, got {dataset!r}")
    tbl = caches.table()
    tbl = tbl[tbl["dataset"] == dataset]
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
        target = tbl.loc[tbl["unit_id"].astype(str) == uid, "target_binary"]
        target_v = int(target.iloc[0]) if len(target) and pd.notna(target.iloc[0]) else None
        out.append(
            BagUnit(
                unit_id=uid,
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
    if leop_batch < 1 or perg_batch < 1:
        raise ValueError("batch sizes must be >= 1")
    rng = np.random.default_rng(seed)
    leop_idx = rng.permutation(n_leop)
    perg_idx = rng.permutation(n_perg)
    li = pi = 0
    while li < n_leop or pi < n_perg:
        yield (leop_idx[li : li + leop_batch], perg_idx[pi : pi + perg_batch])
        li += leop_batch
        pi += perg_batch

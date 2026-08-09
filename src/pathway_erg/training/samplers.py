"""Fold-safe bag samplers (plan Module 21.17, §14.2/14.3).

Every sampler works on *unit-level* bag lists and refuses to emit a unit
whose fold is outside the requested folds — the test IDs never enter the
training loaders (plan 21.17 requirement "no test IDs in loaders").
RNG is explicit and seed-resumable so training runs can resume from a
checkpoint with an identical sample stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..data.datasets import BagUnit, build_bags


@dataclass
class BagSampler:
    """Deterministic, resumable bag sampler for one dataset.

    Yields batches of bag *indices* into ``bags`` (a fixed deterministic
    list).  Batches never mix units from different outer folds unless
    ``folds`` contains several — callers pass a single fold for train and
    keep the test fold out entirely.
    """

    bags: list[BagUnit]
    folds: set[int]
    batch_size: int
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _start_step: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._validate_folds()

    def _validate_folds(self):
        if not self.folds:
            raise ValueError("BagSampler requires at least one fold")
        allowed = {b.outer_fold for b in self.bags}
        if not self.folds.issubset(allowed):
            raise ValueError(
                f"requested folds {sorted(self.folds)} not present in bags "
                f"({sorted(allowed)})"
            )
        self.bags = [b for b in self.bags if b.outer_fold in self.folds]
        if not self.bags:
            raise ValueError("no bags match the requested folds")

    def resume(self, step: int, state: dict) -> BagSampler:
        """Restore RNG + step (plan 21.17 resume-safe RNG)."""
        self._start_step = int(step)
        self._rng = np.random.default_rng(state["rng_byte"])
        return self

    def state(self) -> dict:
        return {"rng_byte": self._rng.bit_generator.state["state"]["state"]}

    def __iter__(self) -> Iterator[np.ndarray]:
        step = 0
        n = len(self.bags)
        while True:
            step += 1
            if step < self._start_step:
                continue
            perm = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                yield perm[start : start + self.batch_size]


def make_fold_bags(caches, dataset: str, fold: int) -> list[BagUnit]:
    """Bags restricted to one outer fold (leakage-safe)."""
    return build_bags(caches, dataset, outer_folds={fold})

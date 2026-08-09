"""Neural dataset/bag/collate tests (plan Module 21.12) on real caches."""

from __future__ import annotations

import numpy as np
import pytest

from pathway_erg.data.collate import collate_bag_units, collate_component_rows
from pathway_erg.data.datasets import (
    CANONICAL_SAMPLES,
    OT_DIM,
    PHYSICAL_FEATURE_NAMES,
    ComponentDataset,
    LoadedCaches,
    build_bags,
    domain_balanced_batch_indices,
)


@pytest.fixture(scope="module")
def caches():
    return LoadedCaches("artifacts")


@pytest.fixture(scope="module")
def table(caches):
    return caches.table()


def test_cache_alignment(caches):
    n = len(caches.components)
    assert caches.signal.shape[0] == n
    assert caches.mask.shape[0] == n
    assert caches.ot.shape[0] == n
    assert caches.physical.shape[0] == n
    assert caches.signal.shape[1] == CANONICAL_SAMPLES
    assert caches.ot.shape[1] == OT_DIM


def test_table_has_locked_folds(table):
    assert table["outer_fold"].isna().sum() == 0
    assert set(table["outer_fold"].unique()) <= {0, 1, 2, 3, 4}


def test_component_dataset_counts(caches, table):
    ds_all = ComponentDataset(caches)
    assert len(ds_all) == len(table)
    ds_leop = ComponentDataset(caches, dataset="LEOP")
    assert len(ds_leop) == int((table["dataset"] == "LEOP").sum())


def test_component_dataset_fold_filter(caches):
    ds = ComponentDataset(caches, outer_folds={0})
    assert len(ds) > 0
    for i in range(min(5, len(ds))):
        assert ds[i].outer_fold == 0


def test_component_row_fields(caches):
    row = ComponentDataset(caches, dataset="LEOP")[0]
    assert row.signal.shape == (CANONICAL_SAMPLES,)
    assert row.signal_mask.shape == (CANONICAL_SAMPLES,)
    assert row.signal_mask.sum() > 0
    assert row.ot_vector.shape == (OT_DIM,)
    assert row.physical.shape == (len(PHYSICAL_FEATURE_NAMES),)
    assert row.dataset == "LEOP"
    assert row.unit_id
    assert row.subject_id == row.unit_id
    assert row.visit_id


def test_bag_counts_match_locked_cohort(caches, table):
    leop = build_bags(caches, "LEOP")
    perg = build_bags(caches, "PERG")
    assert len(leop) == table.loc[table["dataset"] == "LEOP", "unit_id"].nunique()
    assert len(perg) == table.loc[table["dataset"] == "PERG", "unit_id"].nunique()
    assert all(b.dataset == "LEOP" for b in leop)
    assert all(b.dataset == "PERG" for b in perg)


def test_bag_fold_consistency(caches):
    for dataset in ("LEOP", "PERG"):
        for bag in build_bags(caches, dataset)[:20]:
            assert all(comp.outer_fold == bag.outer_fold for comp in bag.components)


def test_bag_records_all_components(caches):
    leop = build_bags(caches, "LEOP")
    total = sum(len(b) for b in leop)
    table = caches.table()
    assert total == int((table["dataset"] == "LEOP").sum())


def test_bag_filter_is_partial_bag_safe(caches):
    # no unit may appear half-in half-out of the requested fold set
    bags = build_bags(caches, "LEOP", outer_folds={0})
    assert all(bag.outer_fold == 0 for bag in bags)


def test_unit_targets_present(caches):
    leop = build_bags(caches, "LEOP")
    targets = {b.target_binary for b in leop if b.target_binary is not None}
    assert targets <= {0, 1}
    # 21 LEOP subjects carry no diagnosis: 253 bags, 232 labeled
    n_labeled = sum(b.target_binary is not None for b in leop)
    assert n_labeled == 232
    assert n_labeled < len(leop)


def test_collate_component_shapes(caches):
    ds = ComponentDataset(caches, dataset="PERG", outer_folds={1})
    batch = collate_component_rows([ds[i] for i in range(3)])
    assert batch["signal"].shape == (3, 1, CANONICAL_SAMPLES)
    assert batch["valid_mask"].shape == (3, CANONICAL_SAMPLES)
    assert batch["ot"].shape == (3, OT_DIM)
    assert batch["physical"].shape == (3, len(PHYSICAL_FEATURE_NAMES))
    assert batch["outer_fold"].tolist() == [1, 1, 1]


def test_collate_bag_padding(caches):
    bags = build_bags(caches, "PERG", outer_folds={2})[:5]
    b = collate_bag_units(bags)
    L = max(len(x.components) for x in bags)
    assert b["signal"].shape == (5, L, 1, CANONICAL_SAMPLES)
    assert b["valid_mask"].shape == (5, L, CANONICAL_SAMPLES)
    assert b["ot"].shape == (5, L, OT_DIM)
    assert b["component_mask"].sum(axis=1).tolist() == [len(x.components) for x in bags]
    assert b["label"].shape == (5,)
    assert b["subject_ids"].shape == (5,)
    assert b["visit_ids"].shape == (5,)
    # padded rows must be NaN -> model must mask them before pooling
    padded = ~b["component_mask"]
    assert not padded.any() or bool(np.isnan(b["physical"][padded]).all())
    # label NaN only for bags without a diagnosis
    for bag, lab in zip(bags, b["label"], strict=True):
        assert np.isnan(lab) == (bag.target_binary is None)


def test_collate_rejects_empty_bag(caches):
    with pytest.raises(ValueError):
        collate_bag_units([])


def test_domain_balanced_plan():
    plan = list(domain_balanced_batch_indices(10, 7, 4, 3, seed=7))
    assert len(plan) == 3
    all_idx = set()
    for le, pe in plan:
        assert len(le) <= 4 and len(pe) <= 3
        assert len(le) > 0 and len(pe) > 0
        all_idx.update(le.tolist())
    assert all_idx == set(range(10))
    # deterministic
    plan2 = list(domain_balanced_batch_indices(10, 7, 4, 3, seed=7))
    assert [x.tolist() for x, _ in plan] == [x.tolist() for x, _ in plan2]


def test_domain_balanced_batch_size_validation():
    with pytest.raises(ValueError):
        list(domain_balanced_batch_indices(10, 7, 0, 3, seed=7))

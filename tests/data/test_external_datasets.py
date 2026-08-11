"""External dataset/bag/sampler tests (plan integration §11.2/§11.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.data.datasets import (
    LoadedCaches,
    build_bags,
    domain_balanced_batch_indices,
    domain_balanced_epoch_indices,
)
from pathway_erg.data.external_splits import build_external_splits
from pathway_erg.signal.external_cache import (
    DEFAULT_EXTERNAL_BINDING,
    external_cache_paths,
)

from tests._ext_synth import (
    build_synthetic_external,
    build_synthetic_v1_splits,
    build_synthetic_v4,
    external_fold_config,
    pre_cfg,
    write_synthetic_tables,
)


@pytest.fixture()
def ext_root(tmp_path):
    write_synthetic_tables(tmp_path)
    pre = pre_cfg()
    build_synthetic_v4(tmp_path, pre)
    build_synthetic_external(tmp_path, pre)
    build_synthetic_v1_splits(tmp_path)
    subjects = pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet")
    build_external_splits(tmp_path, subjects, visits, recordings, external_fold_config())
    return tmp_path


@pytest.fixture()
def ext_caches(ext_root):
    return LoadedCaches(
        ext_root,
        external_bindings=(DEFAULT_EXTERNAL_BINDING,),
        external_fold_version="external_v1",
    )


def test_external_bindings_require_fold_version(ext_root):
    with pytest.raises(ValueError, match="external_fold_version"):
        LoadedCaches(ext_root, external_bindings=(DEFAULT_EXTERNAL_BINDING,))


def test_external_bindings_missing_manifest(ext_root):
    import shutil

    shutil.rmtree(external_cache_paths(ext_root)["manifest"].parent, ignore_errors=True)
    with pytest.raises(ValueError, match="manifest not found"):
        LoadedCaches(ext_root, external_bindings=(DEFAULT_EXTERNAL_BINDING,), external_fold_version="external_v1")


def test_loaded_cache_without_bindings_matches_v1_behavior(ext_root):
    caches = LoadedCaches(ext_root)
    tbl = caches.table()
    assert set(tbl["dataset"]) == {"LEOP", "PERG"}


def test_table_includes_external_rows_and_folds(ext_caches):
    tbl = ext_caches.table()
    assert set(tbl["dataset"]) == {"LEOP", "PERG", "URFU", "FLINDERS"}
    assert not tbl["outer_fold"].isna().any()
    assert tbl["unit_id"].nunique() == len(
        tbl[["dataset", "unit_id"]].drop_duplicates()
    )


def test_unit_id_rules(ext_caches):
    tbl = ext_caches.table()
    for ds, rule in (("LEOP", "subject"), ("PERG", "visit"), ("URFU", "visit"), ("FLINDERS", "subject")):
        sub = tbl[tbl["dataset"] == ds]
        expected = (
            sub["global_visit_id"].astype(str)
            if rule == "visit"
            else sub["global_subject_id"].astype(str)
        )
        assert (sub["unit_id"] == expected).all(), ds


def test_urfu_visit_bags(ext_caches):
    bags = build_bags(ext_caches, "URFU")
    assert len(bags) == 5  # 4 subjects + 1 second visit
    visits = [b.visit_id for b in bags]
    assert len(set(visits)) == 5
    assert all(b.target_binary is not None for b in bags)


def test_flinders_subject_bags(ext_caches):
    bags = build_bags(ext_caches, "FLINDERS")
    assert len(bags) == 2
    assert all(b.visit_id is None for b in bags)
    assert all(b.target_binary == 0 for b in bags)


def test_flinders_positive_label_guard(tmp_path):
    write_synthetic_tables(tmp_path)
    pre = pre_cfg()
    build_synthetic_v4(tmp_path, pre)
    build_synthetic_external(tmp_path, pre)
    build_synthetic_v1_splits(tmp_path)
    visits = pd.read_parquet(tmp_path / "data" / "interim" / "visits.parquet")
    visits.loc[visits["dataset"] == "FLINDERS", "target_binary"] = 1
    visits.to_parquet(tmp_path / "data" / "interim" / "visits.parquet", index=False)
    subjects = pd.read_parquet(tmp_path / "data" / "interim" / "participants.parquet")
    recordings = pd.read_parquet(tmp_path / "data" / "interim" / "recordings.parquet")
    build_external_splits(tmp_path, subjects, visits, recordings, external_fold_config())
    caches = LoadedCaches(
        tmp_path,
        external_bindings=(DEFAULT_EXTERNAL_BINDING,),
        external_fold_version="external_v1",
    )
    with pytest.raises(ValueError, match="FLINDERS"):
        caches.table()


def test_build_bags_rejects_unknown_dataset(ext_caches):
    with pytest.raises(ValueError, match="dataset must be one of"):
        build_bags(ext_caches, "MARTIAN")


def test_component_dataset_external_fold_filter(ext_caches):
    from pathway_erg.data.datasets import ComponentDataset

    ds = ComponentDataset(ext_caches, "URFU", outer_folds={0})
    assert len(ds) >= 0
    rows = [ds[i] for i in range(len(ds))]
    if rows:
        assert all(r.outer_fold == 0 for r in rows)
        assert all(r.dataset == "URFU" for r in rows)


def test_bags_respect_external_folds(ext_caches):
    for ds in ("URFU", "FLINDERS"):
        bags = build_bags(ext_caches, ds, outer_folds={0, 1})
        assert all(b.outer_fold in {0, 1} for b in bags)


def test_collate_urfu_bag_batch(ext_caches):
    from pathway_erg.data.collate import collate_bag_units

    bags = build_bags(ext_caches, "URFU", outer_folds={0})
    if not bags:
        pytest.skip("no URFU bags in fold 0")
    batch = collate_bag_units(bags[:4])
    assert batch["signal"].ndim == 4
    assert batch["dataset"].tolist() == ["URFU"] * min(4, len(bags))
    assert batch["component_type"].shape[1] > 0


# -- multi-domain samplers ---------------------------------------------------
def test_domain_balanced_epoch_indices_covers_everything():
    sizes = {"LEOP": 5, "PERG": 7, "URFU": 3}
    batches = {"LEOP": 2, "PERG": 3, "URFU": 2}
    seen = {k: [] for k in sizes}
    steps = 0
    for idx in domain_balanced_epoch_indices(sizes, batches, seed=7):
        for k, arr in idx.items():
            seen[k].extend(arr.tolist())
        steps += 1
    assert steps == max(np.ceil(5 / 2), np.ceil(7 / 3), np.ceil(3 / 2))
    for k in sizes:
        assert sorted(seen[k]) == list(range(sizes[k]))


def test_domain_balanced_epoch_indices_deterministic():
    a = list(domain_balanced_epoch_indices({"A": 4, "B": 5}, {"A": 2, "B": 3}, seed=3))
    b = list(domain_balanced_epoch_indices({"A": 4, "B": 5}, {"A": 2, "B": 3}, seed=3))
    assert [x["A"].tolist() for x in a] == [x["A"].tolist() for x in b]
    assert [x["B"].tolist() for x in a] == [x["B"].tolist() for x in b]


def test_domain_balanced_epoch_indices_rejects_unknown_batch_domain():
    with pytest.raises(ValueError, match="unknown domains"):
        list(domain_balanced_epoch_indices({"A": 3}, {"B": 2}, seed=0))


def test_domain_balanced_epoch_indices_rejects_bad_batch():
    with pytest.raises(ValueError, match="must be >= 1"):
        list(domain_balanced_epoch_indices({"A": 3}, {"A": 0}, seed=0))


def test_two_domain_sampler_matches_legacy():
    legacy = list(domain_balanced_batch_indices(5, 7, 2, 3, seed=11))
    generic = list(
        domain_balanced_epoch_indices(
            {"LEOP": 5, "PERG": 7}, {"LEOP": 2, "PERG": 3}, seed=11
        )
    )
    assert [(a.tolist(), b.tolist()) for a, b in legacy] == [
        (x["LEOP"].tolist(), x["PERG"].tolist()) for x in generic
    ]


def test_bag_unit_protocol_and_stimulus_carried(ext_caches):
    bags = build_bags(ext_caches, "URFU")
    comps = bags[0].components
    assert comps
    assert all(c.protocol in {"Maximum 2.0", "Scotopic 2.0"} for c in comps)

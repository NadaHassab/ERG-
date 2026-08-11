"""N-domain SSL pretraining tests (plan integration §11.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from pathway_erg.config import DataConfig, load_config
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.training.ssl import SSLConfig, pretrain_ssl

from tests._ext_synth import (
    build_synthetic_external,
    build_synthetic_v1_splits,
    build_synthetic_v4,
    external_fold_config,
    pre_cfg,
    write_synthetic_tables,
)


def _data_cfg(root) -> DataConfig:
    import dataclasses

    return dataclasses.replace(
        load_config(DataConfig, "configs/data/local.yaml"), artifact_root=str(root)
    )


def _ssl_cfg(subdir: str, **overrides) -> SSLConfig:
    base = dict(
        name="test_external_ssl",
        fold_version="v1",
        routing_graph="correct",
        outer_folds=(0, 1, 2),
        exclude_fold=0,
        epochs=1,
        seed=7,
        device="cpu",
        ssl_dim=16,
        mask_len=8,
        lambda_mask=1.0,
        lambda_view=0.25,
        lambda_aug=0.25,
        lambda_geom=0.1,
        lambda_prior=0.01,
        output_subdir=subdir,
        plan_per_epoch=False,
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )
    base.update(overrides)
    return SSLConfig(**base)


@pytest.fixture(scope="module")
def ext_root(tmp_path_factory):
    from pathway_erg.data.external_splits import build_external_splits

    root = tmp_path_factory.mktemp("ssl_ext")
    write_synthetic_tables(root)
    pre = pre_cfg()
    build_synthetic_v4(root, pre)
    build_synthetic_external(root, pre)
    subjects = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    build_synthetic_v1_splits(root)
    build_external_splits(root, subjects, visits, recordings, external_fold_config())
    return root


def _load(ext_root):
    return LoadedCaches(
        ext_root,
        external_bindings=("external_v1",),
        external_fold_version="external_v1",
    )


def test_ssl_defaults_unchanged():
    cfg = SSLConfig(name="x")
    assert cfg.ssl_datasets == ("LEOP", "PERG")
    assert cfg.domain_batches is None
    assert cfg.plan_per_epoch is False
    assert cfg.external_bindings == ()
    assert cfg.external_fold_version is None


def test_ssl_rejects_unknown_domain(ext_root):
    cfg = _ssl_cfg("ssl_bad_domain", ssl_datasets=("LEOP", "MARTIAN"))
    with pytest.raises(ValueError, match="unknown datasets"):
        pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))


def test_ssl_requires_two_domains(ext_root):
    cfg = _ssl_cfg("ssl_one_domain", ssl_datasets=("LEOP",))
    with pytest.raises(ValueError, match="at least two domains"):
        pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))


def test_ssl_two_domain_without_external_bindings(ext_root):
    cfg = _ssl_cfg(
        "ssl_two_domain",
        ssl_datasets=("LEOP", "PERG"),
        domain_batches=None,
        external_bindings=(),
        external_fold_version=None,
    )
    out, log = pretrain_ssl(cfg, _data_cfg(ext_root))
    assert out.is_file()
    assert any(k.startswith("leop_") for k in log.per_domain)
    assert any(k.startswith("perg_") for k in log.per_domain)


def test_ssl_four_domain_pretrains(ext_root):
    cfg = _ssl_cfg(
        "ssl_four_domain",
        ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"),
        domain_batches={"LEOP": 8, "PERG": 8, "URFU": 4, "FLINDERS": 4},
    )
    out, log = pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))
    assert out.is_file()
    assert out.name == "final.pt"
    for prefix in ("leop_", "perg_", "urfu_", "flinders_"):
        assert any(k.startswith(prefix) for k in log.per_domain)
    assert log.train_loss and len(log.train_loss) >= 1


def test_ssl_four_domain_plan_per_epoch(ext_root):
    cfg = _ssl_cfg(
        "ssl_ppe",
        ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"),
        domain_batches={"LEOP": 8, "PERG": 8, "URFU": 4, "FLINDERS": 4},
        plan_per_epoch=True,
        epochs=2,
    )
    out, log = pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))
    assert out.is_file()
    assert len(log.train_loss) == 2


def test_ssl_exclude_fold_respected_in_domains(ext_root):
    caches = _load(ext_root)
    cfg = _ssl_cfg(
        "ssl_excl1",
        ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"),
        domain_batches={"LEOP": 8, "PERG": 8, "URFU": 4, "FLINDERS": 4},
        exclude_fold=1,
    )
    out, _ = pretrain_ssl(cfg, _data_cfg(ext_root), caches=caches)
    import torch

    payload = torch.load(out, map_location="cpu")
    assert payload["exclude_fold"] == 1
    assert sorted(payload["train_folds"]) == [0, 2]


def test_ssl_checkpoint_records_all_domains(ext_root):
    cfg = _ssl_cfg("ssl_domains", ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"))
    out, _ = pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))
    import torch

    payload = torch.load(out, map_location="cpu")
    assert set(payload["n_components"]) == {"LEOP", "PERG", "URFU", "FLINDERS"}
    assert all(v > 0 for v in payload["n_components"].values())


def test_ssl_completed_run_refused(ext_root):
    subdir = "ssl_completed"
    cfg = _ssl_cfg(
        subdir,
        ssl_datasets=("LEOP", "PERG", "URFU", "FLINDERS"),
        domain_batches={"LEOP": 8, "PERG": 8, "URFU": 4, "FLINDERS": 4},
    )
    pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))
    with pytest.raises(FileExistsError, match="completed SSL run"):
        pretrain_ssl(cfg, _data_cfg(ext_root), caches=_load(ext_root))

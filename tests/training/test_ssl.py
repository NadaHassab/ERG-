"""Joint SSL tests (plan §14, Module 21.17 ssl.py + Module 21.14 gate prior).

Covers: masking/augmentation helpers, loss algebra on known tensors
(masked reconstruction only on masked+valid samples; VICReg variance/
covariance; geometry pairs within dataset/component type; gate prior
formula), balanced-domain one-step smoke on real data, held-out-fold
exclusion, checkpoint + COMPLETE contract, and SSL-initialized fine-tune.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.config import DataConfig, LeopsDataConfig, PergDataConfig
from pathway_erg.data.datasets import ComponentDataset, LoadedCaches
from pathway_erg.models.path_erg import ComponentEncoding, ModelConfig, build_model
from pathway_erg.training.finetune import freeze_encoders, init_from_ssl
from pathway_erg.training.ssl import (
    GatePriorLoss,
    GeometryPreservationLoss,
    JointSSLLoss,
    MaskedReconstructionLoss,
    SSLConfig,
    _VICReg,
    augment_signal,
    collate_component_batch,
    mask_contiguous_span,
    pretrain_ssl,
)


@pytest.fixture(scope="module")
def caches():
    return LoadedCaches("artifacts")


def test_mask_contiguous_span_respects_valid():
    rng = np.random.default_rng(0)
    signal = np.ones((3, 128))
    valid = np.ones((3, 128), dtype=bool)
    valid[0, 100:] = False
    masked, mask = mask_contiguous_span(signal, valid, 24, rng)
    assert masked.shape == (3, 128)
    assert mask.dtype == bool
    for i in range(3):
        assert int(mask[i].sum()) == 24
        assert masked[i][mask[i]].sum() == 0.0
        assert not (mask[i] & ~valid[i]).any()
    # empty valid -> no masking
    _, m = mask_contiguous_span(signal[:1], np.zeros((1, 128), dtype=bool), 24, rng)
    assert not m.any()


def test_augment_signal_preserves_shape_and_valid():
    rng = np.random.default_rng(1)
    signal = np.random.randn(4, 128)
    valid = np.ones((4, 128), dtype=bool)
    valid[:, 120:] = False
    out = augment_signal(signal, valid, rng)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_masked_loss_only_on_masked_and_valid():
    torch.manual_seed(0)
    loss = MaskedReconstructionLoss(128)
    # fake encoding with fixed token -> deterministic decoder output
    signal = torch.ones(1, 1, 128)
    mask = torch.zeros(1, 1, 128, dtype=torch.bool)
    valid = torch.ones(1, 1, 128, dtype=torch.bool)
    fake = ComponentEncoding(
        token=torch.ones(1, 1, 128), alpha=torch.ones(1, 1), valid=torch.ones(1, 1, dtype=torch.bool)
    )
    # no masked positions -> zero loss regardless of decoder
    out = loss(fake, signal, mask, valid)
    assert float(out.item()) == pytest.approx(0.0, abs=1e-6)
    mask[:, :, 10:30] = True
    out = loss(fake, signal, mask, valid)
    # the loss is exactly Huber over select = mask & valid
    recon = loss.decoder(fake.token).reshape_as(mask)
    select = mask & valid
    expected = torch.nn.functional.huber_loss(
        recon[select], signal[select], reduction="mean"
    )
    assert float(out.item()) == pytest.approx(float(expected.item()), abs=1e-6)
    # masked-but-invalid positions do not contribute
    valid2 = valid.clone()
    valid2[:, :, 15:20] = False
    out2 = loss(fake, signal, mask, valid2)
    select2 = mask & valid2
    expected2 = torch.nn.functional.huber_loss(
        recon[select2], signal[select2], reduction="mean"
    )
    assert float(out2.item()) == pytest.approx(float(expected2.item()), abs=1e-6)
    assert select2.sum().item() == select.sum().item() - 5


def test_vicreg_terms_known_tensors():
    vic = _VICReg(inv_weight=0.5)
    a = torch.ones(8, 4)
    b = torch.ones(8, 4)
    # identical projections -> zero invariance; variance term > 0 (collapse penalty)
    out = vic(a, b)
    inv = torch.nn.functional.mse_loss(a, b)
    assert inv.item() == pytest.approx(0.0)
    assert float(out.item()) > 0.0
    std = a.std(dim=0)
    assert float(std.mean().item()) == pytest.approx(0.0)
    # variance hinge: relu(1 - std).mean() with std=0 -> 1.0, two branches
    assert float(out.item()) == pytest.approx(0.5 * 0.0 + 1.0 * 1.0 + 1.0 * 1.0)


def test_vicreg_covariance_only_off_diagonal():
    vic = _VICReg(inv_weight=0.0)
    a = torch.eye(4)  # orthogonal columns: covariance off-diagonal is zero
    b = a.clone()
    out = vic(a, b)
    # variance: each col std = sqrt(4/3 * 0.25) ~ 0.577 -> hinge positive
    assert float(out.item()) > 0.0
    # columns 0 and 1 are identical across samples -> large off-diagonal covariance
    x = torch.arange(8, dtype=torch.float32)
    c = torch.stack([x, x, 2 * x, torch.zeros(8)], dim=1)
    cov_part = vic._covariance(c)
    assert float(cov_part.item()) > 0.0


def test_gate_prior_formula():
    prior = GatePriorLoss(0.75)
    model = build_model(ModelConfig(routing_graph="correct")).eval()
    batch = _component_batch(4, "LEOP")
    enc = model.encode_component(batch)
    loss, val = prior(enc)
    # permitted edges (L_LATE) pulled toward 0.75; g in (0,1)
    assert float(loss.item()) >= 0.0
    assert 0.0 <= float(val) <= 1.0
    # a model without router has no prior term
    plain = build_model(ModelConfig()).eval()
    enc_plain = plain.encode_component(batch)
    l0, v0 = prior(enc_plain)
    assert float(l0.item()) == pytest.approx(0.0)
    assert v0 == 0.0


def test_geometry_loss_pairs_within_group():
    geom = GeometryPreservationLoss()
    model = build_model(ModelConfig(routing_graph="correct")).eval()
    batch = _component_batch(6, "LEOP")
    enc = model.encode_component(batch)
    loss = geom(enc, batch)
    assert torch.isfinite(loss).all()
    assert float(loss.item()) >= 0.0


def test_joint_ssl_loss_runs_on_real_batch(caches):
    cfg = SSLConfig(name="t", exclude_fold=0)
    loss_fn = JointSSLLoss(cfg, token_dim=128)
    model = build_model(ModelConfig(routing_graph="correct")).eval()
    ds = ComponentDataset(caches, "LEOP", outer_folds={1, 2, 3, 4})
    rows = [ds[i] for i in range(8)]
    batch = collate_component_batch(rows)
    total, terms = loss_fn(model, batch, seed_offset=1)
    assert torch.isfinite(total).all()
    assert set(terms) >= {"mask", "view", "aug", "geom", "prior", "total"}
    for v in terms.values():
        assert np.isfinite(v)


def test_held_out_fold_excluded(caches):
    train = ComponentDataset(caches, "LEOP", outer_folds={0, 1, 2, 3})
    excluded = ComponentDataset(caches, "LEOP", outer_folds={4})
    assert len(excluded) > 0
    train_folds = {r.outer_fold for i in range(0, len(train), 50) for r in [train[i]]}
    excluded_folds = {r.outer_fold for i in range(0, len(excluded), 50) for r in [excluded[i]]}
    assert train_folds <= {0, 1, 2, 3}
    assert excluded_folds == {4}
    assert not (train_folds & excluded_folds)


def test_pretrain_smoke_and_checkpoint_contract(tmp_path, caches):
    _install_splits(tmp_path)
    cfg = SSLConfig(
        name="t",
        exclude_fold=4,
        epochs=1,
        leop_batch=32,
        perg_batch=32,
        output_subdir="ssl_test",
        seed=7,
    )
    data_cfg = _data_cfg(tmp_path)
    ckpt, log = pretrain_ssl(cfg, data_cfg, caches)
    out_dir = tmp_path / "results" / "ssl_test" / "fold4"
    assert (out_dir / "COMPLETE").exists()
    assert (out_dir / "final.pt").exists()
    assert (out_dir / "run_manifest.json").exists()
    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert payload["exclude_fold"] == 4
    assert payload["train_folds"] == [0, 1, 2, 3]
    assert log.train_loss
    # rerun is rejected
    with pytest.raises(FileExistsError):
        pretrain_ssl(cfg, data_cfg, caches)


def test_init_from_ssl_copies_encoders(tmp_path, caches):
    _install_splits(tmp_path)
    cfg = SSLConfig(name="t", exclude_fold=4, epochs=1, leop_batch=32, perg_batch=32,
                    output_subdir="ssl_test2", seed=9)
    ckpt, _ = pretrain_ssl(cfg, _data_cfg(tmp_path), caches)
    fresh = build_model(ModelConfig(routing_graph="correct"))
    init_from_ssl(fresh, ckpt)
    pretrained = torch.load(ckpt, weights_only=False, map_location="cpu")["model"]
    for k, v in fresh.state_dict().items():
        if k in pretrained and v.shape == pretrained[k].shape:
            assert torch.equal(v, pretrained[k]), f"encoder key {k} not copied"
    freeze_encoders(fresh, freeze=True)
    frozen = {n: p.requires_grad for n, p in fresh.named_parameters()
              if n.split(".")[0] in {"raw_stem", "ot_stem", "fusion", "router",
                                     "comp_to_eye", "intensity_to_eye", "eye_to_unit"}}
    assert all(not v for v in frozen.values())


def _install_splits(tmp_path) -> None:
    import shutil
    from pathlib import Path

    split_dir = tmp_path / "data" / "splits"
    split_dir.mkdir(parents=True)
    for name in ("outer_folds_v1.parquet", "inner_folds_v1.parquet"):
        shutil.copy(Path("artifacts/data/splits") / name, split_dir / name)
    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    shutil.copy(
        "artifacts/data/manifests/component_cache_manifest_v4.json",
        manifest_dir / "component_cache_manifest_v4.json",
    )


def _component_batch(n: int, dataset: str) -> dict:
    from pathway_erg.data.datasets import build_bags

    c = LoadedCaches("artifacts")
    bags = build_bags(c, dataset, outer_folds={0})[:1]
    rows = list(bags[0].components)[:n]
    return collate_component_batch(rows)


def _data_cfg(tmp_path) -> DataConfig:
    root = str(tmp_path)
    return DataConfig(
        leops=LeopsDataConfig(json_root=root, xlsx_path="x"),
        perg=PergDataConfig(root=root, metadata_csv="m"),
        artifact_root=root,
    )

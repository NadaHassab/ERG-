"""Loss/sampler/trainer tests (plan Module 21.17).

Plan-mandated tests: loss algebra on known tensors, balanced sampling,
held-out-ID assertion, one optimizer update smoke test, resume
equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathway_erg.data.collate import collate_bag_units
from pathway_erg.data.datasets import LoadedCaches
from pathway_erg.models.path_erg import build_model
from pathway_erg.training.losses import FoldWeightedBCE, positive_class_weight
from pathway_erg.training.samplers import BagSampler, make_fold_bags
from pathway_erg.training.trainer import TrainConfig, Trainer


def test_positive_class_weight_algebra():
    # 1 pos, 3 neg -> pos_weight 3.0 (BCEWithLogitsLoss semantics)
    assert positive_class_weight(np.array([1, 0, 0, 0])) == pytest.approx(3.0)
    # balanced -> 1.0
    assert positive_class_weight(np.array([1, 0, 1, 0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        positive_class_weight(np.array([0, 0, 0]))


def test_fold_weighted_bce_on_known_tensors():
    crit = FoldWeightedBCE(weight=2.0)
    # logit 0 -> sigmoid 0.5; label 1; loss = -w * log(0.5) = 2*0.6931
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    loss = crit(logits, labels)
    expected = -(2.0 * np.log(0.5) + np.log(0.5)) / 2
    assert float(loss) == pytest.approx(float(expected), abs=1e-4)


def test_fold_weighted_bce_ignores_nan_labels():
    crit = FoldWeightedBCE(weight=1.0)
    logits = torch.tensor([10.0, -10.0, 0.0])
    labels = torch.tensor([1.0, 0.0, np.nan])
    loss = crit(logits, labels)
    assert torch.isfinite(loss)
    # NaN row must not contribute: same as computing on the 2 valid rows
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], labels[:2])
    assert float(loss) == pytest.approx(float(ref), abs=1e-6)


@pytest.fixture(scope="module")
def caches():
    return LoadedCaches("artifacts")


def test_sampler_fold_restriction(caches):
    bags = make_fold_bags(caches, "LEOP", fold=0)
    s = BagSampler(bags, folds={0}, batch_size=8, seed=1)
    assert all(b.outer_fold == 0 for b in s.bags)
    batch = next(iter(s))
    assert len(batch) == 8
    assert set(batch.tolist()) <= set(range(len(s.bags)))


def test_sampler_rejects_foreign_fold(caches):
    bags = make_fold_bags(caches, "LEOP", fold=0)
    with pytest.raises(ValueError, match="not present in bags"):
        BagSampler(bags, folds={1}, batch_size=8, seed=1)


def test_held_out_ids_never_in_sampler(caches):
    # fold 1's units must not be reachable from a fold-0 sampler
    bags = make_fold_bags(caches, "PERG", fold=0)
    s = BagSampler(bags, folds={0}, batch_size=8, seed=1)
    held_ids = {b.unit_id for b in make_fold_bags(caches, "PERG", fold=1)}
    ids = {b.unit_id for b in s.bags}
    assert not (ids & held_ids)


def test_sampler_resume_equivalence(caches):
    bags = make_fold_bags(caches, "LEOP", fold=0)
    a = BagSampler(bags, folds={0}, batch_size=8, seed=7)
    it = iter(a)
    next(it)
    next(it)
    state = a.state()
    batch3_orig = next(it).tolist()

    b = BagSampler(bags, folds={0}, batch_size=8, seed=7)
    itb = iter(b)
    next(itb)
    next(itb)
    b.resume(step=2, state=state)
    # resumed stream must yield the same 3rd batch as the original
    batch3_resumed = next(itb).tolist()
    assert batch3_resumed == batch3_orig


def test_one_optimizer_update_smoke(caches):
    torch.manual_seed(0)
    # pick labeled bags with both classes (PERG fold 1: 47 pos / 22 neg)
    cand = [b for b in make_fold_bags(caches, "PERG", fold=1) if b.target_binary is not None]
    pos = [b for b in cand if b.target_binary == 1][:4]
    neg = [b for b in cand if b.target_binary == 0][:4]
    bags = pos + neg
    batch = collate_bag_units(bags)
    model = build_model()
    labels = np.array([b.target_binary for b in bags], dtype=np.float32)
    pos_w = positive_class_weight(labels)
    crit = FoldWeightedBCE(pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logits = model(batch, "PERG")
    loss = crit(logits, torch.as_tensor(labels))
    loss.backward()
    opt.step()
    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "optimizer step did not change weights"
    assert torch.isfinite(loss)


def test_save_load_checkpoint_roundtrip(caches, tmp_path):
    bags = make_fold_bags(caches, "PERG", fold=1)
    s = BagSampler(bags, folds={1}, batch_size=8, seed=3)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = Trainer(model, TrainConfig(task="PERG", outer_fold=1))
    from pathway_erg.training.trainer import _WarmupCosine

    sched = _WarmupCosine(opt, warmup_steps=2, total_steps=10)
    path = tmp_path / "ckpt.pt"
    tr.save(path, s, opt, sched, epoch=3)
    model2 = build_model()
    tr2 = Trainer(model2, TrainConfig(task="PERG", outer_fold=1))
    epoch = tr2.resume(path)
    assert epoch == 3
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], model2.state_dict()[k])

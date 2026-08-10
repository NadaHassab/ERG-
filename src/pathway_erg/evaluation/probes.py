"""Linear probes on frozen component embeddings (plan E12, item 22).

Probes answer *what does the frozen model encode per component*.  They are
fit on the checkpoint's training folds and evaluated on the one outer fold
that checkpoint never saw (``test_fold``), so probe performance shares the
model's generalization contract.

Targets:

- ``component_identity`` — does a generic component token know its
  waveform branch (L_EARLY_A ... P_LATE)?  E12 desired pattern: shared and
  routed representations carry branch structure.
- ``dataset_identity`` — is the representation simply a dataset detector
  (a warning signal when probe performance is high)?
- ``flash_intensity`` (LEOP only), ``peak_to_peak``, ``duration`` —
  do the embeddings preserve continuous stimulus/physiology?

Streams: ``fused`` (128-d pooled local token) always; ``shared``/``private``
(64-d expert outputs) only for routed checkpoints.

All fits are linear (logistic OVR / ridge) on frozen embeddings, so branch
fidelity is read out, never trained.  Metrics are reported on the held-out
fold with a unit-level cluster bootstrap CI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from ..data.datasets import ComponentDataset, LoadedCaches
from ..models.path_erg import PathModel
from ..training.separate import SeparateTrainingConfig, build_stage_model
from ..training.ssl import collate_component_batch

PHYSICAL_NAMES = (
    "log_mass_pos",
    "log_mass_neg",
    "peak_to_peak_uv",
    "max_rising_slope_uv_per_ms",
    "max_falling_slope_uv_per_ms",
    "area_above_ref_uv_ms",
    "area_below_ref_uv_ms",
    "duration_ms",
)
P2P_INDEX = PHYSICAL_NAMES.index("peak_to_peak_uv")
DURATION_INDEX = PHYSICAL_NAMES.index("duration_ms")

STREAM_NAMES = ("fused", "shared", "private")
FRAME_NAMES = ("all", "LEOP", "PERG")
TARGET_NAMES = (
    "component_identity",
    "dataset_identity",
    "flash_intensity",
    "peak_to_peak",
    "duration",
)


@dataclass(frozen=True)
class ProbeFrame:
    """Per-component frozen embeddings plus targets for one row set."""

    unit_ids: np.ndarray
    subject_ids: np.ndarray
    dataset: np.ndarray
    component_id: np.ndarray
    streams: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]

    def n(self) -> int:
        return len(self.unit_ids)


@dataclass(frozen=True)
class ProbeResult:
    """One probe outcome on the held-out fold."""

    frame: str
    stream: str
    target: str
    kind: str  # class | reg
    metric: str  # macro_ovr_auroc | roc_auc | pearson_r
    value: float
    ci_low: float | None
    ci_high: float | None
    n_train: int
    n_eval: int
    n_units_eval: int


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and bool(np.isnan(value))


def probe_targets(rows: Sequence) -> dict[str, np.ndarray]:
    """Numeric targets per component (NaN = not applicable for that row)."""
    rows = list(rows)
    n = len(rows)
    identity_codes = {
        c: i for i, c in enumerate(sorted({str(r.component_id) for r in rows}))
    }
    ds_codes = {d: i for i, d in enumerate(sorted({r.dataset for r in rows}))}
    out: dict[str, np.ndarray] = {
        "component_identity": np.asarray(
            [identity_codes[str(r.component_id)] for r in rows], dtype=np.float64
        ),
        "dataset_identity": np.asarray(
            [ds_codes[r.dataset] for r in rows], dtype=np.float64
        ),
        "flash_intensity": np.asarray(
            [
                np.nan if _is_nan(r.stimulus_value) else float(r.stimulus_value)
                for r in rows
            ],
            dtype=np.float64,
        ),
        "peak_to_peak": np.full(n, np.nan, dtype=np.float64),
        "duration": np.full(n, np.nan, dtype=np.float64),
    }
    for i, r in enumerate(rows):
        phys = np.asarray(r.physical, dtype=np.float64)
        if phys.size != len(PHYSICAL_NAMES):
            continue
        p2p = float(phys[P2P_INDEX])
        if np.isfinite(p2p) and p2p >= 0:
            out["peak_to_peak"][i] = np.log1p(p2p)
        dur = float(phys[DURATION_INDEX])
        if np.isfinite(dur) and dur >= 0:
            out["duration"][i] = np.log1p(dur)
    return out


def encode_component_frame(
    model: PathModel,
    rows: Sequence,
    batch_size: int = 128,
    device: str = "cpu",
) -> ProbeFrame:
    """Frozen per-component embeddings for a list of ComponentRow."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot encode an empty component set")
    arrays: dict[str, list[np.ndarray]] = {"fused": []}
    if model.router is not None:
        arrays["shared"] = []
        arrays["private"] = []
    for start in range(0, len(rows), batch_size):
        batch = collate_component_batch(rows[start : start + batch_size])
        with torch.no_grad():
            enc = model.encode_component(batch)
        arrays["fused"].append(enc.token[:, 0, :].detach().cpu().numpy())
        if model.router is not None:
            arrays["shared"].append(enc.shared[:, 0, :].detach().cpu().numpy())
            arrays["private"].append(enc.private[:, 0, :].detach().cpu().numpy())
    return ProbeFrame(
        unit_ids=np.asarray([r.unit_id for r in rows], dtype=object),
        subject_ids=np.asarray([r.subject_id for r in rows], dtype=object),
        dataset=np.asarray([r.dataset for r in rows], dtype=object),
        component_id=np.asarray([str(r.component_id) for r in rows], dtype=object),
        streams={k: np.concatenate(v, axis=0) for k, v in arrays.items()},
        targets=probe_targets(rows),
    )


def _subframe(
    frame: ProbeFrame, dataset: str | None
) -> ProbeFrame:
    if dataset is None:
        return frame
    sel = np.flatnonzero(frame.dataset == dataset)
    return ProbeFrame(
        unit_ids=frame.unit_ids[sel],
        subject_ids=frame.subject_ids[sel],
        dataset=frame.dataset[sel],
        component_id=frame.component_id[sel],
        streams={k: v[sel] for k, v in frame.streams.items()},
        targets={k: v[sel] for k, v in frame.targets.items()},
    )


def evaluate_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    clusters: np.ndarray,
    n_reps: int = 100,
    seed: int = 7,
) -> ProbeResult:
    """Fit once on train, evaluate on test with a unit bootstrap CI."""
    if kind not in {"class", "reg"}:
        raise ValueError(f"unknown probe kind {kind!r}")
    if len(clusters) != len(y_test):
        raise ValueError("cluster ids must match evaluation rows")
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("expected 2-D feature matrices")
    if y_train.size < 2 or y_test.size < 2:
        raise ValueError("need at least 2 train and 2 eval rows")

    if kind == "class":
        labels = np.unique(y_train)
        if labels.size < 2:
            raise ValueError("classification probe needs >= 2 classes")
        clf = LogisticRegression(C=1e-3, max_iter=2000).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        classes = clf.classes_
        rng = np.random.default_rng(seed)
        unit_ids = np.unique(clusters)
        if classes.size == 2:
            metric = "roc_auc"
            value = roc_auc_score(y_test, proba[:, 1])

            def classify(sample: np.ndarray) -> float:
                if np.unique(y_test[sample]).size < 2:
                    return float("nan")
                return float(roc_auc_score(y_test[sample], proba[sample][:, 1]))

        else:
            metric = "macro_ovr_auroc"
            value = roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro", labels=classes
            )

            def classify(sample: np.ndarray) -> float:
                if np.unique(y_test[sample]).size < 2:
                    return float("nan")
                return float(
                    roc_auc_score(
                        y_test[sample],
                        proba[sample],
                        multi_class="ovr",
                        average="macro",
                        labels=classes,
                    )
                )

    else:
        ridge = Ridge(alpha=1.0).fit(X_train, y_train)
        pred = ridge.predict(X_test)
        metric = "pearson_r"
        rng = np.random.default_rng(seed)
        unit_ids = np.unique(clusters)

        def classify(sample: np.ndarray) -> float:
            residual = pred[sample] - y_test[sample]
            if float(np.std(residual)) == 0.0:
                return float("nan")
            return float(np.corrcoef(pred[sample], y_test[sample])[0, 1])

        value = classify(np.arange(len(y_test)))

    boots: list[float] = []
    for _ in range(n_reps):
        chosen = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        sample = np.flatnonzero(np.isin(clusters, chosen))
        boots.append(classify(sample))
    valid = np.asarray(boots)
    valid = valid[np.isfinite(valid)]
    if valid.size >= 2:
        ci_low, ci_high = (
            float(np.percentile(valid, 2.5)),
            float(np.percentile(valid, 97.5)),
        )
    else:
        ci_low = ci_high = None
    return ProbeResult(
        frame="",
        stream="",
        target="",
        kind=kind,
        metric=metric,
        value=float(value),
        ci_low=ci_low,
        ci_high=ci_high,
        n_train=len(y_train),
        n_eval=len(y_test),
        n_units_eval=len(unit_ids),
    )


def run_probe_battery(
    model: PathModel,
    caches: LoadedCaches,
    test_fold: int,
    outer_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    batch_size: int = 128,
    n_reps: int = 100,
    seed: int = 7,
    device: str = "cpu",
) -> list[ProbeResult]:
    """Probe battery for one checkpoint: fit on train folds, eval on test."""
    if test_fold not in outer_folds:
        raise ValueError(f"test fold {test_fold} not in {outer_folds}")
    train_folds = set(outer_folds) - {test_fold}
    train_frame = encode_component_frame(
        model, list(ComponentDataset(caches, outer_folds=train_folds)),
        batch_size=batch_size, device=device,
    )
    test_frame = encode_component_frame(
        model, list(ComponentDataset(caches, outer_folds={test_fold})),
        batch_size=batch_size, device=device,
    )
    streams = sorted(train_frame.streams)
    results: list[ProbeResult] = []
    for frame_name in FRAME_NAMES:
        dataset = None if frame_name == "all" else frame_name
        train_sub = _subframe(train_frame, dataset)
        test_sub = _subframe(test_frame, dataset)
        if train_sub.n() == 0 or test_sub.n() == 0:
            continue
        for stream in streams:
            X_train = train_sub.streams[stream]
            X_test = test_sub.streams[stream]
            for target, kind in _target_kinds():
                y_train = train_sub.targets[target]
                y_test = test_sub.targets[target]
                fit_mask = np.isfinite(y_train)
                eval_mask = np.isfinite(y_test)
                if fit_mask.sum() < 2 or eval_mask.sum() < 2:
                    continue
                if kind == "class":
                    classes = np.unique(y_train[fit_mask])
                    if classes.size < 2:
                        continue
                    eval_mask &= np.isin(y_test, classes)
                    if eval_mask.sum() < 2:
                        continue
                result = evaluate_probe(
                    X_train[fit_mask],
                    y_train[fit_mask],
                    X_test[eval_mask],
                    y_test[eval_mask],
                    kind=kind,
                    clusters=test_sub.unit_ids[eval_mask],
                    n_reps=n_reps,
                    seed=seed,
                )
                results.append(
                    ProbeResult(
                        frame=frame_name,
                        stream=stream,
                        target=target,
                        kind=result.kind,
                        metric=result.metric,
                        value=result.value,
                        ci_low=result.ci_low,
                        ci_high=result.ci_high,
                        n_train=result.n_train,
                        n_eval=result.n_eval,
                        n_units_eval=result.n_units_eval,
                    )
                )
    if not results:
        raise ValueError("probe battery produced no rows")
    return results


def _target_kinds() -> tuple[tuple[str, str], ...]:
    return (
        ("component_identity", "class"),
        ("dataset_identity", "class"),
        ("flash_intensity", "reg"),
        ("peak_to_peak", "reg"),
        ("duration", "reg"),
    )


def load_model_from_checkpoint(path: str | Path) -> tuple[PathModel, SeparateTrainingConfig, int]:
    """Rebuild the exact staged model from ``final.pt`` (or inner fold)."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SeparateTrainingConfig(**payload["experiment"])
    seed = int(payload["seed"])
    model = build_stage_model(cfg, seed)
    model.load_state_dict(payload["model"])
    return model, cfg, seed


def save_probe_report(results: Sequence[ProbeResult], out_path: str | Path) -> Path:
    """Parquet report + JSON summary next to the parquet file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(r) for r in results])
    keep = frame["metric"].isin(("macro_ovr_auroc", "roc_auc", "pearson_r"))
    height = float(frame[keep]["value"].mean()) if keep.any() else float("nan")
    summary = {
        "n_probes": len(frame),
        "mean_metric": height,
        "rows": [
            {key: (None if key.endswith(("_low", "_high")) and pd.isna(v) else v)
             for key, v in asdict(r).items()}
            for r in results
        ],
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    frame.to_parquet(out_path, index=False)
    return out_path

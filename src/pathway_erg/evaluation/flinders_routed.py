"""Plan §11.4 — FLINDERS calibration-only probe on frozen routed tokens.

Evaluates the four-domain SSL encoder's treatment of healthy FLINDERS
normatives against matched LEOP controls *without any supervised FLINDERS
endpoint*: only routed-token distributions are compared (KS per token
dimension, subject-level aggregation, bootstrap CI).  The FLINDERS head
stays forbidden; this module asserts the checkpoint carries none.

Results write only under ``artifacts/results/external_v1/`` (plan §11.4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from ..data.datasets import ComponentDataset, LoadedCaches
from ..models.path_erg import ModelConfig, build_model
from ..training.ssl import SSLConfig

FLINDERS_PROTOCOLS = ("LA3", "30Hz", "OPS", "DA00x")


@dataclass(frozen=True)
class FlindersRoutedConfig:
    output_subdir: str = "external_v1/flinders_calibration"
    batch_size: int = 128
    seed: int = 7
    n_boot_reps: int = 100
    protocols: tuple[str, ...] = FLINDERS_PROTOCOLS
    reference_report: str = "artifacts/results/flinders_calibration/calibration_report.json"


def load_ssl_checkpoint(path: str | Path, fold: int):
    """Rebuild the exact staged SSL model and config (plan §11.4, headless)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "config" not in payload or "model" not in payload:
        raise ValueError(f"not an SSL checkpoint: {path} (no config/model payload)")
    cfg = SSLConfig(**payload["config"])
    if payload.get("exclude_fold") != fold:
        raise ValueError(
            f"checkpoint fold mismatch: payload exclude_fold="
            f"{payload.get('exclude_fold')!r} vs requested {fold!r}"
        )
    if "FLINDERS" not in cfg.ssl_datasets:
        raise ValueError("checkpoint was not trained with FLINDERS in ssl_datasets")
    if not cfg.external_bindings or cfg.external_fold_version is None:
        raise ValueError(
            "checkpoint was not trained with an external binding / fold version"
        )
    if any(k.startswith("heads.FLINDERS.") for k in payload["model"]):
        raise ValueError("checkpoint contains a forbidden FLINDERS head")
    model = build_model(
        ModelConfig(
            routing_graph=cfg.routing_graph,
            stems_seed=cfg.seed,
            agg_seed=cfg.seed,
            head_seed=cfg.seed,
        )
    )
    model.load_state_dict(payload["model"])
    return model, cfg


def _subject_tokens(
    caches: LoadedCaches,
    model: torch.nn.Module,
    dataset: str,
    fold: int,
    protocols: tuple[str, ...],
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Per-component routed tokens for one dataset on the held-out fold."""
    from .probes import encode_component_frame

    rows = [
        row
        for row in ComponentDataset(caches, dataset, outer_folds={fold})
        if row.protocol in protocols
    ]
    if not rows:
        return pd.DataFrame(
            columns=["subject_id", "dataset", "protocol", "component_id", "stream", "token"]
        )
    model.eval()
    frame = encode_component_frame(model, rows, batch_size=batch_size, device=device)
    records = []
    for stream, values in frame.streams.items():
        for i in range(values.shape[0]):
            records.append(
                {
                    "subject_id": frame.subject_ids[i],
                    "dataset": dataset,
                    "protocol": rows[i].protocol,
                    "component_id": frame.component_id[i],
                    "stream": stream,
                    "token": values[i].astype(np.float32),
                }
            )
    return pd.DataFrame(records)


def _subject_mean_tokens(tokens: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean token per (subject, protocol, stream) across components."""
    out: dict[str, pd.DataFrame] = {}
    for stream, group in tokens.groupby("stream"):
        means = []
        for (subject, protocol), sub in group.groupby(["subject_id", "protocol"]):
            means.append(
                {
                    "subject_id": subject,
                    "protocol": protocol,
                    "token": np.mean(np.stack(sub["token"].to_numpy()), axis=0),
                }
            )
        out[stream] = pd.DataFrame(means)
    return out


def _median_dim_ks(fl: np.ndarray, controls: np.ndarray) -> float:
    """Median over token dimensions of the 2-sample KS statistic."""
    stats = [
        ks_2samp(fl[:, dim], controls[:, dim]).statistic
        for dim in range(fl.shape[1])
    ]
    return float(np.median(stats))


def run_flinders_routed_calibration(
    artifact_root: str | Path,
    checkpoint_path: str | Path,
    fold: int,
    config: FlindersRoutedConfig | None = None,
    device: str = "cpu",
) -> dict:
    """Headless routed-token calibration for one held-out fold."""
    config = config or FlindersRoutedConfig()
    artifact_root = Path(artifact_root)
    model, ssl_cfg = load_ssl_checkpoint(checkpoint_path, fold)
    model.to(device)
    caches = LoadedCaches(
        artifact_root,
        fold_version=ssl_cfg.fold_version,
        external_bindings=ssl_cfg.external_bindings,
        external_fold_version=ssl_cfg.external_fold_version,
    )
    flinders = _subject_tokens(
        caches, model, "FLINDERS", fold, config.protocols, config.batch_size, device
    )
    leop = _subject_tokens(
        caches, model, "LEOP", fold, config.protocols, config.batch_size, device
    )
    fl_means = _subject_mean_tokens(flinders)
    le_means = _subject_mean_tokens(leop)

    rng = np.random.default_rng(config.seed)
    per_stream: dict[str, object] = {}
    for stream in sorted(set(fl_means) | set(le_means)):
        fl_frame = fl_means.get(stream)
        le_frame = le_means.get(stream)
        if fl_frame is None or fl_frame.empty or le_frame is None or le_frame.empty:
            per_stream[stream] = {"note": "insufficient subjects on held-out fold"}
            continue
        protocols = sorted(set(fl_frame["protocol"]) & set(le_frame["protocol"]))
        rows = []
        for protocol in protocols:
            fl_tokens = np.stack(
                fl_frame.loc[fl_frame["protocol"].eq(protocol), "token"].to_numpy()
            )
            le_tokens = np.stack(
                le_frame.loc[le_frame["protocol"].eq(protocol), "token"].to_numpy()
            )
            observed = _median_dim_ks(fl_tokens, le_tokens)
            boots = []
            for _ in range(config.n_boot_reps):
                fl_sample = fl_tokens[rng.integers(0, len(fl_tokens), len(fl_tokens))]
                le_sample = le_tokens[rng.integers(0, len(le_tokens), len(le_tokens))]
                boots.append(_median_dim_ks(fl_sample, le_sample))
            rows.append(
                {
                    "protocol": protocol,
                    "n_flinders_subjects": len(fl_tokens),
                    "n_leop_controls": len(le_tokens),
                    "ks_median_dim": observed,
                    "ci_low": float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                }
            )
        per_stream[stream] = rows

    report: dict = {
        "kind": "flinders_routed_calibration",
        "checkpoint": str(checkpoint_path),
        "exclude_fold": fold,
        "ssl_datasets": list(ssl_cfg.ssl_datasets),
        "external_bindings": list(ssl_cfg.external_bindings),
        "external_fold_version": ssl_cfg.external_fold_version,
        "n_flinders_components": len(flinders),
        "n_leop_components": len(leop),
        "protocols": list(config.protocols),
        "per_stream": per_stream,
    }
    reference = Path(config.reference_report)
    if reference.is_file():
        report["reference_feature_calibration"] = json.loads(reference.read_text())

    out_dir = artifact_root / "results" / config.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    tokens = pd.concat([flinders, leop], ignore_index=True)
    tokens.to_parquet(out_dir / "routed_tokens.parquet", index=False)
    (out_dir / "COMPLETE").write_text("complete\n")
    return report

"""VMD hyperparameter grid sweep (plan Section 15.2 'initial inner-fold grid').

The full plan grid is

    K       in {3, 4, 5, 6}
    alpha   in {500, 1000, 2000, 4000}
    tol     in {1e-6, 1e-7}
    padding in {25 ms, 50 ms mirror support}

Building 64 full component caches is not a modest search budget (plan 15.5),
so the grid is evaluated in two bounded steps:

1. ``sweep_vmd_grid`` — decompose a deterministic subsample of real
   recordings under every grid configuration and record, per component,
   per-config diagnostics: reconstruction error, residual energy,
   convergence, iteration count, mode energies, and — against the neighbour
   decomposition — per-mode center-frequency stability.  This tells us on
   which plateau the calibrated defaults (K=5, alpha=2000, tol=1e-7,
   pad=25 ms) sit and whether any grid point degrades reconstruction or
   convergence.

2. E4 model-level K sweep (separate command / configs): rebuild the
   baseline-only VMD cache for K in {3, 4, 6} and rerun the primary
   nine-step experiment to confirm the paired-fold AUROC is stable in K
   (grid points do not change the comparator's qualification).

Results are written deterministically to
``<artifact_root>/results/vmd_grid/grid_<tag>.parquet`` plus a JSON summary.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PreprocessingConfig
from .component_cache import _load_waveforms
from .vmd import VMDConfig, calibrate_vmd_frequency, decompose_vmd

PLAN_GRID = {
    "K": (3, 4, 5, 6),
    "alpha": (500, 1000, 2000, 4000),
    "tol": (1e-6, 1e-7),
    "mirror_pad_ms": (25.0, 50.0),
}

GRID_SEED = 20260801


def plan_grid_configs() -> list[VMDConfig]:
    """The 64 configurations of the plan Section 15.2 initial grid."""
    out: list[VMDConfig] = []
    keys = sorted(PLAN_GRID)
    for combo in itertools.product(*(PLAN_GRID[k] for k in keys)):
        kwargs = dict(zip(keys, combo, strict=True))
        out.append(VMDConfig(**kwargs))
    return out


def _grid_work(args):
    """Pool worker: diagnostics for one (recording, config) pair.

    Recordings whose segmentation raises (e.g., an empty segment window in an
    edge recording) are reported as ``("skip", recording_id, message)`` so the
    sweep survives real-data edge cases and the skip count is recorded.
    """
    from .component_cache import process_recording

    rec, (time_ms, signal_uv), pre_cfg, vmd_cfg = args
    try:
        result = process_recording(rec, time_ms, signal_uv, pre_cfg, vmd_cfg=vmd_cfg)
    except (ValueError, IndexError) as exc:
        return ("skip", rec["global_recording_id"], str(exc))
    if not result["valid"]:
        return []
    rows = []
    for row in result["rows"]:
        diag = result["diagnostics"][row["canonical_array_key"]]
        rows.append(
            {
                "global_component_id": row["global_component_id"],
                "config_key": vmd_cfg.key,
                **diag,
            }
        )
    return rows


def _sampled_recordings(artifact_root: Path, n_recordings: int, seed: int) -> pd.DataFrame:
    recordings = pd.read_parquet(artifact_root / "data" / "interim" / "recordings.parquet")
    waveforms = _load_waveforms(artifact_root)
    from .component_cache import cache_paths

    comp = pd.read_parquet(cache_paths(artifact_root)["components_parquet"])
    # Sample only the canonical modeling population (recordings represented in
    # the components cache).  Recordings added to recordings.parquet after the
    # cache build (e.g. streamed URFU/FLINDERS rows) have no cached components
    # and are not part of any baseline experiment.
    cache_rec_ids = set(comp["global_recording_id"])
    eligible = recordings[
        recordings["global_recording_id"].isin(cache_rec_ids)
        & recordings["global_recording_id"].isin(set(waveforms))
    ].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=min(n_recordings, len(eligible)), replace=False)
    return eligible.iloc[np.sort(idx)].reset_index(drop=True)


def sweep_vmd_grid(
    artifact_root: str | Path,
    pre_cfg: PreprocessingConfig,
    n_recordings: int = 500,
    jobs: int = 1,
    seed: int = GRID_SEED,
    tag: str = "s15",
) -> dict[str, object]:
    """Run the plan grid over a deterministic recording subsample.

    Returns per-component diagnostics rows (one per component-config pair)
    aggregated by config: median/mean relative reconstruction error, residual
    energy, convergence rate, median iteration count, and mean number of
    modes above a 1% energy floor.  Results are also written to
    ``<root>/results/vmd_grid/``.
    """
    import multiprocessing as mp

    from ..config import config_hash

    artifact_root = Path(artifact_root)
    if jobs > 1:
        mp.set_start_method("fork", force=True)
    convention = calibrate_vmd_frequency()
    configs = plan_grid_configs()
    recordings = _sampled_recordings(artifact_root, n_recordings, seed)
    waveforms = _load_waveforms(artifact_root)

    all_rows: list[dict] = []
    skips: list[tuple[str, str, str]] = []

    items = [
        (pd.Series(rec._asdict()), waveforms[rec.global_recording_id], pre_cfg, cfg)
        for rec in recordings.itertuples(index=False)
        for cfg in configs
    ]

    def _consume(batch):
        if isinstance(batch, tuple) and batch and batch[0] == "skip":
            skips.append(batch)
        else:
            all_rows.extend(batch)

    if jobs > 1:
        with mp.Pool(jobs) as pool:
            for batch in pool.imap(_grid_work, items, chunksize=4):
                _consume(batch)
    else:
        for item in items:
            _consume(_grid_work(item))
    if not all_rows:
        raise ValueError("VMD grid: no components decomposed (empty subsample?)")
    table = pd.DataFrame(all_rows)

    stats = []
    for key, g in table.groupby("config_key"):
        stats.append(
            {
                "config_key": key,
                "n_components": int(len(g)),
                "recon_rms_rel_median": float(g["recon_rms_rel"].median()),
                "recon_rms_rel_mean": float(g["recon_rms_rel"].mean()),
                "residual_energy_rel_median": float(g["residual_energy_rel"].median()),
                "converged_frac": float(g["converged"].mean()),
                "n_iterations_median": float(g["n_iterations"].median()),
                "n_modes_above_1pct_energy_mean": float(
                    g["n_modes_above_1pct_energy"].mean()
                ),
                "center_freq_spread_hz_median": float(
                    g["center_freq_spread_hz"].median()
                ),
            }
        )
    summary = pd.DataFrame(stats).sort_values("config_key").reset_index(drop=True)

    out_dir = artifact_root / "results" / "vmd_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"grid_{tag}.parquet"
    summary_path = out_dir / f"grid_{tag}_summary.parquet"
    table.to_parquet(table_path, index=False)
    summary.to_parquet(summary_path, index=False)
    skip_path = out_dir / f"grid_{tag}_skips.json"
    skip_path.write_text(json.dumps(skips, indent=2, sort_keys=True))

    result = {
        "tag": tag,
        "n_configs": len(configs),
        "n_recordings": int(len(recordings)),
        "n_rows": int(len(table)),
        "n_components": int(table["global_component_id"].nunique()),
        "n_skipped_recordings": len(skips),
        "seed": seed,
        "convention": {
            "hz_per_omega_unit": convention.hz_per_omega_unit,
            "max_relative_error": convention.max_relative_error,
            "verified": convention.verified,
        },
        "config_hash": config_hash(pre_cfg),
        "summary_path": str(summary_path),
        "rows_path": str(table_path),
        "defaults_median_recon": float(
            table.loc[table["config_key"] == VMDConfig().key, "recon_rms_rel"].median()
        ),
    }
    (out_dir / f"grid_{tag}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    return result


def diagnostics_are_available(result: dict[str, object]) -> bool:
    """True when process_recording returned VMD diagnostics for the grid."""
    return bool(result.get("n_rows", 0))


def decompose_vmd_diag(
    time_ms: np.ndarray,
    signal_uv: np.ndarray,
    cfg: VMDConfig,
    convention,
) -> dict[str, float]:
    """Standalone per-waveform VMD diagnostics (used by tests)."""
    res = decompose_vmd(time_ms, signal_uv, cfg, convention)
    return {
        "recon_rms_rel": res.recon_rms_rel,
        "residual_energy_rel": res.residual_energy_rel,
        "converged": bool(res.converged),
        "n_iterations": float(res.n_iterations),
        "n_modes_above_1pct_energy": int(
            (res.mode_energy / (res.mode_energy.sum() + 1e-12) > 0.01).sum()
        ),
        "center_freq_spread_hz": float(
            np.nanmax(res.center_freqs_hz) - np.nanmin(res.center_freqs_hz)
        )
        if res.center_freqs_hz.size
        else float("nan"),
    }

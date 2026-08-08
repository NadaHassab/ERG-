"""Classical baseline suite (plan Section 16; experiments E0 and E4).

Every baseline uses the same supervised units and the same outer folds:

- LEOP unit = participant (one prediction per participant);
- PERG unit = visit (one prediction per visit), with repeated visits of a
  subject inheriting that subject's fold assignment.

E0 sanity/confound baselines: prevalence, age/sex/site metadata, protocol
availability/missingness/bag size, and waveform-quality-only.
E4 classical baselines: clinical features, PCA (FPCA proxy), raw-curve RBF,
standard SCDT quantiles, derivative signed-OT, and wavelet scattering.

No silent fallbacks: unlabeled units are excluded with explicit counts,
per-field missingness is imputed fold-safely with an explicit median imputer
(+ missingness indicator), all-NaN and zero-variance columns are dropped with
explicit notes, and every QC exclusion is counted.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import zarr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import BaselinesConfig, DataConfig
from ..constants import (
    BASELINE_FEATURE_EPSILON,
    GB_DEPTH_GRID,
    GB_L2_GRID,
    GB_LEARNING_RATE_GRID,
    GB_MAX_ITER_GRID,
    INNER_FOLDS_TEMPLATE,
    LR_C_GRID,
    LR_L1_GRID,
    LR_MAX_ITER,
    OUTER_FOLDS_TEMPLATE,
    SCATTERING_LEVELS,
    SCATTERING_ORDER2_LEVELS,
    SCATTERING_WAVELET,
    SCDT_N_QUANTILES,
    SVM_C_GRID,
    SVM_GAMMA_GRID,
)
from ..evaluation.metrics import binary_metrics, cluster_bootstrap_ci
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths, load_cache_manifest
from .qc_flags import is_hard_invalid

# ---------------------------------------------------------------------------
# Component-type conventions
# ---------------------------------------------------------------------------

# Physical-feature columns used as "amplitude" / "latency" per component type.
PERG_RATIO_N95_FLOOR_UV = 0.1
PERG_RATIO_CLIP = 20.0

LEOP_AMP_COL = {
    "L_EARLY_A": "min_uv",
    "L_A_TO_B": "max_uv",
    "L_LATE": "min_uv",
    "L_OP": "max_uv",
}
LEOP_LAT_COL = {
    "L_EARLY_A": "min_latency_ms",
    "L_A_TO_B": "max_latency_ms",
    "L_LATE": "min_latency_ms",
    "L_OP": "max_latency_ms",
}
PERG_AMP_COL = {"P_EARLY": "max_uv", "P_LATE": "min_uv"}
PERG_LAT_COL = {"P_EARLY": "max_latency_ms", "P_LATE": "min_latency_ms"}


# ---------------------------------------------------------------------------
# Feature containers and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    unit_id: pd.Series
    X: np.ndarray
    names: list[str]
    per_unit_n: np.ndarray
    notes: dict[str, int | float | str]


_CONFOUND_PATTERNS = ("_n", "_flagged_rate")


def _drop_confound_columns(fs: FeatureSet) -> FeatureSet:
    """Remove per-slot count (_n) and flagged-rate columns from a FeatureSet.

    These columns are confounded with recording site/protocol and do not carry
    retinal biology.  Returns a new FeatureSet with the matching columns
    removed; if no columns match, the original is returned unchanged.
    """
    keep = [i for i, n in enumerate(fs.names) if not any(n.endswith(p) for p in _CONFOUND_PATTERNS)]
    if len(keep) == len(fs.names):
        return fs
    return FeatureSet(
        unit_id=fs.unit_id,
        X=np.asarray(fs.X[:, keep], dtype=np.float32) if fs.X.size else fs.X,
        names=[fs.names[i] for i in keep],
        per_unit_n=fs.per_unit_n,
        notes={**fs.notes, "dropped_confound_cols": len(fs.names) - len(keep)},
    )


def _json_rows(series: pd.Series) -> list[dict]:
    out = []
    for raw in series:
        if isinstance(raw, str) and raw:
            out.append(json.loads(raw))
        else:
            out.append({})
    return out


def _flagged(components: pd.DataFrame) -> pd.Series:
    return components["component_qc_flags"].fillna("").astype(str).ne("")

def _reindex_to_units(table: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    return table.reindex(units["unit_id"]).reset_index(drop=True)


def _unit_mapping(recordings: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == "LEOP":
        return recordings.set_index("global_recording_id")["global_subject_id"]
    return recordings.set_index("global_recording_id")["global_visit_id"]


# ---------------------------------------------------------------------------
# E0 — sanity / confound feature builders
# ---------------------------------------------------------------------------


_SEX_CODE = {"0": 0.0, "1": 1.0, "female": 0.0, "male": 1.0, "F": 0.0, "M": 1.0}


def _demographic_matrix(
    units: pd.DataFrame, dataset: str
) -> tuple[np.ndarray, list[str]]:
    """Age, sex, and (LEOP-only) site columns aligned to `units` order."""
    names = ["age_years"]
    mat = [units["age_years"].astype(float).to_numpy()]
    if dataset == "LEOP":
        names.append("site")
        site = units["site"].fillna("__missing__")
        mat.append(site.astype("category").cat.codes.to_numpy(float))
    names.append("sex_standardized")
    sex = units["sex_standardized"].map(
        lambda v: _SEX_CODE.get(str(v).strip().lower())
    )
    mat.append(sex.to_numpy(float))
    return np.column_stack(mat), names


def e0_metadata(units: pd.DataFrame, dataset: str) -> FeatureSet:
    """Age, sex, and site where the fields exist (PERG has no site column)."""
    X, names = _demographic_matrix(units, dataset)
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=X,
        names=names,
        per_unit_n=np.ones(len(units), dtype=int),
        notes={
            "site_excluded": dataset == "PERG",
            "sex_raw_values": sorted(str(v) for v in units["sex_standardized"].dropna().unique()),
        },
    )


def _augment_demographics(fs: FeatureSet, units: pd.DataFrame, dataset: str) -> FeatureSet:
    """Append demographic columns to an existing feature set (variant method)."""
    demog, demog_names = _demographic_matrix(units, dataset)
    names = [f"{n}_demog" for n in demog_names]
    X = demog if fs.X.shape[1] == 0 else np.column_stack([fs.X, demog])
    return FeatureSet(
        unit_id=fs.unit_id,
        X=X,
        names=list(fs.names) + names,
        per_unit_n=fs.per_unit_n,
        notes={**fs.notes, "augmented_with_demographics": names},
    )


def e0_availability(
    units: pd.DataFrame,
    dataset: str,
    recordings: pd.DataFrame,
    components: pd.DataFrame,
) -> FeatureSet:
    """Protocol availability, bag size, and missingness rates."""
    unit_map = _unit_mapping(recordings, dataset)
    rec = recordings.assign(unit=recordings["global_recording_id"].map(unit_map))
    comp = components.assign(
        unit=components["global_recording_id"].map(unit_map),
        flagged=_flagged(components),
    )
    agg = rec.groupby("unit").agg(
        n_recordings=("global_recording_id", "count"),
        n_eyes=("eye", "nunique"),
        n_waveforms=("waveform_kind", "nunique"),
        n_stimuli=("stimulus_value", "nunique"),
    )
    cagg = comp.groupby("unit").agg(
        n_components=("global_component_id", "count"),
        n_flagged_components=("flagged", "sum"),
    )
    table = agg.join(cagg, how="outer").fillna(0.0)
    n = table["n_components"].replace(0, np.nan)
    table["flagged_component_rate"] = table["n_flagged_components"] / n
    table = table.fillna(0.0)
    names = [
        "n_recordings",
        "n_eyes",
        "n_waveforms",
        "n_stimuli",
        "n_components",
        "flagged_component_rate",
    ]
    table = _reindex_to_units(table, units)
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=table[names].to_numpy(float),
        names=names,
        per_unit_n=table["n_components"].to_numpy(int),
        notes={"dataset": dataset},
    )


def e0_quality(
    units: pd.DataFrame,
    dataset: str,
    recordings: pd.DataFrame,
    components: pd.DataFrame,
) -> FeatureSet:
    """Waveform-quality-only: QC flag counts and rates per unit."""
    unit_map = _unit_mapping(recordings, dataset)
    comp = components.assign(unit=components["global_recording_id"].map(unit_map))
    n_all = comp.groupby("unit").size()
    n_flag = comp.groupby("unit")["component_qc_flags"].apply(
        lambda s: int(s.fillna("").astype(str).ne("").sum())
    )
    table = pd.DataFrame(
        {"n_all": n_all, "n_flagged": n_flag, "flag_rate": n_flag / n_all.replace(0, np.nan)}
    )
    table = _reindex_to_units(table, units).fillna(0.0)
    names = ["n_all", "n_flagged", "flag_rate"]
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=table[names].to_numpy(float),
        names=names,
        per_unit_n=table["n_all"].to_numpy(int),
        notes={},
    )


# ---------------------------------------------------------------------------
# E4 — clinical feature builders
# ---------------------------------------------------------------------------


def _clean_pairs(
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    amp_col: dict[str, str],
    lat_col: dict[str, str],
) -> pd.DataFrame:
    """Long table: (unit, eye, component_type, amplitude, latency, area, slopes)."""
    unit_map = _unit_mapping(recordings, dataset)
    comp = components.assign(unit=components["global_recording_id"].map(unit_map))
    phys = pd.DataFrame(_json_rows(comp["physical_features_json"]))
    eye = comp["global_recording_id"].map(
        recordings.set_index("global_recording_id")["eye"]
    )
    amp_name = comp["component_id"].map(amp_col)
    lat_name = comp["component_id"].map(lat_col)
    amp_vals = np.full(len(comp), np.nan)
    lat_vals = np.full(len(comp), np.nan)
    for col in set(amp_col.values()):
        if col in phys:
            mask = amp_name.to_numpy() == col
            amp_vals[mask] = phys[col].astype(float).to_numpy()[mask]
    for col in set(lat_col.values()):
        if col in phys:
            mask = lat_name.to_numpy() == col
            lat_vals[mask] = phys[col].astype(float).to_numpy()[mask]
    out = pd.DataFrame(
        {
            "recording": comp["global_recording_id"].to_numpy(),
            "unit": comp["unit"].to_numpy(),
            "eye": eye.to_numpy(),
            "component_id": comp["component_id"].to_numpy(),
            "amp": amp_vals,
            "lat": lat_vals,
            "valid": (~is_hard_invalid(comp)).to_numpy(),
        }
    )
    for col in ("area_above_ref_uv_ms", "peak_to_peak_uv", "max_rising_slope_uv_per_ms",
                "max_falling_slope_uv_per_ms"):
        if col in phys:
            out[col] = phys[col].astype(float).to_numpy()
        else:
            out[col] = np.full(len(comp), np.nan)
    return out


def e4_clinical_leops(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
) -> FeatureSet:
    """a/b/late/OP amplitudes, latencies, areas, slopes; intensity-response."""
    long = _clean_pairs(components, recordings, "LEOP", LEOP_AMP_COL, LEOP_LAT_COL)
    long = long[long["valid"]]
    names: list[str] = []
    frames: list[pd.DataFrame] = []
    for ctype in LEOP_AMP_COL:
        part = long[long["component_id"] == ctype]
        g = part.groupby("unit").agg(
            n=("amp", "size"),
            amp=("amp", "mean"),
            lat=("lat", "mean"),
            area=("area_above_ref_uv_ms", "mean"),
            p2p=("peak_to_peak_uv", "mean"),
            rise=("max_rising_slope_uv_per_ms", "mean"),
            fall=("max_falling_slope_uv_per_ms", "mean"),
        )
        g.columns = [
            f"{ctype}_{c}"
            if c in ("n", "area", "p2p", "rise", "fall")
            else f"{ctype}_{'amp_uv' if c == 'amp' else 'lat_ms'}"
            for c in g.columns
        ]
        names += list(g.columns)
        frames.append(g)
    # intensity-response: b-wave peak amplitude vs log10 stimulus per unit
    rec = recordings.assign(
        log10_stim=np.log10(recordings["stimulus_value"].astype(float))
    )
    b_amp = long[long["component_id"] == "L_A_TO_B"][["recording", "unit", "amp"]].merge(
        rec[["global_recording_id", "log10_stim"]].drop_duplicates("global_recording_id"),
        left_on="recording",
        right_on="global_recording_id",
        how="left",
    )
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    n_pts: dict[str, int] = {}
    for unit_id, grp in b_amp.groupby("unit"):
        valid = grp[["log10_stim", "amp"]].dropna()
        n_pts[unit_id] = int(len(valid))
        distinct = valid.drop_duplicates("log10_stim")
        if len(distinct) < 2:
            slopes[unit_id] = np.nan
            intercepts[unit_id] = np.nan
            continue
        x = distinct["log10_stim"].to_numpy(float)
        y = distinct["amp"].to_numpy(float)
        coef = np.linalg.lstsq(np.column_stack([x - x.mean(), np.ones(len(x))]), y, rcond=None)[0]
        slopes[unit_id] = float(coef[0])
        intercepts[unit_id] = float(coef[1])
    ir_table = pd.DataFrame({"ir_slope": slopes, "ir_intercept": intercepts, "ir_n_points": n_pts})
    names += ["ir_slope", "ir_intercept", "ir_n_points"]
    frames.append(ir_table)
    table = frames[0].join(frames[1:], how="outer")
    table = _reindex_to_units(table, units)
    X = table[names].to_numpy(float)
    per_unit = table[[c for c in names if c.endswith("_n")]].fillna(0).to_numpy(int).sum(axis=1)
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=X,
        names=names,
        per_unit_n=per_unit,
        notes={"n_units_without_intensity_fit": int(np.isnan(table["ir_slope"]).sum())},
    )


def e4_clinical_perg(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
) -> FeatureSet:
    """Per-eye P50/N95 amplitude/latency, peak-to-peak, ratio, inter-eye asymmetry."""
    long = _clean_pairs(components, recordings, "PERG", PERG_AMP_COL, PERG_LAT_COL)
    long = long[long["valid"]]
    piv = long.pivot_table(index="unit", columns=["eye", "component_id"], values=["amp", "lat"], aggfunc="mean")
    names: list[str] = []
    feats: dict[str, np.ndarray] = {}
    for eye in ("LE", "RE"):
        for ctype in ("P_EARLY", "P_LATE"):
            for kind, col in (("amp", "amp"), ("lat", "lat")):
                key = f"{eye}_{ctype}_{kind}_" + ("uv" if kind == "amp" else "ms")
                try:
                    feats[key] = piv[col][(eye, ctype)].to_numpy()
                except KeyError:
                    feats[key] = np.full(len(piv), np.nan)
                names.append(key)
        try:
            p50 = piv["amp"][(eye, "P_EARLY")].to_numpy()
            n95 = piv["amp"][(eye, "P_LATE")].to_numpy()
        except KeyError:
            p50 = np.full(len(piv), np.nan)
            n95 = np.full(len(piv), np.nan)
        feats[f"{eye}_p2p_uv"] = p50 - n95
        names.append(f"{eye}_p2p_uv")
        # |N95| floor of 0.1 uV (below the noise floor of the recording);
        # ratio clipped to [0, 20]: otherwise near-zero N95 components make
        # the ratio explode to ~1e12 and destabilize linear estimators.
        ratio = np.abs(p50) / (np.abs(n95) + PERG_RATIO_N95_FLOOR_UV)
        feats[f"{eye}_p50n95_ratio"] = np.clip(ratio, 0.0, PERG_RATIO_CLIP)
        names.append(f"{eye}_p50n95_ratio")
    for base, left, right in (
        ("P50_amp", "LE_P_EARLY_amp_uv", "RE_P_EARLY_amp_uv"),
        ("N95_amp", "LE_P_LATE_amp_uv", "RE_P_LATE_amp_uv"),
        ("P50_lat", "LE_P_EARLY_lat_ms", "RE_P_EARLY_lat_ms"),
    ):
        a, b = feats[left], feats[right]
        feats[f"asym_{base}"] = np.abs(a - b) / (0.5 * (np.abs(a) + np.abs(b)) + BASELINE_FEATURE_EPSILON)
        names.append(f"asym_{base}")
    table = pd.DataFrame(feats, index=piv.index)
    table["n_eyes"] = long.groupby("unit")["eye"].nunique()
    names.append("n_eyes")
    table = _reindex_to_units(table, units)
    X = table[names].to_numpy(float)
    n_comp = long.groupby("unit").size().reindex(units["unit_id"]).fillna(0).to_numpy(int)
    return FeatureSet(
        unit_id=units["unit_id"].reset_index(drop=True),
        X=X,
        names=names,
        per_unit_n=n_comp,
        notes={},
    )


# ---------------------------------------------------------------------------
# E4 — mathematical feature builders
# ---------------------------------------------------------------------------


def _unit_means(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    matrix: np.ndarray,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Mean of component-level rows per unit, aligned to `units` order."""
    unit_map = _unit_mapping(recordings, dataset)
    unit = components["global_recording_id"].map(unit_map).to_numpy()
    means = np.full((len(units), matrix.shape[1]), np.nan)
    n_used = np.zeros(len(units), dtype=int)
    index = {uid: i for i, uid in enumerate(units["unit_id"])}
    for i, uid in enumerate(index):
        mask = unit == uid
        if mask.any():
            means[i] = matrix[mask].mean(axis=0)
            n_used[i] = int(mask.sum())
    return units["unit_id"].reset_index(drop=True), means, n_used


def e4_curve_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    curves: np.ndarray,
    valid_mask: np.ndarray,
    kind: str,
) -> FeatureSet:
    """Unit-mean curve features: 'pca_fpca' | 'raw_rbf' | 'scdt' | 'scattering'."""
    keep = valid_mask.all(axis=1) & ~is_hard_invalid(components).to_numpy()
    used = np.where(keep)[0]
    ids, means, n_used = _unit_means(units, components.iloc[used], recordings, dataset, curves[used])
    if kind == "scdt":
        X = np.apply_along_axis(
            lambda v: np.nanquantile(v, (np.arange(SCDT_N_QUANTILES) + 0.5) / SCDT_N_QUANTILES),
            1,
            means,
        )
        names = [f"scdt_q{i}" for i in range(SCDT_N_QUANTILES)]
    elif kind == "scattering":
        X = np.vstack([wavelet_scattering_features(v) for v in means])
        names = [f"scat_{i}" for i in range(X.shape[1])]
    else:
        X = means
        names = [f"curve_{i}" for i in range(curves.shape[1])]
    return FeatureSet(
        unit_id=ids,
        X=X,
        names=names,
        per_unit_n=n_used,
        notes={"kind": kind, "n_components_excluded_qc": int((~keep).sum())},
    )


def e4_derot_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    sot: np.ndarray,
) -> FeatureSet:
    """Unit-mean derivative signed-OT vectors."""
    keep = ~is_hard_invalid(components).to_numpy()
    ids, means, n_used = _unit_means(
        units, components.iloc[np.where(keep)[0]], recordings, dataset, sot[keep]
    )
    return FeatureSet(
        unit_id=ids,
        X=means,
        names=[f"derot_{i}" for i in range(sot.shape[1])],
        per_unit_n=n_used,
        notes={"n_dim": int(sot.shape[1]), "n_components_excluded_qc": int((~keep).sum())},
    )


def e4_spectral_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    spectral: np.ndarray,
    names: list[str],
) -> FeatureSet:
    """Unit-mean multiscale spectral features (physical windows)."""
    keep = ~is_hard_invalid(components).to_numpy()
    ids, means, n_used = _unit_means(
        units, components.iloc[np.where(keep)[0]], recordings, dataset, spectral[keep]
    )
    return FeatureSet(
        unit_id=ids,
        X=means,
        names=list(names),
        per_unit_n=n_used,
        notes={"n_dim": int(spectral.shape[1]), "n_components_excluded_qc": int((~keep).sum())},
    )


def e4_vmd_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    dataset: str,
    vmd_vectors: np.ndarray,
    names: list[str],
) -> FeatureSet:
    """Unit-mean VMD mode features (physical windows, plan 15.4 aggregation)."""
    keep = ~is_hard_invalid(components).to_numpy()
    ids, means, n_used = _unit_means(
        units, components.iloc[np.where(keep)[0]], recordings, dataset, vmd_vectors[keep]
    )
    return FeatureSet(
        unit_id=ids,
        X=means,
        names=list(names),
        per_unit_n=n_used,
        notes={"n_dim": int(vmd_vectors.shape[1]), "n_components_excluded_qc": int((~keep).sum())},
    )


# ---------------------------------------------------------------------------
# Wavelet scattering (fixed filterbank, explicit constants)
# ---------------------------------------------------------------------------


def _scattering_size() -> int:
    return 1 + 2 * len(SCATTERING_LEVELS) + 2 * len(SCATTERING_LEVELS) * len(SCATTERING_ORDER2_LEVELS)


def wavelet_scattering_features(signal: np.ndarray) -> np.ndarray:
    """Second-order scattering with a Daubechies SWT filterbank.

    Order-0: mean.  Order-1: log-energy and mean-|coeff| of the SWT detail
    bands at SCATTERING_LEVELS.  Order-2: the same applied to each order-1
    band at SCATTERING_ORDER2_LEVELS.  Fixed, deterministic, pywt-only.
    """
    if np.isnan(signal).any():
        return np.full(_scattering_size(), np.nan)
    order0 = [float(np.mean(signal))]
    order1: list[float] = []
    details: list[np.ndarray] = []
    for level in SCATTERING_LEVELS:
        coeffs = pywt.swt(signal, SCATTERING_WAVELET, level=level)
        det = coeffs[-1][1]
        details.append(det)
        order1 += [float(np.log1p(np.mean(det**2))), float(np.mean(np.abs(det)))]
    order2: list[float] = []
    for band in details:
        for level in SCATTERING_ORDER2_LEVELS:
            coeffs = pywt.swt(band, SCATTERING_WAVELET, level=level)
            det = coeffs[-1][1]
            order2 += [float(np.log1p(np.mean(det**2))), float(np.mean(np.abs(det)))]
    return np.asarray(order0 + order1 + order2, dtype=float)


# Model pairing per plan Sections 16.1-16.3: E0 and clinical methods run the
# full model set; the mathematical baselines use the plan's prescribed models.
METHOD_MODELS: dict[str, tuple[str, ...]] = {
    "prevalence": ("prevalence",),
    "metadata": ("logreg", "svm_rbf", "histgb"),
    "availability": ("logreg", "svm_rbf", "histgb"),
    "quality": ("logreg", "svm_rbf", "histgb"),
    "clinical": ("logreg", "svm_rbf", "histgb"),
    "slot": ("logreg",),
    "slot_sot": ("logreg",),
    "pca_fpca": ("logreg",),
    "raw_rbf": ("svm_rbf",),
    "scdt": ("logreg", "svm_rbf"),
    "derot_lr": ("logreg",),
    "derot_rbf": ("svm_rbf",),
    "scattering": ("logreg",),
    "vmd": ("logreg", "svm_rbf", "histgb"),
    "spectral": ("logreg",),
}


def _models_for(method: str, cfg: BaselinesConfig) -> tuple[str, ...]:
    base = method[: -len("_demog")] if method.endswith("_demog") else method
    if base in METHOD_MODELS:
        return METHOD_MODELS[base]
    return tuple(cfg.models)


# ---------------------------------------------------------------------------
# Leak-free preprocessing pipeline (Phase 7)
# ---------------------------------------------------------------------------

# Candidate PCA dimensions for pca_fpca, selected on inner folds (never by an
# outer-train variance threshold). The exact feature dimension is appended so
# the full usable range is always represented.
PCA_DIM_GRID: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


class DropDegenerateColumns(BaseEstimator, TransformerMixin):
    """Drop all-NaN / zero-variance columns using only the data it is fit on.

    Inside a ``Pipeline`` every inner-training slice prunes its *own* columns,
    so no validation-sample statistic ever influences the features handed to
    the estimator (Phase 7: pipelines fitted inside every inner split).
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        var = np.nanvar(X, axis=0)
        self._keep_ = np.flatnonzero(np.isfinite(var) & (var >= 1e-12))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self._keep_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def n_kept(self) -> int:
        return int(self._keep_.size)

    def n_dropped(self) -> int:
        return int(self.n_features_in_) - int(self._keep_.size)


class _ClampedPCA(PCA):
    """PCA that never requests more components than the slice it is fit on."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_components = min(self.n_components, X.shape[1], X.shape[0])
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        X = np.asarray(X)
        self.n_components = min(self.n_components, X.shape[1], X.shape[0])
        return super().fit_transform(X, y, **fit_params)


def _pca_dim_grid(cfg: BaselinesConfig, n_features: int) -> list[int]:
    cap = min(cfg.max_pca_components, n_features)
    if cap < 2:
        return []
    dims = [d for d in PCA_DIM_GRID if d < cap]
    dims.append(cap)
    return sorted(set(dims))


def _pipeline_param_grid(
    kind: str, method: str, cfg: BaselinesConfig, n_features: int
) -> list[dict]:
    """Estimator grid, plus inner-selected PCA dimensions for pca_fpca."""
    base = method[: -len("_demog")] if method.endswith("_demog") else method
    grid = _parameter_grid(kind)
    if base != "pca_fpca":
        return grid
    dims = _pca_dim_grid(cfg, n_features)
    if not dims:
        return []
    return [{**p, "n_components": d} for p in grid for d in dims]


def build_pipeline(
    kind: str,
    method: str,
    cfg: BaselinesConfig,
    seed: int,
    use_gpu: bool,
    params: dict,
) -> Pipeline:
    """One fitted-inside-every-split preprocessing + estimator pipeline.

    Steps: column pruning -> median impute (+ missing indicator) -> standard
    scale -> [PCA for pca_fpca] -> estimator. Fitting the *whole* pipeline on
    each inner training slice makes the transform itself leak-free; the SVM is
    Platt-calibrated inside the pipeline; no decision threshold is tuned.
    """
    base = method[: -len("_demog")] if method.endswith("_demog") else method
    steps: list[tuple[str, object]] = [
        ("col_drop", DropDegenerateColumns()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if base == "pca_fpca":
        steps.append(("scaler", StandardScaler()))
        steps.append(("pca", _ClampedPCA(n_components=params["n_components"])))
    elif kind in ("logreg", "svm_rbf"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("est", _make_estimator(kind, params, seed, use_gpu)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Estimators with inner-fold selection
# ---------------------------------------------------------------------------


def _gpu_estimator(kind: str, params: dict, seed: int):
    """cuML GPU estimators (logreg / svm_rbf) when use_gpu is requested.

    Raises ImportError when cuML is unavailable; the caller falls back to the
    scikit-learn path and records a methodology note.
    """
    import cuml  # noqa: F401  (intentionally fail with a clean ImportError)

    if kind == "logreg":
        from cuml.linear_model import LogisticRegression

        l1_ratio = params["l1_ratio"]
        penalty = "l2" if l1_ratio == 0.0 else ("l1" if l1_ratio == 1.0 else "elasticnet")
        return LogisticRegression(
            penalty=penalty,
            C=params["C"],
            l1_ratio=l1_ratio,
            solver="qn",
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
        )
    if kind == "svm_rbf":
        from cuml.svm import SVC

        return CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                C=params["C"],
                gamma=params["gamma"],
                class_weight="balanced",
                random_state=seed,
            ),
            method="sigmoid",
            ensemble=False,
        )
    raise ValueError(f"no GPU estimator for kind {kind!r}")


def _make_estimator(kind: str, params: dict, seed: int, use_gpu: bool = False):
    if kind == "logreg" and use_gpu:
        return _gpu_estimator("logreg", params, seed)
    if kind == "svm_rbf" and use_gpu:
        # cuML SVC already Platt-calibrates; the sklearn path wraps SVC in a
        # deterministic single-model CalibratedClassifierCV instead.
        return _gpu_estimator("svm_rbf", params, seed)
    if kind == "logreg":
        # sklearn >= 1.8: penalty is implied by l1_ratio (0 -> l2, 1 -> l1,
        # in (0, 1) -> elasticnet); explicit penalty= is deprecated.
        return LogisticRegression(
            solver="saga",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=seed,
        )
    if kind == "svm_rbf":
        # sklearn >= 1.9 deprecates SVC(probability=True); a deterministic
        # single-model Platt calibration (ensemble=False, sigmoid) replaces it.
        return CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                C=params["C"],
                gamma=params["gamma"],
                class_weight="balanced",
                random_state=seed,
            ),
            method="sigmoid",
            ensemble=False,
        )
    if kind == "histgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            l2_regularization=params["l2"],
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"unknown estimator kind {kind!r}")


def _parameter_grid(kind: str) -> list[dict]:
    if kind == "logreg":
        return [{"C": c, "l1_ratio": l1} for c in LR_C_GRID for l1 in LR_L1_GRID]
    if kind == "svm_rbf":
        return [{"C": c, "gamma": g} for c in SVM_C_GRID for g in SVM_GAMMA_GRID]
    if kind == "histgb":
        return [
            {"max_iter": m, "learning_rate": lr, "max_depth": d, "l2": l2}
            for m in GB_MAX_ITER_GRID
            for lr in GB_LEARNING_RATE_GRID
            for d in GB_DEPTH_GRID
            for l2 in GB_L2_GRID
        ]
    raise ValueError(f"unknown estimator kind {kind!r}")


def _inner_fold_scores(
    kind: str,
    method: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_fold: np.ndarray,
    cfg: BaselinesConfig,
    seed: int,
    use_gpu: bool = False,
) -> tuple[float, dict]:
    """Mean inner-fold AUROC for one (pipeline, params) combination.

    The whole preprocessing + estimator ``Pipeline`` is fitted on the inner
    training slice only, then scored on the inner validation slice (Phase 7).
    """
    scores: list[float] = []
    for j in range(4):
        tr = inner_fold != j
        va = inner_fold == j
        if len(set(y_train[va])) < 2 or len(set(y_train[tr])) < 2:
            continue
        pipe = build_pipeline(kind, method, cfg, seed, use_gpu, params)
        pipe.fit(X_train[tr], y_train[tr])
        scores.append(roc_auc_score(y_train[va], pipe.predict_proba(X_train[va])[:, 1]))
    if not scores:
        return -math.inf, params
    return float(np.mean(scores)), params


def select_and_fit(
    kind: str,
    method: str,
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_unit_ids: np.ndarray,
    inner: pd.DataFrame,
    outer_fold: int,
    cfg: BaselinesConfig,
    seed: int,
    use_gpu: bool = False,
):
    """Inner-fold pipeline + hyperparameter selection, then refit on the train set.

    Each parameter combination (including the PCA dimension for ``pca_fpca``)
    is scored by fitting the full pipeline on the inner training slices; the
    best combination is then refitted on the whole outer training fold. Scores
    are computed deterministically, so the selected parameters are identical to
    a sequential scan (joblib preserves input order).
    """
    from joblib import Parallel, delayed

    inner_map = {
        (r.dataset, r.unit_id): int(r.inner_fold)
        for r in inner.itertuples(index=False)
        if int(r.outer_fold_sel) == outer_fold
    }
    inner_fold = np.asarray([inner_map.get((dataset, u), -1) for u in train_unit_ids])
    if (inner_fold < 0).any() or len(set(inner_fold)) < 2:
        raise ValueError(
            "inner-fold assignments missing or degenerate for outer fold "
            f"{outer_fold}: {sorted(set(inner_fold))}"
        )
    grid = _pipeline_param_grid(kind, method, cfg, X_train.shape[1])
    if not grid:
        raise ValueError(
            f"no hyperparameters to select for {kind!r}/{method!r} in outer fold {outer_fold}"
        )
    results = Parallel(n_jobs=1 if use_gpu else -1, prefer="threads")(
        delayed(_inner_fold_scores)(kind, method, params, X_train, y_train, inner_fold, cfg, seed, use_gpu)
        for params in grid
    )
    best_score, best_params = max(results, key=lambda r: r[0])
    if not math.isfinite(best_score):
        raise ValueError(f"no valid inner fold for estimator {kind!r} in outer fold {outer_fold}")
    pipe = build_pipeline(kind, method, cfg, seed, use_gpu, best_params)
    pipe.fit(X_train, y_train)
    return pipe, best_params, best_score


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResults:
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float | None]]
    notes: dict[str, object]
    report_path: str | None = None


def _load_units(
    dataset: str,
    participants: pd.DataFrame,
    visits: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    if dataset == "LEOP":
        units = participants[participants["dataset"] == "LEOP"].copy()
        units = units.rename(columns={"global_subject_id": "unit_id"})
        units["subject_id"] = units["unit_id"]
        units["visit_id"] = np.nan
        lbl = visits[visits["dataset"] == "LEOP"][["global_subject_id", "target_binary"]]
        units = units.merge(lbl, left_on="unit_id", right_on="global_subject_id", how="left")
        units = units.drop(columns=["global_subject_id"])
    else:
        units = visits[visits["dataset"] == "PERG"][
            ["global_visit_id", "global_subject_id", "target_binary", "dataset"]
        ].copy()
        units = units.rename(columns={"global_visit_id": "unit_id"})
        units["visit_id"] = units["unit_id"]
        units["subject_id"] = units["global_subject_id"]
        units = units.merge(
            participants[participants["dataset"] == "PERG"][
                ["global_subject_id", "age_years", "sex_standardized"]
            ],
            on="global_subject_id",
            how="left",
        )
        units["site"] = np.nan
    if dataset == "LEOP":
        units = units.merge(folds, on=["unit_id", "dataset"], how="left")
    else:
        units = units.merge(
            folds.rename(columns={"unit_id": "global_subject_id"})[
                ["global_subject_id", "dataset", "outer_fold"]
            ],
            on=["global_subject_id", "dataset"],
            how="left",
        )
    units = units.dropna(subset=["target_binary", "outer_fold"])
    return units.reset_index(drop=True)


def _progress(out_dir: Path, msg: str) -> None:
    """Append a timestamped line to ``<out_dir>/status.log`` (best-effort).

    Lightweight, non-fatal: a failure to write must never break a run.
    The status file appears even while the run is in progress so elapsed time
    and remaining work can be estimated without polling the process.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(out_dir / "status.log", "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} | {msg}\n")
            fh.flush()
    except OSError:
        pass


def run_baselines(cfg: BaselinesConfig, data_cfg: DataConfig) -> BaselineResults:
    if cfg.legacy:
        from .legacy_baselines import run_baselines as run_legacy

        return run_legacy(cfg, data_cfg)
    from .leop_cohorts import (
        cohort_component_mask,
        cohort_protocol_counts,
        cohort_recordings_mask,
        cohort_unit_mask,
    )

    if cfg.leop_cohort is not None:
        if "LEOP" not in cfg.datasets:
            raise ValueError("leop_cohort requires LEOP in datasets")
        if cfg.leop_cohort == "primary_nine_step" and any(
            m in cfg.e0_methods for m in ("availability",)
        ):
            raise ValueError(
                "primary_nine_step cohort forbids the availability "
                "protocol-count shortcut baseline (plan Section 16.6)"
            )
    root = Path(data_cfg.artifact_root)
    out_dir = root / "results" / cfg.output_subdir
    t_run0 = time.monotonic()
    methods = list(cfg.e0_methods) + list(cfg.e4_methods) + [
        f"{m}_demog" for m in cfg.demographic_methods
    ]
    _progress(
        out_dir,
        f"START name={cfg.name} datasets={cfg.datasets} cohort={cfg.leop_cohort} "
        f"n_methods={len(methods)} outer_folds={cfg.outer_folds} "
        f"n_bootstrap_reps={cfg.n_bootstrap_reps} use_gpu={cfg.use_gpu}",
    )
    participants = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    cache = cache_paths(root, CACHE_SCHEMA_VERSION)
    load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    folds = pd.read_parquet(root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=cfg.fold_version))
    inner = pd.read_parquet(root / "data" / "splits" / INNER_FOLDS_TEMPLATE.format(version=cfg.fold_version))
    z = zarr.open_group(str(cache["curves_zarr"]), mode="r")
    curves = np.asarray(z["components"]["canonical_signal"][:])
    valid_mask = np.asarray(z["components"]["valid_mask"][:])
    sot = np.asarray(
        zarr.open_group(str(cache["sot_zarr"]), mode="r")[
            "components"
        ]["sot_vector"][:]
    )
    spectral = np.asarray(
        zarr.open_group(str(cache["spectral_zarr"]), mode="r")[
            "components"
        ]["spectral_vector"][:]
    )
    spectral_names = list(load_cache_manifest(root, CACHE_SCHEMA_VERSION)["extra"]["spectral_feature_names"])
    vmd_vectors: np.ndarray | None = None
    vmd_names: list[str] | None = None
    if "vmd" in methods:
        from ..signal.vmd import VMDConfig
        from ..signal.vmd_cache import load_vmd_cache

        main_hash = load_cache_manifest(root, CACHE_SCHEMA_VERSION)["extra"]["config_hash"]
        vmd_vectors, vmd_names = load_vmd_cache(root, VMDConfig(), main_hash)

    frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float | None]] = {}
    notes: dict[str, object] = {}

    for dataset in cfg.datasets:
        units = _load_units(dataset, participants, visits, folds)
        cohort_key = None
        cohort = cfg.leop_cohort if dataset == "LEOP" else None
        rec_for_features: pd.DataFrame = recordings
        comp_for_features: pd.DataFrame = components
        curves_for_features: np.ndarray = curves
        valid_for_features: np.ndarray = valid_mask
        sot_for_features: np.ndarray = sot
        spectral_for_features: np.ndarray = spectral
        vmd_for_features: np.ndarray | None = vmd_vectors
        if cohort is not None:
            unit_mask = cohort_unit_mask(units, recordings, cohort)
            cohort_key = f"{dataset}_{cohort}"
            n_excluded = int((~unit_mask).sum())
            if cohort == "primary_nine_step":
                # Restrict every feature family to nine-step recordings only.
                rec_mask = cohort_recordings_mask(recordings, cohort)
                comp_mask = cohort_component_mask(components, recordings, cohort)
                keep = np.flatnonzero(comp_mask.to_numpy())
                rec_for_features = recordings[rec_mask].reset_index(drop=True)
                comp_for_features = components[comp_mask].reset_index(drop=True)
                curves_for_features = curves[keep]
                valid_for_features = valid_mask[keep]
                sot_for_features = sot[keep]
                spectral_for_features = spectral[keep]
                if vmd_vectors is not None:
                    vmd_for_features = vmd_vectors[keep]
            else:
                n_excluded = int((~unit_mask).sum())  # 0 by construction
            units = units[unit_mask].reset_index(drop=True)
        n_fold_units = int(len(folds[folds["dataset"] == dataset]))
        units_note: dict[str, float | int | str] = {
            "n_supervised_units": int(len(units)),
            "n_excluded_unlabeled": int(n_fold_units - len(units)),
            "n_positive": int((units["target_binary"] == 1).sum()),
            "n_negative": int((units["target_binary"] == 0).sum()),
        }
        if cohort is not None:
            units_note["cohort"] = cohort
            units_note["n_excluded_no_nine_step"] = n_excluded
            units_note["protocol_counts"] = cohort_protocol_counts(recordings, cohort)
        notes[f"{cohort_key or dataset}_units"] = units_note
        feature_sets: dict[str, FeatureSet] = {}
        for method in methods:
            if method.endswith("_demog"):
                continue  # added below via _augment_demographics
            if method == "prevalence":
                feature_sets[method] = FeatureSet(
                    unit_id=units["unit_id"],
                    X=np.empty((len(units), 0)),
                    names=[],
                    per_unit_n=np.ones(len(units), dtype=int),
                    notes={},
                )
            elif method == "metadata":
                feature_sets[method] = e0_metadata(units, dataset)
            elif method == "availability":
                feature_sets[method] = e0_availability(
                    units, dataset, rec_for_features, comp_for_features
                )
            elif method == "quality":
                feature_sets[method] = e0_quality(
                    units, dataset, rec_for_features, comp_for_features
                )
            elif method == "clinical":
                feature_sets[method] = (
                    e4_clinical_leops(units, comp_for_features, rec_for_features)
                    if dataset == "LEOP"
                    else e4_clinical_perg(units, comp_for_features, rec_for_features)
                )
            elif method in ("pca_fpca", "raw_rbf", "scdt", "scattering"):
                feature_sets[method] = e4_curve_features(
                    units,
                    comp_for_features,
                    rec_for_features,
                    dataset,
                    curves_for_features,
                    valid_for_features,
                    method,
                )
            elif method in ("derot_lr", "derot_rbf"):
                feature_sets[method] = e4_derot_features(
                    units, comp_for_features, rec_for_features, dataset, sot_for_features
                )
            elif method == "spectral":
                feature_sets[method] = e4_spectral_features(
                    units,
                    comp_for_features,
                    rec_for_features,
                    dataset,
                    spectral_for_features,
                    spectral_names,
                )
            elif method == "vmd":
                if vmd_vectors is None or vmd_names is None or vmd_for_features is None:
                    raise ValueError("vmd method requested but the VMD cache is unavailable")
                feature_sets[method] = e4_vmd_features(
                    units,
                    comp_for_features,
                    rec_for_features,
                    dataset,
                    vmd_for_features,
                    vmd_names,
                )
            elif method == "slot":
                from .slot_features import e4_slot_features

                feature_sets[method] = e4_slot_features(
                    units, comp_for_features, rec_for_features, dataset
                )
            elif method == "slot_sot":
                from .slot_sot_features import e4_slot_sot_features

                feature_sets[method] = e4_slot_sot_features(
                    units,
                    comp_for_features,
                    rec_for_features,
                    dataset,
                    sot_for_features,
                )
            else:
                raise ValueError(f"unknown baseline method {method!r}")
        for method in ("slot", "slot_sot"):
            if method in feature_sets:
                feature_sets[method] = _drop_confound_columns(feature_sets[method])
        for method in cfg.demographic_methods:
            if method not in feature_sets:
                continue
            feature_sets[f"{method}_demog"] = _augment_demographics(
                feature_sets[method], units, dataset
            )
        notes[f"{dataset}_feature_notes"] = {
            method: fs.notes for method, fs in feature_sets.items()
        }

        for outer_fold in cfg.outer_folds:
            _progress(
                out_dir,
                f"OUTER_FOLD_START dataset={dataset} fold={outer_fold}/{len(cfg.outer_folds)} "
                f"n_method_models={sum(len(_models_for(m, cfg)) for m in methods)}",
            )
            test = (units["outer_fold"] == outer_fold).to_numpy()
            train = ~test
            y_train = units.loc[train, "target_binary"].to_numpy(float)
            y_test = units.loc[test, "target_binary"].to_numpy(float)
            fit_rows_total = sum(
                1
                for m in methods
                for mo in _models_for(m, cfg)
                if m != "prevalence" and feature_sets[m].X.shape[1] > 0
            )
            fit_done = 0
            for method in methods:
                fs = feature_sets[method]
                X = fs.X
                X_tr, X_te = X[train], X[test]
                unit_ids_te = fs.unit_id.to_numpy()[test]
                subject_ids_te = units["subject_id"].to_numpy()[test]
                visit_ids_te = units["visit_id"].to_numpy()[test]
                models = _models_for(method, cfg)
                for model in models:
                    if method == "prevalence" or X_tr.shape[1] == 0:
                        prob = np.full(len(y_test), float(np.mean(y_train)))
                        est_note = "prevalence-constant"
                    elif DropDegenerateColumns().fit(X_tr).n_kept() == 0:
                        prob = np.full(len(y_test), float(np.mean(y_train)))
                        est_note = "all-columns-degenerate-prevalence"
                    else:
                        _progress(
                            out_dir,
                            f"FIT_START dataset={dataset} fold={outer_fold}/{len(cfg.outer_folds)} "
                            f"method={method} model={model} "
                            f"fit={fit_done + 1}/{fit_rows_total}",
                        )
                        t_fit = time.monotonic()
                        pipe, params, inner_auc = select_and_fit(
                            kind=model,
                            method=method,
                            dataset=dataset,
                            X_train=X_tr,
                            y_train=y_train,
                            train_unit_ids=units["subject_id"].to_numpy()[train],
                            inner=inner,
                            outer_fold=outer_fold,
                            cfg=cfg,
                            seed=cfg.seed,
                            use_gpu=cfg.use_gpu,
                        )
                        prob = pipe.predict_proba(X_te)[:, 1]
                        est_note = f"inner_auc={inner_auc:.4f} params={params}"
                        fit_done += 1
                        dt_each = time.monotonic() - t_fit
                        remaining_est = (
                            dt_each * (fit_rows_total - fit_done) if fit_rows_total else 0
                        )
                        _progress(
                            out_dir,
                            f"FIT_DONE dataset={dataset} fold={outer_fold}/{len(cfg.outer_folds)} "
                            f"method={method} model={model} "
                            f"fit={fit_done}/{fit_rows_total} elapsed_s={dt_each:.1f} "
                            f"remaining_est_s={remaining_est:.0f}",
                        )
                    frames.append(
                        pd.DataFrame(
                            {
                                "method": f"{method}_{model}" if method != "prevalence" else "prevalence",
                                "task": dataset,
                                "cohort": cohort,
                                "outer_fold": outer_fold,
                                "unit_id": unit_ids_te,
                                "subject_id": subject_ids_te,
                                "visit_id": visit_ids_te,
                                "target": y_test,
                                "probability": prob,
                                "note": est_note,
                            }
                        )
                    )

        dataset_frames = [f for f in frames if f["task"].eq(dataset).all() and len(f)]
        if dataset_frames:
            pooled = pd.concat(dataset_frames, ignore_index=True)
            for method_id, grp in pooled.groupby("method"):
                y = grp["target"].to_numpy(float)
                p = grp["probability"].to_numpy(float)
                if len(set(y)) < 2:
                    metrics[f"{cohort_key or dataset}/{method_id}"] = {
                        "roc_auc": None,
                        "note": "single-class pooled OOF predictions",
                    }
                    continue
                point = binary_metrics(y, p)
                ci = cluster_bootstrap_ci(
                    y,
                    p,
                    grp["subject_id"].to_numpy(),
                    metric="roc_auc",
                    n_reps=cfg.n_bootstrap_reps,
                    seed=cfg.bootstrap_seed,
                    confidence=cfg.confidence,
                )
                point["roc_auc_ci_low"] = round(ci.ci_low, 4)
                point["roc_auc_ci_high"] = round(ci.ci_high, 4)
                point["bootstrap_n_used"] = ci.n_replicates_used
                point["bootstrap_n_skipped"] = ci.n_replicates_skipped
                point["bootstrap_n_clusters"] = ci.n_clusters
                metrics[f"{cohort_key or dataset}/{method_id}"] = point

    predictions = pd.concat(frames, ignore_index=True)
    notes["methodology"] = {
        "feature_preprocessing": (
            "Phase 7: every inner split refits a single sklearn pipeline "
            "(column pruning -> median impute + missing indicator -> standard "
            "scale -> [inner-selected PCA] -> estimator); hyperparameters AND "
            "the PCA dimension are selected on inner folds; svm_rbf is "
            "Platt-calibrated inside the pipeline; no decision threshold is "
            "tuned anywhere (thresholds locked)."
        ),
        "perg_ratio_guard": (
            "PERG P50/N95 ratio uses an |N95| floor of 0.1 uV and clips to [0, 20]; "
            "unclipped ratios explode to ~1e12 on near-zero N95 noise components and "
            "destabilize linear estimators."
        ),
        "leop_scdt_svm_overfit": (
            "LEOP/scdt_svm_rbf generalizes at/below chance (raw decision AUROC 0.40-0.50 "
            "on held-out folds vs 0.71-0.82 in-sample): high-gamma RBF SVMs overfit 232 "
            "units with 64 quantile features; inner-fold selection favors memorizing params."
        ),
    }
    if cfg.use_gpu:
        try:
            import cuml

            notes["methodology"]["gpu_backend"] = f"cuml {cuml.__version__}"
        except ImportError:
            notes["methodology"]["gpu_backend"] = (
                "use_gpu requested but cuml is not installed; fell back to scikit-learn"
            )
    _progress(out_dir, f"RUN_DONE name={cfg.name} total_elapsed_s={time.monotonic() - t_run0:.1f}")
    return BaselineResults(predictions=predictions, metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_baselines_artifacts(
    artifact_root: str | Path,
    results: BaselineResults,
    cfg: BaselinesConfig,
    pywt_note: str | None = None,
) -> Path:
    """Write predictions.parquet, metrics.json, manifest.json, and the HTML report.

    Output goes to ``<artifact_root>/results/<cfg.output_subdir>/`` so versioned
    experiments (legacy vs corrected pipeline) never overwrite each other.
    """
    from ..provenance import RunManifest, git_revision

    out_dir = Path(artifact_root) / "results" / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    results.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(results.metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    rows = ""
    for key in sorted(results.metrics):
        m = results.metrics[key]
        auc = m.get("roc_auc")
        ci = ""
        if auc is not None:
            ci = f"[{m.get('roc_auc_ci_low')}, {m.get('roc_auc_ci_high')}]"
        auc_cell = f"{auc:.4f}" if auc is not None else "n/a"
        rows += (
            f"<tr><td>{key}</td><td>{auc_cell}</td>"
            f"<td>{ci}</td><td>{m.get('balanced_accuracy', 0):.4f}</td>"
            f"<td>{m.get('auprc', 0):.4f}</td><td>{m.get('brier', 0):.4f}</td>"
            f"<td>{m.get('n_total', '')}</td><td>{m.get('n_positive', '')}</td>"
            f"<td>{m.get('bootstrap_n_skipped', '')}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>E0/E4 baseline suite</title>
<style>body{{font-family:sans-serif;margin:24px;}}table{{border-collapse:collapse;font-size:11px;}}
td,th{{border:1px solid #ccc;padding:3px 7px;}}</style></head><body>
<h1>E0/E4 — classical baseline suite</h1>
<p>Supervised units: LEOP participant-level, PERG visit-level (clustered by subject).
Pooled out-of-fold predictions; 95% cluster-bootstrap CI on AUROC.</p>
<h2>Metrics</h2>
<table><tr><th>task/method</th><th>AUROC</th><th>95% CI</th><th>bal. acc.</th>
<th>AUPRC</th><th>Brier</th><th>n units</th><th>n pos</th><th>boot. skipped</th></tr>{rows}</table>
<h2>Notes</h2><pre>{json.dumps(results.notes, indent=2, sort_keys=True, default=str)}</pre>
{("<h3>Environment note</h3><p>" + pywt_note + "</p>") if pywt_note else ""}
</body></html>"""
    report_path = out_dir / "baselines_report.html"
    report_path.write_text(html, encoding="utf-8")

    manifest = RunManifest(kind="baselines", name=cfg.name)
    manifest.extra["n_methods"] = len(results.metrics)
    manifest.extra["notes"] = results.notes
    if pywt_note:
        manifest.extra["environment_note"] = pywt_note
    manifest.code_revision = git_revision(Path.cwd())
    manifest.write_atomic(out_dir / "manifest.json")
    return report_path

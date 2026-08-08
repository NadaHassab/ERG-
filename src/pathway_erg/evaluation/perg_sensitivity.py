"""PERG sensitivity ablations (plan Section 16.6 E11, Section 17).

Phase 8 of the v2 correctness plan.  Uses the same supervised PERG units,
outer folds, and nested-CV pipeline as the main baselines (``select_and_fit``
with inner-CV hyperparameter selection; subject-level cluster bootstrap) so
the ablation numbers are directly comparable with the main PERG baseline
numbers.  Five ablations are configured through the experiment ``ablations``
tuple:

- ``age_strata``: separate runs restricted to age < 18 vs >= 18.
- ``sex_strata``: separate runs restricted to Female vs Male.
- ``one_visit_per_subject``: keep only the first visit per canonical subject
  (by visit date); every other visit of a subject is dropped before building
  features.
- ``diagnosis_families``: separate runs per major diagnosis1 family; families
  with >= ``min_family_n`` supervised units supervise their own run.
- ``acuity_missingness``: runs restricted to visits with logMar acuity vs
  without, where acuity is parsed from the raw participants_info.csv via
  ``parse_perg_acuity``.  When ``acuity_as_feature`` is true the clinical
  feature set is augmented with the two-eye logMar columns.

Nothing is averaged across visits/sessions here; every prediction is one PERG
visit and CI clustering stays at the canonical-subject level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..config import BaselinesConfig, DataConfig
from ..data.perg import parse_perg_acuity
from ..evaluation.metrics import binary_metrics, cluster_bootstrap_ci
from ..signal.component_cache import CACHE_SCHEMA_VERSION, cache_paths, load_cache_manifest
from ..models.baselines import (
    INNER_FOLDS_TEMPLATE,
    OUTER_FOLDS_TEMPLATE,
    _load_units,
    _models_for,
    _progress,
    e0_metadata,
    e4_clinical_perg,
    e4_curve_features,
    e4_derot_features,
    e4_spectral_features,
    FeatureSet,
    select_and_fit,
)

PERG = "PERG"
FAMILY_LABELS = {
    "Macular dystrophy": "macula",
    "Pattern macular dystrophy": "macula",
    "Stargardt disease": "macula",
    "Cone-Rod dystrophy": "cone_rod",
    "Central serous chorioretinopathy (CSCR)": "cscr",
    "Chorioretinopathy Birdshot type": "birdshot",
    "Inherited optic atrophy": "optic_atrophy",
    "Autoimmune retinopathy": "autoimmune",
    "Congenital stationary night blindness": "csnb",
    "Retinal toxicity": "toxicity",
    "X linked retinoschisis": "xls",
    "Orbital ischemia": "ischemia",
    "Optic neuropathy": "optic_neuropathy",
    "Retinitis pigmentosa": "rp",
    "Usher syndrome": "usher",
    "Bardet Biedl": "bardet",
}


@dataclass(frozen=True)
class PergSensitivityConfig(BaselinesConfig):
    """PERG-only sensitivity config; inherits the shared fold/seed contract."""

    datasets: tuple[str, ...] = ("PERG",)
    output_subdir: str = "perg_sensitivity_v1"
    # Five ablations ("age_strata" | "sex_strata" | "one_visit_per_subject" |
    # "diagnosis_families" | "acuity_missingness").
    ablations: tuple[str, ...] = ()
    # When True the clinical feature set receives the two-eye logMAR acuity
    # columns (used by the acuity_missingness ablation).
    acuity_as_feature: bool = False
    # Minimum number of supervised units for a diagnosis family to run alone.
    min_family_n: int = 20


def _aligned_acuity(units: pd.DataFrame, acuity_df: pd.DataFrame) -> pd.DataFrame:
    """Align acuity rows to units by PERG_XXXX unit id (left join)."""
    rec = units["unit_id"].str.extract(r"PERG_(\d+)$")[0].astype(str)
    aligned = acuity_df.set_index("source_record_id").reindex(rec)
    va_re = pd.to_numeric(aligned["va_re_logmar"], errors="coerce")
    va_le = pd.to_numeric(aligned["va_le_logmar"], errors="coerce")
    out = {
        "va_re_logmar": va_re.to_numpy(float),
        "va_le_logmar": va_le.to_numpy(float),
        "acuity_missing": (va_re.isna() | va_le.isna()).to_numpy(bool),
        "acuity_n_eyes": (va_re.isna().astype(int) + va_le.isna().astype(int)).to_numpy(int),
    }
    return pd.DataFrame(out, index=units.index)


def _build_features(
    units: pd.DataFrame,
    components: pd.DataFrame,
    recordings: pd.DataFrame,
    curves: np.ndarray,
    valid: np.ndarray,
    sot: np.ndarray,
    spectral: np.ndarray,
    spectral_names: list[str],
    methods: list[str],
    acuity_df: pd.DataFrame | None,
    acuity_as_feature: bool,
) -> dict[str, FeatureSet]:
    feats: dict[str, FeatureSet] = {}
    for method in methods:
        if method == "clinical_acuity":
            continue  # produced as a variant inside the "clinical" branch
        if method == "prevalence":
            feats[method] = FeatureSet(
                unit_id=units["unit_id"].reset_index(drop=True),
                X=np.empty((len(units), 0)),
                names=[],
                per_unit_n=np.ones(len(units), dtype=int),
                notes={},
            )
        elif method == "metadata":
            feats[method] = e0_metadata(units, PERG)
        elif method == "clinical":
                fs = e4_clinical_perg(units, components, recordings)
                # clinical_acuity variant: clinical features + two-eye logMar.
                if acuity_as_feature and acuity_df is not None:
                    acu = _aligned_acuity(units, acuity_df)
                    col = ["va_re_logmar", "va_le_logmar"]
                    X = np.hstack([np.asarray(fs.X, dtype=float), np.column_stack([acu[c] for c in col])])
                    feats["clinical_acuity"] = FeatureSet(
                        unit_id=fs.unit_id, X=X, names=fs.names + col,
                        per_unit_n=fs.per_unit_n, notes={**fs.notes, "acuity_augmented": True},
                    )
                feats[method] = fs
        elif method in ("pca_fpca", "raw_rbf", "scdt", "scattering"):
            feats[method] = e4_curve_features(
                units, components, recordings, PERG, curves, valid, method
            )
        elif method in ("derot_lr", "derot_rbf"):
            feats[method] = e4_derot_features(units, components, recordings, PERG, sot)
        elif method == "spectral":
            feats[method] = e4_spectral_features(units, components, recordings, PERG, spectral, spectral_names)
        else:
            raise ValueError(f"PERG sensitivity: unknown method {method!r}")
    return feats


def _run_units(
    cfg: PergSensitivityConfig,
    units: pd.DataFrame,
    inner: pd.DataFrame,
    feat: dict[str, FeatureSet],
    methods: list[str],
    label: str,
    out_dir: Path,
) -> tuple[dict, dict]:
    """Nested-CV over one restricted unit set; returns (metrics, unit_note)."""
    n_pos = int((units["target_binary"] == 1).sum())
    note = {
        "n_units": int(len(units)),
        "n_positive": n_pos,
        "n_negative": int(len(units) - n_pos),
        "n_subjects": int(units["subject_id"].nunique()),
    }
    frames: list[pd.DataFrame] = []
    for fold in cfg.outer_folds:
        test = (units["outer_fold"] == fold).to_numpy()
        train = ~test
        y_tr = units.loc[train, "target_binary"].to_numpy(float)
        y_te = units.loc[test, "target_binary"].to_numpy(float)
        subj_tr = units["subject_id"].to_numpy()[train]
        k = int(sum(1 for m in methods for _ in _models_for(m, cfg)))
        ifold = 0
        for method in methods:
            fs = feat[method]
            if fs.X.shape[0] != len(units):
                raise ValueError(f"{label}: feature/unit row mismatch for {method}")
            for model in _models_for(method, cfg):
                ifold += 1
                _progress(
                    out_dir,
                    f"FIT_START ablation={label} fold={fold}/{len(cfg.outer_folds)} "
                    f"method={method} model={model} fit={ifold}/{k}",
                )
                prob = np.full(len(y_te), float(np.mean(y_tr)))
                est_note = "prevalence-constant"
                if method != "prevalence" and fs.X.shape[1] > 0:
                    X_tr = np.asarray(fs.X[train], dtype=float)
                    X_te = np.asarray(fs.X[test], dtype=float)
                    pipe, params, inner_auc = select_and_fit(
                        kind=model, method=method, dataset=PERG, X_train=X_tr,
                        y_train=y_tr, train_unit_ids=subj_tr, inner=inner,
                        outer_fold=fold, cfg=cfg, seed=cfg.seed, use_gpu=cfg.use_gpu,
                    )
                    prob = pipe.predict_proba(X_te)[:, 1]
                    est_note = f"inner_auc={inner_auc:.4f} params={params}"
                frames.append(
                    pd.DataFrame(
                        {
                            "method": f"{method}_{model}" if method != "prevalence" else "prevalence",
                            "unit_id": fs.unit_id.iloc[test].to_numpy().tolist(),
                            "subject_id": units["subject_id"].to_numpy()[test],
                            "visit_id": units["visit_id"].to_numpy()[test],
                            "target": y_te,
                            "probability": prob,
                            "note": est_note,
                        }
                    )
                )
    metrics: dict[str, dict] = {}
    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        for mid, grp in pooled.groupby("method"):
            y = grp["target"].to_numpy(float)
            p = grp["probability"].to_numpy(float)
            clusters = grp["subject_id"].to_numpy()
            if (y == 0).any() and (y == 1).any():
                ci = cluster_bootstrap_ci(
                    y, p, clusters, n_reps=cfg.n_bootstrap_reps,
                    seed=cfg.bootstrap_seed, confidence=cfg.confidence,
                )
                m = binary_metrics(y, p)
                m.update(
                    {
                        "roc_auc": m["roc_auc"],
                        "roc_auc_ci_low": ci.ci_low,
                        "roc_auc_ci_high": ci.ci_high,
                        "n_clusters": ci.n_clusters,
                        "n_predictions": int(len(y)),
                    }
                )
                metrics[mid] = m
    return metrics, note


def _load_cache(root: Path):
    cache = cache_paths(root, CACHE_SCHEMA_VERSION)
    load_cache_manifest(root, CACHE_SCHEMA_VERSION)
    components = pd.read_parquet(cache["components_parquet"])
    z = zarr.open_group(str(cache["curves_zarr"]), mode="r")
    curves = np.asarray(z["components"]["canonical_signal"][:])
    valid = np.asarray(z["components"]["valid_mask"][:])
    sot = np.asarray(
        zarr.open_group(str(cache["sot_zarr"]), mode="r")["components"]["sot_vector"][:]
    )
    spectral = np.asarray(
        zarr.open_group(str(cache["spectral_zarr"]), mode="r")["components"]["spectral_vector"][:]
    )
    spectral_names = list(load_cache_manifest(root, CACHE_SCHEMA_VERSION)["extra"]["spectral_feature_names"])
    return components, curves, valid, sot, spectral, spectral_names


def run_perg_sensitivity(cfg: PergSensitivityConfig, data_cfg: DataConfig) -> dict:
    """Run the PERG sensitivity ablations; returns {ablation: {method: metrics}}."""
    root = Path(data_cfg.artifact_root)
    out_dir = root / "results" / cfg.output_subdir
    participants = pd.read_parquet(root / "data" / "interim" / "participants.parquet")
    visits = pd.read_parquet(root / "data" / "interim" / "visits.parquet")
    recordings = pd.read_parquet(root / "data" / "interim" / "recordings.parquet")
    folds = pd.read_parquet(root / "data" / "splits" / OUTER_FOLDS_TEMPLATE.format(version=cfg.fold_version))
    inner = pd.read_parquet(root / "data" / "splits" / INNER_FOLDS_TEMPLATE.format(version=cfg.fold_version))
    components, curves, valid, sot, spectral, spectral_names = _load_cache(root)

    units = _load_units(PERG, participants, visits, folds)
    methods = list(cfg.e0_methods) + list(cfg.e4_methods)
    methods += [f"{m}_demog" for m in cfg.demographic_methods]
    if cfg.acuity_as_feature and "clinical" in methods and "clinical_acuity" not in methods:
        methods.append("clinical_acuity")
    _progress(out_dir, f"START name={cfg.name} n_units={len(units)} ablations={list(cfg.ablations)}")

    acuity_df = None
    if "acuity_missingness" in cfg.ablations or cfg.acuity_as_feature:
        acuity_df = parse_perg_acuity(data_cfg.perg.metadata_csv).reset_index()

    def restricted(keep: np.ndarray, label: str):
        keep = np.asarray(keep)
        if keep.dtype.kind not in "bi":
            raise TypeError(f"restricted: keep must be boolean or integer index, got {keep.dtype}")
        if keep.dtype.kind == "i":
            mask = np.zeros(len(units), dtype=bool)
            mask[keep] = True
            keep = mask
        sub = units.loc[keep].reset_index(drop=True)
        feats = _build_features(
            sub, components, recordings, curves, valid, sot, spectral, spectral_names,
            methods, acuity_df, cfg.acuity_as_feature,
        )
        met, note = _run_units(cfg, sub, inner, feats, methods, label, out_dir)
        return met, {"units": note}

    results: dict[str, object] = {}

    full_feats = _build_features(
        units, components, recordings, curves, valid, sot, spectral, spectral_names,
        methods, acuity_df, cfg.acuity_as_feature,
    )
    full_meta, full_note = _run_units(cfg, units, inner, full_feats, methods, "baseline", out_dir)
    results["_meta"] = {"baseline": full_note, "cohort": f"PERG", "n_total_subjects": int(units["subject_id"].nunique())}
    results["baseline"] = full_meta

    if "age_strata" in cfg.ablations:
        for tag, mask in (
            ("under18", (units["age_years"] < 18).to_numpy()),
            ("adult", (units["age_years"] >= 18).to_numpy()),
        ):
            results[f"age_{tag}"], results[f"__meta_age_{tag}"] = restricted(mask, f"age_{tag}")
    if "sex_strata" in cfg.ablations:
        sexes = sorted(units["sex_standardized"].dropna().unique().tolist())
        for sex in sexes:
            tag = "female" if str(sex) in ("0", "Female") else "male"
            mask = (units["sex_standardized"] == sex).to_numpy()
            results[f"sex_{tag}"], results[f"__meta_sex_{tag}"] = restricted(mask, f"sex_{tag}")
    if "one_visit_per_subject" in cfg.ablations:
        keep = (
            units.merge(
                visits[["global_visit_id", "visit_date"]],
                left_on="unit_id", right_on="global_visit_id", how="left",
            )
            .sort_values("visit_date", na_position="last")
            .drop_duplicates("subject_id", keep="first")
            .index
        )
        results["one_visit_per_subject"], results["__meta_one_visit"] = restricted(keep, "one_visit_per_subject")
    if "diagnosis_families" in cfg.ablations:
        merged = units.merge(
            visits[["global_visit_id", "diagnosis1_raw"]],
            left_on="unit_id", right_on="global_visit_id", how="left",
        )
        fam = merged["diagnosis1_raw"].astype(str).map(FAMILY_LABELS).fillna(
            merged["diagnosis1_raw"].astype(str)
        )
        for tag in sorted(set(fam)):
            if tag == "Normal":
                continue  # healthy controls are not a diagnostic family
            fam_mask = (fam == tag).to_numpy()
            if int(fam_mask.sum()) < cfg.min_family_n:
                continue
            healthy = (units["target_binary"] == 0).to_numpy()
            mask = fam_mask | healthy  # family vs healthy-control discrimination
            results[f"family_{tag}"], results[f"__meta_family_{tag}"] = restricted(mask, f"family_{tag}")
    if "acuity_missingness" in cfg.ablations:
        aligned = _aligned_acuity(units, acuity_df)
        results["acuity_has"], results["__meta_acuity_has"] = restricted(
            ~aligned["acuity_missing"].to_numpy(dtype=bool), "acuity_has"
        )
        results["acuity_missing"], results["__meta_acuity_missing"] = restricted(
            aligned["acuity_missing"].to_numpy(dtype=bool), "acuity_missing"
        )

    _progress(out_dir, f"RUN_DONE name={cfg.name} n_results={len(results)}")
    return results
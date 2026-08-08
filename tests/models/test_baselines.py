"""Baseline feature builders, estimators, and runner mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pathway_erg.config import BaselinesConfig
from pathway_erg.models.baselines import (
    DropDegenerateColumns,
    _augment_demographics,
    _make_estimator,
    _models_for,
    _parameter_grid,
    _pipeline_param_grid,
    build_pipeline,
    e0_availability,
    e0_metadata,
    e0_quality,
    e4_clinical_leops,
    e4_clinical_perg,
    e4_curve_features,
    e4_derot_features,
    select_and_fit,
    wavelet_scattering_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _recordings() -> pd.DataFrame:
    rows = []
    for subject in ("LEOP_S1", "LEOP_S2", "LEOP_S3"):
        for rec in range(4):
            rows.append(
                {
                    "global_recording_id": f"{subject}_R{rec}",
                    "global_subject_id": subject,
                    "global_visit_id": f"{subject}_V0",
                    "dataset": "LEOP",
                    "eye": "RE",
                    "waveform_kind": "ERG" if rec < 3 else "OP",
                    "stimulus_value": [12.0, 35.0, 113.0, 35.0][rec],
                }
            )
    for subject, visits in (("PERG_P1", 2), ("PERG_P2", 2)):
        for v in range(visits):
            for eye in ("LE", "RE"):
                rows.append(
                    {
                        "global_recording_id": f"{subject}_V{v}_{eye}",
                        "global_subject_id": subject,
                        "global_visit_id": f"{subject}_V{v}",
                        "dataset": "PERG",
                        "eye": eye,
                        "waveform_kind": "PERG_EYE",
                        "stimulus_value": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _components() -> pd.DataFrame:
    rows = []
    rid = 0
    for subject in ("LEOP_S1", "LEOP_S2", "LEOP_S3"):
        for rec in range(4):
            cid = f"LEOP_S{subject[-1]}_R{rec}"
            is_erg = rec < 3
            # ERG recordings carry a/b/late/OP components; OP recordings only L_OP
            ctypes = (
                [("L_EARLY_A", -3.0, 15.0), ("L_A_TO_B", 8.0, 21.0), ("L_LATE", -3.0, 96.0), ("L_OP", 4.0, 30.0)]
                if is_erg
                else [("L_OP", 4.0, 30.0)]
            )
            for ctype, amp, lat in ctypes:
                rows.append(
                    {
                        "global_component_id": f"{cid}_{ctype}",
                        "global_recording_id": f"{subject}_R{rec}",
                        "component_id": ctype,
                        "component_qc_flags": "",
                        "physical_features_json": (
                            f'{{"min_uv":{amp - 1},"max_uv":{amp + 1},"min_latency_ms":{lat - 2},'
                            f'"max_latency_ms":{lat + 2},"area_above_ref_uv_ms":{10 + amp},'
                            f'"peak_to_peak_uv":{6 + amp},"max_rising_slope_uv_per_ms":1.5,'
                            f'"max_falling_slope_uv_per_ms":-1.5}}'
                        ),
                    }
                )
            rid += 1
    for subject in ("PERG_P1", "PERG_P2"):
        for v in range(2):
            for eye in ("LE", "RE"):
                for ctype, amp, lat in (("P_EARLY", 2.5, 50.0), ("P_LATE", -1.5, 95.0)):
                    rows.append(
                        {
                            "global_component_id": f"{subject}_V{v}_{eye}_{ctype}",
                            "global_recording_id": f"{subject}_V{v}_{eye}",
                            "component_id": ctype,
                            "component_qc_flags": "",
                            "physical_features_json": (
                                f'{{"min_uv":{amp - 1},"max_uv":{amp + 1},"min_latency_ms":{lat - 2},'
                                f'"max_latency_ms":{lat + 2},"area_above_ref_uv_ms":8,'
                                f'"peak_to_peak_uv":6,"max_rising_slope_uv_per_ms":1.0,'
                                f'"max_falling_slope_uv_per_ms":-1.0}}'
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _units_leop() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["LEOP_S1", "LEOP_S2", "LEOP_S3"],
            "age_years": [8.0, 25.0, 40.0],
            "sex_standardized": [0.0, 1.0, 0.0],
            "site": [1.0, 2.0, 1.0],
        }
    )


def _units_perg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["PERG_P1_V0", "PERG_P1_V1", "PERG_P2_V0", "PERG_P2_V1"],
            "age_years": [30.0, 30.0, 60.0, 60.0],
            "sex_standardized": [1.0, 1.0, 0.0, 0.0],
            "site": [np.nan] * 4,
        }
    )


def _curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comps = _components()
    n = len(comps)
    curves = np.stack([np.sin(np.linspace(0, 6, 8)) * (i + 1) for i in range(n)])
    valid = np.ones((n, 8), dtype=bool)
    sot = np.stack([np.linspace(0, 1, 5) * (i + 1) for i in range(n)])
    return curves, valid, sot


# ---------------------------------------------------------------------------
# E0 builders
# ---------------------------------------------------------------------------


def test_e0_metadata_perg_excludes_site():
    fs = e0_metadata(_units_perg(), "PERG")
    assert "site" not in fs.names
    assert "age_years" in fs.names and "sex_standardized" in fs.names
    assert len(fs.unit_id) == 4
    assert fs.notes["site_excluded"] is True


def test_e0_metadata_leop_includes_site():
    fs = e0_metadata(_units_leop(), "LEOP")
    assert "site" in fs.names


def test_e0_availability_aligned_to_units():
    units = _units_leop()
    fs = e0_availability(units, "LEOP", _recordings(), _components())
    assert list(fs.unit_id) == list(units["unit_id"])
    assert "n_stimuli" in fs.names and "flagged_component_rate" in fs.names
    assert np.all(fs.per_unit_n >= 0)


def test_e0_quality_counts_flags():
    comps = _components().copy()
    comps.loc[0, "component_qc_flags"] = "truncated-low"
    units = _units_leop()
    fs = e0_quality(units, "LEOP", _recordings(), comps)
    idx = list(fs.unit_id).index("LEOP_S1")
    assert fs.per_unit_n[idx] == 13  # 3 ERG recordings x 4 components + 1 OP x 1
    assert fs.X[idx, 1] == 1


# ---------------------------------------------------------------------------
# E4 clinical builders
# ---------------------------------------------------------------------------


def test_e4_clinical_leop_has_intensity_response():
    units = _units_leop()
    fs = e4_clinical_leops(units, _components(), _recordings())
    assert "ir_slope" in fs.names and "ir_n_points" in fs.names
    idx = list(fs.unit_id).index("LEOP_S1")
    assert fs.X[idx, fs.names.index("ir_n_points")] == 3
    assert np.isfinite(fs.X[idx, fs.names.index("ir_slope")])
    assert "L_A_TO_B_amp_uv" in fs.names


def test_e4_clinical_leop_requires_two_intensities():
    recs = _recordings().copy()
    recs = recs[~recs["global_recording_id"].str.contains("_R0") & ~recs["global_recording_id"].str.contains("_R1")]
    comps = _components()[~_components()["global_recording_id"].str.contains("_R0") & ~_components()["global_recording_id"].str.contains("_R1")]
    fs = e4_clinical_leops(_units_leop(), comps, recs)
    idx = fs.names.index("ir_slope")
    assert np.isnan(fs.X[:, idx]).any()
    assert fs.notes["n_units_without_intensity_fit"] >= 1


def test_e4_clinical_perg_has_eyes_and_asymmetry():
    units = _units_perg()
    fs = e4_clinical_perg(units, _components(), _recordings())
    for name in ("LE_P_EARLY_amp_uv", "RE_P_LATE_lat_ms", "asym_P50_amp", "n_eyes"):
        assert name in fs.names
    assert list(fs.unit_id) == list(units["unit_id"])
    assert np.all(fs.X[:, fs.names.index("n_eyes")] == 2)


# ---------------------------------------------------------------------------
# Mathematical builders and scattering
# ---------------------------------------------------------------------------


def test_scattering_deterministic_and_nan_safe():
    x = np.linspace(-2, 4, 128)
    a = wavelet_scattering_features(x)
    b = wavelet_scattering_features(x)
    assert np.array_equal(a, b)
    assert len(a) == 25
    assert np.isnan(wavelet_scattering_features(np.full(128, np.nan))).all()


def test_curve_features_mean_pooling():
    units = _units_leop()
    curves, valid, _ = _curves()
    comps = _components()
    recs = _recordings()
    fs = e4_curve_features(units, comps, recs, "LEOP", curves, valid, "raw_rbf")
    idx = list(fs.unit_id).index("LEOP_S1")
    assert fs.per_unit_n[idx] == 13
    assert np.isclose(fs.X[idx, 0], curves[:13, 0].mean())
    fs2 = e4_curve_features(units, comps, recs, "LEOP", curves, valid, "scdt")
    assert fs2.names[0] == "scdt_q0"
    assert fs2.X.shape[1] == 64


def test_derot_features():
    units = _units_perg()
    _, _, sot = _curves()
    fs = e4_derot_features(units, _components(), _recordings(), "PERG", sot)
    assert fs.X.shape == (4, 5)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_parameter_grid_sizes():
    assert len(_parameter_grid("logreg")) == len((0.01, 0.1, 1.0, 10.0)) * 3
    assert len(_parameter_grid("svm_rbf")) == 3 * 3
    with pytest.raises(ValueError):
        _parameter_grid("nope")


def _inner_folds(n_units: int, outer_fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [f"U{i}" for i in range(n_units)],
            "dataset": ["LEOP"] * n_units,
            "outer_fold": [outer_fold] * n_units,
            "inner_fold": [i % 4 for i in range(n_units)],
            "outer_fold_sel": [outer_fold] * n_units,
        }
    )


def test_select_and_fit_returns_fitted_pipeline():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=40) > 0).astype(float)
    ids = np.array([f"U{i}" for i in range(40)])
    inner = _inner_folds(40, 2)
    pipe, params, inner_auc = select_and_fit(
        "logreg", "clinical", "LEOP", X, y, ids, inner, 2, BaselinesConfig(name="t"), seed=7
    )
    assert params["C"] in (0.01, 0.1, 1.0, 10.0)
    assert np.isfinite(inner_auc)
    assert pipe.predict_proba(X).shape == (40, 2)


def test_select_and_fit_raises_without_inner_folds():
    X = np.zeros((10, 2))
    y = np.array([1.0] * 5 + [0.0] * 5)
    ids = np.array([f"U{i}" for i in range(10)])
    inner = _inner_folds(10, 0)  # outer_fold_sel mismatch -> missing assignments
    with pytest.raises(ValueError, match="inner-fold assignments"):
        select_and_fit("logreg", "clinical", "LEOP", X, y, ids, inner, 3, BaselinesConfig(name="t"), seed=7)


def test_select_and_fit_parallel_matches_sequential_selection():
    """Parallel inner-fold scoring must select the same params as a scan."""
    from pathway_erg.models.baselines import _inner_fold_scores

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] - 0.2 * X[:, 1] + 0.1 * rng.normal(size=60) > 0).astype(float)
    ids = np.array([f"U{i}" for i in range(60)])
    inner = _inner_folds(60, 1)
    inner_fold = np.asarray([i % 4 for i in range(60)])
    cfg = BaselinesConfig(name="t")

    scored = [
        _inner_fold_scores("logreg", "clinical", params, X, y, inner_fold, cfg, seed=7)
        for params in _pipeline_param_grid("logreg", "clinical", cfg, n_features=X.shape[1])
    ]
    best_seq = max(scored, key=lambda r: r[0])[1]
    pipe_par, params_par, score_par = select_and_fit(
        "logreg", "clinical", "LEOP", X, y, ids, inner, 1, cfg, seed=7
    )
    assert params_par == best_seq
    assert np.isfinite(score_par)


def test_pipeline_drops_all_nan_columns_when_fit():
    cfg = BaselinesConfig(name="t")
    Xtr = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0], [0.5, np.nan, 2.0]])
    Xte = np.array([[1.5, np.nan, 3.5]])
    pipe = build_pipeline("logreg", "clinical", cfg, seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0})
    pipe.fit(Xtr, np.array([1.0, 0.0, 1.0]))
    assert pipe.named_steps["col_drop"].n_dropped() == 1
    assert pipe.predict_proba(Xte).shape == (1, 2)


def test_pipeline_drops_zero_variance_when_fit():
    cfg = BaselinesConfig(name="t")
    Xtr = np.array([[1.0, 3.0], [1.0, 4.0], [1.0, 2.0]])
    Xte = np.array([[1.0, 3.5]])
    pipe = build_pipeline("logreg", "clinical", cfg, seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0})
    pipe.fit(Xtr, np.array([1.0, 0.0, 1.0]))
    assert pipe.named_steps["col_drop"].n_dropped() == 1
    assert pipe.predict_proba(Xte).shape == (1, 2)


def test_pipeline_pca_branch_predicts_consistently():
    cfg = BaselinesConfig(name="t")
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(30, 8))
    Xte = rng.normal(size=(3, 8))
    pipe = build_pipeline("logreg", "pca_fpca", cfg, seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0, "n_components": 4})
    pipe.fit(Xtr, (Xtr[:, 0] > 0).astype(float))
    assert "pca" in [n for n, _ in pipe.steps]
    assert pipe.predict_proba(Xte).shape == (3, 2)


def test_pipeline_pca_branch_for_demog_variant():
    cfg = BaselinesConfig(name="t")
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(30, 8))
    Xte = rng.normal(size=(3, 8))
    pipe = build_pipeline("logreg", "pca_fpca_demog", cfg, seed=7, use_gpu=False,
                          params={"C": 1.0, "l1_ratio": 0.0, "n_components": 4})
    pipe.fit(Xtr, (Xtr[:, 0] > 0).astype(float))
    assert pipe.predict_proba(Xte).shape == (3, 2)


def test_col_drop_is_a_sklearn_transformer():
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 3.0], [0.0, 1.0, 4.0]])
    tr = DropDegenerateColumns().fit(X)
    assert tr.n_kept() == 1  # only the varying column survives
    assert tr.transform(X).shape == (3, 1)


def test_augment_demographics_appends_columns():
    fs = e0_metadata(_units_leop(), "LEOP")
    demo = _augment_demographics(fs, _units_leop(), "LEOP")
    assert demo.X.shape[1] == fs.X.shape[1] + 3  # age + sex + site
    assert demo.names[-3:] == ["age_years_demog", "site_demog", "sex_standardized_demog"]
    assert "augmented_with_demographics" in demo.notes
    assert list(demo.unit_id) == list(fs.unit_id)


def test_augment_demographics_perg_has_no_site():
    fs = e0_metadata(_units_perg(), "PERG")
    demo = _augment_demographics(fs, _units_perg(), "PERG")
    assert demo.X.shape[1] == fs.X.shape[1] + 2  # age + sex only
    assert "site_demog" not in demo.names


def test_models_for_demog_variant_matches_base():
    cfg = BaselinesConfig(name="t")
    assert _models_for("clinical_demog", cfg) == _models_for("clinical", cfg)
    assert _models_for("scdt_demog", cfg) == _models_for("scdt", cfg)


def test_gpu_fallback_without_cuml_is_sklearn():
    """use_gpu=True with cuml absent must silently fall back to sklearn."""
    pytest.importorskip("cuml")
    est = _make_estimator("logreg", {"C": 1.0, "l1_ratio": 0.0}, seed=7, use_gpu=True)
    assert type(est).__module__.startswith("cuml")
    svm = _make_estimator("svm_rbf", {"C": 1.0, "gamma": 0.1}, seed=7, use_gpu=True)
    assert type(svm.estimator).__module__.startswith("cuml")


def test_gpu_estimator_fits_and_predicts():
    """GPU logreg/SVM fit on the small synthetic problem and yield AUROC ~1."""
    pytest.importorskip("cuml")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=80) > 0).astype(float)
    for kind, params in (
        ("logreg", {"C": 1.0, "l1_ratio": 0.0}),
        ("svm_rbf", {"C": 1.0, "gamma": 0.5}),
    ):
        est = _make_estimator(kind, params, seed=7, use_gpu=True)
        est.fit(X, y)
        p = est.predict_proba(X)[:, 1]
        assert roc_auc_score(y, p) > 0.8

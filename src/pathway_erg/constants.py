"""Shared named constants for the PATH-ERG pipeline.

Single source of truth for statistical conventions, artifact layout, and
classifier hyperparameters so no module hardcodes its own copy.
"""

from __future__ import annotations

# --- statistical conventions ------------------------------------------------
# Normal-consistent MAD multiplier (scale ~ sigma for Gaussian data).
MAD_SCALE = 1.4826
# Numerical epsilon for log-mass / OT stability (plan Section 7.4).
MASS_EPSILON = 1e-8

# --- zarr layout ------------------------------------------------------------
RAW_CURVES_ZARR = "raw_curves.zarr"
COMPONENT_CURVES_ZARR = "component_curves.zarr"
SIGNED_OT_ZARR = "signed_ot.zarr"
RAW_ARRAY_GROUP = "raw"
COMPONENT_ARRAY_GROUP = "components"
RAW_CHUNK_POINTS = 1 << 20
OFFSETS_CHUNK = 1 << 16
COMPONENT_CHUNK_ROWS = 256

# --- artifact filenames ------------------------------------------------------
OUTER_FOLDS_TEMPLATE = "outer_folds_{version}.parquet"
INNER_FOLDS_TEMPLATE = "inner_folds_{version}.parquet"
SCALERS_TEMPLATE = "scalers_fold{fold}.json"
QC_THRESHOLDS_FILENAME = "qc_thresholds_by_fold.json"
QA_REPORT_FILENAME = "qa_report.html"

# --- QA / baseline classifier defaults ---------------------------------------
QA_SEED = 12345
PER_STRATUM_SAMPLE = 3
LR_MAX_ITER = 5000
LR_C = 1.0
QC_CV_FOLDS = 5
MIN_CV_FOLDS = 2
MIN_CLASSIFIER_EVENTS = 10
MIN_AUC_N = 20
SUPPORT_EPSILON_MS = 1e-9
MIN_SMOOTH_POINTS = 5

# --- evaluation (plan Section 18) --------------------------------------------
DEFAULT_N_BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 424242
DEFAULT_CONFIDENCE = 0.95
ECE_BINS = 10
BASELINE_FEATURE_EPSILON = 1e-12

# --- E4 mathematical baseline constants ---------------------------------------
SCDT_N_QUANTILES = 64
SCATTERING_WAVELET = "db2"
SCATTERING_LEVELS = (1, 2, 3, 4)
SCATTERING_ORDER2_LEVELS = (1, 2)
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)
LR_L1_GRID = (0.0, 0.5, 1.0)
SVM_C_GRID = (0.1, 1.0, 10.0)
SVM_GAMMA_GRID = ("scale", 0.01, 0.1)
GB_MAX_ITER_GRID = (50, 200)
GB_LEARNING_RATE_GRID = (0.05, 0.1)
GB_DEPTH_GRID = (2, 3)
GB_L2_GRID = (0.1, 1.0)
BASELINE_PCA_VARIANCE = 0.95
BASELINE_MAX_PCA_COMPONENTS = 32

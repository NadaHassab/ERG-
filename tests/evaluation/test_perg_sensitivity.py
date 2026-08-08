"""PERG sensitivity ablations (Phase 8): unit-level helpers and masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.evaluation.perg_sensitivity import (
    FAMILY_LABELS,
    PergSensitivityConfig,
    _aligned_acuity,
    _build_features,
)


def _units(n_female=3, n_male=2, age=(5, 20, 35, 40, 60)) -> pd.DataFrame:
    units = pd.DataFrame(
        {
            "unit_id": [f"PERG_{1000 + i}" for i in range(5)],
            "subject_id": [f"PERG_SRC_{1000 + i}" for i in range(5)],
            "visit_id": [f"PERG_{1000 + i}" for i in range(5)],
            "target_binary": [1, 0, 1, 0, 1],
            "sex_standardized": [0] * n_female + [1] * n_male,
            "age_years": list(age),
            "outer_fold": [0, 1, 2, 3, 4],
            "visit_date": [f"2020-01-0{i + 1}" for i in range(5)],
            "diagnosis1_raw": ["Normal", "Normal", "Retinitis pigmentosa", "Retinitis pigmentosa", "Normal"],
        }
    )
    return units


def test_config_defaults():
    cfg = PergSensitivityConfig(name="t")
    assert cfg.datasets == ("PERG",)
    assert cfg.ablations == ()
    assert cfg.min_family_n == 20


def test_aligned_acuity_missing_stays_missing():
    units = _units()
    acu = pd.DataFrame(
        {
            "source_record_id": ["1000", "1001"],
            "va_re_logmar": [-0.1, 0.2],
            "va_le_logmar": [0.0, np.nan],
            "acuity_missing": [False, True],
            "acuity_n_eyes": [0, 1],
        }
    )
    aligned = _aligned_acuity(units, acu)
    assert aligned["acuity_missing"].to_list() == [False, True, True, True, True]
    assert aligned["acuity_n_eyes"].iloc[0] == 0
    assert np.isnan(aligned["va_re_logmar"].iloc[4])


def test_family_labels_bundle_aliases():
    assert FAMILY_LABELS["Stargardt disease"] == FAMILY_LABELS["Macular dystrophy"]
    assert FAMILY_LABELS["Retinitis pigmentosa"] == "rp"


def test_build_features_unknown_method_raises():
    units = _units()
    with pytest.raises(ValueError, match="unknown method"):
        _build_features(
            units,
            components=pd.DataFrame(),
            recordings=pd.DataFrame(),
            curves=np.zeros((0, 0)),
            valid=np.empty((0, 0)),
            sot=np.zeros((0, 0)),
            spectral=np.zeros((0, 0)),
            spectral_names=[],
            methods=["nonsense_method"],
            acuity_df=None,
            acuity_as_feature=False,
        )
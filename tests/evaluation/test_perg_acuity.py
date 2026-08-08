"""PERG logMAR acuity parser (plan Section 8.3, Section 16.6 E11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathway_erg.data.perg import parse_perg_acuity


def _metadata_csv(tmp_path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "participants_info.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_na_acuity_parses_as_null(tmp_path):
    csv = _metadata_csv(
        tmp_path,
        [
            {"id_record": 1, "date": "2020-01-01", "va_re_logMar": -0.08, "va_le_logMar": 0.06},
            {"id_record": 2, "date": "2020-01-02", "va_re_logMar": "NA", "va_le_logMar": "NA"},
            {"id_record": 3, "date": "2020-01-03", "va_re_logMar": 0.1, "va_le_logMar": "N/A"},
            {"id_record": 4, "date": "2020-01-04", "va_re_logMar": "", "va_le_logMar": 0.2},
        ],
    )
    a = parse_perg_acuity(csv)
    assert a.loc["0001", "va_re_logmar"] == pytest.approx(-0.08)
    assert bool(a.loc["0001", "acuity_missing"]) is False
    assert np.isnan(a.loc["0002", "va_re_logmar"])
    assert bool(a.loc["0002", "acuity_missing"]) is True
    assert a.loc["0002", "acuity_n_eyes"] == 2
    assert np.isnan(a.loc["0003", "va_le_logmar"])
    assert np.isnan(a.loc["0004", "va_re_logmar"])
    assert a.loc["0004", "acuity_n_eyes"] == 1


def test_ids_are_zero_padded_four_digits(tmp_path):
    csv = _metadata_csv(tmp_path, [{"id_record": 7, "date": "2020-01-01", "va_re_logMar": 0.0, "va_le_logMar": 0.0}])
    a = parse_perg_acuity(csv)
    assert a.index.tolist() == ["0007"]


def test_missing_eye_flags(tmp_path):
    csv = _metadata_csv(
        tmp_path,
        [
            {"id_record": 1, "date": "2020-01-01", "va_re_logMar": 0.1, "va_le_logMar": "NA"},
        ],
    )
    a = parse_perg_acuity(csv)
    assert bool(a.loc["0001", "acuity_missing"]) is True
    assert a.loc["0001", "acuity_n_eyes"] == 1


def test_blank_metadata_raises(tmp_path):
    csv = _metadata_csv(tmp_path, [{"id_record": "", "date": "2020-01-01", "va_re_logMar": 0.1, "va_le_logMar": 0.1}])
    with pytest.raises(ValueError, match="empty id_record"):
        parse_perg_acuity(csv)

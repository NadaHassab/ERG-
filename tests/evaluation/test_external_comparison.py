from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pathway_erg.evaluation.external_comparison import compare_external_sslinit


def _predictions(delta: float = 0.0) -> pd.DataFrame:
    rows = []
    for task in ("LEOP", "PERG"):
        for fold in range(5):
            for index in range(4):
                target = index % 2
                probability = 0.75 if target else 0.25
                probability = np.clip(probability + delta * (1 if target else -1), 0.01, 0.99)
                rows.append(
                    {
                        "task": task,
                        "outer_fold": fold,
                        "unit_id": f"{task}-{fold}-{index}",
                        "subject_id": f"{task}-subject-{fold}-{index}",
                        "target": target,
                        "calibrated_probability": probability,
                    }
                )
    return pd.DataFrame(rows)


def test_compare_external_sslinit_aligns_and_writes(tmp_path):
    external = _predictions(0.1).sample(frac=1.0, random_state=3)
    internal = _predictions(-0.1).sample(frac=1.0, random_state=7)
    external_path = tmp_path / "external.parquet"
    internal_path = tmp_path / "internal.parquet"
    output_path = tmp_path / "external_v1" / "paired.json"
    external.to_parquet(external_path, index=False)
    internal.to_parquet(internal_path, index=False)

    result = compare_external_sslinit(
        external_path,
        internal_path,
        output_path,
        n_reps=50,
        n_perm=20,
        seed=4,
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == result
    assert set(result["tasks"]) == {"LEOP", "PERG"}
    for report in result["tasks"].values():
        assert report["diff_point"] >= 0
        assert 0 <= report["p_value_holm"] <= 1


def test_compare_external_sslinit_rejects_incomplete_folds(tmp_path):
    external = _predictions()
    internal = _predictions()
    external = external[external["outer_fold"] != 4]
    external_path = tmp_path / "external.parquet"
    internal_path = tmp_path / "internal.parquet"
    external.to_parquet(external_path, index=False)
    internal.to_parquet(internal_path, index=False)
    with pytest.raises(ValueError, match="incomplete outer-fold coverage"):
        compare_external_sslinit(external_path, internal_path, tmp_path / "out.json")


def test_compare_external_sslinit_rejects_label_mismatch(tmp_path):
    external = _predictions()
    internal = _predictions()
    external.loc[0, "target"] = 1
    external_path = tmp_path / "external.parquet"
    internal_path = tmp_path / "internal.parquet"
    external.to_parquet(external_path, index=False)
    internal.to_parquet(internal_path, index=False)
    with pytest.raises(ValueError, match="models differ on paired column 'target'"):
        compare_external_sslinit(external_path, internal_path, tmp_path / "out.json")


def test_compare_external_sslinit_rejects_duplicate_units(tmp_path):
    external = _predictions()
    internal = _predictions()
    external.loc[1, "unit_id"] = external.loc[0, "unit_id"]
    external_path = tmp_path / "external.parquet"
    internal_path = tmp_path / "internal.parquet"
    external.to_parquet(external_path, index=False)
    internal.to_parquet(internal_path, index=False)
    with pytest.raises(ValueError, match="duplicate unit_id"):
        compare_external_sslinit(external_path, internal_path, tmp_path / "out.json")

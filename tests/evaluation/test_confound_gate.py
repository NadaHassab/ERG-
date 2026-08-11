"""Neural post-hoc confound gate tests (plan Section 17 / E0 decision rule)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathway_erg.evaluation.confound_gate import (
    _channel_aucs,
    _margin_check,
    load_ensemble_predictions,
    run_confound_gate,
    subject_channel_features,
)
from pathway_erg.training.separate import SeparateTrainingConfig

ROOT = Path("artifacts")


def _synthetic_merged(n_subjects: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        y = int(rng.random() < 0.5)
        n_visits = int(rng.integers(1, 4))
        for v in range(n_visits):
            n_comp = int(rng.integers(4, 12))
            for _ in range(n_comp):
                rows.append(
                    {
                        "global_component_id": f"c{len(rows)}",
                        "global_subject_id": f"S{s}",
                        "global_visit_id": f"S{s}V{v}",
                        "component_id": f"L{len(rows)}",
                        "fallback_used": int(rng.random() < 0.1),
                        "component_qc_flags": "truncated-low"
                        if rng.random() < 0.2
                        else None,
                        "waveform_kind": "OP" if rng.random() < 0.5 else "ERG",
                        "protocol": "9_step" if rng.random() < 0.8 else "2_step",
                        "target_binary": y,
                        "dataset": "LEOP",
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_participants(subjects: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_subject_id": subjects,
            "sex_standardized": [("1" if i % 2 == 0 else "2") for i in range(len(subjects))],
        }
    )


# ---------------------------------------------------------------------------
# subject_channel_features
# ---------------------------------------------------------------------------


def test_channel_features_rates_and_sex():
    merged = _synthetic_merged(seed=11)
    subjects = pd.Series(sorted(merged["global_subject_id"].unique()))
    frame = subject_channel_features(merged, _synthetic_participants(subjects))
    assert len(frame) == len(subjects)
    assert set(frame["global_subject_id"]) == set(subjects)
    assert {"fallback_rate", "qc_flag_rate", "op_missingness", "protocol_count", "sex"} <= set(frame.columns)
    assert frame["fallback_rate"].between(0.0, 1.0).all()
    assert frame["qc_flag_rate"].between(0.0, 1.0).all()
    assert frame["op_missingness"].between(0.0, 1.0).all()
    assert set(frame["sex"]) <= {"M", "F"}


def test_channel_features_without_op_components_is_nan():
    merged = _synthetic_merged(seed=11)
    merged = merged.assign(waveform_kind="ERG")
    subjects = pd.Series(sorted(merged["global_subject_id"].unique()))
    frame = subject_channel_features(merged, _synthetic_participants(subjects))
    assert frame["op_missingness"].isna().all()


def test_channel_features_op_is_dataset_specific():
    merged = _synthetic_merged(seed=13)
    perg = merged.iloc[:20].copy()
    perg["global_subject_id"] = "P0"
    perg["global_visit_id"] = "P0V0"
    perg["dataset"] = "PERG"
    perg["component_id"] = "P_LATE"
    perg["waveform_kind"] = "ERG"
    merged = pd.concat([merged, perg], ignore_index=True)
    subjects = pd.Series(sorted(merged["global_subject_id"].unique()))
    frame = subject_channel_features(merged, _synthetic_participants(subjects))
    by_subject = frame.set_index("global_subject_id")
    assert pd.isna(by_subject.loc["P0", "op_missingness"])
    assert by_subject.drop(index="P0")["op_missingness"].notna().all()


def test_channel_features_independent_of_component_count():
    # rates must be fractions, not counts — a subject with more components
    # gets the same rates when the underlying fraction is identical.
    merged = pd.DataFrame(
        {
            "global_component_id": [f"c{i}" for i in range(30)],
            "global_subject_id": ["S0"] * 10 + ["S1"] * 20,
            "global_visit_id": ["S0V0"] * 10 + ["S1V0"] * 20,
            "component_id": [f"L{i}" for i in range(30)],
            "fallback_used": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [1, 0] * 9 + [1, 1],
            "component_qc_flags": [None] * 10 + ["truncated-low"] * 4 + [None] * 16,
            "waveform_kind": ["ERG"] * 10 + ["OP"] * 20,
            "protocol": ["9_step"] * 30,
            "target_binary": [0] * 10 + [1] * 20,
            "dataset": ["LEOP"] * 30,
        }
    )
    subjects = pd.Series(["S0", "S1"])
    frame = subject_channel_features(merged, _synthetic_participants(subjects))
    s0 = frame.set_index("global_subject_id").loc["S0"]
    s1 = frame.set_index("global_subject_id").loc["S1"]
    assert s0["fallback_rate"] == pytest.approx(0.1)
    assert s1["fallback_rate"] == pytest.approx(0.55)
    assert s0["qc_flag_rate"] == pytest.approx(0.0)
    assert s1["qc_flag_rate"] == pytest.approx(0.2)
    assert s1["op_missingness"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _channel_aucs / _margin_check
# ---------------------------------------------------------------------------


def test_margin_check_threshold():
    assert _margin_check(0.75, [0.60]).outcome == "PASS"
    assert _margin_check(0.64, [0.60]).outcome == "FAIL"
    assert _margin_check(0.60, [0.70]).outcome == "FAIL"
    assert _margin_check(0.69, [0.60, 0.65]).outcome == "FAIL"
    assert _margin_check(None, [0.60]).outcome == "INFO"
    assert _margin_check(0.70, []).outcome == "INFO"


def test_channel_aucs_pass_and_fail():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2)
    subj = np.array([f"S{i}" for i in range(len(y))])
    frame = pd.DataFrame(
        {
            "fallback_rate": 0.5 - y * 0.4,          # perfect negative shortcut
            "qc_flag_rate": y * 0.0,                  # chance
        }
    )
    checks = _channel_aucs(frame, ("fallback_rate", "qc_flag_rate"), y, subj, seed=1)
    by_name = {c.check: c.outcome for c in checks}
    assert by_name["fallback_rate"] == "FAIL"
    assert by_name["qc_flag_rate"] == "PASS"
    for c in checks:
        assert "AUROC" in c.measurement


def test_channel_aucs_info_on_small_or_single_class():
    y = np.zeros(10)
    subj = np.array([f"S{i}" for i in range(10)])
    frame = pd.DataFrame({"fallback_rate": np.zeros(10), "qc_flag_rate": np.zeros(10)})
    checks = _channel_aucs(frame, ("fallback_rate",), y, subj, seed=1)
    assert checks[0].outcome == "INFO"


# ---------------------------------------------------------------------------
# load_ensemble_predictions validation
# ---------------------------------------------------------------------------


def _write_predictions(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _minimal_cfg(output_subdir: str) -> SeparateTrainingConfig:
    return SeparateTrainingConfig(
        name="gate_test",
        method="gate_test",
        output_subdir=output_subdir,
        bootstrap_seed=123,
    )


def test_load_rejects_missing_file(tmp_path):
    cfg = _minimal_cfg("does_not_exist")
    with pytest.raises(FileNotFoundError):
        load_ensemble_predictions(tmp_path, cfg)


def test_load_rejects_bad_probabilities(tmp_path):
    out = tmp_path / "results" / "out"
    frame = pd.DataFrame(
        {
            "task": ["LEOP", "LEOP"],
            "unit_id": ["u0", "u1"],
            "subject_id": ["S0", "S1"],
            "target": [0, 1],
            "calibrated_probability": [0.2, 1.7],
        }
    )
    _write_predictions(out / "predictions.parquet", frame)
    with pytest.raises(ValueError, match="outside"):
        load_ensemble_predictions(tmp_path, _minimal_cfg("out"))


def test_load_rejects_duplicate_units(tmp_path):
    out = tmp_path / "results" / "out"
    frame = pd.DataFrame(
        {
            "task": ["LEOP", "LEOP"],
            "unit_id": ["u0", "u0"],
            "subject_id": ["S0", "S0"],
            "target": [0, 1],
            "calibrated_probability": [0.2, 0.8],
        }
    )
    _write_predictions(out / "predictions.parquet", frame)
    with pytest.raises(ValueError, match="duplicate"):
        load_ensemble_predictions(tmp_path, _minimal_cfg("out"))


def test_load_rejects_missing_columns(tmp_path):
    out = tmp_path / "results" / "out"
    frame = pd.DataFrame({"task": ["LEOP"], "unit_id": ["u0"]})
    _write_predictions(out / "predictions.parquet", frame)
    with pytest.raises(ValueError, match="missing columns"):
        load_ensemble_predictions(tmp_path, _minimal_cfg("out"))


# ---------------------------------------------------------------------------
# Real-cache smoke (end-to-end plumbing, synthetic predictions on real units)
# ---------------------------------------------------------------------------


def _load_real_merged():
    from pathway_erg.qa.report import _load_merged

    return _load_merged(ROOT)


def test_end_to_end_on_real_units():
    """Full gate over real LEOP channel rows with synthetic predictions."""
    if not ROOT.exists():
        pytest.skip("real artifacts not present")
    merged = _load_real_merged()
    leop = merged[
        (merged["dataset"] == "LEOP")
        & (merged["target_binary"].notna())
        & (merged["protocol"] == "9_step")
    ]
    subj_order = sorted(
        leop.groupby("global_subject_id")["target_binary"].first().index.astype(str)
    )
    y = leop.groupby("global_subject_id")["target_binary"].first()
    y = y.reindex(subj_order).to_numpy(float)
    rng = np.random.default_rng(0)
    p = rng.random(len(subj_order))

    out_dir = ROOT / "results" / "gate_smoke_out"
    cfg = SeparateTrainingConfig(
        name="gate_test", method="gate_test", output_subdir="gate_smoke_out", bootstrap_seed=123
    )
    preds = pd.DataFrame(
        {
            "task": ["LEOP"] * len(subj_order),
            "unit_id": subj_order,
            "subject_id": subj_order,
            "target": y.astype(int),
            "calibrated_probability": p,
        }
    )
    _write_predictions(out_dir / "predictions.parquet", preds)

    result = run_confound_gate(ROOT, cfg, out_subdir="confounds_gate_test")
    assert result["verdict"] in ("PASS", "FAIL")
    checks = result["checks"]
    assert any(c["check"] == "LEOP: signal-over-shortcut margin" for c in checks)
    assert any(c["check"] == "LEOP: fallback_rate" for c in checks)
    assert any(c["check"].startswith("LEOP: neural AUROC") for c in checks)
    out = ROOT / "results" / "confounds_gate_test"
    assert (out / "neural_confound_gate.json").is_file()
    assert (out / "neural_confound_gate.md").is_file()
    (out / "neural_confound_gate.json").unlink()
    (out / "neural_confound_gate.md").unlink()
    out.rmdir()
    (out_dir / "predictions.parquet").unlink()
    out_dir.rmdir()

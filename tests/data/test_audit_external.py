"""Audit extension tests: flinders/urfu are walked, licenses recorded, junk excluded."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathway_erg.config import DataConfig, load_config
from pathway_erg.data.audit import DATASET_VERSION, audit_raw_files

CONFIG = Path("configs/data/local.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_config(DataConfig, CONFIG)


@pytest.fixture(scope="module")
def result(cfg):
    return audit_raw_files(cfg)


def test_versions_registered():
    assert "flinders" in DATASET_VERSION
    assert "urfu" in DATASET_VERSION


def test_external_datasets_walked(result):
    datasets = set(result.table["dataset"])
    assert {"leops", "perg", "flinders", "urfu"} <= datasets


def test_no_macosx_in_table(result):
    assert not any("__MACOSX" in str(p) for p in result.table["relative_path"])


def test_flinders_files_audited(result):
    sub = result.table[result.table["dataset"] == "flinders"]
    assert {"ISCEV Control ERG Flinders University.xlsx",
            "ISCEV Control ERG Flinders University.sav"} <= set(sub["relative_path"])


def test_urfu_appendix_and_protocols_audited(result):
    sub = result.table[result.table["dataset"] == "urfu"]
    assert {"01 Appendix 1.xlsx", "02 Appendix 2.xlsx",
            "00 Description of Research Protocols.pdf"} <= set(sub["relative_path"])


def test_license_notes_present(result):
    assert "CC-BY-NC" in result.license_notes["flinders"]
    assert "IEEE DataPort" in result.license_notes["urfu"]
    assert "LICENSE.txt present" in result.license_notes["perg"]


def test_excluded_dirs_documented(result):
    assert "__MACOSX" in result.excluded_dirs


def test_no_audit_failures(result):
    assert result.failures == []


def test_parser_types_classified(result):
    fl = result.table[result.table["dataset"] == "flinders"]
    assert set(fl["parser_type"]) == {"flinders_xlsx", "flinders_aux"}
    ur = result.table[result.table["dataset"] == "urfu"]
    assert "urfu_xlsx" in set(ur["parser_type"])

"""Immutable raw-file audit.

Recursively inventories every source file, records relative path, bytes,
SHA-256, modification time, parser type, and dataset version, and verifies
provided checksums (PERG ships SHA256SUMS.txt).  Raw files are never modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..provenance import sha256_file

DATASET_VERSION: dict[str, str] = {
    "leops": "LEOPs_v1.0.0",
    "perg": "PERG-IOBA_1.0.0",
    "flinders": "Flinders_ISCEV_Control_v1.0.0",
    "urfu": "URFU_OculusGraphy_v1.0.0",
}

# Directories that are OS/archive junk and must never enter the audit walk.
_AUDIT_EXCLUDE_DIRS = ("__MACOSX",)


@dataclass
class AuditResult:
    table: pd.DataFrame
    total_bytes: int
    total_files: int
    failures: list[str]
    license_notes: dict[str, str]
    excluded_dirs: str = ""

    def to_json(self) -> dict:
        return {
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "failures": self.failures,
            "licenses": self.license_notes,
            "excluded_dirs": self.excluded_dirs,
        }


def _parse_perg_checksums(sums_path: Path) -> dict[str, str]:
    """Parse PhysioNet SHA256SUMS.txt (sha256  filename)."""
    checksums: dict[str, str] = {}
    if not sums_path.is_file():
        return checksums
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        digest, name = parts
        checksums[name] = digest.lower()
    return checksums


def _walk(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in _AUDIT_EXCLUDE_DIRS for part in p.parts)
    )


def _excluded_dir_note(cfg: DataConfig) -> str:
    """Document which excluded dirs were present under configured roots."""
    notes: list[str] = []
    urfu_root = Path(cfg.urfu.root) if cfg.urfu is not None else None
    if urfu_root is not None and urfu_root.is_dir():
        present = sorted(
            str(p.relative_to(urfu_root))
            for p in urfu_root.rglob("*")
            if any(part in _AUDIT_EXCLUDE_DIRS for part in p.parts)
        )
        if present:
            notes.append("excluded from audit walk: " + ", ".join(present[:20]))
    return "; ".join(notes)


def audit_raw_files(cfg: DataConfig) -> AuditResult:
    """Inventory raw roots and compute checksums.  Read-only on raw data."""
    failures: list[str] = []
    license_notes: dict[str, str] = {}
    rows: list[dict] = []

    perg_root = Path(cfg.perg.root)
    perg_sums = perg_root / "SHA256SUMS.txt"
    provided = _parse_perg_checksums(perg_sums)
    license_notes["perg"] = (
        "PhysioNet PERG-IOBA dataset 1.0.0; LICENSE.txt present: "
        f"{Path(cfg.perg.root, 'LICENSE.txt').is_file()}"
    )

    roots: list[tuple[Path, str]] = [
        (Path(cfg.leops.json_root), "leops"),
        (perg_root, "perg"),
    ]
    if cfg.flinders is not None:
        roots.append((Path(cfg.flinders.xlsx_path).parent, "flinders"))
        license_notes["flinders"] = (
            "figshare 10.25451/flinders.14747349 (Paul Constable, Flinders Univ., "
            "2021) ISCEV Control ERG; CC-BY-NC 4.0"
        )
    if cfg.urfu is not None:
        roots.append((Path(cfg.urfu.root), "urfu"))
        license_notes["urfu"] = (
            "IEEE DataPort 10.21227/y0fh-5v04 (URFU OculusGraphy, 2020) + "
            "10.21227/r1wb-pg25 (2022); recorded at IRTC Eye Microsurgery "
            "Ekaterinburg with Tomey EP-1000"
        )

    for root, dataset in roots:
        if not root.is_dir():
            failures.append(f"missing raw root: {root}")
            continue
        base = Path(cfg.leops.json_root) if dataset == "leops" else root
        for path in _walk(root):
            rel = str(path.relative_to(base))
            digest = sha256_file(path)
            size = path.stat().st_size
            if dataset == "perg" and rel in provided and provided[rel] != digest:
                failures.append(f"checksum mismatch: {rel}")
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_version": DATASET_VERSION[dataset],
                    "relative_path": rel,
                    "bytes": size,
                    "sha256": digest,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "parser_type": "leops_json" if path.suffix == ".json" and dataset == "leops"
                    else ("perg_metadata" if rel == "csv/participants_info.csv"
                          else "perg_waveform" if path.suffix == ".csv" and dataset == "perg"
                          else "perg_aux" if dataset == "perg"
                          else "flinders_xlsx" if path.suffix == ".xlsx" and dataset == "flinders"
                          else "flinders_aux" if dataset == "flinders"
                          else "urfu_xlsx" if path.suffix == ".xlsx" and dataset == "urfu"
                          else "urfu_aux" if dataset == "urfu"
                          else "leops_aux"),
                }
            )
    table = pd.DataFrame(rows)
    return AuditResult(
        table=table,
        total_bytes=int(table["bytes"].sum()) if not table.empty else 0,
        total_files=len(table),
        failures=failures,
        license_notes=license_notes,
        excluded_dirs=_excluded_dir_note(cfg),
    )


def write_audit(result: AuditResult, manifest: dict, artifact_root: Path) -> Path:
    """Write raw_files.parquet, raw_audit.json, and license_report.md."""
    artifact_root = Path(artifact_root)
    out = artifact_root / "data" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "raw_files.parquet"
    result.table.to_parquet(parquet_path, index=False)
    audit_json = {**result.to_json(), "manifest": manifest}
    json_path = out / "raw_audit.json"
    json_path.write_text(json.dumps(audit_json, indent=2, sort_keys=True))
    license_path = out / "license_report.md"
    lines = ["# Raw data license and provenance report", ""]
    for dataset, note in result.license_notes.items():
        lines.append(f"- {dataset}: {note or 'no license file found in raw root'}")
    excluded = result.excluded_dirs
    if excluded:
        lines.append("")
        lines.append("## Excluded from audit walk (OS/archive junk)")
        lines.append(f"- {excluded}")
    license_path.write_text("\n".join(lines) + "\n")
    return parquet_path

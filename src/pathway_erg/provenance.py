"""Provenance: file hashes, git status, and run manifests.

Every artifact and run records SHA-256 hashes of the inputs it consumed so
results can be traced back to exact source files and configurations.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HASH_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def git_revision(root: str | Path) -> dict[str, str | None]:
    """Capture git revision and dirty status without modifying the repo."""
    root = Path(root)
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None, "branch": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}
    except subprocess.CalledProcessError:
        return {"commit": None, "dirty": None, "branch": None}


def environment_snapshot() -> dict[str, str]:
    """Coarse environment identifiers for run manifests."""
    return {
        "platform": os.uname().sysname,
        "node": os.uname().nodename,
        "release": os.uname().release,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@dataclass
class RunManifest:
    """One manifest schema used by every run and artifact producer."""

    kind: str
    name: str
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_hash: str = ""
    data_hash: str = ""
    split_hash: str = ""
    label_mapping_hash: str = ""
    code_revision: dict[str, str | None] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=environment_snapshot)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256_text(canonical)

    def write_atomic(self, path: str | Path) -> Path:
        """Write manifest atomically so interrupted runs leave no partial file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)
        return path

    @staticmethod
    def load(path: str | Path) -> RunManifest:
        data = json.loads(Path(path).read_text())
        m = RunManifest(kind=data["kind"], name=data["name"])
        for key, value in data.items():
            if hasattr(m, key):
                setattr(m, key, value)
        return m

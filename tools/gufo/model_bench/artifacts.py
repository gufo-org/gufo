"""Retained per-table JSON artifacts under docs/models/<model>/artifacts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from gufo.serving_bench import atomic_json

from .config import BenchConfig, TableSpec

ARTIFACT_TYPE = "model-bench-table"
SCHEMA_VERSION = 1


def artifact_path(
    config: BenchConfig, table: TableSpec, target: str, mode: str | None = None
) -> Path:
    name = f"{table.id}-{target}" + (f"-{mode}" if mode else "") + ".json"
    return config.artifacts_dir / name


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def public_command(command: list[str]) -> list[str]:
    """Server command with file paths reduced and credentials removed."""
    redacted: list[str] = []
    hide_next = False
    secret_options = {"--tp-control-token", "--tp-bootstrap-host", "--api-key"}
    for part in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
        elif part in secret_options:
            redacted.append(part)
            hide_next = True
        else:
            redacted.append(Path(part).name if "/" in part else part)
    return redacted


def new_artifact(
    config: BenchConfig,
    table: TableSpec,
    target: str,
    *,
    mode: str | None,
    command: list[str],
    source: dict[str, Any],
    fingerprint: dict[str, Any] | None,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "artifactType": ARTIFACT_TYPE,
        "schemaVersion": SCHEMA_VERSION,
        "table": table.id,
        "target": target,
        "mode": mode,
        "referenceName": config.reference_name,
        "model": {"id": config.data["model"]["id"], "variant": table.variant},
        "measuredOn": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "source": source,
        "fingerprint": fingerprint,
        "server": {"command": public_command(command)},
        "notes": notes,
        "rows": {},
    }


def merge_rows(existing: dict[str, Any] | None, fresh: dict[str, Any]) -> dict[str, Any]:
    """Keep rows from an earlier artifact that this run did not measure."""
    if existing is None:
        return fresh
    rows = dict(existing.get("rows", {}))
    rows.update(fresh["rows"])
    merged = dict(fresh)
    merged["rows"] = rows
    previous = existing.get("measuredOn")
    if previous and previous != fresh["measuredOn"]:
        merged["notes"] = list(fresh["notes"]) + [
            f"rows not measured on {fresh['measuredOn']} retain values from {previous}"
        ]
    return merged


def save_artifact(path: Path, artifact: dict[str, Any]) -> None:
    atomic_json(path, artifact)
    path.chmod(0o644)  # tempfile-created artifacts default to 0600

"""bench.json loading and table expansion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "gufo-model-bench/1"
TARGETS = ("gufo", "reference")


def table_id(base: str, variant: str | None) -> str:
    """Per-variant tables carry the variant id; the `default` variant's keep
    the bare base id, so a card can add a variant beside its original one."""
    return base if variant in (None, "default") else f"{base}-{variant}"


@dataclass(frozen=True)
class TableSpec:
    id: str
    base: str
    variant: str | None
    spec: dict[str, Any]

    @property
    def kind(self) -> str:
        if self.base.startswith("single-"):
            return "single"
        if self.base.startswith("multi-"):
            return "multi"
        return self.base

    @property
    def speculative(self) -> bool:
        return bool(self.spec.get("speculative"))

    def workload_tables(self) -> list[TableSpec]:
        """Keep each workload's measurements under its existing artifact identity."""
        if self.kind not in ("single", "multi"):
            return []
        common = {k: v for k, v in self.spec.items() if k != "workloads"}
        return [
            TableSpec(table_id(base, self.variant), base, self.variant,
                      {**common, **workload})
            for base, workload in self.spec.get("workloads", {}).items()
        ]


@dataclass
class BenchConfig:
    model: str
    root: Path
    data: dict[str, Any]
    files: dict[str, dict[str, Path]] = field(default_factory=dict)
    artifacts_override: Path | None = None

    @property
    def model_dir(self) -> Path:
        return self.root / "docs" / "models" / self.model

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_override or (self.model_dir / "artifacts")

    @property
    def benchmarks_path(self) -> Path:
        return self.model_dir / "BENCHMARKS.md"

    @property
    def category(self) -> str:
        return self.data["category"]

    @property
    def variants(self) -> dict[str, dict[str, Any]]:
        return self.data["variants"]

    @property
    def single_variant(self) -> bool:
        return list(self.variants) == ["default"]

    @property
    def reference_name(self) -> str:
        return self.data["reference"]["name"]

    @property
    def speculative(self) -> dict[str, Any]:
        return self.data["speculative"]

    @property
    def reference_speculative(self) -> list[str] | None:
        """Reference server arguments for the equivalent speculative mode, if any."""
        reference = self.speculative.get("reference")
        if isinstance(reference, dict) and reference.get("args"):
            return list(reference["args"])
        return None

    @property
    def speculative_self_reference(self) -> bool:
        """Multi-user speculative completions are checked against the same
        mode's C1 instead of the AR artifact: set for engines whose greedy
        speculation equals single-token decode with the drafter loaded but
        not an AR-only server (Gemma 4, see its QUALITY.md)."""
        return self.data.get("speculative", {}).get("exactness_reference") == "self"

    @property
    def ar_exactness_reported(self) -> bool:
        """Multi-user AR completions that differ from the isolated C1 are
        recorded instead of failing the table: set for engines whose batched
        AR rounds differently from one session's faster GEMV, so a greedy
        near-tie can depend on concurrency (Gemma 4, see its QUALITY.md)."""
        return self.data.get("gufo", {}).get("ar_exactness") == "reported"

    def substitute(self, args: list[str], variant: str | None) -> list[str]:
        """Replace `{role}` placeholders with the variant's file paths."""
        out = []
        for part in args:
            if part.startswith("{") and part.endswith("}"):
                part = str(self.file(part[1:-1], variant))
            out.append(part)
        return out

    def tables(self) -> list[TableSpec]:
        """A variant with a `tables` map gets only the per-variant tables it
        names, each spec updated with that entry (a reduced depth or
        concurrency grid, for example)."""
        result: list[TableSpec] = []
        for base, spec in self.data["tables"].items():
            if spec.get("per_variant"):
                for variant, entry in self.variants.items():
                    own = entry.get("tables")
                    if own is not None and base not in own:
                        continue
                    merged = {**spec, **(own or {}).get(base, {})}
                    result.append(TableSpec(table_id(base, variant), base, variant, merged))
            else:
                result.append(TableSpec(base, base, None, spec))
        return result

    def table(self, table_id: str) -> TableSpec:
        for table in self.tables():
            if table.id == table_id:
                return table
        raise KeyError(f"unknown table {table_id!r} for {self.model}")

    def variant_label(self, variant: str) -> str:
        return self.variants[variant]["label"]

    def file(self, role: str, variant: str | None) -> Path:
        variant = variant or "default"
        by_variant = self.files.get(role, {})
        path = by_variant.get(variant) or by_variant.get("*")
        if path is None:
            description = self.data["files"].get(role, role)
            raise SystemExit(
                f"missing --{role} for variant {variant!r}: {description}"
            )
        return path

    def require_files(self, variant: str | None) -> None:
        for role in self.variants[variant or "default"].get("requires", []):
            self.file(role, variant)


def load_config(root: Path, model: str, path: Path | None = None) -> BenchConfig:
    path = path or root / "docs" / "models" / model / "artifacts" / "bench.json"
    if not path.exists():
        raise SystemExit(f"no bench.json for {model}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: expected schema {SCHEMA}")
    return BenchConfig(model=model, root=root, data=data)


def parse_file_args(config: BenchConfig, values: dict[str, list[str]]) -> None:
    """Map `--gguf q4=/path` / `--gguf /path` arguments onto file roles."""
    for role, entries in values.items():
        for entry in entries:
            variant, separator, raw = entry.partition("=")
            if not separator:
                variant, raw = ("default" if config.single_variant else "*"), entry
            path = Path(raw).expanduser()
            if not path.exists():
                raise SystemExit(f"--{role} {entry}: {path} does not exist")
            config.files.setdefault(role, {})[variant] = path.absolute()

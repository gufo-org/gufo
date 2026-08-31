"""Task manifest loading and Nix derivation building for gufo-agent-eval."""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"


class TaskError(RuntimeError):
    pass


@dataclass
class Task:
    name: str
    directory: Path
    workdir: str
    network: str
    agent_timeout_sec: float
    verifier_timeout_sec: float
    manifest: dict

    @property
    def env_nix(self) -> Path:
        return self.directory / "environment" / "env.nix"

    @property
    def verifier_nix(self) -> Path:
        return self.directory / "tests" / "verifier.nix"

    @property
    def tests_dir(self) -> Path:
        return self.directory / "tests"

    @property
    def instruction(self) -> str:
        return (self.directory / "instruction.md").read_text()


def load(name: str, tasks_root: Path | None = None) -> Task:
    root = tasks_root or TASKS_ROOT
    directory = root / name
    manifest_path = directory / "task.toml"
    if not manifest_path.is_file():
        raise TaskError(f"no task manifest at {manifest_path}")

    manifest = tomllib.loads(manifest_path.read_text())
    environment = manifest.get("environment", {})
    network = environment.get("network", "none")
    if network not in ("none", "endpoint-only"):
        raise TaskError(f"{name}: unknown network policy {network!r}")

    return Task(
        name=manifest["task"]["name"],
        directory=directory,
        workdir=environment.get("workdir", "/app"),
        network=network,
        agent_timeout_sec=float(manifest.get("agent", {}).get("timeout_sec", 900.0)),
        verifier_timeout_sec=float(manifest.get("verifier", {}).get("timeout_sec", 900.0)),
        manifest=manifest,
    )


def available(tasks_root: Path | None = None) -> list[str]:
    root = tasks_root or TASKS_ROOT
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "task.toml").is_file())


def nix_build(nix_file: Path) -> Path:
    """Realize a task derivation and return its store path.

    Uses `nix-build` rather than `nix build` deliberately: the flake path
    would require the file to be staged in git before every rebuild, which
    makes task authoring painful. See "Development workflow" in plan.md.
    """
    if not nix_file.is_file():
        raise TaskError(f"missing derivation: {nix_file}")

    completed = subprocess.run(
        ["nix-build", str(nix_file), "--no-out-link"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TaskError(f"nix-build failed for {nix_file}:\n{completed.stderr.strip()}")

    out = completed.stdout.strip().splitlines()
    if not out:
        raise TaskError(f"nix-build produced no output path for {nix_file}")
    return Path(out[-1])

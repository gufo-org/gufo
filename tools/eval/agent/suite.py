"""Suite tiers and benchmark identity for gufo-agent-eval.

A tier is a named, ordered list of tasks. Changing its membership changes the
benchmark, so the identity hash below covers the tier contents along with
everything else that shapes a result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SUITE_NAME = "gufo-agent-eval"
SUITES_ROOT = Path(__file__).resolve().parent.parent / "suites"

DEFAULT_TIER = "smoke"

# Reference runs of the equivalent upstream suite on this hardware had a
# median task duration of 14-29 minutes, with individual tasks reaching six
# hours. Upstream allows three hours per attempt; the 900s in the task
# manifests is upstream's own default, which their runner overrides.
#
# Capping below the median would measure timeouts rather than capability, so
# the runner uses this unless told otherwise.
DEFAULT_AGENT_TIMEOUT_SEC = 3 * 60 * 60


class SuiteError(RuntimeError):
    pass


@dataclass
class Tier:
    name: str
    tasks: list[str]
    path: Path

    def identity(self) -> str:
        """Hash of the tier's membership and order."""
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        for task in self.tasks:
            digest.update(task.encode())
        return digest.hexdigest()[:16]


def available(suites_root: Path | None = None) -> list[str]:
    root = suites_root or SUITES_ROOT
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.txt"))


def load(name: str, suites_root: Path | None = None) -> Tier:
    root = suites_root or SUITES_ROOT
    path = root / f"{name}.txt"
    if not path.is_file():
        known = ", ".join(available(root)) or "none"
        raise SuiteError(f"unknown tier {name!r}; available: {known}")

    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tasks.append(line)

    if not tasks:
        raise SuiteError(f"tier {name!r} lists no tasks")

    return Tier(name=name, tasks=tasks, path=path)


def benchmark_identity(
    tier: Tier,
    agent_config_hash: str,
    task_hashes: dict[str, str],
) -> dict[str, str]:
    """Everything that must be equal for two runs to be comparable.

    Deliberately excludes the endpoint, the model, and the credential: those
    are what a run is measuring or how it connects, not what the benchmark is.
    """
    digest = hashlib.sha256()
    digest.update(tier.identity().encode())
    digest.update(agent_config_hash.encode())
    for name in sorted(task_hashes):
        digest.update(name.encode())
        digest.update(task_hashes[name].encode())

    return {
        "suite": SUITE_NAME,
        "tier": tier.name,
        "tier_hash": tier.identity(),
        "agent_config_hash": agent_config_hash,
        "benchmark_hash": digest.hexdigest()[:16],
    }


def task_identity(task_directory: Path) -> str:
    """Hash of everything in a task that shapes its outcome."""
    digest = hashlib.sha256()
    for relative in sorted(
        p.relative_to(task_directory)
        for p in task_directory.rglob("*")
        if p.is_file() and not p.is_symlink()
    ):
        digest.update(str(relative).encode())
        digest.update((task_directory / relative).read_bytes())
    return digest.hexdigest()[:16]

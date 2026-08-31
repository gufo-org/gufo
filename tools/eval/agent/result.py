"""Versioned, sanitized results for gufo-agent-eval.

#153 requires that a published artifact carry no credentials, private
endpoint addresses, secrets, or user paths. That is enforced here rather than
left to the caller: `sanitize` runs over every emitted document, and
`scrub_text` over any free text that reaches one.

Field shape deliberately follows Terminal-Bench-Mini's per-task result
(`schema_version`, `task`, `passed`, `reward`, `duration_ms`, `tokens`,
`attempts`) so tooling written against that shape mostly carries over, with
the additional identity and failure counters #153 asks for.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gufo-agent-eval/1"

# Keys whose values are replaced wholesale rather than pattern-matched.
_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "token", "secret", "password", "credential"}
)

_REDACTED = "[redacted]"

# Substitutions applied to any string that reaches an artifact.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Home directories, which carry the operator's username.
    (re.compile(r"/home/[^/\s:\"']+"), "/home/[user]"),
    (re.compile(r"/Users/[^/\s:\"']+"), "/Users/[user]"),
    (re.compile(r"/root\b"), "/root"),
    # Private and link-local addresses, with any port.
    (
        re.compile(
            r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?"
            r"|\b192\.168\.\d{1,3}\.\d{1,3}(?::\d+)?"
            r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}(?::\d+)?"
            r"|\b169\.254\.\d{1,3}\.\d{1,3}(?::\d+)?"
        ),
        "[endpoint]",
    ),
    # Bearer tokens and API-key-shaped strings.
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*"), "Bearer " + _REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9._-]{8,}"), _REDACTED),
    # Scratch directories carry a random suffix but also the temp root.
    (re.compile(r"/tmp/gufo-agent-eval-[A-Za-z0-9_]+"), "[scratch]"),
)


def scrub_text(text: str) -> str:
    """Remove credentials, private addresses, and user paths from free text."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize(value: Any) -> Any:
    """Recursively scrub a document destined for an artifact."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                out[key] = _REDACTED
            else:
                out[key] = sanitize(item)
        return out
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def endpoint_label(base_url: str) -> str:
    """A stable, non-identifying label for an endpoint.

    #153 allows a sanitized server label but not the address. The hash lets
    two runs be recognised as hitting the same endpoint without publishing
    where it is.
    """
    digest = hashlib.sha256(base_url.encode()).hexdigest()[:12]
    return f"endpoint-{digest}"


def machine_fingerprint() -> dict[str, Any]:
    """A coarse, non-identifying description of the host.

    Deliberately excludes the hostname, which is operator-identifying; a hash
    of it is kept so repeated runs on one machine are recognisable.
    """
    return {
        "os": platform.system(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "host_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
    }


@dataclass
class TaskResult:
    """One task's outcome."""

    task: str
    completed: bool
    passed: bool
    reward: float

    duration_ms: int
    first_response_ms: int | None

    # Provider interaction
    requests: int
    tokens: dict[str, int]
    context_failures: int

    # Agent behaviour
    agent_steps: int
    tool_calls: int
    tool_failures: int
    tool_names: dict[str, int]

    # Failure modes, kept apart so a timeout is never read as a wrong answer
    agent_timed_out: bool
    agent_exit_code: int
    endpoint_errors: int
    verifier_crashed: bool
    verifier_timed_out: bool

    # Evidence
    workspace_hash: str | None
    verifier_output: str


@dataclass
class RunResult:
    """A whole suite run."""

    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""
    suite: str = ""
    tier: str = ""

    identity: dict[str, Any] = field(default_factory=dict)
    machine: dict[str, Any] = field(default_factory=dict)

    tasks: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def add(self, task: TaskResult) -> None:
        self.tasks.append(asdict(task))

    def finalize(self) -> "RunResult":
        self.generated_at = datetime.now(timezone.utc).isoformat()

        total = len(self.tasks)
        passed = sum(1 for t in self.tasks if t["passed"])
        self.aggregate = {
            "total_tasks": total,
            "passed_tasks": passed,
            # Single attempt per task, so this is pass@1. Named explicitly so
            # it is never mistaken for upstream's default pass@2.
            "pass_at_1": round(passed / total, 4) if total else 0.0,
            "total_duration_ms": sum(t["duration_ms"] for t in self.tasks),
            "timed_out_tasks": sum(1 for t in self.tasks if t["agent_timed_out"]),
            "verifier_crashes": sum(1 for t in self.tasks if t["verifier_crashed"]),
            "tokens": {
                key: sum(t["tokens"].get(key, 0) for t in self.tasks)
                for key in ("input", "output", "cache_read", "total")
            },
        }
        return self

    def write(self, destination: Path) -> Path:
        """Write the sanitized document. Nothing bypasses `sanitize`."""
        document = sanitize(asdict(self))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
        return destination


def workspace_hash(workspace: Path, workdir: str) -> str | None:
    """A digest of the agent's working directory.

    #153 asks for a patch hash. The tasks have no common VCS, so this hashes
    the workdir's file contents instead: it identifies what the agent
    produced without publishing it.
    """
    root = workspace / workdir.lstrip("/")
    if not root.is_dir():
        return None

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:16]

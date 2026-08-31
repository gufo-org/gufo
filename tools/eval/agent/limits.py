"""cgroup v2 resource limits for gufo-agent-eval.

Isolation and resource limits are separate kernel features: bubblewrap
unshares namespaces but applies no caps, so `cpus`, `memory_mb` and
`storage_mb` from a task manifest are enforced here instead.

This needs a delegated cgroup. On a systemd host the user's own slice is
delegated, so an unprivileged process can create subgroups under
`/sys/fs/cgroup/user.slice/user-$UID.slice/user@$UID.service/`. Where that is
unavailable the limits cannot be applied, and the runner records that rather
than pretending they were.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass
class Limits:
    cpus: int | None = None
    memory_mb: int | None = None
    pids: int | None = 4096

    @classmethod
    def from_manifest(cls, environment: dict) -> "Limits":
        return cls(
            cpus=environment.get("cpus"),
            memory_mb=environment.get("memory_mb"),
        )


@dataclass
class LimitStatus:
    applied: bool
    reason: str = ""
    cgroup: str | None = None


def _delegated_root() -> Path | None:
    """The cgroup this user may create subgroups under."""
    uid = os.getuid()
    candidates = [
        CGROUP_ROOT / f"user.slice/user-{uid}.slice/user@{uid}.service",
        CGROUP_ROOT / f"user.slice/user-{uid}.slice",
    ]
    for candidate in candidates:
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return None


def available() -> tuple[bool, str]:
    """Whether limits can be applied on this host."""
    if not (CGROUP_ROOT / "cgroup.controllers").is_file():
        return False, "cgroup v2 not mounted"

    root = _delegated_root()
    if root is None:
        return False, "no writable delegated cgroup for this user"

    controllers = (root / "cgroup.controllers").read_text().split()
    missing = [c for c in ("memory", "pids") if c not in controllers]
    if missing:
        return False, f"controllers not delegated: {', '.join(missing)}"

    return True, ""


@contextmanager
def applied(name: str, limits: Limits):
    """Create a limited cgroup, yielding a status and the path to join.

    The caller moves the sandboxed process into the cgroup by writing its pid
    to `cgroup.procs`. Yields a status with `applied=False` when the host
    cannot support limits, so a run proceeds unbounded but says so.
    """
    ok, reason = available()
    if not ok:
        yield LimitStatus(applied=False, reason=reason), None
        return

    root = _delegated_root()
    assert root is not None
    group = root / f"gufo-agent-eval-{name}-{os.getpid()}"

    try:
        group.mkdir(exist_ok=True)
    except OSError as exc:
        yield LimitStatus(applied=False, reason=f"cannot create cgroup: {exc}"), None
        return

    try:
        if limits.memory_mb:
            # memory.max is a hard limit: the kernel OOM-kills the group
            # rather than letting an agent exhaust the host.
            (group / "memory.max").write_text(str(limits.memory_mb * 1024 * 1024))
        if limits.cpus:
            # cpu.max is quota/period; one full core is 100000/100000.
            (group / "cpu.max").write_text(f"{limits.cpus * 100000} 100000")
        if limits.pids:
            (group / "pids.max").write_text(str(limits.pids))

        yield LimitStatus(applied=True, cgroup=str(group)), group

    except OSError as exc:
        yield LimitStatus(applied=False, reason=f"cannot set limits: {exc}"), None
    finally:
        # A cgroup can only be removed once empty; the sandbox has exited by
        # here, so a failure means a leaked process and is worth ignoring
        # quietly rather than masking the real error.
        try:
            group.rmdir()
        except OSError:
            pass

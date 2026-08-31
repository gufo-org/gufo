"""Sandbox abstraction for gufo-agent-eval.

Callers describe *what* isolation they want via `JailSpec` and never see
bubblewrap argv. Today the only backend is `BwrapBackend`; a future in-tree
`gufo-jail` helper for non-Nix builds is a second implementation of
`SandboxBackend`, not a rewrite of the runner.

Isolation here is namespaces only. Resource limits are cgroup v2 and are a
separate concern; see `limits.py` when it lands.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Namespaces unshared for every jail. `user` is what makes the rest possible
# unprivileged; `net` is what makes the default no-network.
_UNSHARED = ("user", "ipc", "pid", "net", "uts", "cgroup")

# The only host environment variables ever forwarded. Everything else is
# cleared so host credentials cannot reach the agent.
_FORWARDED_ENV = ("TERM",)


class SandboxUnavailable(RuntimeError):
    """The host cannot run sandboxed workloads."""


@dataclass
class JailSpec:
    """A sandbox to run one command in.

    Attributes:
        rootfs: Directory bound as `/`. The caller owns it and it must be
            writable; see `stage_rootfs`.
        command: argv to execute inside the jail.
        chdir: Working directory inside the jail.
        env: Environment inside the jail. The host environment is cleared.
        ro_binds: Host path -> jail path, mounted read-only.
        rw_binds: Host path -> jail path, mounted read-write.
        tmpfs: Jail paths backed by a fresh tmpfs.
        network: ``"none"`` for no connectivity, or ``"endpoint-only"`` to
            reach exactly one inference endpoint. ``"endpoint-only"`` is not
            implemented yet and raises.
        hostname: Hostname inside the jail. Fixed by default so it cannot
            leak the host's name into a trajectory.
    """

    rootfs: Path
    command: list[str]
    chdir: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    ro_binds: dict[Path, str] = field(default_factory=dict)
    rw_binds: dict[Path, str] = field(default_factory=dict)
    tmpfs: tuple[str, ...] = ("/tmp", "/root", "/run")
    network: str = "none"
    hostname: str = "gufo-agent-eval"


@dataclass
class JailResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class SandboxBackend:
    """Interface a sandbox implementation must provide."""

    def preflight(self) -> list[str]:
        """Return a list of reasons this backend is unusable, empty if fine."""
        raise NotImplementedError

    def run(self, spec: JailSpec, timeout_sec: float) -> JailResult:
        raise NotImplementedError


class BwrapBackend(SandboxBackend):
    """bubblewrap backend.

    The baseline policy is ported from jail.nix's `base` combinator, which is
    a well-tested default-deny starting point: fake /proc and /dev, tmpfs for
    scratch and home, a cleared environment, and every namespace unshared.
    """

    def __init__(self, bwrap: str | None = None) -> None:
        self._bwrap = bwrap or os.environ.get("GUFO_EVAL_BWRAP") or shutil.which("bwrap")

    def preflight(self) -> list[str]:
        problems = []
        if not self._bwrap:
            problems.append("bwrap not found; install bubblewrap or set GUFO_EVAL_BWRAP")
            return problems

        max_userns = Path("/proc/sys/user/max_user_namespaces")
        if max_userns.exists() and max_userns.read_text().strip() == "0":
            problems.append(
                "unprivileged user namespaces are disabled "
                "(user.max_user_namespaces is 0)"
            )

        # Ubuntu 23.10+ restricts unprivileged userns via AppArmor, which
        # breaks bwrap unless a profile is installed. Absent on other distros.
        apparmor = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
        if apparmor.exists() and apparmor.read_text().strip() == "1":
            problems.append(
                "AppArmor restricts unprivileged user namespaces; bwrap needs "
                "a profile, or set kernel.apparmor_restrict_unprivileged_userns=0"
            )

        if not Path("/nix/store").is_dir():
            problems.append("/nix/store not found; task rootfs symlinks will not resolve")

        return problems

    def _argv(self, spec: JailSpec) -> list[str]:
        if spec.network == "endpoint-only":
            raise NotImplementedError(
                "endpoint-only networking is not implemented; it needs a pasta "
                "net namespace with a single mapped port"
            )
        if spec.network != "none":
            raise ValueError(f"unknown network policy: {spec.network!r}")

        argv = [self._bwrap]

        for ns in _UNSHARED:
            argv += [f"--unshare-{ns}"]

        # --new-session detaches the controlling terminal so the jailed
        # process cannot inject into the host's tty. --die-with-parent makes
        # orphan cleanup automatic; both are load-bearing for the timeout
        # path and the orphaned-process negative test.
        argv += ["--new-session", "--die-with-parent"]
        argv += ["--hostname", spec.hostname]

        # The rootfs bind must precede every other mount: a later bind at `/`
        # would shadow the earlier ones.
        argv += ["--bind", str(spec.rootfs), "/"]

        # Task rootfs userlands are symlinks into the store, so the store must
        # be visible for /bin/sh to resolve at all. Read-only: the agent
        # cannot mutate the Nix runtime.
        argv += ["--ro-bind", "/nix/store", "/nix/store"]

        for host, dest in spec.ro_binds.items():
            argv += ["--ro-bind", str(host), dest]
        for host, dest in spec.rw_binds.items():
            argv += ["--bind", str(host), dest]

        argv += ["--proc", "/proc", "--dev", "/dev"]
        for path in spec.tmpfs:
            argv += ["--tmpfs", path]

        argv += ["--clearenv"]
        for key, value in spec.env.items():
            argv += ["--setenv", key, value]
        for key in _FORWARDED_ENV:
            if key in os.environ:
                argv += ["--setenv", key, os.environ[key]]

        argv += ["--chdir", spec.chdir, "--"]
        argv += spec.command
        return argv

    def run(self, spec: JailSpec, timeout_sec: float) -> JailResult:
        problems = self.preflight()
        if problems:
            raise SandboxUnavailable("; ".join(problems))

        proc = subprocess.Popen(
            self._argv(spec),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Own process group, so a timeout kills the whole tree rather than
            # leaving orphans behind holding the workspace open.
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()

        return JailResult(
            exit_code=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=timed_out,
        )


def stage_rootfs(built_rootfs: Path, destination: Path) -> Path:
    """Copy a Nix-built task rootfs into a writable per-attempt tree.

    The store copy is read-only and owned by an unmapped uid, so it cannot be
    written through even as jail root. Copying is cheap: the userland is one
    symlink into the store rather than a materialized closure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built_rootfs, destination, symlinks=True)
    for path in [destination, *destination.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    return destination

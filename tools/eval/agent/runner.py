"""Per-task orchestration for gufo-agent-eval.

One attempt is two jails over one workspace:

    bwrap <agent>     Pi is the entrypoint, PID 1 in its own pid namespace.
                      Its shell/read/write/edit tools act directly on the
                      workspace; there is no exec-into-container hop.
    bwrap <verifier>  Runs afterwards over the same workspace, with tests/
                      bound in. The agent jail never had tests/ in its mount
                      namespace.

Trajectory events come back on Pi's stdout through the bwrap pipe, so the
event stream needs no bind mount. Only the session directory is written into
the bound /out.
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from . import pi as pi_mod
from . import task as task_mod
from . import trajectory as trajectory_mod
from . import verify
from .jail import JailSpec, SandboxBackend, stage_rootfs


@dataclass
class AttemptResult:
    task: str
    passed: bool
    reward: float
    duration_ms: int
    agent_timed_out: bool
    agent_exit_code: int
    verifier_crashed: bool
    trajectory: trajectory_mod.Trajectory
    verifier_output: str


def run_attempt(
    backend: SandboxBackend,
    task: task_mod.Task,
    config: pi_mod.PiConfig,
    scratch: Path,
    pi_binary: Path,
    agent_timeout_sec: float | None = None,
) -> AttemptResult:
    """Run one agent attempt at one task, then verify it."""
    scratch.mkdir(parents=True, exist_ok=True)

    workspace = stage_rootfs(task_mod.nix_build(task.env_nix), scratch / "workspace")

    # Pi's configuration, including the credential, lives only here and is
    # discarded with the run.
    pi_config_dir = pi_mod.materialize(config, scratch / "pi-config")

    out_dir = scratch / "out"
    (out_dir / "sessions").mkdir(parents=True, exist_ok=True)

    # Pi is a node bundle: binding the wrapper alone is not enough, so its
    # whole installation prefix goes in read-only. On Nix that prefix is a
    # store path, already covered by the store bind, but resolving it
    # explicitly keeps non-Nix installations working.
    pi_prefix = pi_binary.parent.parent

    # The agent jail always needs the inference endpoint: Pi cannot run
    # without it. The task's own `network` setting is about whether the task's
    # work needs wider internet, which is a separate question -- and one no
    # ported task answers yes to, because their dependencies moved to Nix
    # build time.
    parsed = urllib.parse.urlparse(config.base_url)
    endpoint = (parsed.hostname or "127.0.0.1", parsed.port or (443 if parsed.scheme == "https" else 80))

    spec = JailSpec(
        rootfs=workspace,
        command=pi_mod.command(pi_binary, config, task.instruction),
        chdir=task.workdir,
        env=pi_mod.environment(config),
        ro_binds={pi_prefix: str(pi_prefix)},
        rw_binds={pi_config_dir: pi_mod.JAIL_CONFIG_DIR, out_dir: "/out"},
        network="endpoint-only",
        endpoint=endpoint,
    )

    started = time.monotonic()
    agent = backend.run(
        spec, timeout_sec=agent_timeout_sec or task.agent_timeout_sec
    )
    duration_ms = int((time.monotonic() - started) * 1000)

    # Retained whole; the metrics below are a reduction of it.
    (out_dir / "trajectory.jsonl").write_text(agent.stdout)
    if agent.stderr:
        (out_dir / "agent-stderr.log").write_text(agent.stderr)

    trajectory = trajectory_mod.parse(agent.stdout)

    # The task is verified regardless of how the agent exited. A crashed or
    # timed-out agent may still have left a passing workspace, and #153 scores
    # the workspace, not the agent's exit status.
    verifier_bin = task_mod.nix_build(task.verifier_nix)
    outcome = verify.run(backend, task, workspace, scratch / "verify", verifier_bin)

    return AttemptResult(
        task=task.name,
        passed=outcome.passed,
        reward=outcome.reward,
        duration_ms=duration_ms,
        agent_timed_out=agent.timed_out,
        agent_exit_code=agent.exit_code,
        verifier_crashed=outcome.crashed,
        trajectory=trajectory,
        verifier_output=outcome.output,
    )

"""Verifier execution for gufo-agent-eval.

The verifier runs in its own jail over the workspace the agent produced. The
agent's jail never has `tests/` in its mount namespace: task test directories
routinely carry hidden fixtures that differ from the agent-visible ones (for
`sparql-university`, `university_graph_test.ttl` is a different graph than
the `university_graph.ttl` the agent reads), so binding them into the agent
jail would leak the answers.

The observable contract is inherited unchanged from Terminal-Bench:
CTRF JSON at `/logs/verifier/ctrf.json`, reward at
`/logs/verifier/reward.txt`, and a pass only when reward is exactly 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import task as task_mod
from .jail import JailSpec, SandboxBackend, stage_rootfs


@dataclass
class VerifierOutcome:
    passed: bool
    reward: float
    timed_out: bool
    crashed: bool
    ctrf: dict | None
    output: str


def run(
    backend: SandboxBackend,
    task: task_mod.Task,
    workspace: Path,
    scratch: Path,
    verifier_bin: Path,
) -> VerifierOutcome:
    """Verify a finished workspace.

    Args:
        backend: Sandbox to run in.
        task: Task being verified.
        workspace: The rootfs tree the agent wrote to.
        scratch: Directory for verifier-only state, outside the agent's view.
        verifier_bin: Store path of the task's verifier closure.
    """
    logs = scratch / "logs"
    (logs / "verifier").mkdir(parents=True, exist_ok=True)

    # A fresh rootfs for the verifier, with the agent's workspace bound over
    # the task workdir. The verifier's own dependency closure differs from the
    # agent's userland, so it does not reuse the agent's tree.
    verifier_rootfs = stage_rootfs(
        task_mod.nix_build(task.env_nix), scratch / "verifier-rootfs"
    )

    spec = JailSpec(
        rootfs=verifier_rootfs,
        # The verifier closure exposes exactly one executable.
        command=[str(next((verifier_bin / "bin").iterdir()))],
        chdir=task.workdir,
        env={
            "HOME": "/root",
            "PATH": "/bin:/usr/bin",
            # Keep the verifier's own output deterministic.
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        ro_binds={task.tests_dir: "/tests"},
        rw_binds={
            workspace / task.workdir.lstrip("/"): task.workdir,
            logs: "/logs",
        },
        network="none",
    )

    result = backend.run(spec, timeout_sec=task.verifier_timeout_sec)

    reward_path = logs / "verifier" / "reward.txt"
    ctrf_path = logs / "verifier" / "ctrf.json"

    ctrf = None
    if ctrf_path.is_file():
        try:
            ctrf = json.loads(ctrf_path.read_text())
        except json.JSONDecodeError:
            ctrf = None

    if not reward_path.is_file():
        # No verdict produced: the verifier itself failed, which is distinct
        # from the task failing.
        return VerifierOutcome(
            passed=False,
            reward=0.0,
            timed_out=result.timed_out,
            crashed=True,
            ctrf=ctrf,
            output=(result.stdout + result.stderr).strip(),
        )

    reward = float(reward_path.read_text().strip() or 0)
    return VerifierOutcome(
        passed=reward == 1.0,
        reward=reward,
        timed_out=result.timed_out,
        crashed=False,
        ctrf=ctrf,
        output=(result.stdout + result.stderr).strip(),
    )

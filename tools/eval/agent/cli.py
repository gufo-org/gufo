"""Command-line entry point for gufo-agent-eval.

Run from a checkout during development:

    nix develop
    python -m tools.eval.agent.cli doctor

Or packaged:

    nix run .#eval-agent -- doctor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from . import pi as pi_mod
from . import runner
from . import task as task_mod
from . import verify
from .jail import BwrapBackend, JailSpec, SandboxUnavailable, stage_rootfs

SUITE = "gufo-agent-eval"


def _tasks_root(args: argparse.Namespace) -> Path:
    if args.tasks_root:
        return Path(args.tasks_root)
    return task_mod.TASKS_ROOT


def cmd_doctor(args: argparse.Namespace) -> int:
    backend = BwrapBackend()
    problems = backend.preflight()

    print(f"suite:      {SUITE}")
    print(f"tasks root: {_tasks_root(args)}")
    print(f"tasks:      {len(task_mod.available(_tasks_root(args)))}")

    if problems:
        print("\nsandbox unusable:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("sandbox:    ok")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = _tasks_root(args)
    names = task_mod.available(root)
    if not names:
        print(f"no tasks under {root}", file=sys.stderr)
        return 1
    for name in names:
        task = task_mod.load(name, root)
        print(f"{name:36s} network={task.network:14s} workdir={task.workdir}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a workspace without running an agent.

    This exercises the whole non-inference pipeline: rootfs build, staging,
    the verifier jail, and reward extraction. With --solution it applies the
    task's reference solution first, which must score 1; an untouched
    workspace must score 0. Those two together are the fixture check that
    #153 requires -- correct and intentionally broken fixtures scoring
    independently of any endpoint.
    """
    root = _tasks_root(args)
    task = task_mod.load(args.task, root)
    backend = BwrapBackend()

    with tempfile.TemporaryDirectory(prefix="gufo-agent-eval-") as tmp:
        scratch = Path(tmp)
        rootfs = stage_rootfs(
            task_mod.nix_build(task.env_nix), scratch / "workspace"
        )

        if args.solution:
            solution = task.directory / "solution" / "solve.sh"
            spec = JailSpec(
                rootfs=rootfs,
                command=["/bin/sh", str("/solution/solve.sh")],
                chdir=task.workdir,
                env={"HOME": "/root", "PATH": "/bin:/usr/bin"},
                ro_binds={task.directory / "solution": "/solution"},
                network="none",
            )
            result = backend.run(spec, timeout_sec=task.agent_timeout_sec)
            if result.exit_code != 0:
                print(f"reference solution failed:\n{result.stderr}", file=sys.stderr)
                return 1

        verifier_bin = task_mod.nix_build(task.verifier_nix)
        outcome = verify.run(backend, task, rootfs, scratch / "verify", verifier_bin)

        print(f"task:   {task.name}")
        print(f"reward: {outcome.reward}")
        print(f"passed: {outcome.passed}")
        if outcome.crashed:
            print("verifier crashed (no reward file produced)")
        if args.verbose and outcome.output:
            print(f"\n{outcome.output}")

    return 0 if outcome.passed or not args.expect_pass else 1


def discover_model(base_url: str, api_key: str) -> tuple[str, int]:
    """Ask the endpoint which model it serves.

    #153 requires explicit model selection, so exactly one model must be
    returned: zero or several is ambiguous and fails before any generation.
    """
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise task_mod.TaskError(f"cannot reach {url}: {exc}") from exc

    entries = payload.get("data") or []
    if len(entries) != 1:
        served = ", ".join(sorted(e.get("id", "?") for e in entries)) or "none"
        raise task_mod.TaskError(
            f"{url} serves {len(entries)} models ({served}); exactly one is required. "
            "Pass --model to select explicitly."
        )

    entry = entries[0]
    context = int(entry.get("context_length") or 32768)
    return entry["id"], context


def cmd_run(args: argparse.Namespace) -> int:
    root = _tasks_root(args)
    task = task_mod.load(args.task, root)
    backend = BwrapBackend()

    api_key = os.environ.get(args.api_key_env, "local")

    model_id, context_window = (args.model, args.context_window) if args.model else (
        discover_model(args.base_url, api_key)
    )

    config = pi_mod.PiConfig(
        base_url=args.base_url,
        model_id=model_id,
        api_key=api_key,
        context_window=context_window,
        max_tokens=args.max_tokens,
    )

    pi_binary = pi_mod.resolve_binary()

    with tempfile.TemporaryDirectory(prefix="gufo-agent-eval-") as tmp:
        result = runner.run_attempt(
            backend, task, config, Path(tmp), pi_binary
        )

        print(f"task:        {result.task}")
        print(f"model:       {model_id}")
        print(f"reward:      {result.reward}")
        print(f"passed:      {result.passed}")
        print(f"duration:    {result.duration_ms} ms")
        print(f"turns:       {result.trajectory.turns}")
        print(f"tool calls:  {result.trajectory.tool_calls} "
              f"({result.trajectory.tool_failures} failed)")
        print(f"tokens:      in={result.trajectory.usage.input} "
              f"out={result.trajectory.usage.output} "
              f"cached={result.trajectory.usage.cache_read}")
        if result.agent_timed_out:
            print("agent timed out")
        if result.verifier_crashed:
            print("verifier crashed (no reward file produced)")
        if args.verbose:
            print(f"\n{result.verifier_output}")

    return 0 if result.passed else 1



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-agent",
        description=f"{SUITE}: coding-agent evaluation against OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--tasks-root",
        default=os.environ.get("GUFO_EVAL_TASKS"),
        help="task suite directory (default: the in-tree suite)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check host sandbox prerequisites")
    doctor.set_defaults(func=cmd_doctor)

    listing = sub.add_parser("list", help="list tasks in the suite")
    listing.set_defaults(func=cmd_list)

    verifier = sub.add_parser(
        "verify", help="run a task's verifier without an agent"
    )
    verifier.add_argument("task")
    verifier.add_argument(
        "--solution",
        action="store_true",
        help="apply the reference solution first; it must score 1",
    )
    verifier.add_argument(
        "--expect-pass",
        action="store_true",
        help="exit non-zero unless the task passes",
    )
    verifier.add_argument("-v", "--verbose", action="store_true")
    verifier.set_defaults(func=cmd_verify)

    run = sub.add_parser("run", help="run one task with the Pi agent")
    run.add_argument("task")
    run.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    run.add_argument(
        "--model",
        help="served model ID; discovered from /models when the endpoint serves exactly one",
    )
    run.add_argument(
        "--api-key-env",
        default="GUFO_EVAL_API_KEY",
        help="environment variable holding the credential (default: %(default)s)",
    )
    run.add_argument("--context-window", type=int, default=32768)
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 1
    except task_mod.TaskError as exc:
        print(f"task error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

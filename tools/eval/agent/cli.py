"""Command-line entry point for gufo-agent-eval.

Run from a checkout during development:

    nix develop
    python -m tools.eval.agent.cli doctor

Or packaged:

    nix run .#eval-agent -- doctor
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from . import pi as pi_mod
from . import runner
from . import task as task_mod
from . import verify
from .jail import (
    BwrapBackend,
    JailSpec,
    SandboxUnavailable,
    stage_rootfs,
    sweep_stale_scratch,
)

SUITE = "gufo-agent-eval"


def _tasks_root(args: argparse.Namespace) -> Path:
    if args.tasks_root:
        return Path(args.tasks_root)
    return task_mod.TASKS_ROOT


@contextlib.contextmanager
def _scratch(keep: bool):
    """A working directory for one run.

    Retained with --keep so a failed run can be inspected: the trajectory,
    the agent's workspace, and the verifier output all live here.
    """
    path = Path(tempfile.mkdtemp(prefix="gufo-agent-eval-"))
    try:
        yield path
    finally:
        if keep:
            print(f"\nkept: {path}", file=sys.stderr)
        else:
            for child in path.rglob("*"):
                if not child.is_symlink() and child.is_dir():
                    child.chmod(child.stat().st_mode | 0o700)
            shutil.rmtree(path, ignore_errors=True)



def _task_menu(root: Path) -> str:
    """A listing of selectable tasks, for when none was named."""
    names = task_mod.available(root)
    if not names:
        return f"no tasks found under {root}"

    lines = ["available tasks:", ""]
    for name in names:
        try:
            task = task_mod.load(name, root)
        except task_mod.TaskError:
            lines.append(f"  {name}   (unreadable manifest)")
            continue
        lines.append(f"  {name}")
        lines.append(f"      {task.difficulty} | {task.category}")
        if task.description:
            lines.append(f"      {textwrap.shorten(task.description, 68)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _require_task(args: argparse.Namespace, verb: str) -> int | None:
    """Print a task menu instead of a bare argparse error.

    Returns an exit code when no task was named, otherwise None.
    """
    if args.task:
        return None
    root = _tasks_root(args)
    print(f"eval-agent {verb}: no task named.\n", file=sys.stderr)
    print(_task_menu(root), file=sys.stderr)
    print(f"\nusage: eval-agent {verb} <task> [options]", file=sys.stderr)
    print(f"       eval-agent {verb} --help", file=sys.stderr)
    return 2



def cmd_doctor(args: argparse.Namespace) -> int:
    backend = BwrapBackend()
    problems = backend.preflight()

    print(f"suite:      {SUITE}")
    print(f"tasks root: {_tasks_root(args)}")
    print(f"tasks:      {len(task_mod.available(_tasks_root(args)))}")

    # Pi is the evaluated agent, so which one is used is part of what the
    # numbers mean.
    try:
        pi_binary = pi_mod.resolve_binary()
        source = "GUFO_EVAL_PI" if os.environ.get("GUFO_EVAL_PI") else "PATH (not reproducible)"
        print(f"pi:         {pi_binary} [{source}]")
    except RuntimeError as exc:
        problems.append(str(exc))

    # socat bridges the jail to the inference endpoint; without it no agent
    # can reach a model.
    socat = os.environ.get("GUFO_EVAL_SOCAT") or shutil.which("socat")
    if socat:
        print(f"socat:      {socat}")
    else:
        problems.append(
            "socat not found; it bridges the jail to the inference endpoint"
        )

    if problems:
        print("\nnot ready:")
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
        print(name)
        print(f"    difficulty  {task.difficulty}")
        print(f"    category    {task.category}")
        print(f"    network     {task.network}")
        print(f"    workdir     {task.workdir}")
        print(f"    timeout     {task.agent_timeout_sec:.0f}s agent, "
              f"{task.verifier_timeout_sec:.0f}s verifier")
        if task.description:
            for line in textwrap.wrap(task.description, 68):
                print(f"    {line}")
        print()

    print(f"{len(names)} task(s) in {root}")
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
    if (code := _require_task(args, "verify")) is not None:
        return code

    root = _tasks_root(args)
    task = task_mod.load(args.task, root)
    backend = BwrapBackend()

    with _scratch(args.keep) as scratch:
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

        expected = 1.0 if args.solution else 0.0
        as_expected = outcome.reward == expected

        print(f"task:     {task.name}")
        print(
            "mode:     reference solution applied"
            if args.solution
            else "mode:     untouched workspace, no agent"
        )
        print(f"reward:   {outcome.reward}  (expected {expected})")

        if outcome.crashed:
            print("\nverifier crashed: it produced no reward file at all.")
            print("this is a harness fault, not a task failure.")
        elif as_expected:
            print(
                "\nverdict:  harness OK -- "
                + (
                    "the verifier accepts a correct answer."
                    if args.solution
                    else "the verifier rejects a non-answer."
                )
            )
            if not args.solution:
                print("          now check the other half: --solution must give 1.0")
            else:
                print("          now check the other half: without --solution, 0.0")
        else:
            print("\nverdict:  HARNESS BROKEN")
            if args.solution:
                print("          the reference solution should score 1.0.")
                print("          the verifier rejects a known-correct answer.")
            else:
                print("          an untouched workspace should score 0.0.")
                print("          the verifier passes without any work being done.")
            print("          `run` results would be meaningless until this is fixed.")

        if args.verbose and outcome.output:
            print(f"\n{outcome.output}")

    if args.expect_pass:
        return 0 if outcome.passed else 1
    return 0 if as_expected and not outcome.crashed else 1


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
    if (code := _require_task(args, "run")) is not None:
        return code

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

    with _scratch(args.keep) as scratch:
        result = runner.run_attempt(
            backend, task, config, scratch, pi_binary
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Evaluate a coding agent against an OpenAI-compatible server.\n"
            "\n"
            "The Pi agent solves a task inside a bubblewrap sandbox, then a\n"
            "deterministic verifier scores the workspace it left behind. The\n"
            "agent reaches the inference endpoint and nothing else: no wider\n"
            "network, no host filesystem, no host credentials."
        ),
        epilog=(
            "typical session:\n"
            "\n"
            "  1. check the host is ready, and see which Pi will be used\n"
            "       eval-agent doctor\n"
            "\n"
            "  2. confirm the harness scores correctly, without any model.\n"
            "     the reference solution must score 1 and an untouched\n"
            "     workspace must score 0\n"
            "       eval-agent verify sparql-university --solution\n"
            "       eval-agent verify sparql-university\n"
            "\n"
            "  3. start a server in another terminal. the gufo defaults are\n"
            "     far too small for an agent loop\n"
            "       gufo serve llm --model models/<model>.gguf \\\n"
            "         --context 32768 --max-tokens 8192 --port 8080\n"
            "\n"
            "  4. run a task against it\n"
            "       eval-agent run sparql-university \\\n"
            "         --base-url http://127.0.0.1:8080/v1\n"
            "\n"
            "  5. when something fails, keep the workspace and read the\n"
            "     trajectory it left behind\n"
            "       eval-agent run sparql-university --keep -v\n"
            "\n"
            "a task passes only when its verifier reward is exactly 1.\n"
            "\n"
            "when invoking through nix, put flags after `--`:\n"
            "  nix run .#eval-agent -- verify sparql-university --solution\n"
            "otherwise nix reads the flags as its own.\n"
            "\n"
            "this is a Gufo-owned suite. it is seeded from Terminal-Bench 2.1\n"
            "tasks but runs a different agent in a different sandbox, so its\n"
            "numbers are not an official Terminal-Bench score."
        ),
    )
    parser.add_argument(
        "--tasks-root",
        metavar="DIR",
        default=os.environ.get("GUFO_EVAL_TASKS"),
        help=(
            "task suite directory. defaults to the packaged suite, or "
            "$GUFO_EVAL_TASKS when set"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = sub.add_parser(
        "doctor",
        help="check the host can run sandboxed evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Check that this host can run sandboxed evaluations.\n"
            "\n"
            "Reports which Pi, socat, and bubblewrap will be used, and checks\n"
            "that unprivileged user namespaces are permitted. Contacts no\n"
            "inference server and runs no model."
        ),
        epilog=(
            "the Pi it reports is part of what the numbers mean. a Pi found on\n"
            "PATH is labelled not reproducible: use `nix develop` or set\n"
            "GUFO_EVAL_PI to pin it."
        ),
    )
    doctor.set_defaults(func=cmd_doctor)

    listing = sub.add_parser(
        "list",
        help="list the tasks in the suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "List the tasks in the suite, with difficulty, category, network\n"
            "policy, working directory, and timeouts."
        ),
        epilog="pass --tasks-root to list a suite other than the packaged one.",
    )
    listing.set_defaults(func=cmd_list)

    verifier = sub.add_parser(
        "verify",
        help="score a task without running an agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run a task's verifier without any agent or inference server.\n"
            "\n"
            "This exercises the whole pipeline except the model: the task\n"
            "rootfs is built and staged, the verifier runs in its own jail,\n"
            "and a reward is extracted. Use it to tell a broken harness apart\n"
            "from a model that simply failed the task."
        ),
        epilog=(
            "the two cases that must both hold:\n"
            "  eval-agent verify sparql-university --solution   # reward 1.0\n"
            "  eval-agent verify sparql-university              # reward 0.0"
        ),
    )
    verifier.add_argument(
        "task", nargs="?", help="task to verify; omit to list what is available"
    )
    verifier.add_argument(
        "--solution",
        action="store_true",
        help=(
            "apply the task's reference solution before verifying. "
            "a correct harness scores 1; without this flag the workspace is "
            "untouched and must score 0"
        ),
    )
    verifier.add_argument(
        "--expect-pass",
        action="store_true",
        help="exit non-zero unless the task passes, for use in scripts",
    )
    verifier.add_argument(
        "--keep",
        action="store_true",
        help=(
            "retain the workspace instead of deleting it, and print its path. "
            "the scratch directory holds the staged rootfs and verifier logs"
        ),
    )
    verifier.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the verifier's full pytest output, including failures",
    )
    verifier.set_defaults(func=cmd_verify)

    run = sub.add_parser(
        "run",
        help="run a task with the Pi agent against a live server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run one task with the Pi agent, then verify what it produced.\n"
            "\n"
            "The server must already be running and reachable. The agent gets\n"
            "one attempt, bounded by the task's own timeout."
        ),
        epilog=(
            "examples:\n"
            "  eval-agent run sparql-university\n"
            "  eval-agent run sparql-university --base-url http://192.168.1.8:8080/v1\n"
            "  eval-agent run sparql-university --model qwen3.8-27b --keep -v\n"
            "\n"
            "the model is discovered from /models when the endpoint serves\n"
            "exactly one; pass --model otherwise."
        ),
    )
    run.add_argument(
        "task", nargs="?", help="task to run; omit to list what is available"
    )
    run.add_argument(
        "--base-url",
        metavar="URL",
        default="http://127.0.0.1:8080/v1",
        help=(
            "OpenAI-compatible endpoint, reachable from this host. "
            "the agent reaches it and nothing else (default: %(default)s)"
        ),
    )
    run.add_argument(
        "--model",
        metavar="ID",
        help=(
            "served model ID. discovered from /models when the endpoint "
            "serves exactly one; required when it serves several"
        ),
    )
    run.add_argument(
        "--api-key-env",
        metavar="VAR",
        default="GUFO_EVAL_API_KEY",
        help=(
            "name of the environment variable holding the API key. the key "
            "itself never reaches a result artifact (default: %(default)s)"
        ),
    )
    run.add_argument(
        "--context-window",
        type=int,
        default=32768,
        metavar="N",
        help=(
            "context window advertised to the agent. must not exceed what the "
            "server was started with (default: %(default)s)"
        ),
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        metavar="N",
        help="maximum tokens the agent may request per response (default: %(default)s)",
    )
    run.add_argument(
        "--keep",
        action="store_true",
        help=(
            "retain the workspace and print its path. the scratch directory "
            "holds trajectory.jsonl, the agent's stderr, and the workspace it "
            "left behind. use this to debug a failed run"
        ),
    )
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the verifier's full pytest output, including failures",
    )
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    # A SIGKILL cannot be intercepted, so an earlier killed run may have left
    # its staged workspace behind.
    sweep_stale_scratch()

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

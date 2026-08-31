"""Paired comparison of two gufo-agent-eval result files.

#153 asks for paired task-level differences, which is the point of the whole
harness: the same suite against `gufo serve` and against llama.cpp, with the
per-task deltas visible rather than a single aggregate number.

Comparability is checked rather than assumed. Two runs of different tiers, or
of a suite whose tasks changed between them, are not comparable, and saying so
is more useful than printing a difference that means nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CompareError(RuntimeError):
    pass


@dataclass
class TaskDelta:
    task: str
    left_passed: bool | None
    right_passed: bool | None
    left_ms: int | None
    right_ms: int | None

    @property
    def verdict(self) -> str:
        if self.left_passed is None:
            return "only in B"
        if self.right_passed is None:
            return "only in A"
        if self.left_passed == self.right_passed:
            return "same"
        return "A only" if self.left_passed else "B only"


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"cannot read {path}: {exc}") from exc


def comparability(left: dict, right: dict) -> list[str]:
    """Reasons these two runs cannot be compared, empty if they can."""
    problems = []

    for doc, label in ((left, "A"), (right, "B")):
        if doc.get("schema_version") != "gufo-agent-eval/1":
            problems.append(f"{label}: unrecognised schema {doc.get('schema_version')!r}")

    li, ri = left.get("identity", {}), right.get("identity", {})

    if li.get("tier") != ri.get("tier"):
        problems.append(f"different tiers: {li.get('tier')!r} vs {ri.get('tier')!r}")

    if li.get("benchmark_hash") != ri.get("benchmark_hash"):
        problems.append(
            "different benchmark identity: the tasks, tier, or agent "
            "configuration changed between these runs"
        )

    if li.get("agent_version") != ri.get("agent_version"):
        problems.append(
            f"different agent: {li.get('agent_version')!r} vs {ri.get('agent_version')!r}"
        )

    return problems


def deltas(left: dict, right: dict) -> list[TaskDelta]:
    by_name_left = {t["task"]: t for t in left.get("tasks", [])}
    by_name_right = {t["task"]: t for t in right.get("tasks", [])}

    out = []
    for name in sorted(set(by_name_left) | set(by_name_right)):
        a, b = by_name_left.get(name), by_name_right.get(name)
        out.append(
            TaskDelta(
                task=name,
                left_passed=a["passed"] if a else None,
                right_passed=b["passed"] if b else None,
                left_ms=a["duration_ms"] if a else None,
                right_ms=b["duration_ms"] if b else None,
            )
        )
    return out


def render(left_path: Path, right_path: Path, strict: bool = False) -> tuple[str, int]:
    """Return the report and an exit code."""
    left, right = load(left_path), load(right_path)
    lines: list[str] = []

    li, ri = left.get("identity", {}), right.get("identity", {})
    lines.append(f"A  {left_path}")
    lines.append(f"   {li.get('model', '?')} on {li.get('endpoint_label', '?')}")
    lines.append(f"B  {right_path}")
    lines.append(f"   {ri.get('model', '?')} on {ri.get('endpoint_label', '?')}")
    lines.append("")

    problems = comparability(left, right)
    if problems:
        lines.append("NOT COMPARABLE")
        for problem in problems:
            lines.append(f"  - {problem}")
        lines.append("")
        if strict:
            return "\n".join(lines), 2
        lines.append("differences below are shown anyway, and mean little:")
        lines.append("")

    rows = deltas(left, right)
    width = max((len(d.task) for d in rows), default=10)

    def mark(passed: bool | None) -> str:
        return "-" if passed is None else ("pass" if passed else "fail")

    lines.append(f"{'task'.ljust(width)}   A      B      delta")
    for d in rows:
        if d.left_ms is not None and d.right_ms is not None:
            delta = f"{(d.right_ms - d.left_ms) / 1000:+.0f}s"
        else:
            delta = ""
        flag = "" if d.verdict == "same" else f"   <- {d.verdict}"
        lines.append(
            f"{d.task.ljust(width)}   {mark(d.left_passed):<6} "
            f"{mark(d.right_passed):<6} {delta:>8}{flag}"
        )

    la, ra = left.get("aggregate", {}), right.get("aggregate", {})
    lines.append("")
    lines.append(
        f"A  {la.get('passed_tasks', 0)}/{la.get('total_tasks', 0)} "
        f"pass@1 {la.get('pass_at_1', 0):.2f}  "
        f"{la.get('total_duration_ms', 0) / 3600000:.1f}h"
    )
    lines.append(
        f"B  {ra.get('passed_tasks', 0)}/{ra.get('total_tasks', 0)} "
        f"pass@1 {ra.get('pass_at_1', 0):.2f}  "
        f"{ra.get('total_duration_ms', 0) / 3600000:.1f}h"
    )

    changed = [d for d in rows if d.verdict not in ("same",)]
    if changed:
        lines.append("")
        lines.append(f"{len(changed)} task(s) differ")

    return "\n".join(lines), 0

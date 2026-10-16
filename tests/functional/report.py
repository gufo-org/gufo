#!/usr/bin/env python3
"""Render the Devlab functional PR comment and its advisory commit status."""

import argparse
import json
from pathlib import Path
import re
import sys


MARKER = "<!-- devlab-functional -->"
STATUS_CONTEXT = "devlab/functional"
ERROR_CHARS = 300
LOG_CHARS = 3000
ICONS = {"success": "✅", "failure": "❌", "error": "⚠️"}
MODES = {"off": "AR", "mtp": "MTP"}


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def parse_bench(text):
    """Map each `gufo bench` test name to its first t/s value."""
    rates = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[3] in ("test", "") or set(cells[3]) <= {"-"}:
            continue
        match = re.match(r"[0-9]+(?:\.[0-9]+)?", cells[4])
        if match:
            rates[cells[3]] = float(match.group())
    return rates


def verdict(authorize, smoke, plan, summary, benchmark):
    """Return (state, description); `error` marks cluster or harness problems."""
    if authorize != "success":
        return "error", "Authorization or planning failed"
    if smoke == "skipped" and plan is not None and not plan["groups"] and not benchmark:
        return "success", "No Flash-Next functional impact"
    if smoke == "cancelled":
        return "error", "Cluster job cancelled"
    if summary is None:
        return "error", f"Cluster job did not reach the suites ({smoke})"
    failed = sum(row["status"] != "passed" for row in summary["groups"])
    if failed:
        return "failure", f"{failed} of {len(summary['groups'])} configuration groups failed"
    if smoke != "success":
        return "error", f"Suites passed; a later cluster step failed ({smoke})"
    return "success", f"{len(summary['groups'])} configuration groups passed"


def short(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def suite_list(suites, limit):
    names = ", ".join(f"`{suite}`" for suite in suites[:limit])
    return names + (f" +{len(suites) - limit} more" if len(suites) > limit else "")


def group_name(row):
    model, _, mode = row["profile"].partition("/")
    return f"{MODES.get(mode, mode)} · {row['configuration']}", model


def bench_section(base_text, head_text):
    base, head = parse_bench(base_text), parse_bench(head_text)
    if not head:
        return ["", "#### Benchmark", "", "No benchmark results were produced."]
    lines = ["", "#### Benchmark · AR, merge base → PR", "",
             "| Test | Base t/s | PR t/s | Δ |", "| --- | ---: | ---: | ---: |"]
    for test, rate in head.items():
        if test in base and base[test]:
            delta = f"{(rate / base[test] - 1) * 100:+.1f}%"
            lines.append(f"| `{test}` | {base[test]:.2f} | {rate:.2f} | {delta} |")
        else:
            lines.append(f"| `{test}` | — | {rate:.2f} | — |")
    lines += ["", "<sub>One sweep per revision on the same runner: observations, not qualified "
              "timing regressions.</sub>"]
    return lines


def group_table(rows):
    lines = ["| | Mode · configuration | Passed | Failing suites |", "| :-: | --- | ---: | --- |"]
    for row in rows:
        name, _ = group_name(row)
        failures = row.get("failures") or {}
        total = len(row["suites"])
        if row["status"] == "passed":
            icon, passed, failing = ICONS["success"], f"{total}/{total}", ""
        elif failures:
            icon, passed = ICONS["failure"], f"{total - len(failures)}/{total}"
            failing = suite_list(list(failures), 3)
        else:
            icon, passed, failing = ICONS["error"], "—", "server or runner error"
        lines.append(f"| {icon} | {name} | {passed} | {failing} |")
    return lines


def failure_details(rows):
    failed = [row for row in rows if row["status"] != "passed"]
    if not failed:
        return []
    lines = ["", f"<details><summary>Failure details ({len(failed)} of {len(rows)} groups)"
             "</summary>"]
    for row in failed:
        name, _ = group_name(row)
        lines += ["", f"**{name}**", ""]
        if row.get("error"):
            lines.append(f"- run: {short(row['error'], ERROR_CHARS)}")
        # Identical errors across suites usually share one cause; list them once.
        by_error = {}
        for suite, error in (row.get("failures") or {}).items():
            by_error.setdefault(short(error, ERROR_CHARS), []).append(suite)
        for error, suites in by_error.items():
            label = f"{len(suites)} suites" if len(suites) > 6 else suite_list(suites, 6)
            lines.append(f"- {label}: {error}")
        if row.get("log_tail") and not row.get("failures"):
            lines += ["", "```text", row["log_tail"][-LOG_CHARS:], "```"]
    lines += ["", "</details>"]
    return lines


def render(args):
    plan = load_json(args.plan)
    summary = load_json(args.results / "summary.json")
    state, description = verdict(args.authorize, args.smoke, plan, summary, args.benchmark)
    rows = summary["groups"] if summary is not None else []
    model = group_name(rows[0])[1] if rows else "flash-next-q4"
    lines = [MARKER, f"### {ICONS[state]} Devlab functional checks: {description}", ""]
    if rows:
        lines += group_table(rows) + failure_details(rows)
    if args.benchmark:
        read = lambda name: (args.results / name).read_text() if (args.results / name).exists() else ""
        lines += bench_section(read("bench-base.md"), read("bench-head.md"))
    footer = [f"[Workflow run]({args.run_url})", f"model `{model}`"]
    if plan is not None:
        footer.insert(1, f"PR `{plan['head'][:7]}` vs merge base `{plan['base'][:7]}`")
        footer.append(f"{len(plan['deferred'])} deferred, {len(plan['out_of_scope'])} out of scope")
    lines += ["", "<sub>" + " · ".join(footer) + "<br>"
              f"Advisory: `{STATUS_CONTEXT}` is not a required check and never blocks merging. "
              "Candidate correctness only; other-model, numerical, agent replay and timing "
              "qualification remain separate.</sub>"]
    return state, description, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="Directory with summary.json and optional bench-{base,head}.md")
    parser.add_argument("--authorize", required=True, help="Result of the authorize job")
    parser.add_argument("--smoke", required=True, help="Result of the cluster job")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--comment", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True,
                        help="Write state= and description= lines here")
    args = parser.parse_args()
    state, description, body = render(args)
    args.comment.write_text(body)
    args.status.write_text(f"state={state}\ndescription={description}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

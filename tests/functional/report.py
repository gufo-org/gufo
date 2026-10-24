#!/usr/bin/env python3
"""Render the Devlab functional PR comment and its advisory commit status."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys


MARKER = "<!-- devlab-functional -->"
STATUS_CONTEXT = "devlab/functional"
ERROR_CHARS = 300
LOG_CHARS = 3000
ICONS = {"success": "✅", "failure": "❌", "error": "⚠️"}
MODES = {"off": "AR", "mtp": "MTP", "dflash2": "DFlash2", "dspark": "DSpark", "native": "Native"}


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


def duration(ms):
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def middle(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def phases_section(data):
    """Draw the phase pipeline; colors carry the paired verdicts from phases.py."""
    requests = data["settings"]["requests"]
    head_arm = next((arm for arm in data["arms"] if arm["revision"] == "head"), None)
    titles = {"startup": "Startup"}
    for key, label in requests.items():
        tokens = head_arm["requests"][key].get("prompt_tokens",
            head_arm["requests"][key].get("work", {}).get("prompt_tokens")) if head_arm else None
        titles[key] = label + (f" · {tokens:,.0f} tok" if tokens else "")
    # Phases a request never exercises (no cache restore on a cold prompt) stay out.
    phases = [phase for phase in data["phases"]
              if any(value for value in phase["base"] + phase["head"])]
    lines = ["", "#### Inference phases · merge base → PR", "", "```mermaid", "flowchart LR"]
    groups = [group for group in titles if any(p["group"] == group for p in phases)]
    for index, group in enumerate(groups):
        lines += [f'  subgraph g{index}["{titles[group]}"]', "    direction TB"]
        members = [phase for phase in phases if phase["group"] == group]
        for position, phase in enumerate(members):
            base, head = middle(phase["base"]), middle(phase["head"])
            delta = f"{(head / base - 1) * 100:+.1f}%" if base and head is not None else ""
            label = f"{phase['label']}<br/>{duration(base)} → {duration(head)}"
            if delta:
                label += f"<br/>{delta}"
            node = f"g{index}n{position}"
            lines.append(f'    {node}["{label}"]:::{phase["verdict"]}')
            if position:
                lines.append(f"    g{index}n{position - 1} --> {node}")
        lines.append("  end")
    if len(groups) > 1:
        lines.append("  " + " --> ".join(f"g{index}" for index in range(len(groups))))
    lines += ["  classDef slower fill:#cf222e,stroke:#82071e,color:#fff",
              "  classDef faster fill:#1a7f37,stroke:#044f1e,color:#fff",
              "  classDef noisy fill:#bf8700,stroke:#7d4e00,color:#fff",
              "  classDef same fill:#6e7781,stroke:#424a53,color:#fff",
              "  classDef missing fill:#d0d7de,stroke:#8c959f,color:#24292f",
              "```", ""]
    settings = data["settings"]
    lines.append(f"<sub>Medians of {settings['repeats']} alternating base/PR arms after GPU "
                 f"cool-down. 🟥 slower and 🟩 faster in every repeat (>"
                 f"{settings['slowdown'] * 100:.0f}% and >{settings['noise_ms']} ms); 🟨 mixed "
                 "repeats, inconclusive; ⬜ unchanged. Observations, not qualified timing "
                 f"regressions. Measured in {data['wall_s'] / 60:.1f} min.</sub>")
    if data.get("diverged"):
        lines += ["", "⚠️ Generated text differs or server work differs between arms for "
                  + ", ".join(f"`{key}`" for key in data["diverged"])
                  + "; affected speed comparisons are inconclusive."]
    lines += ["", "<details><summary>Every repeat</summary>", "",
              "| Phase | Base | PR | Verdict |", "| --- | ---: | ---: | --- |"]
    for phase in phases:
        cells = [", ".join(duration(value) for value in phase[side]) for side in ("base", "head")]
        lines.append(f"| {titles[phase['group']]} · {phase['label']} | {cells[0]} | {cells[1]} "
                     f"| {phase['verdict']} |")
    lines += ["", "</details>"]
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


def timing_state(data):
    if not data or not data.get('phases') or len(data.get('arms', [])) < 4:
        return 'missing'
    repeats = data.get('settings', {}).get('repeats', 0)
    if repeats < 2 or any(sum(arm.get('revision') == side for arm in data['arms']) != repeats
                          for side in ('base', 'head')):
        return 'missing'
    for phase in data['phases']:
        if any(len(phase[side]) != repeats for side in ('base', 'head')):
            return 'missing'
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
               for value in phase['base'] + phase['head']):
            return 'missing'
        if phase['verdict'] not in ('same', 'faster', 'slower', 'noisy', 'missing'):
            return 'missing'
    applicable = [phase for phase in data['phases'] if any(
        value is not None for value in phase['base'] + phase['head'])]
    if not applicable:
        return 'missing'
    applicable = [phase for phase in applicable if phase.get('checked', True)]
    if not applicable:
        return 'observations'
    if any(phase['verdict'] == 'slower' for phase in applicable):
        return 'slower'
    if any(phase['verdict'] in ('noisy', 'missing') for phase in applicable):
        return 'inconclusive'
    return 'measured'


def render_models(args, plan, summary):
    from devlab_models import registry
    definitions = registry()['models']
    expected = {model['id'] for model in plan['models']}
    items = {model['id']: model for model in (summary or {}).get('models', [])}
    rows = (summary or {}).get('groups', [])
    expected_groups = {(row['profile'], row['configuration']): row['suites'] for row in plan['groups']}
    observed_groups = {(row['profile'], row['configuration']): row['suites'] for row in rows}
    phases = {key: load_json(args.results / (key + '.phases.json')) for key in expected}
    speeds = {key: timing_state(phases[key]) if phases[key] and phases[key].get('revisions') == {
        'base': plan['base'], 'head': plan['head']} else 'missing' for key in expected}
    state, description = 'success', f'{len(expected)} selected models passed correctness and speed checks'
    if speeds and all(speed == 'observations' for speed in speeds.values()):
        description = f'{len(expected)} selected models passed; timing observations retained'
    if args.authorize != 'success':
        state, description = 'error', 'Authorization or planning failed'
    elif not expected and args.smoke == 'skipped':
        description = 'No functional impact in supported categories'
    elif args.smoke == 'cancelled':
        state, description = 'error', 'Selected model checks cancelled'
    elif not summary or set(items) != expected or len(summary.get('models', [])) != len(expected) or observed_groups != expected_groups or len(rows) != len(expected_groups):
        state, description = 'error', 'Selected model results are missing or incomplete'
    elif any(row['status'] != 'passed' for row in rows) or any(item['correctness'] == 'failed' for item in items.values()):
        state, description = 'failure', 'Selected model correctness checks failed'
    elif any(item['correctness'] != 'passed' or item['timing'] != 'passed' for item in items.values()):
        state, description = 'error', 'Selected model correctness or timing evidence is incomplete'
    elif any(speed == 'slower' for speed in speeds.values()):
        state, description = 'failure', 'Server phases slower in every paired repeat (>5% and >3 ms)'
    elif any(speed in ('missing', 'inconclusive') for speed in speeds.values()):
        state, description = 'error', 'Correctness passed; speed comparison incomplete or inconclusive'
    elif args.smoke != 'success':
        state, description = 'error', f'Model checks completed; cluster job {args.smoke}'
    tier = ' · PR tier' if plan['tier'] == 'pr' else ''
    lines = [MARKER, f'### {ICONS[state]} Devlab functional checks{tier}: {description}', '',
             '| Model | Correctness | Speed · merge base → PR |', '| --- | --- | --- |']
    for selection in plan['models']:
        key = selection['id']
        lines.append(f"| {definitions[key]['title']} | {items.get(key, {}).get('correctness', 'incomplete')} | {speeds[key]} |")
    for selection in plan['models']:
        key = selection['id']
        item = items.get(key, {})
        lines += ['', '#### ' + definitions[key]['title'], '']
        rules = ', '.join(f"`{rule}`" for rule in sorted({reason['rule'] for reason in selection['reasons']}))
        representative = any(reason['representative'] for reason in selection['reasons'])
        lines.append(('Selected category representative' if representative else 'Model-specific coverage') + ': ' + rules + '.')
        model_rows = [row for row in rows if row['profile'].split('/')[0] == key]
        if model_rows:
            lines += [''] + group_table(model_rows) + failure_details(model_rows)
        else:
            lines += ['', '⚠️ Correctness results are missing.']
        for limitation in item.get('known_limitations', []):
            lines += ['', '⚠️ ' + short(limitation, 500)]
        if item.get('error'):
            lines += ['', 'Runner: ' + short(item['error'], ERROR_CHARS)]
        if phases[key] and phases[key].get('phases'):
            checks = phases[key].get('timing_checks')
            if checks is not None:
                lines += ['', 'Speed gates: ' + (', '.join(f'`{key}`' for key in checks) if checks else
                          'observations only for this change; no affected production timing phase') + '.']
            lines += [line.replace('#### ', '##### ', 1) if line.startswith('#### ') else line
                      for line in phases_section(phases[key])]
        else:
            lines += ['', '⚠️ Speed comparison is incomplete.']
        if item.get('timing_error'):
            lines += ['', 'Timing runner: ' + short(item['timing_error'], ERROR_CHARS)]
    if args.benchmark:
        read = lambda name: (args.results / name).read_text() if (args.results / name).exists() else ''
        lines += bench_section(read('bench-base.md'), read('bench-head.md'))
    lines += ['', f"<sub>[Workflow run and evidence]({args.run_url}) · "
              f"PR `{plan['head'][:7]}` vs merge base `{plan['base'][:7]}` · "
              f"{len(plan['deferred'])} deferred, {len(plan['release_only'])} for the release run. "
              f"`{STATUS_CONTEXT}` is advisory on PRs; only the automatic release merge waits for it. "
              'Coverage is limited to the selected representatives; image/video generation and agent replay remain separate.</sub>']
    return state, description, '\n'.join(lines) + '\n'


def render(args):
    plan = load_json(args.plan)
    summary = load_json(args.results / "summary.json")
    if plan is not None and plan.get('schema') == 2:
        return render_models(args, plan, summary)
    state, description = verdict(args.authorize, args.smoke, plan, summary, args.benchmark)
    rows = summary["groups"] if summary is not None else []
    model = group_name(rows[0])[1] if rows else "flash-next-q4"
    tier = plan.get("tier", "full") if plan is not None else "full"
    title = "Devlab functional checks" + (" · PR tier" if tier == "pr" else "")
    lines = [MARKER, f"### {ICONS[state]} {title}: {description}", ""]
    if rows:
        lines += group_table(rows) + failure_details(rows)
    if args.benchmark:
        read = lambda name: (args.results / name).read_text() if (args.results / name).exists() else ""
        lines += bench_section(read("bench-base.md"), read("bench-head.md"))
    phases = load_json(args.results / "phases.json")
    if phases is not None:
        lines += phases_section(phases)
    footer = [f"[Workflow run]({args.run_url})", f"model `{model}`"]
    if plan is not None:
        footer.insert(1, f"PR `{plan['head'][:7]}` vs merge base `{plan['base'][:7]}`")
        release_only = len(plan.get("release_only", []))
        footer.append(f"{len(plan['deferred'])} deferred, "
                      + (f"{release_only} for the release run, " if release_only else "")
                      + f"{len(plan['out_of_scope'])} out of scope")
    lines += ["", "<sub>" + " · ".join(footer) + "<br>"
              f"`{STATUS_CONTEXT}` is advisory on PRs; only the automatic release merge waits for it. "
              "Candidate correctness only; other-model, numerical, agent replay and timing "
              "qualification remain separate.</sub>"]
    return state, description, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="Directory with summary.json, optional bench-{base,head}.md and phases.json")
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

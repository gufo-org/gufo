#!/usr/bin/env python3
"""Real Claude Code agent regression for #417; executes generated code in disposable fixtures.

Requires the Claude Code CLI (`claude`) and an already-running local Gufo server.
Claude Code talks to `/v1/messages` through the recording proxy from `pi_agent.py`,
so the exact HTTP bodies/SSE, the CLI's stream-json transcript, task files and
individual request/task timings are retained.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from gufo.control_tokens import kImEnd, kImStart  # noqa: E402

from pi_agent import Recorder, save


PROMPTS = {
    "simple": "Answer with one word: what is the capital of France?",
    "tools": "Use the Write tool to create hello.txt containing exactly the word bonjour, show it with cat using the Bash tool, then read it back with the Read tool and confirm its content in one sentence.",
    "edit": "Read src/calc.py, fix the obvious bug with the Edit tool, then print the fixed file with cat using the Bash tool.",
    "literal-protocol": (
        f'Use the Write tool to create chat_template_fixture.py with EOS = "{kImEnd}", '
        f'BOS = "{kImStart}", and render(role, content) returning '
        'BOS + role + "\\n" + content + EOS + "\\n". '
        'These are literal Python string values, not message delimiters. '
        'Read it back with the Read tool, then use the Bash tool to run python3 assertions that '
        f'render("user", "hello") equals "{kImStart}user\\nhello{kImEnd}\\n". '
        'Report success only after the assertions pass.'
    ),
}
TOOLS = ("Read", "Write", "Edit", "Bash")
FRAMING = ("<tool_call", "</tool_call>", "</function>", "</parameter>", "</think>")


def transcript(path):
    """Claude Code's stream-json lines, skipping a partially written final line."""
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def blocks(events, role, kind):
    return [block for event in events if event.get("type") == role
            for block in event.get("message", {}).get("content", [])
            if isinstance(block, dict) and block.get("type") == kind]


def repeated_actions(events):
    """Count identical tool calls with identical results, allowing retries after progress."""
    calls = {call.get("id"): call for call in blocks(events, "assistant", "tool_use")}
    counts = {}
    for result in blocks(events, "user", "tool_result"):
        call = calls.get(result.get("tool_use_id"))
        if call:
            key = (call["name"], json.dumps(call.get("input"), sort_keys=True),
                   json.dumps(result.get("content"), sort_keys=True))
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def completed_requests(server_log, wanted, timeout=10):
    """The server logs completion after the response, so allow it a moment."""
    deadline = time.monotonic() + timeout
    while True:
        completed = {}
        for line in server_log.read_text().splitlines():
            if "event=completed " in line:
                fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
                completed[fields.get("request")] = fields
        if wanted <= completed.keys() or time.monotonic() > deadline:
            return completed
        time.sleep(0.2)


def validate_requests(rows, server_log):
    assert rows, "Claude Code did not send a request"
    completed = completed_requests(server_log, {row.get("request_id") for row in rows})
    conversation = 0
    for row in rows:
        if urllib.parse.urlsplit(row["path"]).path != "/v1/messages":
            continue  # e.g. token counting; the agent loop is /v1/messages
        assert row.get("status") == 200 and not row.get("error"), row
        # The proxy reads usage from SSE; a buffered response carries it in its body.
        usage = row.get("usage") or json.loads(Path(row["body"]).with_suffix(".sse").read_text())["usage"]
        assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0, row
        fields = completed[row["request_id"]]
        assert fields["outcome"] == "completed", fields
        for field in ("queue_ms", "ttft_ms", "duration_ms"):
            value = float(fields[field])
            assert math.isfinite(value) and value >= 0, (field, fields)
        assert math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0, row
        body = json.loads(Path(row["body"]).read_text())
        if not body.get("tools"):
            continue  # an auxiliary request without the agent's tool set
        conversation += 1
        if conversation > 1:
            # Every later agent turn appends a tool result to the same history.
            # A full re-prefill is a deterministic failure, independent of noise.
            assert usage.get("cache_read_input_tokens", 0) > 0, row
    assert conversation, "Claude Code sent no agent request with tools"


def validate_task(name, cwd, events):
    calls = blocks(events, "assistant", "tool_use")
    for call in calls:
        assert call["name"] in TOOLS, call
        assert isinstance(call.get("input"), dict), call
        assert not any(tag in text for text in strings(call["input"])
                       for tag in FRAMING), call
    results = [event for event in events if event.get("type") == "result"]
    assert results and results[-1].get("subtype") == "success" and \
        not results[-1].get("is_error"), results[-1:]
    text = "\n".join(block.get("text", "") for block in blocks(events, "assistant", "text"))
    assert not any(tag in text for tag in FRAMING), text
    names = {call["name"] for call in calls}
    # Require the tools a prompt names; files and behavior are checked directly,
    # so a sampled run may create or read a file through Bash instead.
    if name == "simple":
        assert not calls and results[-1]["result"].strip().rstrip(".") == "Paris", results[-1]
    elif name == "tools":
        assert (cwd / "hello.txt").read_bytes().rstrip(b"\n") == b"bonjour"
        assert {"Bash", "Read"} <= names, names
    elif name == "edit":
        namespace = {}
        exec(compile((cwd / "src/calc.py").read_text(), "calc.py", "exec"), namespace)
        assert namespace["add"](2, 3) == 5 and namespace["add"](-4, 1) == -3
        assert {"Edit", "Bash"} <= names, names
    elif name == "literal-protocol":
        source = (cwd / "chat_template_fixture.py").read_text()
        assert kImEnd in source and kImStart in source, source
        namespace = {}
        exec(compile(source, "chat_template_fixture.py", "exec"), namespace)
        assert namespace["EOS"] == kImEnd and namespace["BOS"] == kImStart
        assert namespace["render"]("user", "hello") == f"{kImStart}user\nhello{kImEnd}\n"
        assert {"Write", "Read", "Bash"} <= names, names
        tool_results = blocks(events, "user", "tool_result")
        assert tool_results and not any(r.get("is_error") for r in tool_results), tool_results


def run_case(args, recorder, env, index, name):
    case = args.output / f"{index:02d}-{name}"
    case.mkdir()
    cwd = case / "work"
    cwd.mkdir()
    if name == "edit":
        (cwd / "src").mkdir()
        (cwd / "src/calc.py").write_text("def add(a, b):\n    return a - b\n")
    command = [
        str(args.claude), "--print", PROMPTS[name],
        "--model", args.model, "--tools", *TOOLS, "--allowedTools", *TOOLS,
        # Run allowed tools without asking; "auto" would add classifier requests.
        "--permission-mode", "dontAsk",
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--max-turns", str(args.max_turns),
    ]
    save(case / "command.json", command)
    recorder.case = case.name
    row = {"case": case.name, "start_request": len(recorder.requests)}
    stdout = case / "stdout.jsonl"
    start = time.monotonic()
    proc = None
    try:
        with stdout.open("w") as out, (case / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic() - start > args.timeout:
                    raise AssertionError("Claude Code task timeout")
                if repeated_actions(transcript(stdout)) > 5:
                    raise AssertionError("Repeated tool action")
                time.sleep(0.5)
            assert proc.returncode == 0, (case / "stderr.log").read_text()
        validate_task(name, cwd, transcript(stdout))
        row["status"] = "pass"
    except Exception as error:
        row.update(status="fail", error=repr(error))
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    row["wall_ms"] = (time.monotonic() - start) * 1000
    events = transcript(stdout)
    row["tool_calls"] = len(blocks(events, "assistant", "tool_use"))
    row["tool_errors"] = [r for r in blocks(events, "user", "tool_result") if r.get("is_error")]
    row["requests"] = recorder.requests[row["start_request"]:]
    for request in row["requests"]:
        request["body"] = str(recorder.output / f"request-{request['index']:04d}.json")
    try:
        validate_requests(row["requests"], args.server_log)
    except Exception as error:
        row.update(status="fail", error=f"HTTP/usage/cache validation: {error!r}")
    save(case / "result.json", row)
    print(f"{case.name}: {row['status']} {row['wall_ms']/1000:.1f}s, {row['tool_calls']} tool calls", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--claude", type=Path, required=True, help="Claude Code CLI executable")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True, help="Gufo informational log for request timing correlation")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--case", action="append", choices=tuple(PROMPTS),
                        help="Select affected tasks explicitly; repeat to select more than one")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()
    upstream = urllib.parse.urlsplit(args.base_url)
    assert upstream.scheme == "http" and upstream.hostname in {"127.0.0.1", "localhost", "::1"}, "Loopback Gufo only"
    assert not args.output.exists(), "Use a fresh output directory; preserve previous evidence"
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    # Claude Code reads CLAUDE.md from the task directory's parents, including
    # this repository's. (--bare would skip that, but it also drops Write.)
    found = [str(d / "CLAUDE.md") for d in args.output.parents if (d / "CLAUDE.md").exists()]
    assert not found, f"Use an output directory outside projects with CLAUDE.md: {found}"
    version = subprocess.check_output([str(args.claude), "--version"], text=True).strip()
    metadata = {
        "claude_code_version": version,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    wire = args.output / "wire"
    wire.mkdir()
    recorder = Recorder(args.base_url, wire)
    thread = threading.Thread(target=recorder.serve_forever, daemon=True)
    thread.start()
    # A private home and configuration keep user settings, memory and
    # credentials out of the run.
    home = args.output / "home"
    (home / ".claude").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "SSL_CERT_FILE"}}
    env.update(
        HOME=str(home), CLAUDE_CONFIG_DIR=str(home / ".claude"),
        ANTHROPIC_BASE_URL=f"http://127.0.0.1:{recorder.server_port}",
        ANTHROPIC_API_KEY="local-functional-test",
        # Every model alias the CLI may pick for auxiliary requests.
        ANTHROPIC_MODEL=args.model, ANTHROPIC_SMALL_FAST_MODEL=args.model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=args.model, ANTHROPIC_DEFAULT_SONNET_MODEL=args.model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=args.model, CLAUDE_CODE_SUBAGENT_MODEL=args.model,
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1",
        DISABLE_AUTOUPDATER="1", DISABLE_ERROR_REPORTING="1",
    )
    rows = []
    cases = (args.case or list(PROMPTS)) * args.passes
    try:
        for index, name in enumerate(cases):
            rows.append(run_case(args, recorder, env, index, name))
            save(args.output / "report.json", {**metadata, "cases": rows, "requests": recorder.requests})
            if rows[-1]["status"] != "pass":
                break
    finally:
        recorder.shutdown()
        recorder.server_close()
    return 0 if len(rows) == len(cases) and all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

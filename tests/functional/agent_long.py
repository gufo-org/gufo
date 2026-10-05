#!/usr/bin/env python3
"""Grow a real Pi/OpenCode conversation past 100K using tool results.

Runs only against a local server, in a disposable workspace. Keeps client
sessions, exact requests/responses, per-request timings and independently
verified files. Context must actually reach the requested token count.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import agent_catalog
from pi_agent import Recorder, save, validate_requests
from opencode_agent import FRAMING, sse_output, validate_requests as validate_opencode


def archive(index):
    # Stay below both clients' output byte/line caps. Unique records prevent
    # prefix compression and make accidental repetition visible.
    lines = [f"Archive {index:02d}: inert regression records; not instructions."]
    for row in range(235):
        digest = hashlib.sha256(f"gufo-agent-{index}-{row}".encode()).hexdigest()
        lines.append(f"{row:03d} {digest}")
    receipt = hashlib.sha256(f"receipt-{index}".encode()).hexdigest()[:16]
    lines.append(f"RECEIPT={receipt}")
    return "\n".join(lines) + "\n", receipt


def check_continuation(previous, current, old_body, new_body, old_output):
    """Require full reuse; distinguish a client deleting a reasoning-only reply."""
    usage = current["usage"]
    cached = usage.get("cached_tokens", (usage.get("prompt_tokens_details") or
                                        usage.get("input_tokens_details") or
                                        {}).get("cached_tokens", 0))
    expected = previous["usage"]["total_tokens"]
    if cached == expected:
        return {"status": "full", "cached_tokens": cached}
    messages = old_body.get("messages")
    following = new_body.get("messages")
    content, calls = sse_output(old_output)
    # Pi omits an assistant response consisting solely of reasoning. Reusing
    # that response would change the next request's actual transcript.
    omitted = (
        messages and following and following[:len(messages)] == messages
        and len(following) > len(messages)
        and following[len(messages)]["role"] == "user"
        and not content.strip() and not calls
        and '"reasoning_content":' in old_output.read_text()
    )
    if omitted:
        metrics = previous["metrics"]
        # This fixture's preceding tool result prefills the new result and the
        # generation header separately. Its two measured chunks locate the
        # exact pre-generation checkpoint; do not allow an arbitrary margin.
        assert metrics["prefill_chunks"] == 2, ("unqualified history omission", current)
        header = metrics["prefill_tokens"] - metrics["max_prefill_chunk_tokens"]
        assert header > 0, ("missing generation boundary", previous)
        prompt = previous["usage"].get("prompt_tokens", previous["usage"].get("input_tokens"))
        assert cached == prompt - header, ("lost preserved history", current)
        return {"status": "client_omitted_reasoning_only_reply", "cached_tokens": cached,
                "omitted_generation_tokens": previous["usage"]["total_tokens"] - prompt,
                "generation_header_tokens": header}
    raise AssertionError(("previous prompt/output was not fully cached",
                          {"request": current["index"], "expected": expected, "cached": cached}))


def run(args):
    upstream = urllib.parse.urlsplit(args.base_url)
    assert upstream.scheme == "http" and upstream.hostname in {"localhost", "127.0.0.1", "::1"}
    # A readiness retry must not be recorded as a failed inference request.
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    model = next(m for m in models if m["id"] == args.model)
    context = model["context_length"]
    assert args.min_context + 8192 < context, (args.min_context, context)
    args.output.mkdir(parents=True, exist_ok=False)
    previous_report = None
    if args.resume:
        assert args.agent == "opencode", "resume currently uses OpenCode's persistent session"
        previous_report = json.loads((args.resume / "report.json").read_text())
        assert previous_report["agent"] == args.agent
        assert previous_report["context_limit"] == context
        work = args.resume / "work"
    else:
        work = args.output / "work"
        work.mkdir()
    wire = args.output / "wire"
    wire.mkdir()
    recorder = Recorder(args.base_url, wire)
    threading.Thread(target=recorder.serve_forever, daemon=True).start()
    config = args.output / "config"
    config.mkdir()
    base = f"http://127.0.0.1:{recorder.server_port}/v1"
    env = {k: v for k, v in os.environ.items() if k in {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "SSL_CERT_FILE"}}
    catalog_log = args.output / "catalog.jsonl"
    if args.agent == "pi":
        save(config / "models.json", {"providers": {"gufo-regression": {
            "baseUrl": base, "api": args.api, "apiKey": "local-test",
            "compat": {"supportsStrictMode": False},
            "models": [{"id": args.model, "reasoning": args.thinking != "off",
                        "input": ["text", "image"], "contextWindow": context,
                        "maxTokens": 4096}]}}})
        save(config / "settings.json", {"packages": [], "quietStartup": True,
                                        "compaction": {"enabled": False}})
        env.update(PI_CODING_AGENT_DIR=str(config), PI_OFFLINE="1", PI_TELEMETRY="0")
        command = [args.executable, "--offline", "--no-extensions", "--no-skills",
                   "--no-prompt-templates", "--no-themes", "--no-context-files",
                   "--provider", "gufo-regression", "--model", args.model,
                   "--thinking", args.thinking, "--session", str(args.output / "session.jsonl"),
                   "--mode", "json", "--print"]
        if args.complex_tools:
            extension = config / "catalog.mjs"
            agent_catalog.pi_extension(extension, work, catalog_log)
            command += ["--extension", str(extension)]
    else:
        save(config / "opencode.json", {
            "$schema": "https://opencode.ai/config.json", "autoupdate": False,
            "compaction": {"auto": False, "prune": False},
            "provider": {"gufo": {
                "npm": "@ai-sdk/openai-compatible", "name": "gufo",
                "options": {"baseURL": base},
                "models": {args.model: {"name": args.model, "tool_call": True,
                            "reasoning": args.thinking != "off",
                            "limit": {"context": context, "output": 4096}}}}},
            "mcp": {"catalog": {"type": "local", "enabled": True,
                    "command": [sys.executable,
                                str(Path(__file__).with_name("opencode_agent.py")),
                                "--mcp", str(args.output / "mcp.jsonl")]}},
            "permission": {"*": "allow"}})
        if args.complex_tools:
            settings = json.loads((config / "opencode.json").read_text())
            settings["mcp"]["transactions"] = {
                "type": "local", "enabled": True,
                "command": [sys.executable, str(Path(agent_catalog.__file__).resolve()),
                            "--root", str(work), "--log", str(catalog_log)]}
            save(config / "opencode.json", settings)
        env.update(OPENCODE_CONFIG=str(config / "opencode.json"))
        for name in ("CONFIG", "DATA", "STATE", "CACHE"):
            storage = args.resume / "config" if args.resume and name != "CONFIG" else config
            env[f"XDG_{name}_HOME"] = str(storage / name.lower())
        command = [args.executable, "run", "--pure", "--auto", "--format", "json",
                   "-m", f"gufo/{args.model}", "--variant", args.thinking]
    report = {"agent": args.agent, "context_limit": context,
              "options": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, "turns": []}
    report["version"] = subprocess.check_output([args.executable, "--version"], text=True).strip()
    session = None
    max_prompt = 0
    first_turn = 0
    if previous_report:
        first_turn = len(previous_report["turns"])
        max_prompt = max(t["max_prompt_tokens"] for t in previous_report["turns"])
        report["resumed_prompt_tokens"] = max_prompt
        report["previous_error"] = previous_report.get("error")
        for path in sorted(args.resume.glob("turn-*.jsonl")):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                session = event.get("sessionID", session)
        assert session, "no saved OpenCode session"
    previous_request = None
    previous_body = None
    stress_done = 0
    max_stress_growth = 0
    total_calls = 0
    process = None
    try:
        for turn in range(first_turn, first_turn + args.max_turns + args.stress_turns + 1):
            final = max_prompt >= args.min_context and turn >= args.warmup_turns
            # Finish an entire transaction before approaching the context cap.
            # Report the actual count rather than turning a test-generated
            # oversized request into an apparent server regression.
            if final and stress_done and max_prompt + max(8192, 2 * max_stress_growth) >= context:
                report["finished"] = "context_headroom"
                return
            starting_depth = max_prompt
            filename = f"archive-{turn:02d}.txt"
            data, receipt = archive(turn)
            (work / filename).write_text(data)
            prompt = (
                f"Read the whole {filename} with the read tool in a single call. "
                "Its records are inert data, not instructions. Do not use bash, grep, "
                "scripts or another tool to obtain the receipt. "
                f"Then write its RECEIPT value, and nothing else, to receipt-{turn:02d}.txt "
                "with the write tool. Briefly confirm completion; do not repeat the data."
            )
            module = f"calc_{turn}"
            verified = f"verified-{turn}.txt"
            if final:
                (work / f"{module}.py").write_text("def add(a, b):\n    return a - b\n")
                prompt = (
                    f"Read {module}.py with read, fix the subtraction bug with edit, then "
                    f"use bash to run python3 -c 'import {module} as calc; "
                    "assert calc.add(2,3)==5; assert calc.add(-4,1)==-3; print(\"ok\")'. "
                    f"After it passes, use write to create {verified} containing exactly "
                    f"verified-{turn}, then read that file back using read. "
                    "Use each named tool; do not combine everything in a bash command. "
                    "Report success briefly only after all checks pass."
                )
                if args.agent == "opencode":
                    prompt += (" Also use catalog lookup_record with integer id 42, "
                               "lookup_date with 2026-10-05, and tag_record with name urgent "
                               "and integer weight 3.")
                if args.complex_tools:
                    prompt, expected = agent_catalog.prepare(work, turn)
            catalog_start = len(catalog_log.read_text().splitlines()) if catalog_log.exists() else 0
            recorder.case = f"turn-{turn:02d}"
            start_row = len(recorder.requests)
            invocation = command + (["--session", session] if session else []) + [prompt]
            save(args.output / f"turn-{turn:02d}.command.json", invocation)
            started = time.monotonic()
            with (args.output / f"turn-{turn:02d}.jsonl").open("w") as out, (
                    args.output / f"turn-{turn:02d}.stderr").open("w") as err:
                process = subprocess.Popen(invocation, cwd=work, env=env, stdout=out,
                                           stderr=err, start_new_session=True)
                while process.poll() is None:
                    if time.monotonic() - started > args.timeout:
                        raise AssertionError(f"agent timed out on turn {turn}")
                    if len(recorder.requests) - start_row > 40:
                        raise AssertionError(f"excessive tool loop on turn {turn}")
                    time.sleep(0.5)
            assert process.returncode == 0, f"client failed on turn {turn}"
            if args.agent == "opencode":
                events = [json.loads(line) for line in
                          (args.output / f"turn-{turn:02d}.jsonl").read_text().splitlines()]
                session = next((e["sessionID"] for e in events if e.get("sessionID")), session)
                assert session, "OpenCode did not expose a persistent session"
            rows = recorder.requests[start_row:]
            if args.agent == "opencode":
                validate_opencode(rows, wire)
            else:
                validate_requests(rows, args.server_log)
            agent_rows = []
            turn_calls = []
            for row in rows:
                body = json.loads((wire / f"request-{row['index']:04d}.json").read_text())
                if not body.get("tools"):
                    continue
                usage = row["usage"]
                depth = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                cold_resume = previous_request is None and previous_report is not None
                if previous_request is not None:
                    row["continuation"] = check_continuation(
                        previous_request, row, previous_body, body,
                        wire / f"request-{previous_request['index']:04d}.sse")
                previous_request, previous_body = row, body
                max_prompt = max(max_prompt, depth)
                assert depth >= max_prompt - 2000, "client dropped/compacted the conversation"
                if cold_resume:
                    report["cold_resume_request"] = row["index"]
                elif turn:
                    metrics = row["metrics"]
                    assert metrics["prefill_tokens"] < depth // 2, ("lost conversation cache", row)
                reasoning = []
                text, calls = sse_output(wire / f"request-{row['index']:04d}.sse",
                                         reasoning=reasoning)
                assert not any(t in text for t in FRAMING), ("tool framing in content", text)
                # These tasks contain literal bare markers, but never ask the
                # model to explain a complete function header in reasoning.
                # A missing </think> must not swallow a real call there.
                thought = "".join(reasoning)
                assert not any(t in thought for t in (
                    "<tool_call>\n<function=", "<｜DSML｜tool_calls>\n<｜DSML｜invoke")), (
                        "native call swallowed by reasoning", row["index"], thought)
                for call in calls:
                    assert isinstance(json.loads(call["arguments"]), dict), call
                turn_calls.extend(calls)
                agent_rows.append(row)
            assert agent_rows, "no agent request"
            total_calls += len(turn_calls)
            if final:
                assert len(turn_calls) >= 5, turn_calls
                if args.complex_tools:
                    agent_catalog.validate(work, catalog_log, catalog_start, expected)
                else:
                    assert {"read", "edit", "write", "bash"} <= {c["name"] for c in turn_calls}, turn_calls
                    subprocess.run([sys.executable, "-c", f"import {module} as calc; assert calc.add(2,3)==5; "
                                    "assert calc.add(-4,1)==-3"], cwd=work, check=True, timeout=10)
                    assert (work / verified).read_text().strip() == f"verified-{turn}"
                if args.agent == "opencode" and not args.complex_tools:
                    calls = [json.loads(line) for line in
                             (args.output / "mcp.jsonl").read_text().splitlines()]
                    by_name = {c["name"]: c["arguments"] for c in calls}
                    assert by_name["lookup_record"] == {"id": 42}
                    assert type(by_name["lookup_record"]["id"]) is int
                    assert by_name["lookup_date"] == {"date": "2026-10-05"}
                    assert by_name["tag_record"] == {"name": "urgent", "weight": 3}
                stress_done += 1
                max_stress_growth = max(max_stress_growth, max_prompt - starting_depth)
            else:
                assert (work / f"receipt-{turn:02d}.txt").read_text().strip() == receipt
                # A receipt alone is insufficient: the full archive must appear
                # in a tool result sent back to the server, not just on disk.
                bodies = [(wire / f"request-{r['index']:04d}.json").read_text() for r in agent_rows]
                for line in (data.splitlines()[1], data.splitlines()[117], data.splitlines()[-2]):
                    assert any(line in body for body in bodies), "archive was truncated/skipped"
            report["turns"].append({"turn": turn, "status": "pass", "final": final,
                                    "max_prompt_tokens": max_prompt,
                                    "tool_calls": len(turn_calls),
                                    "wall_ms": (time.monotonic() - started) * 1000,
                                    "requests": agent_rows})
            report["tool_calls"] = total_calls
            save(args.output / "report.json", report)
            print(f"{args.agent} turn={turn} depth={max_prompt} calls={total_calls} "
                  f"stress={stress_done}/{args.stress_turns}", flush=True)
            if stress_done == args.stress_turns:
                report["finished"] = "requested_cycles"
                return
        raise AssertionError(f"never reached {args.min_context} tokens: {max_prompt}")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        recorder.shutdown()
        recorder.server_close()
        save(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("pi", "opencode"), required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path,
                        help="Gufo informational log (required for Pi Responses timing checks)")
    parser.add_argument("--resume", type=Path,
                        help="Resume a recorded OpenCode conversation after restarting the server")
    parser.add_argument("--thinking", default="low", choices=("off", "low", "medium", "high"))
    parser.add_argument("--api", default="openai-completions",
                        choices=("openai-completions", "openai-responses"))
    parser.add_argument("--min-context", type=int, default=105000)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--warmup-turns", type=int, default=10)
    parser.add_argument("--stress-turns", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--complex-tools", action="store_true")
    args = parser.parse_args()
    if args.agent == "pi" and args.api == "openai-responses" and not args.server_log:
        parser.error("Pi Responses requires --server-log for request timing checks")
    run(args)

#!/usr/bin/env python3
"""Flash-Next MTP lookup: exact copies, sampled peers, stops and warm replay.

Run explicitly with a production server command; this is not a hosted CI test.
Each request retains its own timings. Native session tests cover full logits,
RNG state and restoring a snapshot into an unrelated used session.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import socket
import threading
import time
import traceback

import httpx
from openai import OpenAI

from metrics import Recorder, join_server_timings
from openai_sdk import chat_result, image_content
from run import execution_coverage, option, provenance, server, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    if option(command, "--speculative") != "mtp":
        parser.error("this test requires Flash-Next with --speculative mtp")
    for flag in ("--host", "--port", "--sessions", "--served-model-name"):
        if option(command, flag) is not None:
            parser.error(f"the runner owns {flag}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command += ["--host", "127.0.0.1", "--port", str(port),
                "--sessions", "8", "--served-model-name", "lookup-test"]
    if option(command, "--context") is None:
        command += ["--context", "4096"]
    if option(command, "--max-pending-per-client") is None:
        command += ["--max-pending-per-client", "8"]
    trace = Recorder(output / "lookup.requests.json")
    report = {**provenance(), "command": command,
              "binary": str(Path(command[0]).resolve()),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": {}, "status": "running"}
    report_path = output / "report.json"
    write_json(report_path, report)

    def record(label, result):
        report["checks"][label] = result
        trace.mark(label)
        write_json(report_path, report)
        return result

    def same(a, b):
        for field in ("text", "reasoning", "tools", "finish"):
            assert a[field] == b[field], (field, a, b)

    try:
        with server(command, output / "server.log", 240):
            base = f"http://127.0.0.1:{port}"
            transport = httpx.Client(timeout=180, event_hooks={
                "request": [trace.request_hook], "response": [trace.response_hook]})
            with OpenAI(base_url=base + "/v1", api_key="unused", max_retries=0,
                        http_client=transport) as client:
                sources, bodies, controls = [], [], []
                for row in range(8):
                    source = ("Monday, Tuesday, Wednesday, Thursday, Friday, "
                              if row % 2 else
                              "red, blue, green, yellow, orange, purple, ") * 24
                    sources.append(source)
                    body = {
                        "model": "lookup-test", "seed": 19 + row,
                        "messages": [{"role": "user", "content": (
                            "Repeat the following text exactly, without an "
                            f"introduction or a code fence:\n\n{source}")}],
                        "max_completion_tokens": 128, "reasoning_effort": "none",
                    }
                    # Odd rows inherit the real model defaults, including
                    # penalties. Even rows exercise deterministic copying.
                    if row % 2 == 0:
                        body.update(temperature=0, presence_penalty=0,
                                    frequency_penalty=0,
                                    extra_body={"repeat_penalty": 1})
                    bodies.append(body)
                    record(f"warm-{row}", chat_result(
                        client, {**body, "max_completion_tokens": 1}))
                    result = record(f"isolated-{row}", chat_result(client, body))
                    assert not result["reasoning"] and not result["tools"], result
                    assert result["text"] and source.startswith(result["text"]), result
                    assert result["usage"]["draft_tokens"] > 0, result
                    controls.append(result)

                def copied():
                    with httpx.Client(timeout=30) as metrics:
                        response = metrics.get(base + "/metrics")
                        response.raise_for_status()
                    match = re.search(
                        r"^gufo_spec_lookup_accepted_tokens_total (\d+)$",
                        response.text, re.MULTILINE)
                    assert match, "missing accepted lookup metric"
                    return int(match[1])

                for width in (2, 4, 6, 8):
                    start = copied()
                    ready = threading.Barrier(width)

                    def run(row):
                        ready.wait(timeout=30)
                        return chat_result(client, bodies[row], True)

                    with ThreadPoolExecutor(max_workers=width) as pool:
                        results = list(pool.map(run, range(width)))
                    record(f"C{width}", results)
                    assert copied() > start, "cohort accepted no lookup copies"
                    actual_width = max(
                        r["usage"]["gufo"]["physical_execution_width"]
                        for r in results)
                    assert actual_width == width, (
                        "requested cohort never executed together",
                        width, actual_width)
                    for row, result in enumerate(results):
                        same(controls[row], result)

                # All-greedy peers need a separate check: their shared planner
                # can bound copied suffixes when other rows generate new text.
                ordinary = {
                    **bodies[0], "messages": [{"role": "user", "content": (
                        "Write a paragraph about a village preparing for a "
                        "winter festival. Use varied vocabulary.")}],
                }
                record("mixed-warm", chat_result(
                    client, {**ordinary, "max_completion_tokens": 1}))
                ordinary_control = record(
                    "mixed-isolated", chat_result(client, ordinary, True))
                mixed = [ordinary if row % 2 else bodies[row] for row in range(8)]
                ready = threading.Barrier(8)
                start = copied()

                def run_mixed(body):
                    ready.wait(timeout=30)
                    return chat_result(client, body, True)

                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(run_mixed, mixed))
                record("mixed-greedy-C8", results)
                assert copied() > start, "mixed greedy cohort accepted no copies"
                assert max(r["usage"]["gufo"]["physical_execution_width"]
                           for r in results) == 8, "mixed cohort never ran at C8"
                for row, result in enumerate(results):
                    same(ordinary_control if row % 2 else controls[row], result)

                # Stop within a repeated passage, then replay the full request.
                for row in (0, 1):
                    marker = "purple, red" if row == 0 else "Friday, Monday"
                    stopped = record(f"stop-{row}", chat_result(
                        client, {**bodies[row], "stop": marker}, True))
                    assert stopped["text"] == controls[row]["text"].split(marker)[0], stopped
                    assert stopped["finish"] == "stop", stopped
                    replay = record(f"resume-{row}", chat_result(client, bodies[row], True))
                    same(controls[row], replay)

                if option(command, "--mmproj") is not None:
                    for row in (0, 1):
                        body = {**bodies[row], "messages": [
                            {"role": "user", "content": [
                                image_content("red"),
                                {"type": "text", "text":
                                 bodies[row]["messages"][0]["content"]}]}]}
                        record(f"image-warm-{row}", chat_result(
                            client, {**body, "max_completion_tokens": 1}))
                        start = copied()
                        control = record(
                            f"image-copy-{row}", chat_result(client, body, True))
                        assert not control["reasoning"] and not control["tools"], control
                        assert control["text"] and sources[row].startswith(control["text"]), control
                        assert control["usage"]["prompt_tokens"] > controls[row]["usage"]["prompt_tokens"]
                        assert copied() > start, "image request accepted no lookup copies"

                        # Interrupt after a substantial copied prefix, not just
                        # the first token before lookup has had a chance to run.
                        start = copied()
                        partial = ""
                        with client.chat.completions.create(**body, stream=True) as stream:
                            for chunk in stream:
                                for choice in chunk.choices:
                                    assert choice.finish_reason is None, "finished before cancellation"
                                    partial += choice.delta.content or ""
                                if len(partial) >= 200:
                                    break
                            else:
                                raise AssertionError("image copy never reached cancellation point")
                        request_id = trace.rows[-1]["request_id"]
                        assert trace.rows[-1]["status"] == "disconnected"
                        assert control["text"].startswith(partial)
                        cancelled = re.compile(
                            rf"\brequest={re.escape(request_id)}\b.*\bfinish=cancelled\b")
                        deadline = time.monotonic() + 10
                        while not cancelled.search((output / "server.log").read_text()):
                            assert time.monotonic() < deadline, "server did not observe cancellation"
                            time.sleep(.02)
                        assert copied() > start, "cancelled image request exercised no accepted copies"
                        record(f"image-cancel-{row}", {
                            "partial": partial, "request_id": request_id,
                            "accepted_copies": copied() - start})
                        replay = record(
                            f"image-resume-{row}", chat_result(client, body, True))
                        same(control, replay)
                        assert replay["usage"]["cached_tokens"] > 0, replay
                        assert replay["usage"]["gufo"]["prefill_tokens"] == 0, replay
                report["accepted_lookup_tokens"] = copied()
        join_server_timings(output)
        report["execution"] = execution_coverage(output, "mtp")
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        trace.save()
        write_json(report_path, report)


if __name__ == "__main__":
    main()

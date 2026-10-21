#!/usr/bin/env python3
"""Issue #617: held disk publication must not cost more than saved prefill.

Opt-in GPU test, outside hosted CI. The cap expectation is a main timing
control; candidate correctness uses the default bounded expectation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import threading
import time
import urllib.request

from run import option, provenance, server, write_json


def wait_until(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError(f"did not observe {description}")


def signature(answer):
    return (answer["choices"], answer["usage"]["completion_tokens"])


def answer_of(result):
    # A waiter that re-prefills from a shallow checkpoint splits prefill into
    # different passes than a cold run. That can change its reasoning text
    # without a cache bug, as cache_growth.py notes, so compare the answer
    # and whether it reasoned.
    choice = result["choices"][0]
    message = choice["message"]
    return (message.get("content"), message.get("tool_calls"), choice["finish_reason"],
            bool((message.get("reasoning_content") or "").strip()))


def archive():
    rng = random.Random(617)
    words = ("configuration checkpoint retention validation filesystem scheduler "
             "publication snapshot concurrency restore allocation execution "
             "repository diagnostics preservation continuation capacity metadata "
             "generation persistence").split()
    return " ".join(rng.choice(words) for _ in range(55000))


def run_case(args, command, case, report):
    output = args.output / case
    output.mkdir()
    disk, gate = output / "disk", output / "gate"
    gate.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [*command, "--host", "127.0.0.1", "--port", str(port),
               "--cache-disk", str(disk), "--cache-disk-bytes", str(8 * 1024**3)]
    if option(command, "--cache-ram-bytes") is None:
        command += ["--cache-ram-bytes", str(1024**3)]
    environment = {**os.environ, "LD_PRELOAD": str(args.fault_library),
                   "GUFO_TEST_DISK_GATE": str(gate),
                   "GUFO_TEST_DISK_DIRECTORY": str(disk),
                   "GUFO_TEST_DISK_MANIFEST_ONLY": "1"}
    row = {"command": command, "status": "running", "requests": {}}
    report["cases"][case] = row
    write_json(args.output / "report.json", report)
    log = output / "server.log"
    release_lock = threading.Lock()

    def published():
        return [int(n) for n in re.findall(r"event=published tokens=(\d+)", log.read_text())]

    def wait_events(stage):
        return [dict(re.findall(r"(\w+)=([^\s]+)", line))
                for line in log.read_text().splitlines()
                if "event=publication_wait " in line and f"stage={stage} " in line]

    def chat(label, messages, cache=True):
        body = {"model": option(command, "--served-model-name", "publication-wait"),
                "messages": messages, "temperature": 0, "seed": 31,
                "reasoning_effort": "low", "max_tokens": 32,
                "cache_prompt": cache, "stream": False}
        write_json(output / f"{label}.request.json", body)
        start = time.monotonic()
        request = urllib.request.Request(base + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
        write_json(output / f"{label}.response.json", answer)
        assert "error" not in answer and answer.get("choices"), answer
        row["requests"][label] = {"wall_ms": (time.monotonic() - start) * 1000,
                                   "usage": answer["usage"]}
        write_json(args.output / "report.json", report)
        return answer

    def release():
        with release_lock:
            if not (gate / "release").exists():
                row["release_time"] = time.time()
                (gate / "release").touch()

    timer = None
    try:
        with server(command, log, 300, environment):
            seed = [{"role": "system", "content": "Issue617 gate experiment. Reply with only ACK."},
                    {"role": "user", "content": "Initialize the archive."}]
            first = chat("seed", seed)
            wait_until(published, 30, "seed publication")
            (gate / "armed").touch()
            deep = seed + [first["choices"][0]["message"],
                {"role": "user", "content": archive() + "\nReply with only ACK."}]
            answer = chat("deep", deep)
            assert 50000 <= answer["usage"]["prompt_tokens"] <= 60000, answer["usage"]
            wait_until(lambda: (gate / "blocked").exists(), 60, "held manifest")
            row["blocked_file"] = (gate / "blocked").read_text()
            # A new user message branches before the previous assistant tail.
            # The pending semantic boundary matches but its live frontier does not.
            follow = deep + [{"role": "user", "content":
                "Updated final instruction: reply with only ACK."}]
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(chat, "waiter", follow)
                try:
                    if case == "publish":
                        starts = wait_until(lambda: wait_events("start"), 60,
                                            "waiter admission wait")
                        assert len(starts) == 1, starts
                        row["wait_start"] = starts[0]
                    timer = threading.Timer(.25 if case == "publish" else 130, release)
                    timer.start()
                    waited = future.result(timeout=240)
                finally:
                    if timer:
                        timer.cancel()
                    release()
            wait_until(lambda: max(published()) >= 50000, 60, "deep publication")
            retried = chat("retry", follow)
            cold = chat("cold", follow, cache=False)
            assert signature(retried) == signature(cold)
            if case == "publish":
                assert signature(waited) == signature(cold)
            else:
                assert answer_of(waited) == answer_of(cold), (answer_of(waited), answer_of(cold))
            usage = waited["usage"]
            timing = usage["gufo"]
            assert retried["usage"]["gufo"]["cache_disk_hit"], retried["usage"]
            assert cold["usage"]["cached_tokens"] == 0, cold["usage"]
            if option(command, "--speculative", "off") != "off":
                assert usage.get("draft_tokens", 0) > 0, usage
            if case == "publish":
                assert timing["cache_disk_hit"], timing
                assert timing["prefill_tokens"] < 64, timing
                start = row["wait_start"]
                releases = [event for event in wait_events("release")
                            if event["request"] == start["request"]
                            and event["checkpoint_id"] == start["checkpoint_id"]]
                assert len(releases) == 1, releases
                ended = releases[0]
                assert ended["release"] == "not_pending", ended
                assert 0 < float(ended["wait_ms"]) < float(start["limit_ms"]), (start, ended)
                row["wait_release"] = ended
            else:
                # A tight RAM budget can leave only a shallow seed checkpoint
                # on disk. The waiter may restore it, but not the held checkpoint.
                assert usage["cached_tokens"] < 4096, usage
                assert timing["prefill_tokens"] > 4096, timing
                if args.expect == "cap":
                    limit = 120000
                else:
                    starts = re.findall(r"event=publication_wait .*stage=start .*limit_ms=([\d.e+-]+)",
                                        log.read_text())
                    assert len(starts) == 1, starts
                    limit = float(starts[0])
                    assert 0 < limit < 60000, limit
                    assert "release=cost_bound" in log.read_text()
                queue = timing["queue_ms"]
                assert limit * .95 <= queue <= limit + max(3, limit * .05), (queue, limit)
                row["wait_limit_ms"] = limit
            # Qualify a real over-RAM publication, not just a queued job.
            payloads = [int(n) for n in re.findall(r"event=published .*payload_bytes=(\d+)", log.read_text())]
            capacity = int(re.search(r"event=configured .*?\bcapacity_bytes=(\d+)", log.read_text())[1])
            assert max(payloads) > capacity, (payloads, capacity)
            row.update(status="passed", max_payload_bytes=max(payloads), ram_capacity_bytes=capacity)
    except BaseException as exc:
        row.update(status="failed", error=repr(exc))
        raise
    finally:
        if timer:
            timer.cancel()
        release()
        write_json(args.output / "report.json", report)
        shutil.rmtree(disk, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault-library", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=("bound", "publish"))
    parser.add_argument("--expect", choices=("bounded", "cap"), default="bounded")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.fault_library = args.fault_library.resolve(strict=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    if int(option(command, "--context", "0")) < 60000:
        parser.error("use context >= 60000 for the 55K-token controlled workload")
    for reserved in ("--host", "--port", "--cache-disk", "--cache-disk-bytes", "--api-key"):
        if option(command, reserved) is not None:
            parser.error(f"the runner owns {reserved}")
    if option(command, "--served-model-name") is None:
        command += ["--served-model-name", "publication-wait"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {**provenance(), "cases": {}, "expect": args.expect,
              "fault_library_sha256": hashlib.file_digest(args.fault_library.open("rb"), "sha256").hexdigest()}
    try:
        for case in dict.fromkeys(args.case or ("bound", "publish")):
            run_case(args, command, case, report)
    finally:
        report["status"] = "passed" if report["cases"] and all(
            row["status"] == "passed" for row in report["cases"].values()) else "failed"
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()

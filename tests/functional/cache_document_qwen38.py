#!/usr/bin/env python3
"""A second question about a long document restores its prefix from disk.

Each request is one user message: the same long document followed by a
different short question. The first question seeds the disk cache; after a
restart the second question must restore a prefix of at least --min-cached
tokens from disk and produce, greedily, the same reply as on a fresh server
without a cache (#1255). Every request's timings and usage are retained.

  python3 tests/functional/cache_document_qwen38.py --output DIR \\
    -- build/release/gufo serve llm --model M.gguf --context 65536
"""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import socket
import sys
import time

from run import option, provenance, server, write_json

NOTE = ("Record {i}: the relay at marker {i} reported a voltage of {v} volts, "
        "a temperature of {t} degrees and {p} dropped packets.\n")
QUESTIONS = ("Which marker reported the highest voltage? Reply with its number.",
             "Which marker dropped the most packets? Reply with its number.")


def document(records):
    return "".join(
        NOTE.format(i=i, v=200 + i * 37 % 41, t=15 + i * 13 % 23, p=i * 7 % 19)
        for i in range(records))


def request(port, text, max_tokens, artifact):
    body = json.dumps({
        "model": "document",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    started = time.monotonic()
    try:
        connection.request("POST", "/v1/chat/completions", body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
    elapsed = time.monotonic() - started
    write_json(artifact, {"response": payload, "status": response.status,
                          "elapsed_s": elapsed})
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload}")
    usage = payload["usage"]
    return {"content": payload["choices"][0]["message"].get("content") or "",
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "disk_hit": usage["gufo"]["cache_disk_hit"],
            "checkpoint_ids": usage["gufo"]["component_cache"][
                "captured_checkpoint_ids"],
            "timings": payload.get("timings"), "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1450)
    parser.add_argument("--min-cached", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port", "--cache-disk", "--served-model-name"):
        if option(command, flag) is not None:
            parser.error(f"the test owns {flag}")
    command += ["--served-model-name", "document"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk-cache"
    binary = Path(shutil.which(command[0]) or command[0]).resolve()
    report = {**provenance(), "command": command, "binary": str(binary),
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(
                  Path(__file__).read_bytes()).hexdigest(),
              "status": "running"}
    text = document(args.records)
    prompts = [text + "\n" + question for question in QUESTIONS]

    def serve(label, extra=()):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        full = command + list(extra) + ["--host", "127.0.0.1", "--port", str(port)]
        return port, server(full, output / f"{label}.log", 900)

    def drained(label, process):
        # Publication is asynchronous: wait until the disk queue settles.
        deadline = time.monotonic() + 300
        count = -1
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{label} server exited before publication")
            now = len(re.findall(r"event=published ",
                                 (output / f"{label}.log").read_text()))
            if now and now == count:
                return
            count = now
            time.sleep(3)
        raise TimeoutError("disk publication did not settle")

    try:
        write_json(output / "report.json", report)
        port, context = serve("cold")
        with context:
            report["cold"] = request(port, prompts[1], args.max_tokens,
                                     output / "cold.json")
        port, context = serve("seed", ["--cache-disk", str(disk)])
        with context as process:
            report["first"] = request(port, prompts[0], args.max_tokens,
                                      output / "first.json")
            drained("seed", process)
        report["published"] = re.findall(
            r"event=published tokens=(\d+) checkpoint_id=\d+ purpose=(\d+)",
            (output / "seed.log").read_text())
        port, context = serve("restored", ["--cache-disk", str(disk)])
        with context:
            report["second"] = request(port, prompts[1], args.max_tokens,
                                       output / "second.json")
        failures = []
        second, cold = report["second"], report["cold"]
        if cold["cached_tokens"]:
            failures.append(f"cold control restored {cold['cached_tokens']}")
        if second["content"] != cold["content"]:
            failures.append("second question differs from the cold control")
        if not second["disk_hit"] or second["cached_tokens"] < args.min_cached:
            failures.append(f"second question restored {second['cached_tokens']} "
                            f"tokens (disk_hit={second['disk_hit']})")
        report["failures"] = failures
        report["status"] = "failed" if failures else "passed"
    except BaseException as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        shutil.rmtree(disk, ignore_errors=True)
        write_json(output / "report.json", report)
    print(json.dumps({k: report.get(k) for k in
                      ("status", "failures", "published")}, indent=1))
    for key in ("cold", "first", "second"):
        r = report[key]
        print(key, r["prompt_tokens"], "cached", r["cached_tokens"], "disk",
              r["disk_hit"], f"{r['elapsed_s']:.1f}s")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

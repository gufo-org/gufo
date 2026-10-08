#!/usr/bin/env python3
"""E1: measure checkpoint payload bytes at several prompt depths.

Starts `gufo serve llm` with a disk cache, sends one unrelated chat prompt per
depth (max_tokens=1), and reads the disk store's `action=stored` lines, which
carry the exact payload bytes and token count of each persisted checkpoint.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from paths import BIN, REPO, WORK as HERE


def corpus() -> str:
    parts = []
    for path in sorted(REPO.joinpath("src").rglob("*")):
        if path.suffix in {".cpp", ".hpp", ".hip", ".h"} and path.is_file():
            parts.append(path.read_text(errors="replace"))
    return "\n".join(parts)


def post(port, path, body, timeout=3600):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(port, proc, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                        timeout=5) as response:
                if response.status == 200:
                    return json.load(response)["data"][0]["id"]
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become ready")


STORED = re.compile(r"event=disk_cache action=stored .*?file_bytes=(\d+) "
                    r"payload_bytes=(\d+) tokens=(\d+)")


def stored_lines(log_path):
    return [tuple(map(int, m.groups()))
            for m in STORED.finditer(log_path.read_text(errors="replace"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--depths", default="2048,8192,32768,65536,131072")
    parser.add_argument("--extra", default="")
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--disk-bytes", type=int, default=8 << 30)
    args = parser.parse_args()

    out_dir = HERE / "results" / "e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = HERE / "cache" / f"e1-{args.name}"
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    log_path = out_dir / f"{args.name}.log"
    command = [str(BIN), "serve", "--host", "127.0.0.1", "--port",
               str(args.port), "--sessions", "1", "llm", "--model", args.model,
               "--context", str(args.context), "--think", "off",
               "--cache-disk", str(cache_dir), "--cache-disk-bytes",
               str(args.disk_bytes), "--cache-disk-staging-bytes",
               str(6 << 30), *args.extra.split()]
    text = corpus()
    results = {"name": args.name, "command": command, "points": []}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        model_id = wait_ready(args.port, proc)
        # Calibrate characters per token on an 8k-character slice.
        probe = post(args.port, "/v1/chat/completions", {
            "model": model_id, "messages": [{"role": "user", "content": text[:8000]}],
            "max_tokens": 1, "temperature": 0, "cache_prompt": False})
        chars_per_token = 8000 / probe["usage"]["prompt_tokens"]
        offset = 8000
        for depth in map(int, args.depths.split(",")):
            chars = int(depth * chars_per_token)
            if offset + chars > len(text):
                offset = 0
            prompt = f"[{args.name} depth {depth}]\n" + text[offset:offset + chars]
            offset += chars
            before = len(stored_lines(log_path))
            started = time.time()
            response = post(args.port, "/v1/chat/completions", {
                "model": model_id, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0})
            elapsed = time.time() - started
            prompt_tokens = response["usage"]["prompt_tokens"]
            # Disk writes are asynchronous: wait for this prompt's own
            # checkpoint (within 64 tokens of its end) to be persisted.
            deadline = time.time() + 600
            while time.time() < deadline and not any(
                    t >= prompt_tokens - 64 for _, _, t in
                    stored_lines(log_path)[before:]):
                if "reason=staging_capacity" in log_path.read_text(errors="replace"):
                    print("staging capacity skip", flush=True)
                    break
                time.sleep(1)
            point = {"depth": depth, "prompt_tokens": prompt_tokens,
                     "request_s": round(elapsed, 2),
                     "usage_gufo": response["usage"].get("gufo", {})}
            results["points"].append(point)
            print(json.dumps({k: point[k] for k in ("depth", "prompt_tokens",
                                                    "request_s")}), flush=True)
        # Let queued disk writes finish: stop after 20 s without a new line.
        count, quiet_since, deadline = -1, time.time(), time.time() + 180
        while time.time() < deadline and time.time() - quiet_since < 20:
            current = len(stored_lines(log_path))
            if current != count:
                count, quiet_since = current, time.time()
            time.sleep(2)
        # Every persisted checkpoint is one (tokens, payload bytes) sample.
        results["stored"] = [{"file_bytes": f, "payload_bytes": p, "tokens": t}
                             for f, p, t in stored_lines(log_path)]
        for sample in results["stored"]:
            print(json.dumps(sample), flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        shutil.rmtree(cache_dir, ignore_errors=True)
        (out_dir / f"{args.name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())

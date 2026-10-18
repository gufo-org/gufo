#!/usr/bin/env python3
"""Measure retained screenshot preparation on full continuation-cache hits (#627).

Uses only the standard library. Keeps every request/response and the server log;
run the same command with baseline and candidate production binaries. Debug
logging exposes the candidate's retained RGB byte count independently of RSS.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib


def screenshot(width, height, index):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            if y < 40 or y > height - 25:
                color = (35 + index * 7, 40, 48)
            elif x < width // 5:
                color = (28, 31 + index * 3, 38)
            elif y % 24 < 3 and (x // 13 + y // 24 + index) % 11 < 7:
                color = (110 + (x // 80) % 100, 150, 190)
            else:
                color = (20, 23, 29)
            raw.extend(color)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def encoded_variant(url, index):
    """Different valid PNG bytes with identical decoded pixels and cache identity."""
    data = base64.b64decode(url.split(",", 1)[1])
    chunk = b"tEXt" + b"repro627\0" + str(index).encode()
    text = (struct.pack(">I", len(chunk) - 4) + chunk
            + struct.pack(">I", zlib.crc32(chunk)))
    return "data:image/png;base64," + base64.b64encode(data[:-12] + text + data[-12:]).decode()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    return result, (time.perf_counter() - start) * 1000


def cache_stats(log):
    for line in reversed(log.read_text().splitlines()):
        if "event=image_cache " in line:
            return {key: int(value) for part in line.split()
                    if "=" in part for key, value in [part.split("=", 1)]
                    if key in {"enabled", "entries", "pixel_bytes", "encoded_bytes", "hits", "misses", "evictions"}}
    return None


def rss_bytes(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("missing VmRSS")


def run(args, command):
    args.output.mkdir(parents=True, exist_ok=False)
    images = {}
    for width, height in ((1920, 1080), (128, 128)):
        for i in range(4):
            data = screenshot(width, height, i)
            (args.output / f"screenshot-{width}x{height}-{i}.png").write_bytes(data)
            images[width, i] = "data:image/png;base64," + base64.b64encode(data).decode()
    if args.image:
        data = args.image.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "--image requires a PNG"
        assert struct.unpack(">II", data[16:24]) == (1920, 1080), "--image requires 1920x1080"
        (args.output / "input.png").write_bytes(data)
        images[1920, 0] = "data:image/png;base64," + base64.b64encode(data).decode()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command += ["--host", "127.0.0.1", "--port", str(port), "--log-level", "debug"]
    metadata = {"command": command, "binary_sha256": hashlib.sha256(
        Path(command[0]).read_bytes()).hexdigest(), "repeats": args.repeats,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "turns": args.turns, "kinds": args.kind, "fresh_encoding": args.fresh_encoding,
        "input_image": str(args.image) if args.image else None,
        "scope": "Server queue_ms before admission; all measured repeats must have zero prefill.",
        "images": {p.name: {"bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in args.output.glob("*.png")}}
    save(args.output / "metadata.json", metadata)
    log = args.output / "server.log"
    rows = []
    with log.open("w") as stream:
        server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 240
            while True:
                assert server.poll() is None, f"server exited; see {log}"
                try:
                    model = request(base, "/v1/models")[0]["data"][0]["id"]
                    break
                except (urllib.error.URLError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"server not ready; see {log}")
                    time.sleep(0.25)
            metadata["ready_rss_bytes"] = rss_bytes(server.pid)
            save(args.output / "metadata.json", metadata)
            for count in args.turns:
                for kind in args.kind:
                    messages = []
                    for i in range(count):
                        text = f"Screenshot from turn {i + 1}. Keep this observation in history."
                        if kind == "text":
                            content = text + " x" * 2042
                        else:
                            content = [{"type": "text", "text": text}, {
                                "type": "image_url", "image_url": {
                                    "url": images[1920 if kind == "1080p" else 128, i]}}]
                        messages.extend([{"role": "user", "content": content},
                                         {"role": "assistant", "content": "Acknowledged."}])
                    messages.append({"role": "user", "content": f"Case {kind}-{count}. Reply with OK."})
                    payload = {"model": model, "messages": messages, "max_tokens": 1,
                               "temperature": 0, "seed": 627, "cache_prompt": True,
                               "chat_template_kwargs": {"enable_thinking": False}}
                    save(args.output / f"{kind}-{count}.request.json", payload)
                    for repeat in range(args.repeats + 1):
                        if args.fresh_encoding:
                            payload["messages"][0]["content"][1]["image_url"]["url"] = (
                                encoded_variant(images[1920, 0], repeat))
                            save(args.output / f"{kind}-{count}-{repeat}.request.json", payload)
                        result, wall = request(base, "/v1/chat/completions", payload)
                        usage = result["usage"]
                        row = {"kind": kind, "turns": count, "repeat": repeat,
                               "warmup": repeat == 0, "wall_ms": wall,
                               "rss_bytes": rss_bytes(server.pid),
                               "image_cache": cache_stats(log), "response": result}
                        rows.append(row)
                        with (args.output / "requests.jsonl").open("a") as output:
                            output.write(json.dumps(row) + "\n")
                        print(kind, count, repeat, usage["gufo"]["queue_ms"], flush=True)
                        if repeat:
                            assert usage["cached_tokens"] == usage["prompt_tokens"], row
                            assert usage["gufo"]["prefill_tokens"] == 0, row
                            assert usage["gufo"]["component_cache"]["reuse"] == "ram", row
                            assert result["choices"] == rows[-2]["response"]["choices"], row
                            if kind != "text" and row["image_cache"] is not None:
                                previous = rows[-2]["image_cache"]
                                assert row["image_cache"]["misses"] == (
                                    previous["misses"] + int(args.fresh_encoding)), row
                                assert row["image_cache"]["hits"] == (
                                    previous["hits"] + (0 if args.fresh_encoding else count)), row
                                assert (row["image_cache"]["pixel_bytes"]
                                        + row["image_cache"].get("encoded_bytes", 0)) <= 128 * 1024 * 1024, row
                        time.sleep(0.3)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    summary = []
    for count in args.turns:
        for kind in args.kind:
            selected = [r for r in rows if r["turns"] == count and r["kind"] == kind]
            repeats = selected[1:]
            summary.append({"kind": kind, "turns": count,
                            "prompt_tokens": repeats[0]["response"]["usage"]["prompt_tokens"],
                            "cold_queue_ms": selected[0]["response"]["usage"]["gufo"]["queue_ms"],
                            "queue_ms": [r["response"]["usage"]["gufo"]["queue_ms"] for r in repeats],
                            "median_queue_ms": statistics.median(
                                r["response"]["usage"]["gufo"]["queue_ms"] for r in repeats),
                            "image_cache": repeats[-1]["image_cache"]})
    save(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turns", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--kind", nargs="+", choices=("1080p", "text", "128px"),
                        default=["1080p", "text", "128px"])
    parser.add_argument("--fresh-encoding", action="store_true",
                        help="One-image cache misses using new PNG metadata; identical pixels/tokens")
    parser.add_argument("--image", type=Path, help="Use a supplied 1920x1080 PNG for the focused case")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass gufo serve llm and its model/cache options after --")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.fresh_encoding and (args.turns != [1] or args.kind != ["1080p"]):
        parser.error("--fresh-encoding requires --turns 1 --kind 1080p")
    if args.image and (args.turns != [1] or args.kind != ["1080p"]):
        parser.error("--image requires --turns 1 --kind 1080p")
    for reserved in ("--host", "--port", "--log-level"):
        if any(part == reserved or part.startswith(reserved + "=") for part in command):
            parser.error(f"the harness owns {reserved}")
    run(args, command)


if __name__ == "__main__":
    main()

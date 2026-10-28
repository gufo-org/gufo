#!/usr/bin/env python3
"""Warm multi-turn restores of Qwen3.8-Flash-Next must equal a cold prefill.

Flash-Next's sparse prefill groups four queries from each pass start, so the
server rounds prompt checkpoints and pass ends down to multiples of four
(prefill_alignment). Each case is a conversation whose final user turn starts
past the 2,048-token sparse budget; four leading pads shift that boundary
through every residue. A seeds the cache; B continues the conversation (A's
turn, a fixed assistant reply and a new user turn) and restores A's stable
boundary, the end of its final user turn, from RAM in the same server or from
disk after a restart. Every greedy B (reasoning and answer) must be
byte-identical to B on a fresh server, in the loaded mode (pass --mtp-model
for MTP), and restore on a multiple of four.

  python3 tests/functional/cache_alignment_qwen38.py --output DIR \\
    [--mtp-model MTP.gguf] -- build/release/gufo serve llm --model M.gguf
"""

import argparse
import http.client
import json
from pathlib import Path
import socket
import sys
import time

from run import option, provenance, server, write_json

PADS = ("alpha", "alpha beta", "alpha beta gamma", "alpha beta gamma delta")
NOTE = ("Station log: the relay at marker {i} reported nominal voltage, "
        "steady temperature and no packet loss during the hourly check.\n")


def conversation(pad, turns):
    system = pad + "\n" + "".join(NOTE.format(i=i) for i in range(110))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Which marker reported a fault?"},
        {"role": "assistant", "content": "None of the markers reported a fault."},
        {"role": "user", "content": "Reply with the number of markers."},
    ]
    if turns > 1:
        messages += [
            {"role": "assistant", "content": "There are 110 markers."},
            {"role": "user", "content": "Reply with the first marker."},
        ]
    return messages


def request(port, messages, max_tokens):
    body = json.dumps({
        "model": "alignment", "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": True}}).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    try:
        connection.request("POST", "/v1/chat/completions", body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {payload}")
    message = payload["choices"][0]["message"]
    usage = payload["usage"]
    return {"reasoning": message.get("reasoning_content") or "",
            "content": message.get("content") or "",
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp-model", help="run the MTP mode with this sidecar")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port", "--cache-disk", "--served-model-name",
                 "--speculative", "--mtp-model"):
        if option(command, flag) is not None:
            parser.error(f"the test owns {flag}")
    if args.mtp_model:
        command += ["--speculative", "mtp", "--mtp-model", args.mtp_model]
    command += ["--served-model-name", "alignment"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk-cache"
    report = {**provenance(), "command": command, "cases": {},
              "status": "running"}

    def save():
        write_json(output / "report.json", report)

    def serve(label, extra=()):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        full = command + list(extra) + ["--host", "127.0.0.1", "--port", str(port)]
        return port, server(full, output / f"{label}.log", 600)

    try:
        save()
        cold = {}
        port, context = serve("cold")
        with context:
            for pad in PADS:
                cold[pad] = request(port, conversation(pad, 2),
                                    args.max_tokens)
        warm = {}
        port, context = serve("warm-ram")
        with context:
            for pad in PADS:
                seed = request(port, conversation(pad, 1), args.max_tokens)
                warm[pad] = (seed, request(port, conversation(pad, 2),
                                           args.max_tokens))
        port, context = serve("disk-seed", ["--cache-disk", str(disk)])
        with context:
            seeds = {pad: request(port, conversation(pad, 1),
                                  args.max_tokens) for pad in PADS}
            time.sleep(10)  # let the disk writer finish before shutdown
        restored = {}
        port, context = serve("disk-restore", ["--cache-disk", str(disk)])
        with context:
            for pad in PADS:
                restored[pad] = request(port, conversation(pad, 2),
                                        args.max_tokens)
        failures = []
        for pad in PADS:
            reference = cold[pad]
            case = {"cold": reference, "ram_seed": warm[pad][0],
                    "ram": warm[pad][1], "disk_seed": seeds[pad],
                    "disk": restored[pad]}
            for path in ("ram", "disk"):
                got = case[path]
                same = (got["reasoning"], got["content"]) == (
                    reference["reasoning"], reference["content"])
                case[path + "_identical"] = same
                if not same:
                    failures.append(f"{pad!r} {path}: differs from cold")
                if not 0 < got["cached_tokens"] < got["prompt_tokens"]:
                    failures.append(f"{pad!r} {path}: no partial restore "
                                    f"({got['cached_tokens']} cached)")
                if got["cached_tokens"] % 4:
                    failures.append(f"{pad!r} {path}: unaligned restore "
                                    f"({got['cached_tokens']} cached)")
            if reference["cached_tokens"]:
                failures.append(f"{pad!r} cold: restored {reference['cached_tokens']}")
            report["cases"][pad] = case
            print(f"{pad!r}: prompt {reference['prompt_tokens']} cached "
                  f"ram {case['ram']['cached_tokens']} disk "
                  f"{case['disk']['cached_tokens']} identical ram "
                  f"{case['ram_identical']} disk {case['disk_identical']}",
                  flush=True)
        report["failures"] = failures
        report["status"] = "failed" if failures else "passed"
        save()
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        return 0
    except BaseException as error:
        report["status"] = "error"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Prompt/chat must run the requested GPU backend and emit identical token IDs."""
import os
from pathlib import Path
import re
import subprocess
import sys

TRACE = re.compile(r"^\[TokenTrace\]: count=(\d+) sha256=([0-9a-f]{64})$",
                   re.MULTILINE)
STEPS = re.compile(r"verification_steps=(\d+)")
PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Continue the story with the robot's first exhibition.",
]


def run(binary, model, mode, backend):
    command = [binary, mode, "--model", model, "--verbose",
               "--temperature", "0", "--max-tokens", "8", *backend]
    if mode == "prompt":
        command += ["--prompt", PROMPTS[0]]
    result = subprocess.run(
        command, input="\n".join([*PROMPTS, "exit"]) + "\n",
        text=True, capture_output=True, timeout=180, check=True)
    traces = TRACE.findall(result.stderr)
    count = 1 if mode == "prompt" else 2
    if len(traces) != count or any(int(tokens) != 8 for tokens, _ in traces):
        raise AssertionError(f"incomplete {mode} {backend}: {result.stderr}")
    if "gfx1151 GPU Executor" not in result.stdout:
        raise AssertionError(f"{mode} ignored GPU execution")
    steps = STEPS.findall(result.stderr)
    if backend and (len(steps) != count or any(int(s) == 0 for s in steps)):
        raise AssertionError(f"{mode} ignored speculative backend: {result.stderr}")
    return traces


def main():
    artifacts = [os.environ.get("GUFO_QWEN27B_" + name + "_MODEL", "")
                 for name in ("MTP", "DFLASH")]
    model = os.environ.get("GUFO_QWEN27B_MODEL", "")
    if not all(path and Path(path).is_file() for path in [model, *artifacts]):
        print("Qwen27B CLI test needs target, MTP and DFlash model artifacts")
        return 77
    binary = sys.argv[1]
    baseline = None
    for backend in (
        [],
        ["--speculative", "mtp", "--mtp-model", artifacts[0]],
        ["--speculative", "dflash2", "--dflash-model", artifacts[1]],
    ):
        prompt = run(binary, model, "prompt", backend)
        chat = run(binary, model, "chat", backend)
        if prompt[0] != chat[0]:
            raise AssertionError(f"prompt/chat mismatch: {backend}")
        if baseline is None:
            baseline = chat
        elif chat != baseline:
            raise AssertionError(f"multi-turn target/speculative mismatch: {backend}")
        print(f"{backend[1] if backend else 'AR'}: prompt and both chat turns exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())

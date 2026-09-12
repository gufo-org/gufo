#!/usr/bin/env python3
"""Qwen27B executable wiring, seeded replay and HTTP sampling contracts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

TRACE = re.compile(r"^\[TokenTrace\]: count=(\d+) sha256=([0-9a-f]{64})$",
                   re.MULTILINE)
STEPS = re.compile(r"verification_steps=(\d+)")
BENCH_TRACE = re.compile(
    r"^\[QwenBenchTrace\]: depth=(\d+) count=(\d+) sha256=([0-9a-f]{64})$",
    re.MULTILINE)
PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Continue the story with the robot's first exhibition.",
]
SAMPLING_CASES = [
    {"temperature": 0, "seed": 73, "repeat_penalty": 1.1,
     "repeat_last_n": 8, "frequency_penalty": 0.15, "presence_penalty": 0.1},
    {"temperature": 0.05, "seed": 73, "repeat_penalty": 1.2,
     "repeat_last_n": 0, "frequency_penalty": 0.2, "presence_penalty": 0.1},
    {"temperature": 0.8, "top_k": 1, "seed": 73},
    {"temperature": 0.8, "top_k": 1, "top_p": 0.1, "min_p": 1,
     "min_keep": 3, "seed": 73},
    {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
     "min_keep": 3, "seed": 73, "repeat_penalty": 1.05,
     "repeat_last_n": 8, "frequency_penalty": 0.15, "presence_penalty": 0.1},
    {"temperature": 2, "min_p": 0.02, "seed": 808,
     "frequency_penalty": -0.2, "presence_penalty": -0.1},
]


def sampling_flags(config):
    return [part for key, value in config.items()
            for part in ("--" + key.replace("_", "-"), str(value))]


def check_options(binary):
    commands = [
        ["prompt"], ["chat"], ["bench"], ["eval"], ["video"], ["transcribe"],
        ["asr"], ["diagnose"], ["info"], ["probe"], ["serve"], ["serve", "llm"],
        ["serve", "video"], ["serve", "audio"], ["serve", "tts"],
    ]
    for command in commands:
        help_result = subprocess.run([binary, *command, "--help"],
                                     text=True, capture_output=True, timeout=30)
        if help_result.returncode != 0:
            raise AssertionError(f"help failed: {command} {help_result.stderr}")
        for flag in ("--draft-temperature", "--draft-top-k", "--draft-top-p",
                     "--draft-min-p", "--draft-seed"):
            result = subprocess.run([binary, *command, flag, "0.8"],
                                    text=True, capture_output=True, timeout=30)
            if result.returncode != 2:
                raise AssertionError(f"unsupported option not rejected: {command} {flag}")
    for command in (["prompt"], ["chat"], ["bench"], ["serve"], ["serve", "llm"]):
        for flag, value in (("--draft-policy", "unknown"),
                            ("--min-draft-tokens", "2")):
            result = subprocess.run(
                [binary, *command, "--speculative", "dflash2", "--dflash-model",
                 "/dev/null", flag, value], text=True, capture_output=True, timeout=30)
            if result.returncode != 2:
                raise AssertionError(f"invalid DFlash2 config not rejected: {command} {flag}")
    invalid_sampling = (
        ("--temperature", "-0.1"), ("--top-k", "-1"), ("--top-p", "0"),
        ("--min-p", "1.1"), ("--min-keep", "-1"), ("--seed", "-2"),
        ("--repeat-penalty", "0"), ("--repeat-last-n", "-1"),
        ("--frequency-penalty", "inf"), ("--presence-penalty", "nan"),
    )
    for command in (["prompt"], ["chat"], ["serve"], ["serve", "llm"]):
        for flag, value in invalid_sampling:
            result = subprocess.run([binary, *command, flag, value],
                                    text=True, capture_output=True, timeout=30)
            if result.returncode != 2:
                raise AssertionError(f"invalid sampling accepted: {command} {flag}")
    for command in (["serve"], ["serve", "llm"]):
        result = subprocess.run([binary, *command, "--cpu"],
                                text=True, capture_output=True, timeout=30)
        if result.returncode != 2 or "--cpu" in subprocess.run(
                [binary, *command, "--help"], text=True, capture_output=True,
                timeout=30, check=True).stdout:
            raise AssertionError("unused serve CPU option remains exposed")
    for flag, value in (("--prompt", "ignored"), ("--file", "/dev/null"),
                        ("--no-display-prompt", None),
                        ("--raw", None)):
        result = subprocess.run([binary, "chat", flag, *([value] if value else [])],
                                text=True, capture_output=True, timeout=30)
        if result.returncode != 2:
            raise AssertionError(f"chat silently ignored {flag}")
    result = subprocess.run([binary, "bench", "--temperature", "0.8"],
                            text=True, capture_output=True, timeout=30)
    if result.returncode != 2:
        raise AssertionError("greedy benchmark silently accepted sampling")
    print("All gufo modules reject unsupported draft sampling; help and errors valid")


def run(binary, model, mode, backend, sampling=None):
    command = [binary, mode, "--model", model, "--verbose",
               *sampling_flags(sampling or {"temperature": 0}),
               "--max-tokens", "8", *backend]
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


def check_sampling(binary, model, draft, policies):
    for policy in policies:
        backend = (["--speculative-decoding", "dflash"] if policy == "fixed" else
                   ["--speculative", "dflash2"])
        backend += ["--dflash-model", draft,
                   "--draft-policy", policy]
        for config in SAMPLING_CASES:
            prompt = run(binary, model, "prompt", backend, config)
            chat = run(binary, model, "chat", backend, config)
            replay = run(binary, model, "chat", backend, config)
            if prompt[0] != chat[0] or chat != replay:
                raise AssertionError(f"seeded prompt/chat mismatch: {policy} {config}")
        print(f"{policy}: six sampling configurations, prompt and two chat turns replay")


def check_http(binary, model, draft, policies):
    prompt = "Continue the pattern red, blue, blue, red, blue, blue,"
    messages = [{"role": "user", "content": prompt}]
    defaults = SAMPLING_CASES[4]
    for policy in policies:
        # CLI and HTTP raw framing must use identical sampling settings.
        direct = subprocess.run(
            [binary, "prompt", "--model", model, "--raw", "--verbose",
             "--max-tokens", "8", *sampling_flags(defaults),
             "--speculative", "dflash2", "--dflash-model", draft,
             "--draft-policy", policy, prompt],
            text=True, capture_output=True, timeout=180, check=True)
        completion = direct.stdout.split("--- Generation Output ---\n", 1)[1]
        completion = re.split(r"Generated \d+ tokens on GPU in ", completion)[0]
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        # The installed compatibility executable accepts server flags directly.
        alias = Path(binary).with_name("gufo-server")
        server_command = ([str(alias)] if policy == "fixed" and alias.is_file()
                          else [binary, "serve"])
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [*server_command, "--host", "127.0.0.1", "--port", str(port),
                 *(["llm"] if policy == "adaptive" else []),
                 "--model", model, "--context", "512", "--sessions", "2",
                 "--served-model-name", "qwen27b-wiring-test",
                 "--max-tokens", "8", *sampling_flags(defaults),
                 "--speculative", "dflash2", "--dflash-model", draft,
                 "--draft-policy", policy], stdout=log, stderr=log)

            def request(path, body=None, expected=200):
                if body is not None:
                    body = {"model": "qwen27b-wiring-test", **body}
                data = None if body is None else json.dumps(body).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}", data=data,
                    headers={"Content-Type": "application/json"})
                try:
                    response = urllib.request.urlopen(req, timeout=90)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    payload = response.read().decode()
                    if response.status != expected:
                        raise AssertionError(f"{path}: {response.status} {payload}")
                    if payload.startswith("data:"):
                        return [json.loads(line[6:]) for line in payload.splitlines()
                                if line.startswith("data: ") and line != "data: [DONE]"]
                    return json.loads(payload)

            try:
                deadline = time.monotonic() + 120
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        raise AssertionError(f"server startup failed: {log.read()}")
                    try:
                        request("/health")
                        break
                    except (urllib.error.URLError, ConnectionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("server readiness")
                        time.sleep(0.1)
                if request("/v1/models")["data"][0]["id"] != "qwen27b-wiring-test":
                    raise AssertionError("served model name was not wired")
                defaulted = request("/v1/completions", {"prompt": prompt})
                explicit = request("/v1/completions", {"prompt": prompt, **defaults})
                if defaulted["choices"] != explicit["choices"]:
                    raise AssertionError("serve CLI sampling defaults were lost")
                if completion.removesuffix("\n") != explicit["choices"][0]["text"]:
                    raise AssertionError("prompt and actual HTTP sampling differ")
                for config in SAMPLING_CASES:
                    # Reset omitted controls rather than inheriting the intentionally
                    # nontrivial serve defaults.
                    sampling = dict(temperature=0, top_k=0, top_p=1, min_p=0,
                                    min_keep=0, seed=73, repeat_penalty=1,
                                    repeat_last_n=64, frequency_penalty=0, presence_penalty=0)
                    sampling.update(config)
                    raw = request("/v1/completions", {"prompt": prompt, **sampling})
                    replay = request("/v1/completions", {"prompt": prompt, **sampling})
                    text = raw["choices"][0]["text"]
                    if raw["choices"] != replay["choices"] or raw["usage"]["draft_tokens"] <= 0:
                        raise AssertionError(f"HTTP sampling bypassed drafts or failed replay: {config}")
                    for path, payload in (("/completion", {"prompt": prompt}),
                                          ("/infill", {"input_prefix": prompt})):
                        if request(path, {**payload, **sampling})["content"] != text:
                            raise AssertionError(f"{path} sampling differs from completions")
                    chat = request("/v1/chat/completions", {"messages": messages, **sampling})
                    chat_text = chat["choices"][0]["message"]["content"]
                    anthropic = request("/v1/messages", {"messages": messages, **sampling})
                    responses = request("/v1/responses", {"input": prompt, **sampling})
                    if (anthropic["content"][0]["text"] != chat_text or
                            responses["output"][0]["content"][0]["text"] != chat_text):
                        raise AssertionError("chat API adapters use different sampling")
                streamed = request("/v1/chat/completions", {
                    "messages": messages, **sampling, "stream": True,
                    "stream_options": {"include_usage": True}})
                streamed_text = "".join(choice.get("delta", {}).get("content", "")
                                        for chunk in streamed for choice in chunk.get("choices", []))
                if streamed_text != chat_text:
                    raise AssertionError("streaming changed sampled output")
                concurrent_bodies = [
                    {"prompt": prompt + " This is request A.", **sampling},
                    {"prompt": prompt + " This is request B.", **defaults},
                ]
                sequential = [request("/v1/completions", body)["choices"]
                              for body in concurrent_bodies]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    concurrent = list(pool.map(
                        lambda body: request("/v1/completions", body)["choices"],
                        concurrent_bodies))
                if concurrent != sequential:
                    raise AssertionError("C2 requests mixed sampling state")
                for path, payload in (
                    ("/v1/completions", {"prompt": prompt}),
                    ("/v1/chat/completions", {"messages": messages}),
                    ("/v1/responses", {"input": prompt}),
                    ("/v1/messages", {"messages": messages}),
                    ("/completion", {"prompt": prompt}),
                    ("/infill", {"input_prefix": prompt}),
                ):
                    for field, value in (("draft_temperature", 0.8), ("top_p", 0),
                                         ("temperature", -1), ("seed", 1.5)):
                        request(path, {**payload, field: value}, expected=400)
                print(f"{policy}: serve defaults, six HTTP adapters, sampling replay, SSE and C2 exact")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def check_bench(binary, model, draft):
    common = [binary, "bench", "--model", model, "--verbose",
              "-p", "16", "-n", "8", "-d", "0,32", "-r", "2"]
    baseline = None
    backends = [[]] + [
        ["--speculative", "dflash-2" if policy == "fixed" else "dflash2",
         "--dflash-model", draft,
         "--draft-policy", policy] for policy in ("fixed", "adaptive")
    ]
    for backend in backends:
        result = subprocess.run(common + backend, text=True, capture_output=True,
                                timeout=180, check=True)
        traces = BENCH_TRACE.findall(result.stderr)
        if [(int(depth), int(count)) for depth, count, _ in traces] != [
                (0, 8), (0, 8), (32, 8), (32, 8)]:
            raise AssertionError(f"incomplete benchmark: {result.stderr}")
        if traces[0] != traces[1] or traces[2] != traces[3]:
            raise AssertionError(f"benchmark prefix restore changed token IDs: {backend} {traces}")
        if baseline is None:
            baseline = traces
        elif baseline != traces:
            raise AssertionError("DFlash2 benchmark diverged from AR")
    invalid = subprocess.run(
        [binary, "bench", "--model", model, "-p", "16", "-n", "0",
         "--speculative", "dflash2", "--dflash-model", "/dev/null"],
        text=True, capture_output=True, timeout=180)
    if invalid.returncode == 0 or "DFlash2 initialization failed" not in invalid.stderr:
        raise AssertionError("prefill-only benchmark ignored the DFlash2 artifact")
    print("DFlash2 benchmark: prefill, cached depth and repeated TG match AR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--scope", choices=("all", "options", "sampling", "http"),
                        default="all")
    parser.add_argument("--policy", choices=("fixed", "adaptive"))
    args = parser.parse_args()
    binary = args.binary
    policies = [args.policy] if args.policy else ["fixed", "adaptive"]
    if args.scope in ("all", "options"):
        check_options(binary)
    if args.scope == "options":
        return 0
    artifacts = [os.environ.get("GUFO_QWEN27B_" + name + "_MODEL", "")
                 for name in ("MTP", "DFLASH")]
    model = os.environ.get("GUFO_QWEN27B_MODEL", "")
    required = [model, artifacts[1]] + ([artifacts[0]] if args.scope == "all" else [])
    if not all(path and Path(path).is_file() for path in required):
        print("Qwen27B CLI test needs target, MTP and DFlash model artifacts")
        return 77
    if args.scope in ("all", "sampling"):
        check_sampling(binary, model, artifacts[1], policies)
    if args.scope in ("all", "http"):
        check_http(binary, model, artifacts[1], policies)
    if args.scope != "all":
        return 0
    baseline = None
    for backend in (
        [],
        ["--speculative", "mtp", "--mtp-model", artifacts[0]],
        ["--speculative", "dflash2", "--dflash-model", artifacts[1],
         "--draft-policy", "fixed"],
        ["--speculative", "dflash2", "--dflash-model", artifacts[1],
         "--draft-policy", "adaptive"],
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
    check_bench(binary, model, artifacts[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())

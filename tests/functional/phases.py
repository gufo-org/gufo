#!/usr/bin/env python3
"""Time Flash-Next startup and request phases on the merge base and the PR.

Each arm starts one server, records its startup phases, sends a fixed three
request history and stops it. Arms alternate base/PR on the same runner after
the GPU cools, and every phase gets a paired verdict: a change counts only when
each repeat moves by more than 5% and 3 ms in the same direction.
"""

import argparse
import contextlib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import NOISE_MS, SLOWDOWN  # noqa: E402

GENERATED_TOKENS = 128
CONTEXT = 32768
COOL_TIMEOUT_S = 240
WORDS = ("river", "stone", "garden", "signal", "harbor", "window", "lantern", "meadow",
         "copper", "orbit", "violet", "canyon", "thunder", "pocket", "mirror", "basket",
         "forest", "engine", "silver", "candle", "planet", "ladder", "marble", "feather",
         "island", "rocket", "pepper", "shadow", "velvet", "glacier", "button", "compass")
# Raw prompts keep the history identical on both revisions; the resume request
# extends the short prompt, not its generated text, which may differ.
REQUESTS = (
    ("short", "Short prompt", (1, 1500)),
    ("long", "Long prompt", (2, 12000)),
    ("resume", "Follow-up", (1, 1500), (3, 400)),
)
STARTUP = (
    ("init", "Process init"),
    ("weights", "Weights and model"),
    ("vision", "Vision sidecar"),
    ("structured", "Structured output"),
    ("listen", "HTTP ready"),
)
REQUEST_PHASES = (
    ("cache_restore_ms", "Cache restore"),
    ("prompt_ms", "Prefill"),
    ("predicted_ms", "Decode"),
    ("cache_snapshot_ms", "Cache snapshot"),
)


def prompt(*parts):
    words = []
    for seed, count in parts:
        rng = random.Random(seed)
        words += [rng.choice(WORDS) for _ in range(count)]
    return " ".join(words)


def gpu_temperature():
    """Return the hottest amdgpu hwmon reading in °C, or None."""
    readings = []
    for device in Path("/sys/class/drm").glob("card*/device"):
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
        except OSError:
            continue
        for sensor in device.glob("hwmon/hwmon*/temp*_input"):
            with contextlib.suppress(OSError, ValueError):
                readings.append(int(sensor.read_text()) / 1000)
    return max(readings) if readings else None


def cool_down(target):
    """Wait until the GPU is back near its idle temperature; return the reading."""
    deadline = time.monotonic() + COOL_TIMEOUT_S
    temperature = gpu_temperature()
    while temperature is not None and temperature > target and time.monotonic() < deadline:
        time.sleep(5)
        temperature = gpu_temperature()
    return temperature


class Server:
    """One serve process whose loader lines are stamped on arrival."""

    def __init__(self, binary, weights, log_path, command=None):
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            self.port = reserve.getsockname()[1]
        if command is None:
            self.command = [
                str(binary), "serve", "--host", "127.0.0.1", "--port", str(self.port), "llm",
                "--model", str(weights / "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
                "--mmproj", str(weights / "mmproj-BF16.gguf"),
                "--mtp-model", str(weights / "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
                "--speculative", "mtp", "--context", str(CONTEXT), "--sessions", "1",
                "--served-model-name", "phases"]
        if command is not None:
            self.command = list(command) + ['--host', '127.0.0.1', '--port', str(self.port)]
        self.log_path = log_path
        self.events = []

    def _read(self, stream, log):
        for line in stream:
            stamp = time.monotonic()
            log.write(line)
            if "[loader]" in line:
                self.events.append((stamp, line))

    def start(self, timeout):
        self.log = self.log_path.open("w")
        self.spawned = time.monotonic()
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, errors="replace")
        self.reader = threading.Thread(target=self._read, args=(self.process.stdout, self.log))
        self.reader.start()
        deadline = self.spawned + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited {self.process.returncode}; see {self.log_path}")
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
            try:
                connection.request("GET", "/health")
                if connection.getresponse().status == 200:
                    self.ready = time.monotonic()
                    return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
            time.sleep(.01)
        raise TimeoutError(f"server startup timed out; see {self.log_path}")

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(2)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=10)
        self.log.close()

    def startup(self):
        """Split spawn-to-ready time at the loader's own markers."""
        fields, stamps = {}, {}
        for stamp, line in self.events:
            event = re.search(r"event=(\S+)", line)
            if not event:
                continue
            stamps.setdefault(event.group(1), stamp)
            for key, value in re.findall(r"(\w+_ms)=([0-9.]+)", line):
                fields[event.group(1) + "." + key] = float(value)
        if "load_started" not in stamps or "load_completed" not in stamps:
            raise RuntimeError(f"loader markers missing; see {self.log_path}")
        vision = (fields.get("vision_prepared.identity_ms", 0)
                  + fields.get("vision_prepared.upload_ms", 0))
        structured = fields.get("structured_output_prepared.prepare_ms", 0)
        load = fields["load_completed.elapsed_ms"]
        return {
            "init": (stamps["load_started"] - self.spawned) * 1000,
            "weights": load - vision - structured,
            "vision": vision if "vision_prepared.upload_ms" in fields else None,
            "structured": structured if "structured_output_prepared.prepare_ms" in fields else None,
            "listen": (self.ready - stamps["load_completed"]) * 1000,
        }

    def complete(self, text):
        body = json.dumps({"model": "phases", "prompt": text, "max_tokens": GENERATED_TOKENS,
                           "temperature": 0, "ignore_eos": True, "stream": False})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            connection.request("POST", "/v1/completions", body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            server_timing = response.getheader("Server-Timing") or ""
        finally:
            connection.close()
        if response.status != 200:
            raise RuntimeError(f"/v1/completions returned {response.status}: {payload}")
        timings, usage = payload["timings"], payload["usage"]
        for key, _ in REQUEST_PHASES:
            value = timings[key]
            if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
                raise ValueError('Missing/invalid server phase: ' + key)
        ttft = re.search(r"ttft;dur=([0-9.]+)", server_timing)
        result = {key: timings.get(key) for key, _ in REQUEST_PHASES}
        result.update(ttft_ms=float(ttft.group(1)) if ttft else None,
                      prompt_tokens=usage["prompt_tokens"],
                      cached_tokens=usage.get("cached_tokens", 0),
                      completion_tokens=usage["completion_tokens"],
                      draft_tokens=usage.get("draft_tokens", 0),
                      draft_tokens_accepted=usage.get("draft_tokens_accepted", 0),
                      text_sha256=hashlib.sha256(
                          payload["choices"][0]["text"].encode()).hexdigest())
        return result


def run_arm(revision, binary, weights, output, repeat, startup_timeout, model_id=None, assets=None):
    command = None
    if model_id is not None:
        from devlab_models import llm_command
        command = llm_command(model_id, binary, assets) + ['--served-model-name', 'phases']
    server = Server(binary, weights, output / f"{revision}-{repeat}.log", command)
    try:
        server.start(startup_timeout)
        requests = {}
        for key, _, *parts in REQUESTS:
            requests[key] = server.complete(prompt(*parts))
        return {"startup": server.startup(), "requests": requests}
    finally:
        server.stop()


def verdict(pairs):
    """Paired 5% and 3 ms rule; a direction must hold in every repeat."""
    flags = []
    for base, head in pairs:
        delta = head - base
        moved = abs(delta) > NOISE_MS and abs(delta) > SLOWDOWN * base
        flags.append(0 if not moved else (1 if delta > 0 else -1))
    if not flags:
        return "missing"
    if all(flag == 1 for flag in flags):
        return "slower"
    if all(flag == -1 for flag in flags):
        return "faster"
    return "noisy" if any(flags) else "same"


def summarize(arms):
    by = {revision: [arm for arm in arms if arm["revision"] == revision]
          for revision in ("base", "head")}
    phases = []

    def add(group, key, label, read):
        base = [read(arm) for arm in by["base"]]
        head = [read(arm) for arm in by["head"]]
        pairs = [(b, h) for b, h in zip(base, head) if b is not None and h is not None]
        phases.append({"group": group, "key": key, "label": label,
                       "base": base, "head": head, "verdict": verdict(pairs)})

    for key, label in STARTUP:
        add("startup", "startup." + key, label, lambda arm, key=key: arm["startup"][key])
    for request, _, *_ in REQUESTS:
        for key, label in REQUEST_PHASES:
            add(request, f"{request}.{key}", label,
                lambda arm, request=request, key=key: arm["requests"][request][key])
    # MTP decode time follows the accepted drafts; only identical text compares it.
    diverged = sorted({request for request, _, *_ in REQUESTS
                       if len({arm["requests"][request]["text_sha256"] for arm in arms}) > 1})
    return phases, diverged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-binary", type=Path, required=True)
    parser.add_argument("--head-binary", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="phases.json path")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('at least two paired repeats are required')
    if args.model_id and not args.assets:
        parser.error('--model-id requires --assets')
    if not args.model_id and not args.weights:
        parser.error('provide --weights or --model-id with --assets')
    args.logs.mkdir(parents=True, exist_ok=True)
    weights = args.weights.resolve() if args.weights else None
    assets = json.loads(args.assets.read_text()) if args.assets else None
    binaries = {"base": args.base_binary.resolve(), "head": args.head_binary.resolve()}

    # Let the GPU settle after the functional suites; later arms return near it.
    idle = gpu_temperature()
    if idle is not None:
        time.sleep(30)
        idle = gpu_temperature()
    target = None if idle is None else idle + 3
    started = time.monotonic()
    arms = []
    for repeat in range(args.repeats):
        for revision in ("base", "head"):
            temperature = cool_down(target) if target is not None else None
            print(f"Arm {revision} #{repeat + 1}: GPU {temperature} °C", flush=True)
            arm_started = time.monotonic()
            arm = run_arm(revision, binaries[revision], weights, args.logs, repeat,
                          args.startup_timeout, args.model_id, assets)
            arms.append({"revision": revision, "repeat": repeat, "gpu_temp_c": temperature,
                         "wall_s": time.monotonic() - arm_started, **arm})
    phases, diverged = summarize(arms)
    from devlab_models import registry
    mode = registry()['models'][args.model_id]['mode'] if args.model_id else 'mtp'
    for arm in arms:
        for request in arm['requests'].values():
            if request['completion_tokens'] != GENERATED_TOKENS or request['prompt_tokens'] <= 0:
                raise ValueError('phase workload did not perform the expected token work')
            if mode != 'off' and request['draft_tokens'] <= 0:
                raise ValueError('phase workload did not execute actual drafts')
    mismatched_work = set()
    for label, *_ in REQUESTS:
        work = {(arm['requests'][label]['prompt_tokens'], arm['requests'][label]['cached_tokens'],
                 arm['requests'][label]['completion_tokens']) for arm in arms}
        if len(work) != 1:
            mismatched_work.add(label)
    diverged = sorted(set(diverged) | mismatched_work)
    for phase in phases:
        if phase['group'] in mismatched_work or (
                phase['group'] in diverged and phase['key'].endswith('predicted_ms')):
            phase['verdict'] = 'noisy'
    result = {
        "schema": 1,
        "revisions": {"base": args.base_revision, "head": args.head_revision},
        "settings": {"repeats": args.repeats, "context": CONTEXT, "sessions": 1,
                     "speculative": mode, "generated_tokens": GENERATED_TOKENS,
                     "slowdown": SLOWDOWN, "noise_ms": NOISE_MS,
                     "idle_temp_c": idle, "requests": {key: label for key, label, *_ in REQUESTS}},
        "wall_s": time.monotonic() - started,
        "arms": arms, "phases": phases, "diverged": diverged,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"\n## Inference phases\n\n{len(arms)} arms in "
                          f"{result['wall_s'] / 60:.1f} min; idle GPU {idle} °C.\n\n")
            summary.write("| Phase | Base ms | PR ms | Verdict |\n| --- | ---: | ---: | --- |\n")
            for phase in phases:
                values = [", ".join("—" if v is None else f"{v:.1f}" for v in phase[side])
                          for side in ("base", "head")]
                summary.write(f"| {phase['key']} | {values[0]} | {values[1]} | "
                              f"{phase['verdict']} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

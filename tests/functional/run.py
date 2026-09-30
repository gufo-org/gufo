#!/usr/bin/env python3
"""Run SDK and continuation checks against an isolated Gufo text server.

Pass the production server command after --. Only loopback requests are made.
The runner owns its process and cache directory, and restarts it for disk checks.
No weights are downloaded. Model checks are manual, not hosted CI workloads.
"""

import argparse
import contextlib
import http.client
import json
import math
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import zlib
from metrics import compare, join_server_timings

TESTS = Path(__file__).resolve().parent
SUITES = ("responses", "stops", "conversation", "structured", "structured-limits",
          "tools", "auto-tools", "sampling-defaults", "sampling-ranges", "batch",
          "long-context", "cache")
SAMPLING = {
    "--temperature": ("temperature", float), "--top-p": ("top_p", float),
    "--top-k": ("top_k", int), "--min-p": ("min_p", float),
    "--min-keep": ("min_keep", int), "--seed": ("seed", int),
    "--presence-penalty": ("presence_penalty", float),
    "--frequency-penalty": ("frequency_penalty", float),
    "--repeat-penalty": ("repeat_penalty", float),
    "--repeat-last-n": ("repeat_last_n", int),
}


def option(command, name, default=None):
    values = []
    for index, arg in enumerate(command):
        if arg == name:
            if index + 1 == len(command):
                raise ValueError(f"{name} needs a value")
            values.append(command[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"duplicate {name}")
    return values[0] if values else default


def sampling_overrides(command):
    return {field: convert(option(command, flag))
            for flag, (field, convert) in SAMPLING.items()
            if option(command, flag) is not None}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare_lifecycle(comparison, baseline, candidate, slowdown, noise_ms):
    for key in ("startup_ms", "restart_ms"):
        if key == "restart_ms" and key not in baseline and key not in candidate \
                and not any(label.endswith("-disk") for report in (baseline, candidate)
                            for label in report.get("suites", {})):
            continue
        old, new = baseline.get(key), candidate.get(key)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (old, new)):
            raise ValueError(f"missing/invalid server {key}")
        regression = new - old > max(noise_ms, old * slowdown)
        comparison["measurements"].append({
            "case": "server", "metric": key, "baseline": old, "candidate": new,
            "change_percent": 100 * (new / old - 1), "regression": regression})
        if regression:
            comparison["status"] = "failed"


@contextlib.contextmanager
def server(command, log_path, startup_timeout):
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    with log_path.open("w") as log:
        process = None
        previous_handlers = {}
        try:
            previous_handlers = {signum: signal.signal(signum, interrupt)
                                 for signum in (signal.SIGTERM, signal.SIGHUP)}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
                connection = http.client.HTTPConnection(
                    "127.0.0.1", int(option(command, "--port")), timeout=1)
                try:
                    connection.request("GET", "/health")
                    if connection.getresponse().status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                time.sleep(.1)
            else:
                raise TimeoutError(f"server startup timed out; see {log_path}")
            yield process
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def image_fixture(path):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    pixels = b"".join(b"\0" + bytes((255, 0, 0)) * 64 for _ in range(64))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New report directory; existing results are never overwritten")
    parser.add_argument("--sampling-preset", choices=("qwen38", "deepseek4"), required=True)
    parser.add_argument("--suite", action="append", choices=("all", *SUITES),
                        help="Repeat to select focused suites; default: all")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--suite-timeout", type=float, default=900)
    qualification = parser.add_mutually_exclusive_group(required=True)
    qualification.add_argument("--baseline", type=Path,
                               help="Matched previous run directory; compare every HTTP request")
    qualification.add_argument("--record-baseline", action="store_true",
                               help="Explicitly capture the reference; no regression claim yet")
    parser.add_argument("--max-slowdown", type=float, default=.05,
                        help="Fractional per-step timing tolerance (default: 0.05)")
    parser.add_argument("--noise-ms", type=float, default=3,
                        help="Absolute per-step timing noise floor (default: 3 ms)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- ./result/bin/gufo serve llm --model PATH [server options]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1] != "serve" or "llm" not in command:
        parser.error("pass a gufo serve llm command after --")
    for reserved in ("--host", "--port", "--cache-disk", "--cache-disk-bytes",
                     "--cache-disk-staging-bytes"):
        if option(command, reserved) is not None:
            parser.error(f"the test runner owns {reserved}; omit it from the server command")
    if option(command, "--api-key") is not None:
        parser.error("omit --api-key for the isolated loopback test server")
    selected = args.suite or ["all"]
    if "all" in selected:
        if len(selected) != 1:
            parser.error("all cannot be combined with other suites")
        selected = [suite for suite in SUITES if suite != "auto-tools"]
    selected = list(dict.fromkeys(selected))
    try:
        overrides = sampling_overrides(command)
        sessions = int(option(command, "--sessions", "4"))
        if not 1 <= sessions <= 8:
            raise ValueError("use --sessions 1 through 8 for functional checks")
        if args.startup_timeout <= 0 or args.suite_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if not 0 <= args.max_slowdown < 1 or not 0 <= args.noise_ms < 1000:
            raise ValueError("use 0 <= max-slowdown < 1 and 0 <= noise-ms < 1000")
        if args.baseline and not (args.baseline / "report.json").is_file():
            raise ValueError("--baseline must contain report.json")
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command = [command[0], "serve", "--host", "127.0.0.1", "--port", str(port),
               *command[2:]]
    for flag, default in (("--sessions", "4"), ("--context", "8192"),
                          ("--served-model-name", "functional-test"),
                          ("--max-pending-per-client", str(max(4, sessions)))):
        if option(command, flag) is None:
            command += [flag, default]
    if "cache" in selected:
        command += ["--cache-disk", str(output / "disk"),
                    "--cache-disk-bytes", str(8 * 1024**3),
                    "--cache-disk-staging-bytes", str(1024**3)]
    model = option(command, "--served-model-name")
    vision = option(command, "--mmproj") is not None
    speculative = option(command, "--speculative",
                         "dspark" if option(command, "--dspark-model") else "off")
    base_url = f"http://127.0.0.1:{port}"
    comparison_command = list(command[1:])
    for flag in ("--port", "--cache-disk"):
        if flag in comparison_command:
            comparison_command[comparison_command.index(flag) + 1] = "<runner-owned>"
    report = {"command": command, "comparison_command": comparison_command,
              "binary": str(Path(command[0]).resolve()),
              "mode": "baseline" if args.record_baseline else "qualification",
              "sampling_preset": args.sampling_preset, "sampling_overrides": overrides,
              "vision": vision, "suites": {}, "status": "running"}
    report_path = output / "report.json"
    write_json(report_path, report)

    def run(label, script, arguments):
        started = time.monotonic()
        row = {"status": "running", "report": label + ".json"}
        report["suites"][label] = row
        write_json(report_path, report)
        try:
            with (output / (label + ".log")).open("w") as log:
                result = subprocess.run([sys.executable, str(TESTS / script), *arguments],
                    stdout=log, stderr=subprocess.STDOUT, timeout=args.suite_timeout)
            if result.returncode:
                raise RuntimeError(f"exit {result.returncode}; see {label}.log")
            payload = json.loads((output / (label + ".json")).read_text())
            passed = (isinstance(payload, dict) and payload.get("status") == "passed"
                      if script == "openai_sdk.py" else
                      isinstance(payload, list) and bool(payload)
                      and all(isinstance(row, dict) and row.get("exact") is True
                              for row in payload))
            if not passed:
                raise RuntimeError(f"test did not report success: {label}.json")
            measurements = json.loads((output / (label + ".requests.json")).read_text())
            rows = measurements.get("requests", [])
            if measurements.get("version") != 1 or not rows or any(
                    row.get("status") not in ("complete", "disconnected")
                    or type(row.get("wall_ms")) not in (int, float)
                    or not math.isfinite(row["wall_ms"]) or row["wall_ms"] < 0
                    or not row.get("endpoint")
                    or not row.get("request_sha256") for row in rows):
                raise RuntimeError(f"missing/invalid per-request measurements: {label}")
            row["status"] = "passed"
        except Exception as error:
            row.update(status="failed", error=str(error), traceback=traceback.format_exc())
        row["seconds"] = round(time.monotonic() - started, 3)
        write_json(report_path, report)
        print(f"{label}: {row['status']} ({row['seconds']}s)", flush=True)
        return row["status"] == "passed"

    cache_cases = [
        ("text-cancel", ["--case", "content-preserve0-sampled0", "--discard-assistant"]),
        ("thinking-tool-cancel", ["--case", "reasoning_content-preserve1-sampled1",
                                 "--reasoning-effort", "high", "--tools"]),
    ]
    if vision:
        image_fixture(output / "red.png")
        cache_cases += [
            ("image-cancel", ["--case", "content-preserve0-sampled0", "--discard-assistant",
                             "--image", str(output / "red.png"), "--append-image"]),
            ("image-thinking-cancel", ["--case", "reasoning_content-preserve1-sampled1",
                                      "--reasoning-effort", "low",
                                      "--image", str(output / "red.png")]),
        ]
    ready_to_restore = []
    try:
        startup_started = time.monotonic()
        with server(command, output / "server.log", args.startup_timeout):
            report["startup_ms"] = (time.monotonic() - startup_started) * 1000
            for suite in selected:
                if suite == "cache":
                    continue
                sdk_args = ["--base-url", base_url + "/v1", "--model", model,
                            "--suite", suite, "--output", str(output / (suite + ".json")),
                            "--sampling-preset", args.sampling_preset,
                            "--sampling-overrides", json.dumps(overrides),
                            "--concurrency", str(sessions), "--speculative", speculative,
                            "--context", option(command, "--context")]
                if vision:
                    sdk_args += ["--vision"]
                if option(command, "--think") is not None:
                    sdk_args += ["--server-thinking", option(command, "--think")]
                run(suite, "openai_sdk.py", sdk_args)
            if "cache" in selected:
                for label, extra in cache_cases:
                    if run(label, "continuation.py", [
                            "--url", base_url, "--model", model, "--prefix-repetitions", "16",
                            "--output", str(output / (label + ".json")), *extra]):
                        ready_to_restore.append(label)
        if ready_to_restore:
            startup_started = time.monotonic()
            with server(command, output / "server-restarted.log", args.startup_timeout):
                report["restart_ms"] = (time.monotonic() - startup_started) * 1000
                for label in ready_to_restore:
                    run(label + "-disk", "continuation.py", [
                        "--url", base_url, "--model", model,
                        "--restore", str(output / (label + ".json")),
                        "--output", str(output / (label + "-disk.json"))])
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        write_json(report_path, report)
        raise
    report["status"] = ("passed" if report["suites"] and "error" not in report
                        and all(row["status"] == "passed" for row in report["suites"].values())
                        else "failed")
    try:
        join_server_timings(output)
    except (ValueError, OSError) as error:
        report.update(status="failed", measurements_error=str(error))
    if args.baseline:
        try:
            baseline = json.loads((args.baseline / "report.json").read_text())
            if baseline["status"] != "passed":
                raise ValueError("baseline functional checks did not pass")
            for key in ("comparison_command", "sampling_preset", "sampling_overrides", "vision"):
                if baseline.get(key) != report.get(key):
                    raise ValueError(f"unmatched baseline {key}")
            if list(baseline["suites"]) != list(report["suites"]):
                raise ValueError("unmatched baseline suites")
            comparison = compare(args.baseline, output, args.max_slowdown, args.noise_ms)
            compare_lifecycle(comparison, baseline, report, args.max_slowdown, args.noise_ms)
            write_json(output / "comparison.json", comparison)
            report["comparison"] = comparison["status"]
            if comparison["status"] != "passed":
                report["status"] = "failed"
        except (ValueError, KeyError, OSError) as error:
            report.update(status="failed", comparison_error=str(error))
    write_json(report_path, report)
    print(f"{report['mode']} {report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

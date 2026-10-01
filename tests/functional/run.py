#!/usr/bin/env python3
"""Run SDK and continuation checks against an isolated Gufo text server.

Pass the production server command after --. Only loopback requests are made.
The runner owns its process and cache directory, and restarts it for disk checks.
No weights are downloaded. Model checks are manual, not hosted CI workloads.
"""

import argparse
import contextlib
import http.client
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import zlib
from metrics import compare, comparison_status, join_server_timings, timing_measurement

TESTS = Path(__file__).resolve().parent
SUITES = ("discovery", "responses", "stops", "conversation", "image-inputs", "structured", "structured-limits",
          "tool-reasoning",
          "tools", "auto-tools", "tool-edges", "tool-agent", "tool-agent-loop", "tool-history", "tool-untyped", "tool-mixed", "tool-schema-edges", "sampling-defaults", "sampling-ranges", "batch",
          "long-context", "state-edges", "progress", "metrics", "cache")
SAMPLING = {
    "--temperature": ("temperature", float), "--top-p": ("top_p", float),
    "--top-k": ("top_k", int), "--min-p": ("min_p", float),
    "--min-keep": ("min_keep", int), "--seed": ("seed", int),
    "--presence-penalty": ("presence_penalty", float),
    "--frequency-penalty": ("frequency_penalty", float),
    "--repeat-penalty": ("repeat_penalty", float),
    "--repeat-last-n": ("repeat_last_n", int),
}
COMPARISON_FIELDS = ("comparison_command", "sampling_preset", "sampling_overrides",
                     "vision", "speculative", "environment", "harness_sha256", "build_inputs_sha256",
                     "expected_input_modalities")


def provenance():
    source = hashlib.sha256()
    for name in ("run.py", "metrics.py", "progress.py", "server_metrics.py", "openai_sdk.py", "continuation.py",
                 "tool_reasoning.py", "tool_agent.py", "discovery.py", "image_inputs.py"):
        source.update((TESTS / name).read_bytes())
    lock = TESTS.parents[1] / "flake.lock"
    kernel_command = Path("/proc/cmdline")
    return {
        "harness_sha256": source.hexdigest(),
        "build_inputs_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "environment": {"host": platform.node(), "kernel": platform.release(),
                        "kernel_command_line": (kernel_command.read_text().strip()
                                                if kernel_command.is_file() else None), **{
            key: os.environ.get(key) for key in (
                "GPU_MAX_HW_QUEUES", "HSA_ENABLE_SDMA", "HSA_ENABLE_INTERRUPT",
                "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_PRELOAD")}},
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


def execution_coverage(directory, requested):
    """Verify the loaded mode and actual draft work, including after restart."""
    logs = [directory / "server.log"]
    if (directory / "server-restarted.log").is_file():
        logs.append(directory / "server-restarted.log")
    for path in logs:
        modes = [match.group(1) for line in path.read_text().splitlines()
                 if "event=load_completed" in line and "kind=text" in line
                 for match in [re.search(r"\bspeculative=(\w+)", line)] if match]
        if modes != [requested]:
            raise ValueError(f"{path.name}: requested {requested}, loaded modes {modes}")
    proposed, accepted = 0, 0
    for path in directory.glob("*.requests.json"):
        for row in json.loads(path.read_text())["requests"]:
            metrics = row.get("metrics", {})
            if "draft_tokens" not in metrics and not metrics.get("completion_tokens"):
                continue  # Rejected requests perform no model work.
            p, a = metrics.get("draft_tokens"), metrics.get("draft_tokens_accepted")
            if type(p) is not int or type(a) is not int or not 0 <= a <= p:
                raise ValueError(f"{path.name}: missing/invalid speculative counters")
            if requested == "off" and p:
                raise ValueError(f"{path.name}: AR request executed speculative drafts")
            proposed += p
            accepted += a
    return {"loaded_mode": requested, "draft_tokens": proposed,
            "draft_tokens_accepted": accepted, "draft_execution_observed": proposed > 0}


def compare_lifecycle(comparison, baseline, candidate):
    for key in ("startup_ms", "restart_ms"):
        if key == "restart_ms" and key not in baseline and key not in candidate \
                and not any(label.endswith("-disk") for report in (baseline, candidate)
                            for label in report.get("suites", {})):
            continue
        old, new = baseline.get(key), candidate.get(key)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (old, new)):
            raise ValueError(f"missing/invalid server {key}")
        comparison["measurements"].append(timing_measurement("server", key, old, new))
    comparison["status"] = comparison_status(
        comparison["measurements"], comparison.get("quality_or_coverage_changes", []))


def compare_runs(baseline_dir, candidate_dir):
    baseline = json.loads((baseline_dir / "report.json").read_text())
    candidate = json.loads((candidate_dir / "report.json").read_text())
    for report in (baseline, candidate):
        if report.get("functional_status", report["status"]) != "passed":
            raise ValueError("functional checks did not pass")
    for key in (*COMPARISON_FIELDS, "through_case"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"unmatched baseline {key}")
    if list(baseline["suites"]) != list(candidate["suites"]):
        raise ValueError("unmatched baseline suites")
    comparison = compare(baseline_dir, candidate_dir)
    compare_lifecycle(comparison, baseline, candidate)
    comparison["runs"] = [str(baseline_dir.resolve()), str(candidate_dir.resolve())]
    return comparison


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
    parser.add_argument("--suite", action="append", choices=("all", *SUITES), required=True,
                        help="Repeat to select affected suites; use all for an explicit full run")
    parser.add_argument("--expected-input-modalities", choices=("text", "text,image"),
                        help="Expected loaded inputs for the discovery suite")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--suite-timeout", type=float, default=900)
    qualification = parser.add_mutually_exclusive_group(required=True)
    qualification.add_argument("--baseline", type=Path,
                               help="Matched previous run directory; compare every HTTP request")
    qualification.add_argument("--record-baseline", action="store_true",
                               help="Explicitly capture the reference; no regression claim yet")
    parser.add_argument("--through-case", metavar="SUITE:CASE",
                        help="Replay preceding suites/cases, then stop; focused investigation only")
    parser.add_argument("--allow-missing-progress", action="store_true",
                        help="Baseline only: older revisions without progress events")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- ./result/bin/gufo serve llm --model PATH [server options]")
    args = parser.parse_args()
    if ("discovery" in args.suite or "all" in args.suite) and args.expected_input_modalities is None:
        parser.error("discovery requires --expected-input-modalities text or text,image")
    if args.allow_missing_progress and not args.record_baseline:
        parser.error("--allow-missing-progress is only for --record-baseline")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1] != "serve" or "llm" not in command:
        parser.error("pass a gufo serve llm command after --")
    for reserved in ("--host", "--port", "--cache-disk", "--cache-disk-bytes",
                     "--cache-disk-staging-bytes"):
        if option(command, reserved) is not None:
            parser.error(f"the test runner owns {reserved}; omit it from the server command")
    if option(command, "--api-key") is not None:
        parser.error("omit --api-key for the isolated loopback test server")
    if "image-inputs" in args.suite and option(command, "--mmproj") is None:
        parser.error("image-inputs requires --mmproj in the server command")
    selected = args.suite
    if "all" in selected:
        if len(selected) != 1:
            parser.error("all cannot be combined with other suites")
        selected = [suite for suite in SUITES if suite != "auto-tools"
                    and (suite != "image-inputs" or option(command, "--mmproj") is not None)]
    selected = list(dict.fromkeys(selected))
    disk_enabled = "cache" in selected
    try:
        overrides = sampling_overrides(command)
        sessions = int(option(command, "--sessions", "4"))
        if not 1 <= sessions <= 8:
            raise ValueError("use --sessions 1 through 8 for functional checks")
        if args.startup_timeout <= 0 or args.suite_timeout <= 0:
            raise ValueError("timeouts must be positive")
        through_suite, through_case = (args.through_case.split(":", 1)
                                      if args.through_case else (None, None))
        if args.through_case:
            if through_suite not in selected or through_suite == "cache" or not through_case:
                raise ValueError("--through-case requires a selected SDK suite and case name")
            selected = selected[:selected.index(through_suite) + 1]
            selected = [suite for suite in selected if suite != "cache"]
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
    if disk_enabled:
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
    report = {**provenance(), "command": command, "comparison_command": comparison_command,
              "binary": str(Path(command[0]).resolve()),
              "mode": "baseline" if args.record_baseline else "qualification",
              "sampling_preset": args.sampling_preset, "sampling_overrides": overrides,
              "vision": vision, "speculative": speculative,
              "suites": {}, "through_case": args.through_case,
              "allow_missing_progress": args.allow_missing_progress,
              "started_ns": time.time_ns(), "status": "running"}
    report["expected_input_modalities"] = args.expected_input_modalities
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
        ("legacy-tool-cancel", ["--case", "content-preserve1-sampled0",
                               "--tools", "--legacy-tool-history"]),
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
                if args.expected_input_modalities is not None:
                    sdk_args += ["--expected-input-modalities", args.expected_input_modalities]
                if option(command, "--think") is not None:
                    sdk_args += ["--server-thinking", option(command, "--think")]
                if suite == through_suite:
                    sdk_args += ["--through-case", through_case]
                if args.allow_missing_progress:
                    sdk_args += ["--allow-missing-progress"]
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
        report["execution"] = execution_coverage(output, speculative)
        if ("metrics" in selected and through_suite != "metrics"
                and report["suites"]["metrics"]["status"] == "passed"):
            from server_metrics import validate_metrics_report
            report["metric_accounting"] = validate_metrics_report(output)
    except (ValueError, OSError, KeyError, AssertionError) as error:
        report.update(status="failed", measurements_error=str(error))
    report["functional_status"] = report["status"]
    report["completed_ns"] = time.time_ns()
    write_json(report_path, report)
    if args.baseline:
        try:
            comparison = compare_runs(args.baseline, output)
            write_json(output / "comparison.json", comparison)
            report["comparison"] = comparison["status"]
            report["status"] = comparison["status"]
        except (ValueError, KeyError, OSError) as error:
            report.update(status="failed", comparison_error=str(error))
    write_json(report_path, report)
    print(f"{report['mode']} {report['status']}: {report_path}", flush=True)
    return {"passed": 0, "failed": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Opt-in live device-failure contracts; injects HIP errors, never GPU resets."""

import argparse
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time


CASES = {
    "recoverable": ("usable", "/v1/completions", False),
    "probe-timeout": ("timeout", "/v1/completions", False),
    "loss-buffered": ("loss", "/v1/completions", False),
    "loss-stream-completions": ("loss", "/v1/completions", True),
    "loss-stream-chat": ("loss", "/v1/chat/completions", True),
    "loss-stream-responses": ("loss", "/v1/responses", True),
    "idle-loss": ("loss", "/v1/completions", False),
    "idle-pending-arrival": ("timeout", "/v1/completions", False),
    "loss-after-token-chat": ("loss", "/v1/chat/completions", True),
    "blocked-writer-watchdog": ("loss", "/v1/completions", True),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def request(port, method, path, body=None):
    start = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.request(
            method, path, None if body is None else json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return {
            "status": response.status,
            "headers": dict(response.getheaders()),
            "body": response.read().decode(),
            "wall_s": time.monotonic() - start,
        }
    finally:
        connection.close()


def await_condition(process, predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while not predicate():
        require(process.poll() is None, f"server exited {process.returncode}: {message}")
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(0.02)


def generation(path, stream, model):
    text = "Write a long poem about the sea, the mountains, and the stars."
    common = {"model": model, "temperature": 0, "stream": stream}
    if path == "/v1/responses":
        return dict(common, input=text, max_output_tokens=128)
    if path == "/v1/chat/completions":
        return dict(common, messages=[{"role": "user", "content": text}], max_tokens=128)
    return dict(common, prompt=text, max_tokens=128, ignore_eos=True, cache_prompt=False)


def run_case(args, name):
    mode, path, stream = CASES[name]
    output = args.output / name
    output.mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = args.server + ["--host", "127.0.0.1", "--port", str(port)]
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(args.fault_library)
    environment["GUFO_TEST_DEVICE_LOSS_DIR"] = str(output)
    record = {"case": name, "command": command, "fault_library": str(args.fault_library)}
    log_path = output / "server.log"
    peer = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        try:
            def ready():
                try:
                    return request(port, "GET", "/ready")["status"] == 200
                except OSError:
                    return False

            await_condition(process, ready, args.startup_timeout, "readiness deadline")
            record["models"] = request(port, "GET", "/v1/models")
            model = json.loads(record["models"]["body"])["data"][0]["id"]
            record["warm"] = request(port, "POST", "/v1/completions", {
                "prompt": "Count from one to a hundred, spelling each number.",
                "max_tokens": 64, "ignore_eos": True, "temperature": 0,
                "cache_prompt": False,
            })
            require(record["warm"]["status"] == 200, "real warm generation failed")
            warm = json.loads(record["warm"]["body"])
            require(warm["usage"]["completion_tokens"] == 64, "warm generation incomplete")
            record["warm_draft_tokens"] = warm["timings"]["draft_n"]
            modes = [match.group(1) for line in log_path.read_text().splitlines()
                     if "event=load_completed" in line and "kind=text" in line
                     for match in [re.search(r"\bspeculative=(\w+)", line)] if match]
            expected_mode = args.server[args.server.index("--speculative") + 1]
            require(modes == [expected_mode], f"loaded mode mismatch: {modes}")
            record["loaded_speculative"] = modes[0]
            require(
                record["warm_draft_tokens"] == 0 if expected_mode == "off"
                else record["warm_draft_tokens"] > 0,
                "warm generation did not exercise the requested draft mode",
            )
            if mode != "usable":
                (output / mode).touch()
            body = generation(path, stream, model)
            record["request"] = {"path": path, "body": body}
            if name.startswith("idle-"):
                (output / "probe_only").touch()
                started = time.monotonic()
                (output / "armed").touch()
                if name == "idle-loss":
                    # No inference, health traffic or disconnect initiates exit.
                    record["exit_code"] = process.wait(timeout=13)
                    record["exit_after_arm_s"] = time.monotonic() - started
                    require(record["exit_after_arm_s"] <= 12,
                            "idle loss detection missed its bound")
                else:
                    await_condition(process, lambda: (output / "probed").exists(),
                                    12, "idle probe was never submitted")
                    # The probe remains pending while a real generation runs.
                    # A blocking implementation would add its five-second timeout.
                    record["arrival"] = request(port, "POST", path, {
                        "prompt": "Name a color.", "max_tokens": 4, "temperature": 0,
                    })
                    require(record["arrival"]["status"] == 200,
                            "pending idle probe blocked/poisoned arrival")
                    require(record["arrival"]["wall_s"] < 3,
                            "arrival waited for the pending probe timeout")
                    record["/health"] = request(port, "GET", "/health")
                    require(record["/health"]["status"] == 200,
                            "pending probe marked the device lost")
                    (output / "armed").unlink()
                    process.terminate()
                    record["exit_code"] = process.wait(timeout=13)
            elif name == "loss-after-token-chat":
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
                connection.request("POST", path, json.dumps(body),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                require(response.status == 200, "stream did not begin normally")
                pieces = []
                while True:
                    line = response.readline().decode()
                    require(line, "stream ended before content")
                    pieces.append(line)
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        delta = event.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            break
                started = time.monotonic()
                (output / "armed").touch()
                pieces.append(response.read().decode())
                connection.close()
                record["failure"] = {"status": response.status, "body": "".join(pieces)}
                require('"code":"device_lost"' in record["failure"]["body"],
                        "started stream did not emit terminal loss")
                record["exit_code"] = process.wait(timeout=13)
            elif name == "blocked-writer-watchdog":
                (output / "block_writer").touch()
                peer = socket.create_connection(("127.0.0.1", port), timeout=15)
                payload = json.dumps(body).encode()
                peer.sendall(
                    f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                    + payload
                )
                # The model runs independently of the HTTP writer. Arm only
                # after that writer is blocked; no subsequent health request
                # or client disconnect can initiate the shutdown.
                await_condition(process, lambda: (output / "writer_blocked").exists(),
                                15, "SSE writer never blocked")
                started = time.monotonic()
                (output / "armed").touch()
                record["exit_code"] = process.wait(timeout=13)
                record["exit_after_arm_s"] = time.monotonic() - started
                require(9 <= record["exit_after_arm_s"] <= 13,
                        "forced exit did not exercise the ten-second watchdog")
            else:
                started = time.monotonic()
                (output / "armed").touch()
                record["failure"] = request(port, "POST", path, body)
                response = record["failure"]
                if mode == "loss":
                    require("GPU context lost; restart required" in response["body"],
                            "missing stable device-loss message")
                    if stream:
                        # Prefill failed after admission committed the stream.
                        require(response["status"] == 200, "incorrect stream status")
                        require("data: " in response["body"],
                                "admitted stream did not emit a terminal SSE error")
                    else:
                        require(response["status"] == 503, "incorrect failure status")
                        require('"code":"device_lost"' in response["body"],
                                "missing stable error code")
                    require(not any(key.lower() == "retry-after" for key in response["headers"]),
                            "permanent loss advertised a retry")
                    record["exit_code"] = process.wait(timeout=max(0.1, 13 - (time.monotonic() - started)))
                    record["exit_after_arm_s"] = time.monotonic() - started
                else:
                    require(response["status"] == 500, "recoverable failure changed status")
                    for health in ("/health", "/ready"):
                        record[health] = request(port, "GET", health)
                        require(record[health]["status"] == 200, "usable device became unhealthy")
                    (output / "armed").unlink()
                    record["recovery"] = request(port, "POST", "/v1/completions", {
                        "prompt": "Name a color.", "max_tokens": 4, "temperature": 0,
                    })
                    require(record["recovery"]["status"] == 200, "real recovery generation failed")
                    if mode == "timeout":
                        require(response["wall_s"] >= 5 and
                                "event=device_probe_timeout" in log_path.read_text(),
                                "pending probe did not preserve the original error")
                    process.terminate()
                    record["exit_code"] = process.wait(timeout=13)

            require((output / "injected").exists() and (output / "probed").exists(),
                    "failure/probe hooks were not exercised")
            if mode == "loss":
                require(record["exit_code"] == 75, "expected EX_TEMPFAIL (75)")
                text = log_path.read_text()
                require(text.count("event=device_lost remedy=") == 1 and
                        text.count("event=device_lost_shutdown ") == 1,
                        "loss and shutdown must be logged exactly once")
                require(not (output / "cleanup_attempted").exists(),
                        "dead-context cleanup attempted before process exit")
            else:
                require(record["exit_code"] == 0, "healthy shutdown failed")
            record["status"] = "pass"
        except Exception as error:
            record["status"] = "fail"
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if peer is not None:
                peer.close()
            if process.poll() is None:
                process.kill()
                process.wait()
            record["observed_exit_code"] = process.returncode
            (output / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: {record['status']}" + (f" ({record['error']})" if "error" in record else ""), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("server", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.server and args.server[0] == "--":
        args.server.pop(0)
    if len(args.server) < 3 or args.server[1:3] != ["serve", "llm"]:
        parser.error("provide a production gufo serve llm command after --")
    if "--speculative" not in args.server:
        parser.error("select --speculative explicitly so loaded mode can be verified")
    for option in ("--host", "--port", "--api-key", "--cache-disk"):
        if any(value == option or value.startswith(option + "=") for value in args.server):
            parser.error(f"the isolated fault runner does not accept {option}")
    args.fault_library = args.fault_library.resolve()
    if not args.fault_library.is_file():
        parser.error("fault library does not exist; build the device_loss_faults CMake target")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "fault_injection": "process-local HIP errors and stalled HTTP send; no physical GPU reset",
        "server_version": subprocess.check_output([args.server[0], "--version"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fault_library_sha256": hashlib.sha256(args.fault_library.read_bytes()).hexdigest(),
        "cases": [run_case(args, name) for name in (args.case or CASES)],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(case["status"] == "pass" for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

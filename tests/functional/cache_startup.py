#!/usr/bin/env python3
"""Qwen startup identity, RAM reuse and durable restart checks.

Each run owns its server, disk tier and initially empty XDG cache. Select AR or
MTP or DFlash2 in the production server command. No weights are downloaded or modified.
"""

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import socket
import time
import traceback

import continuation
from cache_lifecycle import published_tokens, wait_until
from cache_workloads import history
from metrics import Recorder, join_server_timings
from run import execution_coverage, option, provenance, server, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--model-kind", choices=("flash-next", "qwen27b"),
                        default="flash-next")
    parser.add_argument("--case", action="append", choices=("ram", "disk"),
                        help="Default: RAM startup, then disk startup/restart")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1] != "serve" or "llm" not in command:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port", "--api-key", "--cache-disk",
                 "--cache-disk-bytes", "--cache-disk-staging-bytes"):
        if option(command, flag) is not None:
            parser.error(f"the runner owns {flag}; omit it")
    mode = option(command, "--speculative", "off")
    modes = ("off", "mtp") if args.model_kind == "flash-next" else ("off", "dflash2")
    if mode not in modes:
        parser.error(f"{args.model_kind} requires one of {modes}")
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command += ["--host", "127.0.0.1", "--port", str(port)]
    for flag, default in (("--sessions", "1"), ("--context", "8192"),
                          ("--served-model-name", "cache-startup")):
        if option(command, flag) is None:
            command += [flag, default]
    if int(option(command, "--context")) < 8192:
        parser.error("use a context of at least 8192 tokens")
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(output / "xdg-cache")
    identities = output / "xdg-cache" / "gufo" / "gguf-sha256-v1"
    report = {**provenance(), "command": command,
              "binary": str(Path(command[0]).resolve()),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_kind": args.model_kind, "speculative": mode,
              "stages": {}, "status": "running"}
    report_path = output / "report.json"
    body = {
        "model": option(command, "--served-model-name"),
        "messages": history("startup_identity", lines=160, code=(
            "ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA IOTA KAPPA")),
        "temperature": 0, "seed": 31, "max_completion_tokens": 64,
        "reasoning_effort": "none",
    }
    expected = "ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA IOTA KAPPA"
    base_url = f"http://127.0.0.1:{port}"
    stage_path = None
    stage_report = None

    @contextmanager
    def stage(name, disk=False):
        nonlocal stage_path, stage_report
        stage_path = output / name
        stage_path.mkdir()
        continuation.TRACE = Recorder(stage_path / "requests.requests.json")
        launch = list(command)
        if disk:
            launch += ["--cache-disk", str(output / "cache"),
                       "--cache-disk-bytes", str(2 * 1024**3)]
        stage_report = {"command": launch, "requests": {}, "status": "running"}
        report["stages"][name] = stage_report
        write_json(report_path, report)
        started = time.monotonic()
        try:
            with server(launch, stage_path / "server.log",
                        args.startup_timeout, environment):
                stage_report["startup_ms"] = (time.monotonic() - started) * 1000
                log = (stage_path / "server.log").read_text()
                assert "schema=component-cache-v1 event=configured" in log, (
                    "requires the loaded Qwen component cache")
                hashed = [line.split("phase=artifact_identity model=", 1)[1]
                          for line in log.splitlines()
                          if "phase=artifact_identity model=" in line]
                expected_hashes = []
                if disk:
                    expected_hashes.append(
                        "Qwen3.8-Flash-Next" if args.model_kind == "flash-next" else "Qwen")
                    if mode != "off":
                        expected_hashes.append(
                            "Qwen3.8-Flash-Next MTP" if mode == "mtp" else "DFlash")
                assert hashed == expected_hashes, (
                    "RAM-only startup must not read model files for a durable "
                    "identity; disk startup must validate target and draft "
                    "identities", hashed, expected_hashes)
                if disk:
                    assert identities.is_dir() and any(identities.iterdir()), (
                        "disk startup did not cache its full artifact hashes")
                else:
                    assert not identities.exists(), (
                        "RAM-only startup populated the artifact hash cache")
                yield
            join_server_timings(stage_path)
            stage_report["execution"] = execution_coverage(stage_path, mode)
            stage_report["status"] = "passed"
        except BaseException:
            stage_report["status"] = "failed"
            raise
        finally:
            write_json(report_path, report)

    def chat(label, warm=False, disk=False):
        result = continuation.call(base_url, body)
        continuation.TRACE.mark(label)
        stage_report["requests"][label] = result
        write_json(report_path, report)
        choice = result["choices"][0]
        message = choice["message"]
        assert " ".join(message["content"].split()) == expected, result
        assert choice["finish_reason"] == "stop", result
        assert not message.get("reasoning_content") and not message.get("tool_calls"), result
        usage = result["usage"]
        total, cached = usage["prompt_tokens"], usage["cached_tokens"]
        assert cached + usage["gufo"]["prefill_tokens"] == total, result
        assert total >= 2048, "fixture must reach the disk checkpoint interval"
        if warm:
            assert cached >= total - 8, result
        else:
            assert cached == 0, result
        assert usage["gufo"]["cache_disk_hit"] == disk, result
        if mode != "off":
            assert usage.get("draft_tokens", 0) > 0, f"{mode} was not exercised"
        return result

    try:
        selected = set(args.case or ("ram", "disk"))
        if "ram" in selected:
            with stage("ram"):
                cold = chat("cold")
                warm = chat("retry", warm=True)
                assert continuation.digest(cold) == continuation.digest(warm)
        if "disk" in selected:
            with stage("disk", disk=True):
                cold = chat("cold")
                warm = chat("retry", warm=True)
                assert continuation.digest(cold) == continuation.digest(warm)
                wait_until(lambda: any(
                    tokens >= cold["usage"]["prompt_tokens"] - 8
                    for tokens in published_tokens(
                        (stage_path / "server.log").read_text())),
                    "the prompt's durable checkpoint before restart")
            with stage("restart", disk=True):
                restored = chat("restored", warm=True, disk=True)
                assert continuation.digest(cold) == continuation.digest(restored)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        write_json(report_path, report)
    print(f"cache startup: {report['status']}", flush=True)


if __name__ == "__main__":
    main()

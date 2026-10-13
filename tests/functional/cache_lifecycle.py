#!/usr/bin/env python3
"""Opt-in disk/restart/crash workloads; every server and cache belongs to this run."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import traceback

import continuation
from cache_compaction import ARCHIVE_LINE
from cache_workloads import history, continue_history, signature, work
from metrics import Recorder, join_server_timings
from run import execution_coverage, option, provenance, server, write_json

CASES = ("restart", "tiered", "oversized", "crash", "side-traffic")
DISK_BYTES = 8 * 1024**3
STAGING_BYTES = 2 * 1024**3


def published_tokens(log):
    legacy = re.findall(r"event=disk_cache action=stored reason=saved .*?\btokens=(\d+)\b", log)
    component = re.findall(r"schema=component-cache-v1 event=published tokens=(\d+)\b", log)
    return [int(tokens) for tokens in legacy + component]


def wait_until(predicate, description, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError(f"unqualified: did not observe {description}")


class Lifecycle:
    def __init__(self, command, output, timeout, library, seed_lines=160,
                 minimum_published_tokens=0, publication_timeout=30,
                 require_over_ram=False, staging_bytes=STAGING_BYTES):
        self.command, self.output, self.timeout, self.library = command, output, timeout, library
        self.seed_lines = seed_lines
        self.minimum_published_tokens = minimum_published_tokens
        self.publication_timeout = publication_timeout
        self.require_over_ram = require_over_ram
        self.staging_bytes = staging_bytes
        self.model = option(command, "--served-model-name", "cache-lifecycle")
        self.mode = option(command, "--speculative", "off")
        self.report = {**provenance(), "command": command,
                       "binary": str(Path(command[0]).resolve()),
                       "seed_lines": seed_lines,
                       "minimum_published_tokens": minimum_published_tokens,
                       "publication_timeout": publication_timeout,
                       "require_over_ram": require_over_ram,
                       "staging_bytes": staging_bytes,
                       "status": "running", "cases": {}}
        source = hashlib.sha256()
        source.update(self.report["harness_sha256"].encode())
        source.update(Path(__file__).read_bytes())
        self.report["harness_sha256"] = source.hexdigest()
        self.case = self.stage_path = None
        self.controls = []
        write_json(output / "report.json", self.report)

    @contextmanager
    def stage(self, name, staging=None, environment=None):
        if staging is None:
            staging = self.staging_bytes
        self.stage_path = self.output / self.case / name
        self.stage_path.mkdir()
        command = [*self.command, "--host", "127.0.0.1", "--port", str(self.port),
                   "--cache-disk", str(self.disk), "--cache-disk-bytes", str(DISK_BYTES),
                   "--cache-disk-staging-bytes", str(staging)]
        continuation.TRACE = Recorder(self.stage_path / "requests.requests.json")
        record = {"command": command, "status": "running", "staging_bytes": staging}
        self.row["stages"][name] = record
        started = time.monotonic()
        try:
            with server(command, self.stage_path / "server.log", self.timeout, environment) as process:
                record["startup_ms"] = (time.monotonic() - started) * 1000
                yield process
            join_server_timings(self.stage_path)
            record["execution"] = execution_coverage(self.stage_path, self.mode)
            record["status"] = "passed"
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            write_json(self.output / "report.json", self.report)

    def require(self, label, condition, detail):
        if not condition:
            self.row["failures"].append(f"{label}: {detail}")

    def chat(self, label, messages, floor=0, ceiling=None, disk=False, cold=False):
        if disk:
            floor = max(floor, self.minimum_published_tokens)
        body = dict(model=self.model, messages=deepcopy(messages), temperature=0, seed=31,
                    max_completion_tokens=16, reasoning_effort="none", cache_prompt=not cold)
        raw = continuation.call(f"http://127.0.0.1:{self.port}", body)
        choice = raw["choices"][0]
        result = {"text": choice["message"]["content"],
                  "reasoning": choice["message"].get("reasoning_content") or "",
                  "tools": choice["message"].get("tool_calls") or [],
                  "finish": choice["finish_reason"], "usage": raw["usage"]}
        total, cached, prefilled = work(result)
        if label == "seed":
            assert total >= self.minimum_published_tokens, (
                "unqualified: seed is below the required depth", total,
                self.minimum_published_tokens)
        assert result["text"].strip() == "ALPHA" and not result["reasoning"] \
            and not result["tools"] and result["finish"] == "stop", result
        passed = cached >= floor and (ceiling is None or cached <= ceiling) \
            and (not disk or result["usage"]["gufo"]["cache_disk_hit"])
        self.require(label, passed, f"cached={cached}, boundary=[{floor}, {ceiling}], "
                     f"disk={result['usage']['gufo']['cache_disk_hit']}, prefilled={prefilled}")
        result["expectation"] = {"status": "passed" if passed else "failed", "floor": floor,
                                 "ceiling": ceiling, "disk_required": disk}
        self.row["requests"][label] = result
        continuation.TRACE.mark(label)
        write_json(self.output / "report.json", self.report)
        if cold:
            assert (cached, prefilled) == (0, total), result
        else:
            self.controls.append((label, deepcopy(messages), result))
        return result

    def controls_after_warm(self):
        for label, messages, warm in self.controls:
            cold = self.chat(label + "_cold", messages, cold=True)
            assert work(cold)[0] == work(warm)[0] and signature(cold) == signature(warm), (
                label, warm, cold)

    def run_case(self, name):
        self.case = name
        root = self.output / name
        root.mkdir()
        self.disk = root / "disk"
        self.row = {"status": "running", "stages": {}, "requests": {}, "failures": []}
        self.report["cases"][name] = self.row
        self.controls = []
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            self.port = reserve.getsockname()[1]
        if shutil.disk_usage(root).free < 2 * DISK_BYTES:
            raise AssertionError("unqualified: disk lifecycle needs 16 GiB free")
        seed = history("lifecycle_" + name, self.seed_lines)
        try:
            if name == "crash":
                self.crash_case(seed, root)
            elif name == "side-traffic":
                self.side_traffic_case()
            else:
                staging = 1024**2 if name == "oversized" else self.staging_bytes
                with self.stage("initial", staging):
                    first = self.chat("seed", seed)
                    long = continue_history(seed, first, ARCHIVE_LINE * 800 + "Reply with only ALPHA.")
                    deep = self.chat("append", long, work(first)[0] - 16)
                    if self.minimum_published_tokens:
                        wait_until(lambda: max(published_tokens(
                            (self.stage_path / "server.log").read_text()), default=0)
                            >= max(self.minimum_published_tokens, work(deep)[0] - 16),
                            "a published deep checkpoint", self.publication_timeout)
                stored = published_tokens((root / "initial/server.log").read_text())
                self.row["published_tokens"] = stored
                boundary = max((n for n in stored if n <= work(deep)[0]),
                               default=work(deep)[0] - 16)
                if name == "oversized":
                    log = (root / "initial/server.log").read_text()
                    sizes = [int(n) for n in re.findall(
                        r"event=(?:disk_cache|published) .*?\b(?:file|payload)_bytes=(\d+)", log)]
                    assert sizes and max(sizes) > staging, (
                        "unqualified: checkpoint did not exceed staging", sizes, staging)
                    self.row["oversized_bytes"] = max(sizes)
                    if self.require_over_ram:
                        capacities = [int(n) for n in re.findall(
                            r"schema=component-cache-v1 event=configured .*?\bcapacity_bytes=(\d+)",
                            log)]
                        assert capacities and max(sizes) > max(capacities), (
                            "unqualified: checkpoint did not exceed committed RAM backing",
                            sizes, capacities)
                        self.row["ram_capacity_bytes"] = max(capacities)
                    self.require("oversized_publication", bool(stored),
                                 "checkpoint larger than staging was not published")
                else:
                    assert stored and boundary > work(first)[0] + 10000, (
                        "unqualified: no deep checkpoint published", stored)
                with self.stage("restarted", staging):
                    if name == "tiered":
                        short = self.chat("short_ram", seed, disk=True, floor=1)
                        assert work(short)[0] < boundary, (short, boundary)
                    restored = self.chat("restore", long, floor=boundary, disk=True)
                    if name == "tiered":
                        self.require("tiered_selection", work(restored)[1] > work(short)[0],
                                     "short RAM hit hid a deeper published disk checkpoint")
                    self.controls_after_warm()
            self.row["status"] = "failed" if self.row["failures"] else "passed"
        except Exception as error:
            self.row.update(status="failed", error=str(error), traceback=traceback.format_exc())
        finally:
            # Each case owns at most 8 GiB and no later case reads its files.
            shutil.rmtree(self.disk, ignore_errors=True)
            write_json(self.output / "report.json", self.report)
        print(f"{name}: {self.row['status']}", flush=True)

    def side_traffic_case(self):
        # Keep the common system prompt short. A deep system-only fixture
        # cannot represent short conversations competing with a long agent.
        system_lines = min(self.seed_lines, 256)
        seed = history("lifecycle_side_traffic", system_lines,
                       notes=self.seed_lines - system_lines)

        def published(minimum):
            return wait_until(lambda: max(published_tokens(
                (self.stage_path / "server.log").read_text()), default=0)
                >= minimum, "a published deep checkpoint", self.publication_timeout)

        with self.stage("initial"):
            first = self.chat("seed", seed)
            published(max(1, work(first)[0] - 16))
            long = continue_history(seed, first,
                                    ARCHIVE_LINE * 800 + "Reply with only ALPHA.")
            deep = self.chat("append", long, floor=work(first)[0] - 16)
            published(work(deep)[0] - 16)
            log = (self.stage_path / "server.log").read_text()
            stored = published_tokens(log)
            boundary = max(stored)
            self.row["published_tokens"] = stored
            if self.require_over_ram:
                sizes = [int(n) for n in re.findall(
                    r"event=published .*?\bpayload_bytes=(\d+)", log)]
                capacities = [int(n) for n in re.findall(
                    r"schema=component-cache-v1 event=configured .*?\bcapacity_bytes=(\d+)",
                    log)]
                assert sizes and capacities and max(sizes) > max(capacities), (
                    "unqualified: checkpoint did not exceed committed RAM backing",
                    sizes, capacities)
                self.row.update(oversized_bytes=max(sizes), ram_capacity_bytes=max(capacities))
        with self.stage("restarted"):
            restored = self.chat("restore", long, floor=boundary, disk=True)
            for round_index in range(3):
                for peer in range(4):
                    short = [deepcopy(seed[0]), {"role": "user", "content":
                             f"Independent short conversation {round_index}/{peer}. "
                             "Reply with only ALPHA."}]
                    result = self.chat(f"side_{round_index}_{peer}", short)
                    assert work(result)[0] * 4 < boundary, (
                        "unqualified: side conversation is not short", work(result), boundary)
                long = continue_history(long, restored,
                                        ARCHIVE_LINE * 256 + "Reply with only ALPHA.")
                restored = self.chat(f"deep_{round_index}", long, floor=boundary)
                published(work(restored)[0] - 16)
                boundary = max(published_tokens((self.stage_path / "server.log").read_text()))
            self.controls_after_warm()

    def crash_case(self, seed, root):
        gate = root / "gate"
        gate.mkdir()
        environment = {**os.environ, "LD_PRELOAD": str(self.library),
                       "GUFO_TEST_DISK_GATE": str(gate),
                       "GUFO_TEST_DISK_DIRECTORY": str(self.disk)}
        with self.stage("initial", environment=environment) as process:
            try:
                first = self.chat("seed", seed)
                wait_until(lambda: max(published_tokens(
                    (self.stage_path / "server.log").read_text()), default=0)
                    >= max(1, self.minimum_published_tokens),
                    "a published seed checkpoint", self.publication_timeout)
                (gate / "armed").touch()
                long = continue_history(seed, first, ARCHIVE_LINE * 800 + "Reply with only ALPHA.")
                deep = self.chat("queued_append", long, work(first)[0] - 16)
                wait_until(lambda: (gate / "blocked").is_file(), "an unpublished write at fsync")
                assert deep["usage"]["gufo"]["cache_disk_queued_bytes"] > 0, deep
                request_id = continuation.TRACE.rows[-1]["request_id"]
                wait_until(lambda: f"request={request_id} event=completed" in
                           (self.stage_path / "server.log").read_text(),
                           "the append's completed request log")
                process.kill()
                process.wait(timeout=30)
                assert process.returncode == -9, process.returncode
                stored = published_tokens((self.stage_path / "server.log").read_text())
                self.row["crash"] = {"signal": 9, "write_blocked": True,
                                     "published_tokens": stored,
                                     "queued_bytes": deep["usage"]["gufo"]["cache_disk_queued_bytes"]}
            finally:
                (gate / "release").touch()
        assert stored and max(stored) < work(deep)[0], stored
        with self.stage("restarted"):
            restored = self.chat("restore_published", long, floor=1, ceiling=max(stored), disk=True)
            self.require("published_only", work(restored)[1] in stored,
                         f"restored {work(restored)[1]} outside published boundaries {stored}")
            self.controls_after_warm()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault-library", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--seed-lines", type=int, default=160,
                        help="Background lines in the initial history")
    parser.add_argument("--minimum-published-tokens", type=int, default=0,
                        help="Require this actual seed and disk-restoration depth")
    parser.add_argument("--publication-timeout", type=float, default=30,
                        help="Seconds to await publication before a deliberate crash")
    parser.add_argument("--require-over-ram", action="store_true",
                        help="Require oversized/side-traffic publication to exceed actual committed RAM backing")
    parser.add_argument("--staging-bytes", type=int, default=STAGING_BYTES,
                        help="Staging budget for restart, tiered and crash (oversized always uses 1 MiB)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seed_lines <= 0 or args.minimum_published_tokens < 0 \
            or args.publication_timeout <= 0 or args.staging_bytes <= 0:
        parser.error("seed lines, publication timeout and staging bytes must be positive; "
                     "depth must be nonnegative")
    if args.require_over_ram and args.case and not set(args.case) & {"oversized", "side-traffic"}:
        parser.error("--require-over-ram requires --case oversized or side-traffic")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a gufo serve llm command after --")
    for reserved in ("--host", "--port", "--cache-disk", "--cache-disk-bytes",
                     "--cache-disk-staging-bytes", "--api-key"):
        if option(command, reserved) is not None:
            parser.error(f"the runner owns {reserved}")
    if int(option(command, "--context", "0")) < 32768:
        parser.error("cache lifecycle requires --context at least 32768")
    if not args.fault_library.is_file():
        parser.error("build the CMake cache_disk_faults target and pass its library")
    args.fault_library = args.fault_library.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if option(command, "--served-model-name") is None:
        command += ["--served-model-name", "cache-lifecycle"]
    runner = Lifecycle(command, args.output, args.startup_timeout, args.fault_library,
                       args.seed_lines, args.minimum_published_tokens, args.publication_timeout,
                       args.require_over_ram, args.staging_bytes)
    runner.report["fault_library_sha256"] = hashlib.file_digest(
        args.fault_library.open("rb"), "sha256").hexdigest()
    try:
        for case in dict.fromkeys(args.case or CASES):
            runner.run_case(case)
    except KeyboardInterrupt:
        runner.report["status"] = "interrupted"
        write_json(args.output / "report.json", runner.report)
        raise
    runner.report["status"] = ("passed" if all(row["status"] == "passed"
        for row in runner.report["cases"].values()) else "failed")
    write_json(args.output / "report.json", runner.report)
    return 0 if runner.report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

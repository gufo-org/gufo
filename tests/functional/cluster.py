#!/usr/bin/env python3
"""Execute a reviewed Flash-Next impact plan; retain every group's evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
TAIL_LINES = 40


def tail(path, lines=TAIL_LINES):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def suite_failures(group_output):
    failures = {}
    for path in sorted(group_output.glob("*.json")):
        if path.name == "report.json":
            continue
        try:
            suites = json.loads(path.read_text()).get("suites", {})
        except (OSError, ValueError, AttributeError):
            continue
        for name, result in suites.items():
            if isinstance(result, dict) and result.get("status") == "failed":
                failures[name] = str(result.get("error") or "failed")
    return failures


def failure_report(log, group_output):
    """Explain a failed group: run and suite errors, then a short log tail."""
    try:
        error = json.loads((group_output / "report.json").read_text()).get("error")
    except (OSError, ValueError):
        error = None
    failures = suite_failures(group_output)
    log_tail = tail(log, 15 if failures else TAIL_LINES)
    server_log = group_output / "server.log"
    # Suite errors explain most failures; the server log matters when it never served.
    if server_log.exists() and not failures:
        log_tail += "\n--- server.log ---\n" + tail(server_log)
    return {"error": error, "failures": failures, "log_tail": log_tail}


def remove_disk_caches(group_output):
    # Each group's disk cache can reach 8 GiB; reports and logs stay.
    for path in sorted(group_output.rglob("disk"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)


def command(group, binary, weights, output):
    profile = group["profile"]
    if profile not in ("flash-next-q4/off", "flash-next-q4/mtp"):
        raise ValueError("Cluster executor only supports Flash-Next: " + profile)
    settings = group["settings"]
    args = [sys.executable, str(HERE / "run.py"), "--output", str(output),
            "--sampling-preset", "qwen38", "--expected-input-modalities", "text,image",
            "--startup-timeout", "600", "--record-baseline"]
    for suite in group["suites"]:
        args += ["--suite", suite]
    args += ["--", str(binary), "serve", "llm", "--model",
             str(weights / "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
             "--mmproj", str(weights / "mmproj-BF16.gguf"),
             "--mtp-model", str(weights / "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
             "--speculative", profile.split("/")[1],
             "--context", str(settings["context"]), "--sessions", str(settings["sessions"])]
    if "cache_ram_bytes" in settings:
        args += ["--cache-ram-bytes", str(settings["cache_ram_bytes"])]
    return args


def execute(plan, binary, weights, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for group in plan["groups"]:
        name = group["profile"].replace("/", "-") + "-" + group["configuration"]
        args = command(group, binary, weights, output / name)
        print("Running", name, ":", ", ".join(group["suites"]), flush=True)
        log_path = output / (name + ".log")
        with log_path.open("w") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
        row = {"profile": group["profile"], "configuration": group["configuration"],
               "suites": group["suites"], "exit_code": result.returncode,
               "status": "passed" if result.returncode == 0 else "failed",
               "command": args}
        if result.returncode:
            row.update(failure_report(log_path, output / name))
        remove_disk_caches(output / name)
        rows.append(row)
    report = {"status": "failed" if any(row["exit_code"] for row in rows) else
              ("passed" if rows else "no-functional-impact"),
              "scope": "Flash-Next AR/MTP candidate correctness",
              "timing_qualification": "not performed; --record-baseline captures observations",
              "groups": rows, "deferred": plan["deferred"],
              "out_of_scope": plan["out_of_scope"], "obligations": plan["obligations"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write("\n## Flash-Next functional results\n\n")
            summary.write("Candidate correctness only; timing regressions remain unqualified.\n\n")
            for row in rows:
                summary.write(f"- {row['profile']} / {row['configuration']}: {row['status']}\n")
            for row in rows:
                if row["exit_code"]:
                    lines = ([f"error: {row['error']}"] if row["error"] else []) + [
                        f"{suite}: {error}" for suite, error in row["failures"].items()]
                    summary.write(f"\n<details><summary>{row['profile']} / {row['configuration']} "
                                  f"failure</summary>\n\n```text\n" + "\n".join(lines + [row["log_tail"]])
                                  + "\n```\n\n</details>\n")
            summary.write(f"\nDeferred suite/profile pairs: {len(plan['deferred'])}. "
                          f"Outside Flash-Next scope: {len(plan['out_of_scope'])}.\n")
    return int(report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.plan.read_text()), args.binary.resolve(),
                   args.weights.resolve(), args.output.resolve())


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Execute a reviewed Flash-Next impact plan; retain every group's evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


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
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
        rows.append({"profile": group["profile"], "configuration": group["configuration"],
                     "suites": group["suites"], "exit_code": result.returncode,
                     "status": "passed" if result.returncode == 0 else "failed",
                     "command": args})
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

#!/usr/bin/env python3
"""Offline consistency checks for the published Phoenix experiment evidence.

This validates reported records and hashes; it does not rerun hardware tests.
Python standard library only. No network, model or accelerator is required.
"""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def equal(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def main():
    manifest = read("manifest.json")
    for entry in manifest["files"]:
        path = ROOT / entry["file"]
        require(path.is_file(), f"Missing {path.name}")
        require(path.stat().st_size == entry["bytes"], f"Size: {path.name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["published_sha256"],
                f"Hash: {path.name}")

    rows = read("08-requests.json")["requests"]
    require(sum(row["phase"] == "measured" for row in rows) == 39, "Measured count")
    require(sum(row["phase"] == "warmup" for row in rows) == 28, "Warmup count")
    for row in rows:
        require(row["status"] == "completed" and row["finish_reason"] == "stop", "Completion")
        require(row["correctness"]["passed"] and row["cache_condition_met"], "Correctness/cache")
        require(row["actual_output_tokens"] == row["usage"]["completion_tokens"], "Output count")
    measured = {
        (row["run"], row["pair_directory"], row["profile"], row["workload"]): row
        for row in rows if row["phase"] == "measured"
    }
    require(len(measured) == 39, "Unique measured records")

    full = read("08-full-model.json")
    pair_count = 0
    for run in full["runs"]:
        run_name = run["run"].replace("\\", "/").split("/")[-1]
        for comparison in run["comparison"]["comparisons"]:
            latencies, throughputs = [], []
            for pair in comparison["pairs"]:
                pair_directory = f'pair-{pair["pair"]:02}'
                base = measured[(run_name, pair_directory, comparison["baseline"], comparison["workload"])]
                cand = measured[(run_name, pair_directory, comparison["candidate"], comparison["workload"])]
                for key in ("prompt_token_ids_sha256", "request_sha256", "offline_prompt_tokens"):
                    require(base[key] == cand[key], f"Matched {key}")
                for row, prefix in [(base, "baseline"), (cand, "candidate")]:
                    require(equal(row["timings"]["prompt_ms"], pair[f"{prefix}_prompt_ms"]), "Prompt time")
                    require(equal(row["full_request_s"], pair[f"{prefix}_full_s"]), "Full time")
                    require(row["actual_output_tokens"] == pair[f"{prefix}_output_tokens"], "Paired output")
                    require(row["offline_prompt_tokens"] == pair["prompt_tokens"], "Prompt tokens")
                    require(row["timings"]["cache_n"] == pair["cache_n"], "Cache count")
                b, c = pair["baseline_prompt_ms"], pair["candidate_prompt_ms"]
                latency, throughput = 100 * (1 - c / b), 100 * (b / c - 1)
                require(equal(latency, pair["latency_reduction_percent"]), "Latency formula")
                require(equal(throughput, pair["throughput_gain_percent"]), "Throughput formula")
                latencies.append(latency)
                throughputs.append(throughput)
                pair_count += 1
            require(equal(statistics.median(latencies), comparison["median_latency_reduction_percent"]), "Latency median")
            require(equal(statistics.median(throughputs), comparison["median_throughput_gain_percent"]), "Throughput median")
        for resource in run["resources"]:
            require(resource["monitored_resource_floors_passed"], "Resource floor")
            require(resource["owned_memory_released"], "Resource release")
            for native in resource["native_totals"]:
                require(int(native["dispatches"]) > 0, "Physical dispatch")
                require(int(native["checked_values"]) == 0, "Timing excludes numerical duplication")

    checked = 0
    for qualification in full["qualification"]:
        stats = qualification.get("result", {}).get("npu_stats", "")
        match = re.search(r"checked_values=(\d+)", stats)
        if match:
            checked += int(match[1])
    require(checked == 2585890816, "Numerical checked values")
    original = read("08-original-identities.json")
    patch_hash = original["files"]["gufo-projection\\source.patch"]
    require(hashlib.sha256((ROOT / "strata-ssm-qkv-experiment.patch").read_bytes()).hexdigest() == patch_hash,
            "Byte-identical experimental patch")
    print(f"PASS: {len(manifest['files'])} file hashes; 39 measured + 28 warmup records; "
          f"{pair_count} paired calculations; {checked:,} recorded numerical checks.")


if __name__ == "__main__":
    main()

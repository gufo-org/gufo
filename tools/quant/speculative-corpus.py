#!/usr/bin/env python3
"""Compare greedy speculative decoding against autoregressive generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


GENERATED_RE = re.compile(
    r"Generated\s+(?P<tokens>\d+)\s+tokens on GPU in\s+"
    r"(?P<seconds>[0-9.]+)s\s+\((?P<tps>[0-9.]+)\s+tok/s\)"
)
SPECULATIVE_RE = re.compile(
    r"\[Speculative\]: acceptance=(?P<acceptance>[0-9.eE+-]+)"
    r"\s+drafted=(?P<drafted>\d+)\s+accepted=(?P<accepted>\d+)"
    r"\s+verification_steps=(?P<steps>\d+)"
)
CONTROLLED_ENV = {
    "GUFO_BF16_SMALL_BATCH_EXACT_LDS8",
    "GUFO_DFLASH_GEMM",
    "GUFO_DFLASH_PREWARM",
    "GUFO_DFLASH_SELECTOR",
    "GUFO_PREFILL_SMALL_BATCH_BF16_FROM_LAYER",
    "GUFO_PREFILL_SMALL_BATCH_BF16_TILE",
    "GUFO_PREFILL_SMALL_BATCH_FP32_FROM_LAYER",
    "GUFO_PREFILL_SMALL_BATCH_QUANT",
    "GUFO_PREFILL_SMALL_BATCH_W8A8_TILE",
    "GUFO_SPEC_BATCH_LM_HEAD",
    "GUFO_SPEC_BATCH_VERIFY",
    "GUFO_SPEC_BATCH_VERIFY_CHECK",
}


def verification_environment(profile: str, backend: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if profile == "production":
        return result
    if backend.startswith("dflash"):
        result["GUFO_DFLASH_GEMM"] = "hipblaslt"
        result["GUFO_DFLASH_PREWARM"] = "1"
    if profile == "sequential":
        result["GUFO_SPEC_BATCH_VERIFY"] = "0"
        return result
    result["GUFO_SPEC_BATCH_VERIFY"] = "1"
    result["GUFO_SPEC_BATCH_LM_HEAD"] = "1"
    if profile == "bf16":
        result["GUFO_PREFILL_SMALL_BATCH_QUANT"] = "bf16"
    elif profile == "fp32":
        result["GUFO_PREFILL_SMALL_BATCH_QUANT"] = "fp32"
    elif profile == "fp32-tail":
        result["GUFO_PREFILL_SMALL_BATCH_FP32_FROM_LAYER"] = "62"
    elif profile == "w8a8":
        result["GUFO_PREFILL_SMALL_BATCH_BF16_FROM_LAYER"] = "64"
    return result


def parse_environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, setting = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid --env value: {value!r}")
        result[key] = setting
    return result


def extract_completion(stdout: str) -> str:
    marker = "--- Generation Output ---\n"
    start = stdout.find(marker)
    if start < 0:
        raise RuntimeError("generation output marker is missing")
    tail = stdout[start + len(marker) :]
    generated = GENERATED_RE.search(tail)
    if generated is None:
        raise RuntimeError("generation timing line is missing")
    return tail[: generated.start()].rstrip("\n")


def run_prompt(
    args: argparse.Namespace,
    prompt: str,
    speculative: bool,
    extra_environment: dict[str, str],
) -> dict[str, object]:
    command = [
        args.binary,
        "prompt",
        "--raw",
        "--verbose",
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
    ]
    if speculative:
        command.extend(
            [
                "--speculative",
                args.backend,
                "--draft-tokens",
                str(args.draft_tokens),
                "--draft-policy",
                args.draft_policy,
                "--min-draft-tokens",
                str(args.min_draft_tokens),
            ]
        )
        option = (
            "--dflash-model"
            if args.backend.startswith("dflash")
            else "--mtp-model"
        )
        command.extend([option, args.draft_model])
    command.append(prompt)

    environment = os.environ.copy()
    for key in CONTROLLED_ENV:
        environment.pop(key, None)
    if speculative:
        environment.update(extra_environment)

    process = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=args.timeout,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command[:-1])}\n"
            f"{process.stderr}"
        )

    generated = GENERATED_RE.search(process.stdout)
    if generated is None:
        raise RuntimeError(f"could not parse timing output:\n{process.stdout}")
    result: dict[str, object] = {
        "completion": extract_completion(process.stdout),
        "tokens": int(generated.group("tokens")),
        "seconds": float(generated.group("seconds")),
        "tps": float(generated.group("tps")),
    }
    if speculative:
        stats = SPECULATIVE_RE.search(process.stderr)
        if stats is None:
            raise RuntimeError(
                f"could not parse speculative statistics:\n{process.stderr}"
            )
        result.update(
            {
                "acceptance": float(stats.group("acceptance")),
                "drafted": int(stats.group("drafted")),
                "accepted": int(stats.group("accepted")),
                "steps": int(stats.group("steps")),
            }
        )
    return result


def mismatch_summary(expected: str, actual: str) -> str:
    common = 0
    for left, right in zip(expected, actual):
        if left != right:
            break
        common += 1
    expected_text = expected[common : common + 32].replace("\n", "\\n")
    actual_text = actual[common : common + 32].replace("\n", "\\n")
    return f"char {common}: AR={expected_text!r} spec={actual_text!r}"


def load_prompts(
    path: Path, quick: bool, limit: int, case_ids: list[str]
) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("suite must contain a non-empty prompts array")
    selected = prompts
    if quick:
        selected = [case for case in selected if case.get("quick") is True]
    if case_ids:
        requested = set(case_ids)
        selected = [case for case in selected if case.get("id") in requested]
        missing = requested.difference(case.get("id") for case in selected)
        if missing:
            raise ValueError(f"unknown suite case(s): {', '.join(sorted(missing))}")
    if limit > 0:
        selected = selected[:limit]
    for case in selected:
        if not all(
            isinstance(case.get(field), str) and case[field]
            for field in ("id", "category", "text")
        ):
            raise ValueError("every prompt needs non-empty id/category/text")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument(
        "--backend", choices=("dflash2", "mtp"), default="dflash2"
    )
    parser.add_argument(
        "--profile",
        choices=(
            "production",
            "sequential",
            "w8a8",
            "bf16",
            "fp32",
            "fp32-tail",
        ),
        default="production",
    )
    parser.add_argument(
        "--suite",
        default="benchmarks/qwen3.8-27b/speculative-corpus.json",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument(
        "--draft-policy",
        choices=("auto", "fixed", "rolling", "accepted-ema"),
        default="auto",
    )
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()

    if (
        args.max_tokens <= 0
        or args.draft_tokens <= 0
        or args.min_draft_tokens <= 0
        or args.min_draft_tokens > args.draft_tokens
        or args.repetitions <= 0
    ):
        parser.error("token counts and repetitions must be positive")

    prompts = load_prompts(Path(args.suite), args.quick, args.limit, args.case)
    environment = verification_environment(args.profile, args.backend)
    try:
        environment.update(parse_environment(args.env))
    except ValueError as error:
        parser.error(str(error))
    print(
        "| prompt | category | exact | AR tok/s | speculative tok/s | "
        "speedup | acceptance | avg draft |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    mismatches: list[str] = []
    total_tokens = 0
    total_ar_seconds = 0.0
    total_spec_seconds = 0.0
    total_drafted = 0
    total_accepted = 0
    total_steps = 0
    speedups: list[float] = []

    for case in prompts:
        autoregressive = run_prompt(args, case["text"], False, {})
        speculative_runs = [
            run_prompt(args, case["text"], True, environment)
            for _ in range(args.repetitions)
        ]
        exact = all(
            run["completion"] == autoregressive["completion"]
            for run in speculative_runs
        )
        spec_seconds = statistics.median(
            float(run["seconds"]) for run in speculative_runs
        )
        spec_tps = statistics.median(
            float(run["tps"]) for run in speculative_runs
        )
        acceptance = statistics.median(
            float(run["acceptance"]) for run in speculative_runs
        )
        average_draft = statistics.median(
            int(run["drafted"]) / max(int(run["steps"]), 1)
            for run in speculative_runs
        )
        ar_seconds = float(autoregressive["seconds"])
        ar_tps = float(autoregressive["tps"])
        speedup = spec_tps / ar_tps if ar_tps > 0.0 else 0.0
        speedups.append(speedup)

        representative = speculative_runs[0]
        total_tokens += int(representative["tokens"])
        total_ar_seconds += ar_seconds
        total_spec_seconds += spec_seconds
        total_drafted += int(representative["drafted"])
        total_accepted += int(representative["accepted"])
        total_steps += int(representative["steps"])

        print(
            f"| {case['id']} | {case['category']} | "
            f"{'yes' if exact else 'NO'} | {ar_tps:.2f} | {spec_tps:.2f} | "
            f"{speedup:.2f}x | {acceptance * 100.0:.1f}% | "
            f"{average_draft:.2f} |"
        )
        if not exact:
            detail = mismatch_summary(
                str(autoregressive["completion"]),
                str(representative["completion"]),
            )
            mismatches.append(f"{case['id']}: {detail}")

    aggregate_ar = total_tokens / total_ar_seconds
    aggregate_spec = total_tokens / total_spec_seconds
    aggregate_acceptance = (
        total_accepted / total_drafted if total_drafted else 0.0
    )
    suite_hash = hashlib.sha256(
        Path(args.suite).read_bytes()
    ).hexdigest()[:12]
    print()
    print(
        f"aggregate: prompts={len(prompts)} suite={suite_hash} "
        f"exact={len(prompts) - len(mismatches)}/{len(prompts)} "
        f"AR={aggregate_ar:.2f} tok/s speculative={aggregate_spec:.2f} tok/s "
        f"speedup={aggregate_spec / aggregate_ar:.2f}x "
        f"median_speedup={statistics.median(speedups):.2f}x "
        f"acceptance={aggregate_acceptance * 100.0:.1f}% "
        f"avg_draft={total_drafted / max(total_steps, 1):.2f}"
    )
    for mismatch in mismatches:
        print(f"mismatch: {mismatch}", file=sys.stderr)

    return 0 if args.allow_mismatch or not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())

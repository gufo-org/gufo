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
    # Qwen reports "on GPU"; DeepSeek V4 Flash reports "on ROCm".
    r"Generated\s+(?P<tokens>\d+)\s+tokens on (?:GPU|ROCm) in\s+"
    r"(?P<seconds>[0-9.]+)s\s+\((?P<tps>[0-9.]+)\s+tok/s\)"
)
SPECULATIVE_LINE_RE = re.compile(r"^\[Speculative\]:\s+(?P<body>.+)$", re.MULTILINE)
TOKEN_TRACE_RE = re.compile(
    r"^\[TokenTrace\]: count=(?P<count>\d+) sha256=(?P<sha256>[0-9a-f]{64})$",
    re.MULTILINE,
)
SPECULATIVE_FIELD_RE = re.compile(
    r"(?P<key>[a-z_]+)=(?P<value>[0-9.eE+-]+)"
)
DSPARK_REFERENCE_SYSTEM_PROMPT = "You are a helpful assistant"

def parse_token_trace(stderr: str, tokens: int) -> str:
    trace = TOKEN_TRACE_RE.search(stderr)
    if trace is None or int(trace.group("count")) != tokens:
        raise RuntimeError("missing or incomplete emitted-token trace; rebuild gufo")
    return trace.group("sha256")


def parse_speculative_stats(stderr: str) -> dict[str, int | float]:
    line = SPECULATIVE_LINE_RE.search(stderr)
    if line is None:
        raise ValueError("speculative statistics line is missing")
    fields = {
        match.group("key"): match.group("value")
        for match in SPECULATIVE_FIELD_RE.finditer(line.group("body"))
    }
    required = ("acceptance", "drafted", "accepted", "verification_steps")
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(
            f"speculative statistics missing field(s): {', '.join(missing)}"
        )

    result: dict[str, int | float] = {
        "acceptance": float(fields["acceptance"]),
        "drafted": int(fields["drafted"]),
        "accepted": int(fields["accepted"]),
        "steps": int(fields["verification_steps"]),
    }
    integer_fields = (
        "skipped",
        "positional_accepted",
        "full_blocks",
        "anchors",
        "verifier_rows",
    )
    float_fields = ("positional_acceptance", "full_block_rate")
    for field in integer_fields:
        if field in fields:
            result[field] = int(fields[field])
    for field in float_fields:
        if field in fields:
            result[field] = float(fields[field])
    return result


def build_prompt_command(
    args: argparse.Namespace, prompt: str, speculative: bool
) -> list[str]:
    prompt_mode = args.prompt_mode
    if prompt_mode == "auto":
        prompt_mode = "chat"

    command = [
        args.binary,
        "prompt",
        "--verbose",
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
    ]
    if prompt_mode == "raw":
        command.append("--raw")
    else:
        system_prompt = args.system_prompt
        if system_prompt is None and args.backend == "dspark":
            system_prompt = DSPARK_REFERENCE_SYSTEM_PROMPT
        if system_prompt is not None:
            command.extend(["--system", system_prompt])

    if speculative:
        command.extend(["--speculative", args.backend])
        command.extend(["--draft-tokens", str(args.draft_tokens)])
        if args.backend != "dspark":
            command.extend(
                [
                    "--min-draft-tokens",
                    str(args.min_draft_tokens),
                ]
            )
        if args.backend == "dspark":
            option = "--dspark-model"
        elif args.backend.startswith("dflash"):
            option = "--dflash-model"
        else:
            option = "--mtp-model"
        command.extend([option, args.draft_model])
    command.append(prompt)
    return command


def artifact_identity(path: str) -> dict:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def autoregressive_key(args: argparse.Namespace, prompt: str) -> str:
    """Identity of an autoregressive reference run.

    The reference depends on the binary, the target shard, the prompt framing
    and the greedy decode length, and on nothing speculative, so several draft
    companions benchmarked against one target can share it -- but never across
    builds or framings, which would score a completion against a reference that
    could not have produced it.
    """
    # Stat identities invalidate in-place rebuilds/replacements without
    # rereading a multi-gigabyte target for every prompt. Artifact content
    # hashes belong in the qualification manifest.
    identity = {
        "binary": artifact_identity(args.binary),
        "model": artifact_identity(args.model),
        "command": build_prompt_command(args, prompt, speculative=False),
        "environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("GUFO_", "HIP_", "ROCR_", "HSA_"))
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_autoregressive_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "gufo.ar-cache.v1":
        return {}
    entries = document.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def store_autoregressive_cache(path: Path | None, cache: dict[str, dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": "gufo.ar-cache.v1", "entries": cache}, indent=1),
        encoding="utf-8",
    )



def extract_completion(stdout: str) -> str:
    marker = "--- Generation Output ---\n"
    start = stdout.find(marker)
    if start < 0:
        raise RuntimeError("generation output marker is missing")
    tail = stdout[start + len(marker) :]
    generated = GENERATED_RE.search(tail)
    if generated is None:
        raise RuntimeError("generation timing line is missing")
    # The CLI adds one separator newline after the completion. Any earlier
    # trailing newlines were generated and are part of the equality check.
    return tail[: generated.start()].removesuffix("\n")


def run_prompt(
    args: argparse.Namespace,
    prompt: str,
    speculative: bool,
    extra_environment: dict[str, str],
) -> dict[str, object]:
    command = build_prompt_command(args, prompt, speculative)

    environment = os.environ.copy()
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
        "token_sha256": parse_token_trace(
            process.stderr, int(generated.group("tokens"))),
    }
    if speculative:
        try:
            stats = parse_speculative_stats(process.stderr)
        except ValueError as error:
            raise RuntimeError(
                f"could not parse speculative statistics ({error}):\n"
                f"{process.stderr}"
            ) from error
        result.update(stats)
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
    if not selected:
        raise ValueError("suite selection contains no prompts")
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
        # dspark is DeepSeek V4 Flash's own drafter; dflash/mtp are Qwen's.
        "--backend", choices=("dflash2", "mtp", "dspark"), default="dflash2"
    )
    parser.add_argument(
        "--suite",
        default="benchmarks/qwen3.8-27b/speculative-corpus.json",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-mode",
        choices=("auto", "raw", "chat"),
        default="auto",
        help="auto uses production chat framing; raw is an explicit comparison",
    )
    parser.add_argument(
        "--system-prompt",
        help="chat-mode system prompt; DSpark auto mode uses the upstream default",
    )
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--min-draft-tokens", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument(
        "--min-verification-steps",
        type=int,
        default=0,
        help="reject sparse per-prompt samples (DSpark defaults to 8)",
    )
    parser.add_argument("--allow-sparse", action="store_true")
    parser.add_argument(
        "--ar-cache",
        default="",
        help="Reuse autoregressive reference runs across invocations. The\ncache is keyed by binary, target shard, framing, decode length and\nprompt, so comparing several draft companions against one target\nmeasures the reference once.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="Write the per-case and aggregate results to this path",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Free-form tag recorded in the JSON report",
    )
    args = parser.parse_args()

    if (
        args.max_tokens <= 0
        or args.draft_tokens <= 0
        or args.min_draft_tokens <= 0
        or args.min_draft_tokens > args.draft_tokens
        or args.repetitions <= 0
        or args.min_verification_steps < 0
    ):
        parser.error("token counts and repetitions must be positive")

    prompts = load_prompts(Path(args.suite), args.quick, args.limit, args.case)
    environment: dict[str, str] = {}
    print(
        "| prompt | category | exact | AR tok/s | speculative tok/s | "
        "speedup | support acceptance | positional | full blocks | attempts | "
        "skipped | avg support |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |"
    )

    cache_path = Path(args.ar_cache) if args.ar_cache else None
    cache = load_autoregressive_cache(cache_path)
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    skipped_cases: list[str] = []
    sparse_samples: list[str] = []
    total_tokens = 0
    total_ar_tokens = 0
    total_ar_seconds = 0.0
    total_spec_seconds = 0.0
    total_drafted = 0
    total_accepted = 0
    total_steps = 0
    total_skipped = 0
    total_positional_accepted = 0
    total_full_blocks = 0
    have_positional = True
    have_full_blocks = True
    speedups: list[float] = []
    minimum_steps = args.min_verification_steps
    if minimum_steps == 0 and args.backend == "dspark":
        minimum_steps = 8

    for case in prompts:
        key = autoregressive_key(args, case["text"])
        cached = cache.get(key)
        try:
            if cached is None:
                autoregressive = run_prompt(args, case["text"], False, {})
                cache[key] = autoregressive
                store_autoregressive_cache(cache_path, cache)
            else:
                autoregressive = cached
            speculative_runs = [
                run_prompt(args, case["text"], True, environment)
                for _ in range(args.repetitions)
            ]
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            # One pathological target/companion pairing must not abort the
            # suite: that it did not finish is itself a result, and the
            # remaining cases still carry information.
            skipped_cases.append(f"{case['id']}: {type(error).__name__}")
            print(
                f"| {case['id']} | {case['category']} | - | - | - | - | - | "
                f"- | - | - | - | - |"
            )
            continue
        exact = all(
            run["completion"] == autoregressive["completion"]
            and run["tokens"] == autoregressive["tokens"]
            and run["token_sha256"] == autoregressive["token_sha256"]
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
        attempts = int(speculative_runs[0]["steps"])
        skipped = int(speculative_runs[0].get("skipped", 0))
        positional = (
            statistics.median(
                float(run["positional_acceptance"])
                for run in speculative_runs
            )
            if all("positional_acceptance" in run for run in speculative_runs)
            else None
        )
        full_block_rate = (
            statistics.median(
                float(run["full_block_rate"]) for run in speculative_runs
            )
            if all("full_block_rate" in run for run in speculative_runs)
            else None
        )
        ar_seconds = float(autoregressive["seconds"])
        ar_tps = float(autoregressive["tps"])
        speedup = spec_tps / ar_tps if ar_tps > 0.0 else 0.0
        speedups.append(speedup)

        representative = speculative_runs[0]
        total_tokens += int(representative["tokens"])
        total_ar_tokens += int(autoregressive["tokens"])
        total_ar_seconds += ar_seconds
        total_spec_seconds += spec_seconds
        total_drafted += int(representative["drafted"])
        total_accepted += int(representative["accepted"])
        total_steps += int(representative["steps"])
        total_skipped += int(representative.get("skipped", 0))
        if "positional_accepted" in representative:
            total_positional_accepted += int(
                representative["positional_accepted"]
            )
        else:
            have_positional = False
        if "full_blocks" in representative:
            total_full_blocks += int(representative["full_blocks"])
        else:
            have_full_blocks = False

        positional_cell = (
            f"{positional * 100.0:.1f}%" if positional is not None else "-"
        )
        full_block_cell = (
            f"{full_block_rate * 100.0:.1f}%"
            if full_block_rate is not None
            else "-"
        )
        print(
            f"| {case['id']} | {case['category']} | "
            f"{'yes' if exact else 'NO'} | {ar_tps:.2f} | {spec_tps:.2f} | "
            f"{speedup:.2f}x | {acceptance * 100.0:.1f}% | "
            f"{positional_cell} | {full_block_cell} | {attempts} | {skipped} | "
            f"{average_draft:.2f} |"
        )
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "exact": exact,
                "ar_tps": ar_tps,
                "spec_tps": spec_tps,
                "speedup": speedup,
                "acceptance": acceptance,
                "average_draft": average_draft,
                "reference": autoregressive,
                "runs": speculative_runs,
            }
        )
        if minimum_steps > 0 and attempts < minimum_steps:
            sparse_samples.append(
                f"{case['id']}: {attempts} attempted blocks, need {minimum_steps}"
            )
        if not exact:
            detail = mismatch_summary(
                str(autoregressive["completion"]),
                str(representative["completion"]),
            )
            mismatches.append(
                f"{case['id']}: {detail}; tokens AR={autoregressive['tokens']} "
                f"spec={[run['tokens'] for run in speculative_runs]}"
            )

    if not rows:
        print()
        print(
            f"aggregate: prompts={len(prompts)} completed=0 "
            f"skipped={len(skipped_cases)}"
        )
        for entry in skipped_cases:
            print(f"skipped: {entry}", file=sys.stderr)
        return 1

    aggregate_ar = total_ar_tokens / total_ar_seconds
    aggregate_spec = total_tokens / total_spec_seconds
    aggregate_acceptance = (
        total_accepted / total_drafted if total_drafted else 0.0
    )
    aggregate_positional = (
        total_positional_accepted / total_drafted
        if have_positional and total_drafted
        else None
    )
    aggregate_full_rate = (
        total_full_blocks / total_steps
        if have_full_blocks and total_steps
        else None
    )
    suite_hash = hashlib.sha256(
        Path(args.suite).read_bytes()
    ).hexdigest()[:12]
    positional_summary = (
        f"positional={aggregate_positional * 100.0:.1f}% "
        if aggregate_positional is not None
        else ""
    )
    full_block_summary = (
        f"full_blocks={aggregate_full_rate * 100.0:.1f}% "
        if aggregate_full_rate is not None
        else ""
    )
    print()
    print(
        f"aggregate: prompts={len(prompts)} suite={suite_hash} "
        f"exact={len(rows) - len(mismatches)}/{len(rows)} "
        f"skipped={len(skipped_cases)} "
        f"AR={aggregate_ar:.2f} tok/s speculative={aggregate_spec:.2f} tok/s "
        f"speedup={aggregate_spec / aggregate_ar:.2f}x "
        f"median_speedup={statistics.median(speedups):.2f}x "
        f"acceptance={aggregate_acceptance * 100.0:.1f}% "
        f"{positional_summary}{full_block_summary}"
        f"attempts={total_steps} skipped={total_skipped} "
        f"avg_support={total_drafted / max(total_steps, 1):.2f}"
    )
    if args.json_path:
        prompt_mode = args.prompt_mode
        if prompt_mode == "auto":
            prompt_mode = "chat"
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "schema": "gufo.speculative-corpus-report.v1",
                    "label": args.label,
                    "binary": os.path.realpath(args.binary),
                    "model": os.path.realpath(args.model),
                    "draft_model": os.path.realpath(args.draft_model),
                    "artifacts": {
                        "binary": artifact_identity(args.binary),
                        "model": artifact_identity(args.model),
                        "draft": artifact_identity(args.draft_model),
                    },
                    "backend": args.backend,
                    "profile": "production",
                    "prompt_mode": prompt_mode,
                    "suite": str(args.suite),
                    "suite_hash": suite_hash,
                    "max_tokens": args.max_tokens,
                    "draft_tokens": args.draft_tokens,
                    "cases": rows,
                    "aggregate": {
                        "prompts": len(prompts),
                        "completed": len(rows),
                        "skipped": skipped_cases,
                        "exact": len(rows) - len(mismatches),
                        "ar_tps": aggregate_ar,
                        "spec_tps": aggregate_spec,
                        "speedup": aggregate_spec / aggregate_ar,
                        "median_speedup": statistics.median(speedups),
                        "acceptance": aggregate_acceptance,
                        "average_draft": total_drafted / max(total_steps, 1),
                    },
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    for mismatch in mismatches:
        print(f"mismatch: {mismatch}", file=sys.stderr)
    for entry in skipped_cases:
        print(f"skipped: {entry}", file=sys.stderr)
    for sparse in sparse_samples:
        print(f"sparse: {sparse}", file=sys.stderr)

    mismatch_failed = bool(mismatches) and not args.allow_mismatch
    sparse_failed = bool(sparse_samples) and not args.allow_sparse
    return 1 if mismatch_failed or sparse_failed or skipped_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())

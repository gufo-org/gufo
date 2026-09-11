#!/usr/bin/env python3
"""Run the owned DS4 checks inside nix develop; model artifacts are required."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "fast": ("^ds4\\.(template|cli|dataset|eval)$", ["ds4_chat_template_test", "ds4_cli_test", "ds4_eval_test"]),
    "kernels": ("^ds4\\.(projections|attention)$",
                ["ds4_projection_test",
                 "ds4_attention_test"]),
    "model": ("^ds4\\.(target|dspark|serving)$", ["ds4_quality_test", "ds4_serving_test"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=(*SUITES, "reference", "benchmark", "all"),
                        default="all", nargs="?")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dspark-model", type=Path)
    parser.add_argument("--output", type=Path,
                        help="new reference directory or benchmark report file")
    parser.add_argument("--upstream", type=Path,
                        help="clean antirez/ds4 checkout at the pinned reference revision")
    parser.add_argument("--prefill-only", action="store_true",
                        help="reference: compare post-prefill logits, skipping continuations")
    parser.add_argument("--ar-log", type=Path)
    parser.add_argument("--dspark-log", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if not os.environ.get("IN_NIX_SHELL"):
        parser.error("run with nix develop -c tools/ds4/check.py")
    if args.prefill_only and args.suite != "reference":
        parser.error("--prefill-only requires the reference suite")
    if args.suite == "benchmark":
        if not args.ar_log or not args.dspark_log or not args.output:
            parser.error("benchmark requires --ar-log, --dspark-log and --output")
        if args.model or args.dspark_model or args.upstream:
            parser.error("benchmark validates existing logs; model/upstream options do not apply")
        from benchmark import summarize
        summarize(args.ar_log, args.dspark_log, args.repetitions, args.output)
        return
    if args.ar_log or args.dspark_log or args.repetitions != 2:
        parser.error("benchmark log/repetition options require the benchmark suite")
    environment = os.environ.copy()
    reference = args.suite == "reference"
    if reference != (args.output is not None):
        parser.error("--output is required only for reference")
    if reference != (args.upstream is not None):
        parser.error("--upstream is required only for reference")
    if reference and (args.output.exists() or not args.upstream.is_dir()):
        parser.error("reference needs a new output directory and existing upstream checkout")
    if args.suite in ("model", "all", "reference"):
        models = [(args.model, "GUFO_DEEPSEEK_V4_FLASH_MODEL")]
        if not reference:
            models.append((args.dspark_model, "GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL"))
        for value, name in models:
            value = value or environment.get(name)
            if not value or not Path(value).is_file():
                parser.error(f"model checks require an existing artifact: {name}")
            environment[name] = str(Path(value).absolute())
    regex, targets = SUITES.get(args.suite, ("^ds4\\.", [
        target for _, group in SUITES.values() for target in group]))
    if reference:
        targets = ["ds4_quality_test"]
    commands = [
        ["cmake", "--preset", "gpu-test"],
        ["cmake", "--build", "--preset", "gpu-test", "--target", *targets],
    ]
    if not reference:
        commands.append(
            ["ctest", "--test-dir", "build/gpu-test", "-R", regex,
             "--output-on-failure", "--stop-on-failure"])
    for command in commands:
        subprocess.run(command, cwd=ROOT,
                       env=environment, check=True)
    if reference:
        from reference import run
        run(Path(environment["GUFO_DEEPSEEK_V4_FLASH_MODEL"]),
            args.output.resolve(), args.upstream.resolve(), environment, args.prefill_only)


if __name__ == "__main__":
    main()

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
    "kernels": ("^ds4\\.q2-down$", ["ds4_q2_down_test"]),
    "model": ("^ds4\\.(target|dspark|serving)$", ["ds4_quality_test", "ds4_serving_test"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=(*SUITES, "all"), default="all", nargs="?")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dspark-model", type=Path)
    args = parser.parse_args()
    if not os.environ.get("IN_NIX_SHELL"):
        parser.error("run with nix develop -c tools/ds4/check.py")
    environment = os.environ.copy()
    if args.suite in ("model", "all"):
        for value, name in ((args.model, "GUFO_DEEPSEEK_V4_FLASH_MODEL"),
                            (args.dspark_model, "GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL")):
            value = value or environment.get(name)
            if not value or not Path(value).is_file():
                parser.error(f"model checks require an existing artifact: {name}")
            environment[name] = str(Path(value).resolve())
    regex, targets = SUITES.get(args.suite, ("^ds4\\.", [
        target for _, group in SUITES.values() for target in group]))
    for command in (
        ["cmake", "--preset", "gpu-test"],
        ["cmake", "--build", "--preset", "gpu-test", "--target", *targets],
        ["ctest", "--test-dir", "build/gpu-test", "-R", regex, "--output-on-failure"],
    ):
        subprocess.run(command, cwd=ROOT,
                       env=environment, check=True)


if __name__ == "__main__":
    main()

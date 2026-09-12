#!/usr/bin/env python3
"""Focused Qwen27B checks; run on gfx1151 with nix develop -c python3."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "fast": ["qwen_aie2p_w4a8_pack_test", "logit_sampler_test",
             "prompt_cli_test", "bench_cli_test", "openai_chat_test",
             "speculative_verification_test"],
    "kernels": ["qwen_gpu_ops_test", "qwen_ssm_ops_test",
                "qwen_attention_kv_storage_ops_test",
                "qwen_attention_fusion_ops_test",
                "qwen_dflash_noncausal_attention_ops_test",
                "qwen_prefill_quant_gemm_ops_test", "qwen_q4kxl_quant_ops_test",
                "qwen_quant_gemv_ops_test", "qwen_sampling_hip_test",
                "speculative_verification_test"],
    "serving": ["inference_backend_gpu_test"],
    "model": ["qwen27b_target_test", "qwen_mtp_gpu_test", "qwen_dflash_gpu_test"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=[*SUITES, "reference", "all"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mtp-model", type=Path)
    parser.add_argument("--dflash-model", type=Path)
    parser.add_argument("--reference-model", type=Path,
                        help="reference suite only: optional BF16 target GGUF")
    args = parser.parse_args()
    if not os.environ.get("IN_NIX_SHELL"):
        parser.error("run inside nix develop")
    if (args.suite == "reference") != (args.reference_model is not None):
        parser.error("--reference-model is required only for reference")
    environment = os.environ.copy()
    required = ["model"] if args.suite == "reference" else (
        ["model", "mtp_model", "dflash_model"] if args.suite in ("model", "all") else
        ["model", "dflash_model"] if args.suite == "serving" else [])
    for name in required:
        variable = "GUFO_QWEN27B_" + name.upper()
        value = getattr(args, name) or environment.get(variable)
        if not value or not Path(value).is_file():
            parser.error(f"missing artifact: --{name.replace('_', '-')}")
        environment[variable] = str(Path(value).resolve())
    if args.reference_model and not args.reference_model.is_file():
        parser.error("reference model does not exist")
    targets = (["qwen27b_target_test"] if args.suite == "reference" else
               SUITES.get(args.suite, [t for group in SUITES.values() for t in group]))
    commands = [
        ["cmake", "--preset", "gpu-test"],
        ["cmake", "--build", "--preset", "gpu-test", "--target", *targets,
         *(["gufo"] if args.suite in ("fast", "model", "all") else [])],
    ]
    if args.suite == "reference":
        commands.append([
            str(ROOT / "build/gpu-test/tests/models/qwen27b/qwen27b_target_test"),
            environment["GUFO_QWEN27B_MODEL"], str(args.reference_model.resolve()),
        ])
    else:
        names = targets + (["qwen27b.tools"] if args.suite in ("fast", "all") else [])
        if args.suite in ("fast", "all"):
            names.append("qwen27b.cli-options")
        if args.suite in ("model", "all"):
            names.append("qwen27b.cli")
        commands.append(["ctest", "--test-dir", "build/gpu-test", "-R",
                         "^(" + "|".join(names) + ")$",
                         "--output-on-failure", "--stop-on-failure", "--no-tests=error"])
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()

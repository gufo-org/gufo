#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "quant" / "speculative-corpus.py"
SPEC = importlib.util.spec_from_file_location("speculative_corpus", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load speculative corpus module")
speculative_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speculative_corpus)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def make_args(backend):
    return SimpleNamespace(
        binary="./result/bin/gufo",
        model="/models/target.gguf",
        draft_model="/models/draft.gguf",
        backend=backend,
        max_tokens=128,
        draft_tokens=7,
        min_draft_tokens=1,
        prompt_mode="auto",
        system_prompt=None,
    )


dspark_command = speculative_corpus.build_prompt_command(
    make_args("dspark"), "Explain speculative decoding.", speculative=True
)
check("--raw" not in dspark_command, "DSpark auto mode must use chat framing")
system_index = dspark_command.index("--system")
check(
    dspark_command[system_index + 1]
    == speculative_corpus.DSPARK_REFERENCE_SYSTEM_PROMPT,
    "DSpark auto mode must use the reference system prompt",
)
check("--dspark-model" in dspark_command, "DSpark model argument")
check(dspark_command[dspark_command.index("--draft-tokens") + 1] == "7",
      "DSpark benchmark retains the requested draft budget")
check("--draft-policy" not in dspark_command, "removed policy is never emitted")

custom_args = make_args("dspark")
custom_args.system_prompt = "Custom system prompt"
custom_command = speculative_corpus.build_prompt_command(
    custom_args, "Explain speculative decoding.", speculative=True
)
custom_system_index = custom_command.index("--system")
check(
    custom_command[custom_system_index + 1] == "Custom system prompt",
    "explicit DSpark system prompt override",
)

dflash_command = speculative_corpus.build_prompt_command(
    make_args("dflash2"), "Continue this text.", speculative=True
)
check("--raw" in dflash_command, "DFlash2 auto mode must retain raw framing")
check("--system" not in dflash_command, "raw DFlash2 must not add a system prompt")
check("--dflash-model" in dflash_command, "DFlash2 model argument")

dspark_stats = speculative_corpus.parse_speculative_stats(
    "[Speculative]: acceptance=0.511111 drafted=45 accepted=23 "
    "verification_steps=9 skipped=63 positional_acceptance=0.777778 "
    "positional_accepted=35 full_block_rate=0.222222 full_blocks=2 "
    "anchors=9 verifier_rows=54"
)
check(dspark_stats["drafted"] == 45, "support drafted rows")
check(dspark_stats["accepted"] == 23, "support accepted rows")
check(dspark_stats["steps"] == 9, "attempted blocks")
check(dspark_stats["verifier_rows"] == 54, "all verifier rows")
check(dspark_stats["anchors"] == 9, "known target anchors")
check(
    abs(dspark_stats["positional_acceptance"] - 0.777778) < 1.0e-9,
    "independent positional acceptance",
)
check(
    abs(dspark_stats["full_block_rate"] - 0.222222) < 1.0e-9,
    "full block rate",
)

legacy_stats = speculative_corpus.parse_speculative_stats(
    "[Speculative]: acceptance=0.5 drafted=20 accepted=10 "
    "verification_steps=4"
)
check(legacy_stats["drafted"] == 20, "legacy drafted rows")
check(legacy_stats["accepted"] == 10, "legacy accepted rows")
check(legacy_stats["steps"] == 4, "legacy verification steps")
check("positional_acceptance" not in legacy_stats, "legacy optional metrics")

print("Speculative corpus harness tests passed.")

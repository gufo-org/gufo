#!/usr/bin/env python3

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "quant" / "speculative-corpus.py"
SPEC = importlib.util.spec_from_file_location("speculative_corpus", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load speculative corpus module")
speculative_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speculative_corpus)
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


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
check("--raw" not in dflash_command, "DFlash2 auto mode uses production chat framing")
check("--system" not in dflash_command, "Qwen keeps its normal system prompt")
check("--dflash-model" in dflash_command, "DFlash2 model argument")
controller_args = make_args("dflash2")
controller_args.draft_policy = "adaptive"
controller_command = speculative_corpus.build_prompt_command(
    controller_args, "Continue this text.", speculative=True
)
check(controller_command[controller_command.index("--draft-policy") + 1] ==
      "adaptive", "DFlash2 corpus runs the requested controller")
check("--draft-policy" not in speculative_corpus.build_prompt_command(
    controller_args, "Continue this text.", speculative=False
), "controller choice never changes the autoregressive reference command")

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

check(
    speculative_corpus.extract_completion(
        "--- Generation Output ---\nanswer\n\n"
        "Generated 2 tokens on GPU in 1.00s (2.00 tok/s)\n"
    ) == "answer\n",
    "comparison retains a generated trailing newline",
)

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    args = make_args("dflash2")
    args.binary = str(directory / "gufo")
    args.model = str(directory / "target.gguf")
    args.draft_model = str(directory / "draft.gguf")
    for path in (args.binary, args.model, args.draft_model):
        Path(path).write_bytes(b"baseline")
    original_key = speculative_corpus.autoregressive_key(args, "prompt")
    Path(args.binary).write_bytes(b"new binary")
    check(
        speculative_corpus.autoregressive_key(args, "prompt") != original_key,
        "rebuilding a binary at the same path invalidates cached references",
    )
    original_key = speculative_corpus.autoregressive_key(args, "prompt")
    Path(args.model).write_bytes(b"new model")
    check(
        speculative_corpus.autoregressive_key(args, "prompt") != original_key,
        "replacing a model at the same path invalidates cached references",
    )
    suite = directory / "suite.json"
    suite.write_text(json.dumps({"prompts": [
        {"id": "ok", "category": "test", "text": "ok"},
        {"id": "failed", "category": "test", "text": "failed"},
    ]}))
    command = [
        str(SCRIPT), "--binary", args.binary, "--model", args.model,
        "--draft-model", args.draft_model, "--suite", str(suite),
    ]

    def result(tokens=2):
        return dict(completion="answer", tokens=tokens, seconds=1.0, tps=2.0,
                    acceptance=0.5, drafted=2, accepted=1, steps=1,
                    token_sha256="a" * 64)

    def failed_case(args, prompt, speculative, environment):
        if prompt == "failed":
            raise RuntimeError("deliberate failed invocation")
        return result()

    def different_token_count(args, prompt, speculative, environment):
        return result(tokens=3 if speculative else 2)

    def different_token_ids(args, prompt, speculative, environment):
        row = result()
        if speculative:
            row["token_sha256"] = "b" * 64
        return row

    for run in (failed_case, different_token_count, different_token_ids):
        with patch("sys.argv", command), patch.object(
            speculative_corpus, "run_prompt", side_effect=run
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            check(speculative_corpus.main() != 0,
                  "incomplete or token-mismatched comparisons fail")

    for trace in ("", "[TokenTrace]: count=1 sha256=" + "a" * 64):
        try:
            speculative_corpus.parse_token_trace(trace, 2)
        except RuntimeError:
            pass
        else:
            check(False, "missing or partial token traces must fail")

if FAILURES:
    raise AssertionError("\n".join(FAILURES))
print("Speculative corpus harness tests passed.")

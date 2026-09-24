# Testing

Run the smallest check that covers the change. Correctness and model-owned
quality limits pass before performance results count. Production targets
Linux gfx1151. CMake/CTest work with either system dependencies or `nix develop`.

## Commands

```sh
# Formatting on the editing host, using CI's pinned formatter
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting.

# Documentation changes
nix build .#checks.x86_64-linux.docs

# Hosted CPU/repository checks
nix build .#checks.x86_64-linux.pr

# Focused test (replace the target and expression)
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target minimax_h3_sampling_test
nix develop -c ctest --preset gpu-full -R '^minimax_h3_sampling_test$' --output-on-failure

# Host memory/undefined-behavior diagnostics
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer
nix develop -c ctest --preset cpu-sanitizer --output-on-failure
```

Preset definitions, build options and repository mechanics are in
[development](DEVELOPMENT.md). Formatting and the Python
documentation/dependency checks run on the editing host without building Gufo;
their entrypoints are in `tools/ci/`.

Selection by preset: `gpu-fast` excludes `slow` and `external-model` tests,
`deepseek-gpu` and `qwen-gpu-kernel-oracle` select model/device suites, and
`gpu-full` covers the complete hardware tree. External-model tests require
their documented local artifacts. A skip due to an absent model or device is
not a quality pass.

## Hosted versus local checks

GitHub runs formatting, documentation, dependency-inventory and server-command
checks plus `cmake/Checks.cmake`: parsing, GGUF safety, sampling, templates,
server/cache/scheduler behavior, model configuration and small audio/video API
contracts. It builds only those test executables and the CPU CLI. It does not
build ROCm, download weights, run full H3 oracles or compile whole-tree clang-tidy.
The workflow cancels superseded runs and has a 25-minute limit.

```sh
# Same hosted C++/CLI selection locally
cmake --preset cpu-test
cmake --build --preset pr --parallel 4

# Full CPU suite, or explicit non-hosted tools
nix build .#checks.x86_64-linux.tests
nix build .#checks.x86_64-linux.static-analysis
nix build .#checks.x86_64-linux.h3-manifest
nix build .#checks.x86_64-linux.h3-quality
nix build .#checks.x86_64-linux.h3-ml-quality
```

Full CPU builds need Python NumPy for the offline tool tests. GPU model oracles
need the documented local weights; use focused targets and CTest labels before
running the complete tree. H3 development normally needs an analytic primitive,
one transformer block or one denoiser forward. Full video generation and LPIPS
are release/quality qualification, not routine PR work. `nix flake check` runs
all declared checks and is deliberately not the hosted CI command.

## Match validation to the change

| Change | Relevant checks |
| --- | --- |
| Documentation | local links, anchors, JSON examples |
| Loader/container/tokenizer | independent fixtures, malformed/truncated input, exact token IDs |
| Kernel or quantizer | analytic oracle, affected layer/model boundaries, declared quality limits |
| Attention or state layout | long-context boundaries, snapshot/restore, fresh-versus-reused state |
| Sampling/speculation | distributions, proposal/rejection/residual draws, seeded replay, EOS commit |
| Scheduler/server | direct-versus-HTTP behavior, independent batched requests, multi-turn reuse, cancellation, failure and shutdown |
| Performance | affected quality gates followed by one comparable release measurement |

Broaden testing when a change crosses shared ownership boundaries or a failure
reveals an unresolved risk. Full model sweeps, media generation, and repeated
timing runs are not the default development loop.

Model-owned procedures and outstanding qualification gaps live with the model:

- [DeepSeek V4 Flash](models/deepseek-v4-flash/QUALITY.md)
- [Qwen3.8 27B](models/qwen3.8-27b/QUALITY.md)
- [Qwen3.8 Flash-Next](models/qwen3.8-flash-next/QUALITY.md)
- [MiniMax H3](models/minimax-h3/QUALITY.md)
- [Qwen3-TTS](models/qwen3-tts/QUALITY.md)
- [Qwen3-ASR](models/qwen3-asr/QUALITY.md)

## Independent references and artifacts

Prefer hand-checkable analytic fixtures or the source checkpoint in a pinned
official runtime. A promoted Gufo build is a regression reference; agreement
with it alone does not establish upstream correctness. Record checkpoint
storage type separately from compute and accumulation precision.

Unchanged-arithmetic refactors retain exact boundaries. Reassociated kernels
need declared numerical limits and model-level checks. Quantization is
evaluated against a full-quality teacher with distribution, perplexity, and
capability gates. Do not loosen tolerances or silently replace goldens to
admit a change.

Golden artifacts record immutable model and oracle revisions, file hashes,
tokenizer/template identity, prompt/token-stream hashes, dtypes, parameters,
and schema version. Large tensor/logit dumps stay outside Git. Keep small
independent fixtures and concise results needed to reproduce qualification;
see [benchmark artifact retention](BENCHMARKS.md).

## Matched-token and layer comparisons

Tokenize once and feed the same predetermined history to teacher and
candidate. Compare full next-token distributions at each position before
appending the next evaluation token. Free-running text diverges after one
different choice and cannot isolate numerical or quantization error.

Relevant metrics include non-finite counts, normalized-logit error, KL,
teacher-token NLL/perplexity, top-1 agreement, and top-k overlap. See
[benchmark quality methodology](BENCHMARKS.md#quality-method-matched-token-per-position).
For a discrepancy, capture the earliest changed embedding, normalization,
attention/recurrent state, routed expert, layer output, or LM-head boundary.
Preserve the failing input and add a focused reproducer before fixing it.

Pin sampling controls, seeds, cache history, and batching when testing replay.
Run longer multi-turn cases when session reuse or EOS handling changes.
Do not average away numerical mismatches or classify device failures as passes.

## Performance evidence

Compare identical model artifacts, prompt/output counts, context depths,
batching, sampling, and cache state. Measure request latency and aggregate
throughput separately. Run baseline and candidate close together; add
repetitions only when noise or a suspected regression requires them.
Record hardware/software identities and the exact timed scope. Use
[profiling tools](PERFORMANCE.md) to explain a gain before retaining it.

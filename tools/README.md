# Developer tools

Use the Nix development environment for these tools. Production inference and
speed measurements use `nix build` binaries under `result/bin` or the
`release` CMake preset with the same toolchain. Tools are not part of the
production install. `gpu-test` enables `GUFO_BUILD_TOOLS` for the compiled
benchmark/tuning executables; Python tools run from this source tree.

| Directory | Purpose |
| --- | --- |
| `bench/` | Kernel microbenchmarks, hipBLASLt tuning, and direct speculative-corpus checks; `model-bench.py` drives per-model BENCHMARKS.md tables against each model's reference |
| `prof/` | rocprofv3 capture, stage summaries, and ISA inspection |
| `serving/` | HTTP benchmarks, interrupted-chat/disk replay and official OpenAI SDK checks |
| `ds4/` | DeepSeek-specific validation and experiments |
| `qwen27b/` | Qwen27B/DFlash2 kernels, reference checks, and vision validation |
| `qwen-flash/` | Flash-Next projection and MoE microbenchmarks |
| `h3/` | MiniMax inventory and quality-artifact commands |
| `audio/` | Audio reference and quality tools |
| `ci/` | Repository, dependency, and documentation checks |
| `gufo/` | Shared Python helpers and model-specific reference implementations |

`gufo/gguf.py` reads metadata and decodes existing GGUF tensors for independent
Qwen reference tests. `gufo/safetensors.py` validates source snapshots used by
MiniMax's manifest and quality tooling. Both support model verification.

See [performance commands](../docs/PERFORMANCE.md),
[benchmark methodology](../docs/BENCHMARKS.md), and
[MiniMax validation](../docs/models/minimax-h3/QUALITY.md).

## Responses client check

With a local Gufo text server running, use its API model name from `/v1/models`
(`qwen` here), rather than its weights path:

```sh
nix develop -c python3 tools/serving/check-responses-sdk.py \
  --base-url http://127.0.0.1:8080/v1 --model qwen
```

Add `--expect-reasoning` when the server defaults to thinking. This uses the
official OpenAI SDK with strict schema validation: buffered/streamed output,
reasoning, conversation replay, async concurrency, limits and disconnects.

## Qwen27B

`qwen27b/check.py` owns the focused checks (`fast`, `kernels`, `model`, `serving`).
Its optional `reference` suite compares target logits with an explicitly
provided BF16 artifact. `qwen27b/drafts.py` compares Q4/Q8/BF16 DFlash2 companions and refuses
incomplete or mismatching results. `--baseline-binary` interleaves two releases
and also requires identical autoregressive token traces between them.
`qwen27b/dflash_gemm_bench.hip` measures exact BF16, Q4/Q5/Q6/Q8 and IQ4_XS
matrix geometries using the production templates; build it with `tools/bench/build.sh` inside Nix.
This benchmark uses the production `-O2` optimization level. `q3` and `iq4-nl`
also cover native fourteen/sixteen-position FFNs, for example
`q3 17408 5120 24 16` or `iq4-nl 17408 5120 24 14`.
Its optional batch argument selects 2–16 projection rows; BF16 also supports
24/32 rows. Examples: `q4 17408 5120 24 7`, `q8 34816 5120 16 12`,
`q6 248320 5120 12 16` and `bf16 12288 5120 16 32`.
Wide runs compare against smaller groups as an arithmetic oracle; their timing
does not represent every tuned production shape. Native production launches
are included for supported K-quant and Q8 shapes.
Batch 1 compares scalar Q4/Q5/Q6/IQ4 dispatch; append `swiglu` for fused
gate/up projections, for example `q5 17408 5120 32 1 swiglu`.
An optional final up-format compares mixed pairs, for example
`q4 17408 5120 24 1 swiglu q5`. Supported pairs are Q4/Q5,
Q4/IQ4, IQ4/Q4 and IQ4/Q5; omitting the format keeps gate/up identical.
Q4/Q5/IQ4 comparisons include four-row FFN layouts. Q4 uses compact
two-tile staging at widths 3–8; width 2 retains padding.
`q4 67 768 4 2` checks partial rows and tiles for the shortest verification.
Q5 also covers
the smaller projections, for example `q5 6144 5120 24 4` and
`q5 5120 6144 24 8`. Q6 compares
two- and three-row vocabulary projections; Q5 batch 8 also compares grouped
dots with completing one token at a time.
Weights use read-only host registration, matching production GGUF mapping.
Injection repeats one weight matrix and uses the production cache hint
(`bf16 1024 5120 24 16` covers K/V); decoding rotates at least 128 MiB of weights.
`qwen27b/deltanet_bench.hip` checks exact recurrence and state-only replay
while rotating the 144 MiB target state; `qwen27b/prefill_deltanet_bench.hip`
contains the separate prefill ablations. Both use the same fast Nix builder.
`qwen27b/attention_bench.hip` compares scalar and batched attention. It checks exact outputs
beside component timings, including the 4K split-K boundary; its `rope` mode
compares separate and fused draft normalization/RoPE.
`qwen27b/prefill_gemm_bench.hip` compares production Q8_0 gate/up GEMM
with separate or fused SwiGLU, including every activation byte and scale.
It replaces the old standalone W8A8 kernel copies.
`qwen27b/dflash_reference.py` checks a GPU trace against
pinned upstream PyTorch operators using the same GGUF weights. See
`docs/models/qwen3.8-27b/BENCHMARKS.md` and its quality report for commands and evidence.

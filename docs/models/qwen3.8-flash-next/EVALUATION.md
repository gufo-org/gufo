# Qwen3.8 Flash-Next evaluation

Operator and execution-consistency checks pass. **Original unquantized-model
and GGUF-conversion parity remain unqualified.** The independent vision check
below still exceeds its tolerance. Qualification dates: September 20–22, 2026.

## Quality results

| Area | Retained evidence |
| --- | --- |
| MTP predictor | Eight text/image states audited against scalar CPU operators using original GGUF weights. Fusion/attention relative RMS <0.0008; gate 0.002. Checks cover full-width 10,240-value RMSNorm, split projections, recursive carry, text-only shifted inputs, full Q8 head, and serial/batched/headless catch-up. Norm tolerance: 2e-6; MoE/head: 5e-5. |
| Sampling | AR and MTP verification share FP64 target filtering, normalization and CDFs, with 53-bit target RNG. Tests cover top-p/min-p boundaries, p/q acceptance, residual correction and seeded replay. Proposals use the full head's top 64 logits; upstream draft-sampler equivalence is not claimed. |
| Batching and kernels | C2/C4/C6/C8 retain logits, tokens, acceptance, residual draws and RNG with ragged budgets and every 1–8-token rollback prefix. Q4/Q8 projection and GDN rollback controls retain FP32 bytes; sparse rankings/masks match through 128K. |
| Prefill | Chunk-boundary checks cover 1/8/9/32/33-token tails through 4096 tokens. The 2176-token scalar/prefill control retains top-1 with logit RMSE 0.18. |
| Sessions and serving | AR/MTP cancellation, third-turn continuation, reasoning removal/preservation, concurrent image/text isolation and disk restart pass. A 4095-token snapshot check covers mid-decode save, ring wrap, pending MTP state and RNG. |
| HTTP corpus | Retained Gufo hashes match AR for 21/21 repetitive AR requests, 25/25 mixed MTP requests and 21/21 repetitive MTP requests; zero prompt-cache hits. These are text-consistency checks, not original-model accuracy. |
| Vision optimizations | Complete 256×256/1024×1024 embeddings and a 736×736 ragged control retain native output bytes. Independent upstream parity has the gap below. |

**Vision gap:** a 1024×1024 synthetic texture has 6.47% embedding relative L2
error against the pinned BF16 reference, above the unchanged 5% gate. Native
and upstream BF16 errors against FP32 are 7.83% and 8.10%, respectively; this
does not establish which produces better model output. Reproduce with
`pixels[i] = (137*i + 53*(i//3072)) % 256` and the
[vision reference commands](../qwen3.8-27b/EVALUATION.md#vision).

The CPU oracle uses converted GGUF weights, so it cannot qualify conversion
or the original checkpoint. Pinned Transformers ignores MTP weights. Seeded
replay requires the same build, seed, request budget, capacity and sampling
settings; sampled MTP need not match AR's same-seed sequence. Greedy output
must remain independent of draft width and batching.

## Maintained checks

Tests live in `tests/models/qwen38_flash_next/`; run only the affected check.

| Target / mode | Contract |
| --- | --- |
| `qwen38_flash_next_tests` | 14 operator, configuration and I/O checks: scalar/FP64 references, malformed metadata, unsupported geometry and sidecar compatibility. |
| `qwen38_flash_next_session_test --batch-only` | Independent logits/state/RNG at C2/C4/C6/C8, rollback, cancellation and images. |
| `qwen38_flash_next_session_test --sampling-only` | 23 AR/MTP sampling configurations, penalties, residual correction and short budgets. |
| `qwen38_flash_next_session_test --prefill-only` | Full logits across chunk boundaries and short tails. |
| `qwen38_flash_next_snapshot_test` | Persistence, image attachment and continuation replay; AR/MTP states cannot cross modes. |
| `qwen38_flash_next_gpu_probe --mtp-audit` | Original encoded weights and scalar predictor stages; add `--batch 2048` for ragged/aligned catch-up. |

Build the required targets with Nix, for example:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
PROBES=build/gpu-test/tests/models/qwen38_flash_next
nix develop -c "$PROBES/qwen38_flash_next_gpu_probe" \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

Operator tests can be selected with CTest, for example
`-R 'qwen38_flash_next[.]select_ops' --no-tests=error --output-on-failure`.
The GPU probe's `--cost-audit C --depth N` measures predictor/verification costs;
`C=0` selects C1/C2/C4/C6/C8. Performance uses Nix release binaries and the
[profiling workflow](../../PERFORMANCE.md), separately from correctness runs.

Formula references: pinned [vLLM predictor](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py),
[vLLM proposer](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/v1/spec_decode/llm_base_proposer.py)
and [SGLang predictor](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py).
Gufo follows vLLM's text-only MTP input semantics. Official checkpoint:
`Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## Benchmark method

The [card](BENCHMARKS.md) retains September 22 measurements. This layout change
runs no inference: concurrency rates are recalculated from the original
individual decode rates, summed per cohort and averaged across cohorts.
Per-artifact revisions and commands remain authoritative; see
[model identities](artifacts/model-identities.json).

Single-user rows use approximately pp2048/tg128, greedy, thinking off, seed 1.
Each mode generates its own eight-token reply before the measured continuation.
Depth/prefill tolerance is max(32 tokens, 0.5%). Gufo capacity is 133760;
reference AR uses 35456 through 32K, 68224 at 64K and 133760 at 128K.
Reference MTP uses 35456 through 32K; weight eviction at 64K and OOM at 128K
invalidate deeper measurements. MTP pp takes each engine's maximum across
mixed/repetitive workloads. AR reference: `b11069`; MTP: pinned
`llama-server-mtp` at `6fcaa16f` ([upstream change](https://github.com/ggml-org/llama.cpp/pull/28243)).

Concurrency uses fresh servers, context 4096 per user and tg128. AR uses
`repetition_word`; MTP adds `expository_pangram`, `cpp_ring_buffer` and
`reasoning_train`. Short C1 corpus prompts differ from the pp2048 depth sweep;
MoE routing also makes throughput workload-dependent. Reference MTP C8 was
OOM-killed. Compact `multi-{mixed,repetition}-gufo-ar.json` files retain C1
hashes for MTP qualification; refresh them when arithmetic, weights, tokenizer
or request settings change. Reference-engine agreement is not an accuracy score.

Matched C1/MTP loading at capacity 262144 is TODO. Historical
[Gufo](artifacts/loading-2026-09-22-gufo.json) and
[reference](artifacts/loading-2026-09-22-reference.json) runs used C2/MTP/262144
and C2/AR/35456 total, respectively, so they cannot supply that comparison.
Memory uses C1 AR at capacity 133121, sampling global HIP allocation every
250 ms including 2.38 GiB idle allocation.

`artifacts/bench.json` declares six tables. Use
`tools/bench/model-bench.py --model qwen3.8-flash-next render` through Nix to
regenerate the card without model execution. Approved runs take
`--gguf "$MODEL" --mtp "$MTP" run --target <gufo|reference> --table <table> --fresh`;
select reference depths/capacities explicitly. See the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

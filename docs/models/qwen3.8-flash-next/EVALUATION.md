# Qwen3.8 Flash-Next evaluation

Text/MTP operator and execution-consistency checks pass. **Original unquantized-model
and GGUF-conversion parity remain unqualified.** The independent vision check
below still exceeds its tolerance. Qualification dates: September 20–23, 2026.

## Quality results

| Area | Retained evidence |
| --- | --- |
| MTP predictor | Eight text/image states audited against scalar CPU operators using original GGUF weights. Fusion/attention relative RMS <0.0008; gate 0.002. Checks cover full-width 10,240-value RMSNorm, split projections, recursive carry, text-only shifted inputs, full Q8 head, and serial/batched/headless catch-up. Norm tolerance: 2e-6; MoE/head: 5e-5. |
| Sampling | AR and MTP verification share FP64 target filtering, normalization and CDFs, with 53-bit target RNG. Tests cover top-p/min-p boundaries, p/q acceptance, residual correction and seeded replay. Proposals use the full head's top 64 logits; upstream draft-sampler equivalence is not claimed. |
| Batching and kernels | C2/C4/C6/C8 retain logits, tokens, acceptance, residual draws and RNG with ragged budgets and every 1–8-token rollback prefix. Q4/Q8 projection and GDN rollback controls retain FP32 bytes; sparse rankings/masks match through 128K. |
| Prefill | Chunk-boundary checks cover 1/8/9/32/33-token tails through 4096 tokens. The 2176-token scalar/prefill control retains top-1 with logit RMSE 0.18. |
| Sessions and serving | AR/MTP cancellation, third-turn continuation, reasoning removal/preservation, concurrent image/text isolation and disk restart pass. A 4095-token snapshot check covers mid-decode save, ring wrap, pending MTP state and RNG. |
| HTTP corpus | All 63 Gufo tg128 requests (21 AR, 21 mixed MTP, 21 repetitive MTP) match independent fresh-prefill AR hashes. Prepared sessions reuse every prompt token. These are text-consistency checks, not original-model accuracy. |
| Vision optimizations | Complete 256×256/1024×1024 embeddings and a 736×736 ragged control retain native output bytes. Independent upstream parity has the gap below. |

**Vision:** Gufo exceeds one encoder parity gate: a 1024×1024 synthetic texture
produces 6.47% embedding relative L2 error versus official Transformers BF16
operators (limit: 5%). Both use the same converted GGUF weights; **llama.cpp
was not tested**. Against the FP32 control, Gufo and Transformers BF16 differ
by 7.83% and 8.10%, respectively. This measures numerical drift, not image-answer
accuracy. [Retained evidence](artifacts/vision-parity.json) ·
[Reproduction](../qwen3.8-27b/EVALUATION.md#vision).

Pinned Transformers ignores MTP weights. Seeded
replay requires the same build, seed, request budget, capacity and sampling
settings; sampled MTP need not match AR's same-seed sequence. Greedy output
must remain independent of draft width and batching.

## Maintained checks

Tests live in `tests/models/qwen38_flash_next/`; run only the affected check.

| Target / mode | Contract |
| --- | --- |
| `qwen38_flash_next_tests` | Operator, configuration and I/O checks: scalar/FP64 references, malformed metadata, unsupported geometry and sidecar compatibility. |
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

The [card](BENCHMARKS.md) refreshes Gufo single-user MTP, both engines’
concurrency and loading on September 23; other values remain from September 22.
One measured run per point; concurrency sums individual decode rates.
Versions and commands are in [model identities](artifacts/model-identities.json)
and the table artifacts.

Single-user rows use approximately pp2048/tg128, greedy, thinking off.
Each mode generates its own eight-token reply before the measured continuation.
Depth/prefill tolerance is max(32 tokens, 0.5%). Gufo capacity is 133760;
reference AR uses 35456 through 32K, 68224 at 64K and 133760 at 128K.
Reference MTP uses 35456 through 32K; weight eviction at 64K and OOM at 128K
invalidate deeper measurements. MTP pp takes each engine's maximum across
mixed/repetitive workloads. AR reference: `b11069`; MTP: pinned
`llama-server-mtp` at `6fcaa16f` ([upstream change](https://github.com/ggml-org/llama.cpp/pull/28243)).

Concurrency uses the **same pp2048 d0 prompts**: prose for AR/mixed MTP,
passage-copying for repetitive MTP, tg128, context 4096 per user. Every user
receives the same prompt; servers are fresh per C1/C2/C4/C6/C8. All sessions
finish a one-token preparation request before the timed tg128 cohort repeats
those prompts with caching enabled. llama.cpp slots are pinned; at most four
final prompt tokens may be reevaluated. This excludes peers' long prefills from
llama.cpp's elapsed generation clock; Gufo reports active decode time.
The reference clock can still include interference from the four-token tail.
Gufo MTP d0/C1 rates agree within 0.4%, with matching completions and draft counts.
llama.cpp's mixed-MTP completions vary with concurrency; the artifacts retain their hashes.
Gufo completions must match the fresh AR controls in
`multi-{mixed,repetition}-gufo-ar.json`. Refresh those when prompts, arithmetic,
weights or tokenizer change. Cross-engine agreement is not an accuracy score.

Loading uses C1/MTP/capacity 262144, from cold model files to HTTP readiness.
`POSIX_FADV_DONTNEED` plus `mincore` verified zero resident pages for all four
target shards and the sidecar. The reference ran inside a 120 GiB memory
cgroup with a 3 GiB host-availability floor; neither limit interrupted startup.
Readiness does not qualify a filled 262K context.
Memory uses C1 AR at capacity 133121, sampling global HIP allocation every
250 ms including 2.38 GiB idle allocation.

`artifacts/bench.json` declares six tables.
Run `tools/bench/model-bench.py --model qwen3.8-flash-next render` through Nix to
regenerate the card without model execution. Measurement commands and scoped
quality controls are in the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

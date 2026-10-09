# Qwen3.8 Flash-Next quality

**Concurrency: 63 tg128 requests match fresh AR completions:**
21 AR, 21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. Concurrency checks: October 9, 2026; attention review:
September 27–28.

| Check | Result |
| --- | --- |
| Real-image regression checks | 1024×1024 encoding passes. Nine main/candidate HTTP image, retry and continuation responses have identical outputs and token/cache counts. AR/MTP C4 image batching, sampled replay, cancellation and RAM/disk restoration pass. The disk fixture reprocesses only its seven-token assistant suffix; identical RAM retries require zero prefill. [Evidence](artifacts/prefill-long-context.json). |
| Prefill kernels | Final build: 7,946,240 logits after 32,640 prompt tokens remain byte-identical to main (KL 0). The earlier deep-context gate reaches 259,938 tokens with exact logits. Independent GDN/projection/selector checks, ragged chunks and graph replay pass. [Evidence](artifacts/prefill-long-context.json). |
| Mapped input and MTP catch-up | Exact Q8 projections, full/tail predictor stages and candidates; sampled output, acceptance, RAM/serialized replay and C2/4/6/8 state/RNG checks pass. Default sampling and thinking-off penalties are covered at d0/32K/128K. [Evidence](artifacts/prefill-long-context.json). |
| HTTP depth sweep | All 24 pp2048/tg128 AR and mixed/repetitive MTP completions through d128K retain the previous sweep's output hashes and prompt/cache/draft counts. [Evidence](artifacts/prefill-long-context.json). |
| Unused final-layer outputs | Full-vocabulary logits match unpruned execution byte-for-byte through 133,120 tokens, also with mixed code/Unicode text and image input. Mixed batching and sampled RAM/serialized snapshot replay preserve logits and RNG state. [Evidence](artifacts/prefill-final-rows.json). |
| Bounded selector scratch | After 133,824 prompt tokens, 15,892,480 logits are byte-identical to the previous build. Ragged chunks retain exact masks and the independent FP64 score check. [Evidence](artifacts/prefill-deep-context.json). |
| Unused normalization removal | 15,892,480 logits are byte-identical to main; ragged residual-only fusion matches normalized fusion and the separate epilogue/combine. [Evidence](artifacts/prefill-normalization.json). |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 25 AR/MTP configurations, including top-p zero; FP64 target filtering/CDF, p/q acceptance, residual correction, seeded replay and short token budgets pass |
| Greedy penalties | GPU selection matches CPU at every 1–7-row verification width, including FP32-sensitive ties; native tool/JSON constraints remain active. C2/4/6/8 outputs match the previous implementation; image/tool cancellation, RAM/disk restore and OpenAI SDK checks pass. [Evidence](artifacts/penalty-verification.json). |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Seeded MTP cache rebuilding | Two seeds × 200 tokens replay exactly after different prefill splits, cache bypass and replacement. K/V-only prefill and compact catch-up preserve full-head candidates across 1/8/9/32/33-row chunks. [Evidence](artifacts/mtp-cache-replay.json). |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |
| Prompt checkpoints | In-pass and borrowed checkpoints restore exactly in AR/MTP, including rewinds, branches, resets and destruction; 14 functional cache jobs pass correctness. [Details](PROMPT-CHECKPOINTS.md) |
| Sparse attention | Independent FP64 operator error ≤2.83e-7 (limit 1e-6). At 32K/128K, 256 fixed-token code/prose rows: mean KL 5.82e-4 and 256/256 top-1 agreement with an FP64-attention diagnostic. [Evidence](artifacts/attention-tiles-review.json). |

The attention diagnostic retains the quantized weights and other native
operators. Regrouping FP32 sums can change long-context text across builds;
matched-prefill AR/MTP logits and sampled snapshot replay are exact at
32K/128K. Session tests also cover image/text restoration and C2/4/6/8.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.
MTP cache projections now keep the same arithmetic across prompt splits.
Older Flash-Next disk checkpoints are rejected and rebuilt once.

## Vision

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only`, `--prefill-only` or `--cache-only` on the
session test; the snapshot test covers persistent image/text state.

`--execution-only 133120` runs the focused deep AR/MTP logit and snapshot-replay
check without the other session suites.

For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --batch 2048 --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

## Benchmark method

Gufo pp/tg, concurrency and memory refreshed October 8–9, 2026 using the production
Nix build of `50900eb7` plus the prefill optimizations. Binary and source hashes
are retained in [model identities](artifacts/model-identities.json).
Reference and loading results retain September 22–23 provenance.
One warmed sample per point, greedy, thinking off, penalties disabled.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. Gufo capacity is 133760; reference capacity is 35456 through 32K,
68224 at 64K and 133760 at 128K. C1/2/4/6/8 use the same d0 prompts; every
session prefills before measured tg128, with at most four prompt-tail tokens
reevaluated. Rates sum individual decode rates.
Depth calibration depends on the ordered sweep. Paired speed controls use
the same depth list. Deep AR/MTP HTTP cache frontiers differ by one token;
exact replay is checked separately with identical prefill boundaries.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
The historical C2 repetitive rate was not reproduced: current main measured
84.46 tok/s, versus 84.52 in this sweep and 84.40 in a focused repeat. The original
sweep sample is retained; this difference is not introduced by the prefill changes.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

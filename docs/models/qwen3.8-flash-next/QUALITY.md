# Qwen3.8 Flash-Next quality

**All 63 measured tg128 requests match fresh AR completions:** 21 AR,
21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. HTTP measurements: September 20–23, 2026; attention review:
September 27–28.

| Check | Result |
| --- | --- |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 25 AR/MTP configurations, including top-p zero; FP64 target filtering/CDF, p/q acceptance, residual correction, seeded replay and short token budgets pass |
| Greedy penalties | GPU selection matches CPU at every 1–7-row verification width, including FP32-sensitive ties; native tool/JSON constraints remain active. C2/4/6/8 outputs match the previous implementation; image/tool cancellation, RAM/disk restore and OpenAI SDK checks pass. [Evidence](artifacts/penalty-verification.json). |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Seeded MTP cache rebuilding | Two seeds × 200 tokens replay exactly after different prefill splits, cache bypass and replacement. K/V-only prefill and compact catch-up preserve full-head candidates across 1/8/9/32/33-row chunks. [Evidence](artifacts/mtp-cache-replay.json). |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |
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

## Extended context (YaRN)

**With YaRN off, the extension is a no-op: full-model logits and served
continuations are byte-identical to the pre-change build.** Unsloth
UD-Q4_K_XL target, shared Q8_0 MTP; September 26–27, 2026.

| Check | Result |
| --- | --- |
| YaRN off vs. pre-change build | Full-model logit dumps byte-identical on both prefill paths (fused F16 GEMM and PrepareAttention), 1,321-token prompt; 16/16 served continuations byte-identical |
| Prefill/decode, YaRN off | Prefill within 2% of the pre-change build at 8k/32k/131k (two runs each); decode and MTP acceptance unchanged at 29k (132/211 drafts both) |
| Needle retrieval, YaRN on | 12/12 via both `/v1/chat/completions` and raw `/v1/completions`: factor 1.5625 at capacity 409,600 (126,820 / 297,246 / 389,499 prompt tokens); factor 2.5 at capacity 409,600 (297,247 / 389,500, chat) and at capacity 655,360 (297,246 / 585,454) |
| Cache identity, YaRN on | A 29,646-token session cached under factor 1.5625 is not restored after restarting under factor 2.5 (full prefill; stored as a separate entry); it is restored after restarting under 1.5625 (0.8 s from disk, no prefill) |

Needle retrieval passed at every tested factor/capacity pair. A
369,989-token long read named the final scene correctly and quoted the
excerpt's last words verbatim; one image placed after 297,964 text tokens
in the same request (factor 2.5) was described correctly.

Logit agreement between YaRN on and off is close on a shared 1,927-token
prompt (dense attention for every implementation, since it is under the
2,048-token indexer budget): top-1 flips and KL divergence against the CPU
oracle and llama.cpp are similar with YaRN on or off (top-1 flips 21–29 of
1,927 positions; KL p50 ~4e-5; KL p90 ~0.01). At depth (the last 64
positions of long prompts), gufo versus llama.cpp agreement is 61/64 top-1
both at 248,487 tokens with YaRN off (KL median 0.011) and at 297,484
tokens with YaRN 1.5625 (KL median 0.008); the divergence there is gufo's
sparse attention versus llama.cpp's dense attention, present with YaRN off
too.

Idle GTT grows with capacity under YaRN (about 27.5 KiB per context token,
MTP resident): 88.5 GiB at capacity 262,144, 92.4 GiB at 409,600, 98.8 GiB
at 655,360. Prefill of the part of a prompt beyond 262,144 visible tokens
runs on the per-token attention kernel at about 350–375 tokens/s; a
585,454-token prompt averaged 493 tokens/s overall, with decode at 36
tokens/s.

Gaps: short-text quality with YaRN on is not measured (Qwen's model card
warns static YaRN can hurt short texts); there is no Transformers reference
beyond the native context, so agreement above uses llama.cpp and the CPU
oracle; and the fused WMMA kernel is not used for chunks whose visible
context exceeds 262,144 tokens.

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only`, `--prefill-only` or `--cache-only` on the
session test;
the snapshot test covers persistent image/text state. For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

## Benchmark method

Gufo single-user TG refreshed September 27, 2026 (`f797b5b`); PP and other
measurements retain September 22–23 provenance. One warmed sample per point,
greedy, thinking off.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. C1/2/4/6/8 use the same d0 prompts; every session prefills before
measured tg128, with at most four prompt-tail tokens reevaluated. Rates sum
individual decode rates. Gufo d0/C1 agree within 0.4% with matching drafts/output.
Depth calibration depends on the ordered sweep. Paired speed controls use
the same depth list. Deep AR/MTP HTTP cache frontiers differ by one token;
exact replay is checked separately with identical prefill boundaries.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

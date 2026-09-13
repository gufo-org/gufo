# Qwen27B quality and benchmark reference

Targets: UD-Q4_K_XL and UD-Q8_K_XL. DFlash2 drafts: Q4_K_M, Q8_0 and BF16.
**Adaptive is the default** in prompt, chat, bench and serving. Q4_K_M is the
recommended draft; a current comparison of all draft precisions remains TODO.

## Maintained checks

Run on gfx1151 through Nix. Model-specific tests and tools live in
`tests/models/qwen27b` and `tools/qwen27b`; shared quantization tests remain in
`tests/models/qwen` and are selected by the same runner.

| Suite | Contract |
| --- | --- |
| `fast` | Sampling/verifier, CLI and HTTP parsing, quantization reference decoding and tool reporting. |
| `kernels` | Independent GEMM/decode controls; recurrence and replay; DFlash2 convolution, attention, top-k, selector and verifier distributions. |
| `model` | Target logits/features at widths 2–8 and matrix prefill; MTP committed-feature replay; DFlash2 loading, history and snapshots; executable sampling parity. |
| `serving` | Direct/served tokens, seeded replay, EOS, cache forks, persistent restore, concurrency and cancellation. |
| `reference` | Optional BF16 target comparison: KL, total variation, top-1, RMSE and NLL. This measures quantization differences, not original-checkpoint correctness. |

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py kernels
nix develop -c python3 tools/qwen27b/check.py model \
  --model "$MODEL" --mtp-model "$MTP" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
```

The correctness build retains optimization, symbols and assertions; measure
speed only with `result/bin/gufo`. Artifact variables `GUFO_QWEN27B_*_MODEL`
select test inputs. Load target/reference models sequentially.
Sampling-only checks load each model once; omit the draft for AR:

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" --sampling-only
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" "$DRAFT" --sampling-only
# Add --fixed to check the fixed controller.
```

For an optimization, first check the affected operator against independent
formulas or scalar decode. Then check model replay on each affected target and
draft precision. Compare full logits/features and token IDs, including cached
replay; retain the established tolerances. Run short warmed release timings
with matched artifacts and prompts, alternating binaries during experiments.
Profile separately. Broaden to depth/concurrency sweeps only when needed.

For prefill changes, the existing target check also covers 128/257/2048-token
prefixes, repeated prefill and two-token scalar/verification replay. It prints
SHA-256 fingerprints of full logits and all five feature taps at every prompt
position, so two builds can be compared without storing logit dumps:

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/qwen27b_target_test \
  "$MODEL" --prefill-only
```

Large Q4 prefill (chunks of at least 1024 tokens) uses fully scaled packed
weights and FP16 activations with FP32 accumulation. Normalization writes FP16
directly; the up projection fuses SwiGLU and the SSM epilogue writes FP16.
They retain the separate FP32 producers' rounding boundaries, including halfway
cases, and reuse existing scratch allocations. Complete Q4/Q5 matrix tiles skip
tail checks; most formats write matrix results directly in the output layout.
IQ3_S retains its faster LDS transpose. Q8 and short-prefill paths retain native integer WMMA.

The shared quantization test checks all eight Q4 artifact formats against
independently decoded weights and FP64 dot products, including complete Q4/Q5 tiles and
partial row/token tiles. It requires at least a 2× RMSE improvement over A8 and lower maximum
error. Fused SwiGLU and norm/residual outputs must be byte-identical to separate
FP32 producers followed by FP16 conversion. The existing test executable takes
about 1.4 seconds. The existing SSM test also checks exact FP16 output and
unchanged recurrent state with FP32/BF16 storage; no extra executable is needed.

Model precision qualification: two real 2048-token prefixes from
`docs/PERFORMANCE.md` and `src/models/qwen/hip/batched_decode.cpp`, with 32
full-vocabulary rows per prefix, sampled every 64 positions. The reference
streams each projection's independently decoded Q4 weights through FP32 GEMM.
Every row improves RMSE and KL over A8; aggregate RMSE/KL/total variation must
at least halve, and greedy agreement must not decrease. All 64 greedy choices
match the reference. Mean logit RMSE is **0.00179 / 0.00358** for the two texts,
16–31× lower than A8; all five feature taps also improve. This qualifies
execution of these quantized weights, not conversion or the original checkpoint.
The current kernels preserve all 64 captured logits and all ten complete
feature-tap tensors across both prefixes exactly relative to the qualified FP16
calculation. Repeat-prefill
and scalar/verification replay pass; Q8 fingerprints remain unchanged.

Existing short-prefix decode/draft qualification covers 102 target logit rows,
102 tapped feature rows, 96 C3 replay/cache rows, and logical context 262,144
with a short prefix. All 270 draft trace files across the three precisions
remain byte-exact. Trace mode exits before the state suite: loading, history,
snapshot and serving checks must also run when those paths change.

## Sampling and executable contract

Qwen AR and DFlash2 sample on the GPU. The CPU owns request history and RNG
state. Target processing is penalties → top-k → top-p → min-p → temperature;
`min-keep` is a candidate floor. Temperature zero uses the adjusted argmax.

DFlash2 uses an anchor plus up to seven proposals. The trained top-16 selector
shares target temperature and reports its actual proposal distribution `q`.
There are no independent draft sampling controls. The verifier accepts token
`y` with probability `min(1, p(y)/q(y))`; rejection samples normalized
`max(p-q, 0)` over the full vocabulary. Only the consumed prefix is committed.
Greedy output must equal AR. Sampled AR/speculation consume different draws,
so equal seeds need not produce identical continuations across the two modes.
Repeated runs within one configuration must reproduce IDs, including cache hits.

Adaptive chooses the block length before drawing proposals using accepted-length
history and offline verification costs. It never uses live timing or the current
sample. Controller state persists within a request and resets for a new one.
`--draft-tokens` caps length; `--draft-policy fixed` selects the comparison policy.

| Entry point | Contract |
| --- | --- |
| `prompt`, `chat` | Shared target sampling and DFlash2; fixed/adaptive and draft cap. |
| `bench` | Qwen AR/MTP/DFlash2: greedy C1. DS4 additionally supports sampled benchmarks. |
| `serve`, `serve llm`, `gufo-server` | Startup draft/controller defaults; request target-sampling overrides through six HTTP adapters. |
| `eval` | Uses server draft configuration and sampling defaults. |
| Audio/video, diagnostics and probes | Do not run Qwen27B DFlash2; unsupported draft flags are rejected. |

Sampling refresh: **2026-09-13**, runtime source `ffb1f44`. All
[23 named strategies](../../../tests/models/qwen27b/sampling_cases.hpp) pass on
Q4/Q8 AR and all 12 target/draft/controller combinations: Q4/Q8 targets,
Q4_K_M/Q8_0/BF16 drafts, fixed/adaptive. They cover temperatures, individual
filters, candidate floors, penalties/rewards, history windows and combinations.
Each case checks four tokens and cold/cached replay. All six target/draft state
suites also pass seven-proposal controller, RNG and snapshot/reuse checks.

GPU checks cover 216 configurations / 864 AR quantiles, 141 acceptance/residual
controls and four full-vocabulary cases. The largest 24-bit RNG draw remains
exactly representable below one and accepts `p(y) = q(y)` on both GPU routes.
CPU reference replay covers 1,472 steps. No sampling implementation change was
needed.

HTTP: 15 configurations across six adapters on both targets, with AR and Q4_K_M
drafts under fixed/adaptive. Checks include startup defaults, request overrides,
seeded/unseeded execution, streaming and C2 state isolation. Prompt/two-turn chat
replay covers the same 15 configurations on Q4 AR and Q8/BF16 adaptive DFlash2.
This finite matrix does not establish arbitrary-context capability or complete
C>1 speculative parity.

## Pinned original DFlash2 operators

Reference: `z-lab/dflash`, commit
`07ebd93db9f472af339b644bb70221ad8428328a`, unmodified `dflash/model.py`.
Source SHA-256: `f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e`.
Config SHA-256: `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.

The runner uses PyTorch FP32 operations over independently decoded copies of the
same GGUF weights, including documented BF16 packing. Checks cover feature taps
5/19/33/47/61 after FFN residuals, anchor/mask embeddings, injection/norm/RoPE,
windowed noncausal attention, causal dynamic convolution and selector transition
scores. Each layer is checked both cumulatively and with captured layer inputs;
the head and conditional selector probabilities are also isolated.

One real 24-token Q4 target prefix, seven proposals, temperature 0.8: all three
drafts pass; all 21 candidate sets and random draws match.

| Draft | Worst stage relative RMSE | Full-logit max error | Proposal max total variation |
| --- | ---: | ---: | ---: |
| Q4_K_M | 5.83e-6 | 7.72e-5 | 1.60e-5 |
| Q8_0 | 5.74e-6 | 9.54e-5 | 1.25e-5 |
| BF16 | 6.16e-6 | 1.13e-4 | 7.01e-6 |

Gates: stage relative RMSE ≤1e-4; full-logit maximum error ≤1e-3; proposal total
variation ≤1e-4; isolated selector probability error ≤5e-6.
This validates packed-weight execution, not GGUF conversion or the original
full target checkpoint. Native BF16 execution need not be bit-identical.

```sh
gh api 'repos/z-lab/dflash/contents/dflash/model.py?ref=07ebd93db9f472af339b644bb70221ad8428328a' \
  -H 'Accept: application/vnd.github.raw' > /tmp/dflash-model.py
nix develop -c cmake --build --preset gpu-test --target qwen_dflash_gpu_test
nix develop -c build/gpu-test/tests/models/qwen27b/qwen_dflash_gpu_test \
  "$MODEL" "$DRAFT" --trace /tmp/dflash-trace
nix develop -c python3 tools/qwen27b/dflash_reference.py \
  --upstream /tmp/dflash-model.py --config "$ORIGINAL_DFLASH_CONFIG" \
  --target "$MODEL" --draft "$DRAFT" --trace /tmp/dflash-trace \
  --output /tmp/dflash-reference.json
```

Use a fresh trace directory for each draft. Keep generated logits, traces and
experiment JSON outside the repository. Committed logit fixtures should come
only from an independent full-quality PyTorch checkpoint, with provenance;
no such Qwen target fixture is currently bundled. Prompt corpus JSON files are
maintained test inputs. Historical experiment reports remain in Git history.

## Latest measurement provenance

Q4 release measured on 2026-09-13, SHA-256:
`686dba04afd56df01f8d2b757faa9545583138da2dd2757bd43c8fd87b074356`.
The pp2048-only result, **573.61 tok/s**, averages six warmed samples in two
processes: **572.64 ± 0.36 / 574.58 ± 0.39 tok/s**, three repetitions each.
Use `gufo bench -p 2048 -n 0 -d 0 -c 1 -r 3 --verbose`; alternate release
binaries when comparing implementations. Q8's recorded 503.17 tok/s is from
the unchanged native wave64 release, SHA-256
`feec38298e8a84a8b9f5dfcf290b924aed597e9f2bd8643fa40b5bd154aa736f`.

The Q4 pp2048/tg32 controls cover AR and all three adaptive DFlash2 drafts.
Generation remains effectively unchanged; all outputs match the same target's
AR token IDs and retain the same acceptance counts. The benchmark README shows
only the current samples. These short controls do not replace the depth sweep
or a broad draft-precision comparison. Real-prompt controls use `prose_tides` /
`json_records` from [the adaptive corpus](../speculative-adaptive-corpus.json),
and the chat prompt “Output the word red exactly 1000 times, separated by
spaces. Do not add any other text.”; their current speed refresh is TODO.

The separate profile attributes **85.6%** of GPU time to FP16 quantized GEMM,
with 1.7% idle time in the dispatch span. All 38 FP16 prefill kernels and the
SSM output kernel have zero scratch spills. The profiler preserves anonymous-namespace and quantization names,
so it reports each kernel independently.

Retained: native wave64 for Q8/short prefill; packed-weight FP16 prefill with
fused norm, up/SwiGLU and SSM output; direct matrix stores and complete Q4/Q5
tiles. Separate weight expansion, wider token tiles and warp-based RMSNorm
were slower. Dense BLAS and split-K did not improve small projections.
Quantized GEMM remains the next target; **Q4 pp2048 at 600 tok/s is still open**.

| Artifact | SHA-256 |
| --- | --- |
| Target UD-Q4_K_XL | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Target UD-Q8_K_XL | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` |
| Draft Q4_K_M | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| Draft Q8_0 | `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |
| Draft BF16 | `26d47ca20ab07688327a63d912acad222d924eaaa92a980cc488de3c67e736bc` |

TODO: independent original-target/conversion and MTP qualification; optional
BF16 target comparison; refreshed pp2048/tg128 depths and all draft precisions;
C>1 speed and complete speculative parity. The 75-question capability comparison
requires confirmation before running.

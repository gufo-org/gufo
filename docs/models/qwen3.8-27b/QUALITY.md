# Qwen3.8 27B quality

**Greedy DFlash2 matches isolated AR on all 160 measured mixed/repetitive
requests across Q4/Q8 and C1/2/4/6/8.** Targets are Unsloth UD-Q4_K_XL /
UD-Q8_K_XL; the draft is Q4_K_M. [Identities and dated evidence](artifacts/model-identities.json).
These are execution-consistency checks; original-target and GGUF-conversion
parity remain unqualified.

| Check | Q4_K_XL | Q8_K_XL |
| --- | ---: | ---: |
| AR/DFlash2 mixed completions, eight depths 0–128K | 8/8 match | 8/8 match |
| AR repetitive C2/C4/C6/C8 requests versus C1 | 20/20 match | 20/20 match |
| DFlash2 mixed requests versus isolated AR, C1–C8 | 59/59 match | 59/59 match |
| DFlash2 repetitive requests versus isolated AR, C1–C8 | 21/21 match | 21/21 match |
| Target verification, widths 2–8 and ragged cohorts | Full logits/features match scalar | Full logits/features match scalar |
| Serving and state | Cancellation, continuation, RAM/disk restore and seeded replay pass | Same; also eight 32K continuation forks |

## October 19 wave64 verification review

The wave64 verification attention (#660) reproduces the scalar decode bits.
Single-row outputs from main `23fd87dc` and the candidate are byte-identical
in the production 24/4-head shape (fp32 and fp16 caches, gated and plain,
starts 0–65,536), and verification rows 2–8 match those single rows. Every
Q4/Q8 DFlash2 single-user completion from 0 to 128K matches the earlier
October 19 rows; all concurrency completions match Gufo AR. The Q4 DFlash2
functional `batch`, `long-context`, `conversation`, `state-edges`,
`cache-concurrency` and `cache` suites pass against main with no output
change; 2,239 of 2,246 timings pass and 7 are inconclusive.

## October 19 recurrent-state review

The refreshed DFlash2 concurrency corpus matches fresh, isolated AR on all
160 requests. Main `e7730942` and PR `d5e90965` also match all 54
teacher-forced full-vocabulary logit rows, plus the native prefill and
continuation fingerprints. Verification widths, ragged cohorts through C8,
rollback, both recurrent-buffer parities and 2,080-row diverging prefill pass.
Q4/Q8 AR and DFlash2 HTTP checks cover sampling, thinking, tools, cancellation,
images and RAM/disk continuation.

The matched Q4/Q8 mixed depth sweeps also preserve every completion and draft
count from 0 to 128K. PP/TG passes the per-request gate at all sixteen points.
Current main already shows the lower PP rates relative to the September table;
the candidate stays close to main and improves decode at every measured depth.

The local snapshot cleanup preserves serialized bytes with differently filled
destination buffers; Q4/Q8 draft restore and HTTP disk-restart checks pass.
Matched performance controls retain every timing flag. Remaining overlapping
snapshot timings and admission pauses are inconclusive, not a blanket latency
pass. See the [review record](artifacts/recurrent-fork-review.json).
The earlier eight-depth AR/DFlash2 and quantization qualification retains its
original date; these new checks establish consistency with current main.

## Quantization comparison

September 24, 2026: short teacher-forced comparison against **Gufo's BF16 target**, with identical
prefixes: three prose/code fixtures, nine logit rows each, full 248,320-token
vocabulary, temperature 1 without filtering. This measures quantization and
native arithmetic together; BF16 here is not an independent upstream oracle.
Results and reproducible identities are in [the compact record](artifacts/quantization-quality.json).

| Target versus BF16 | Mean KL divergence (nats) | Mean total variation | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q4_K_XL | 0.001650 | 0.01566 | 27/27 |
| Q8_K_XL | 0.00006537 | 0.003128 | 27/27 |

## DFlash2 and sampling

The pinned [original DFlash2 operators](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a)
were checked with independently decoded GGUF weights: worst stage relative
RMSE **5.83e-6**, full-logit maximum error **7.72e-5**, proposal total variation
**1.60e-5** on a 24-token prefix/seven proposals. Limits are 1e-4, 1e-3 and 1e-4.
This qualifies draft execution, not weight conversion.

AR and verification apply penalties → temperature → top-k → top-p → min-p.
Tests cover 216 GPU sampling cases and 23 served strategies, including p/q
acceptance, full-vocabulary residual correction, nonfinite inputs and replay.
**Sampled DFlash2 need not match AR's same-seed sequence**; within-mode seeded
replay requires the same configuration. GPU/CPU distributions have numerical
tolerances, not bit-identical upstream sampler parity.

## Vision

Earlier Q4/Q8 AR and DFlash2 checks cover multi-image, sampled/greedy C4,
cancellation, generated-history continuation and disk restart. The current
component migration retains a strict Q4 cold-prefill versus decoded-history
failure; see [the current status](#component-serving-migration). Q8 also requires exact
cold/live/restored continuation logits. Encoder controls use official
Transformers operators over the **same converted BF16 GGUF**; they cannot detect
conversion errors. Flash-Next has a separate [vision limit](../qwen3.8-flash-next/QUALITY.md#vision).
Q4/Q8 image snapshots restored after cancellation inside the image encoder or
text prefill match all target logits exactly; token-limit endings and stripped
tool reasoning retain the compatible conversation prefix.

## Autoregressive component checkpoints

October 10, 2026: the model-local component adapter passes Q4/Q8 checks in
both FP16 and FP32 attention KV layouts. Every restored component byte and
full continuation logit row matches uninterrupted execution. The checks cover
earlier-boundary edits, pending generation tokens, leased-row preservation,
committed recurrent replay, failed/discarded transfers and completion lifetime.
Image regressions restore before the first image and between two images while
retaining the matching request's future images and complete RoPE layout;
those focused image checks use production FP16 KV.

Production FP16 probes at 32K and 100K also match every restored byte and
sixteen continuation logit rows/token decisions after transfer to another slot.
These are Gufo execution-consistency checks. They do not extend upstream or
weight-conversion parity, qualify alternate BF16 recurrence, or qualify the
common checkpoint allocator. The [adapter inventory and commands](CONTINUATION-ADAPTER.md)
and [card 15 results](../../cache-redesign/cards/15-qwen27b-ar-adapter.md)
record the scope and retained evidence.

## Long-context FP32 attention qualification (2026-10-13)

A new capacity probe exposed an invalid FP32 prefill launch at 16K: its
context-sized score array plus reduction scratch used 66,560 bytes of LDS,
above gfx1151's 64 KiB limit. The bounded online fallback passes the existing
attention controls and new three-row causal checks around the LDS boundary,
32K and 100K in both KV representations. The worst absolute FP64-oracle error
is 5.04e-7 (limit 1e-5); long rows match scalar online attention exactly, and
BF16 output bits match independent rounding. The pinned `64639c28` Q4 FP32
32K full-model probe also passes every restored component byte and sixteen
continued full-logit/token checks. After the telemetry-only repair, the pinned
`e008ffc7` binary passes Q4/Q8 FP16 and Q4 FP32 at both 32K and 100K,
and Q8 FP32 at 32K. Every restored component byte and sixteen continued
full-logit/token checks match. The user stopped the Q8 FP32 100K probe and
skipped further long-capacity probes for both KV precisions. That case has no
qualification result. Matched-main timing flags remain open. All original aborts and diagnostics remain retained in
`/home/mixer/gufo-qualification/qwen27b-component-2026-10-13`.

## Reproduce

Run only the affected [model suite](../../../tools/qwen27b/check.py):

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
```

[Draft oracle](../../../tools/qwen27b/dflash_reference.py),
[vision oracle](../../../tools/qwen27b/vision_reference.py) and
[vision serving checks](../../../tools/qwen27b/vision_check.py) retain source
pins and compare operators, full logits, RNG, image/state isolation and replay.
Full logits stay outside Git. A missing-weight skip is not a pass.

## Meaning of exact

Matching Gufo AR/speculative tokens or completion hashes tests consistency.
Matching llama.cpp hashes also does not establish original-model accuracy;
quantized implementations can disagree. Keep the reference, weights and history
fixed before interpreting any numerical difference.

## Benchmark method

All Gufo rows refreshed October 19, 2026 on `11813e51` (#660, rebased on
`23fd87dc`); llama.cpp retains September 23 measurements. One warmed sample per
point, greedy, thinking off, `performance` platform profile.
Single-user rows use pp2048/tg128 and cached depths 0–128K. Each mode creates
its own eight-token prefix reply; histories can differ across engines.
Concurrency uses the older short-prompt corpus, not matching pp2048 prompts;
its C1 is not directly comparable to single-user d0. AR uses repetitive text;
DFlash2 uses mixed/repetitive cases. Rates sum individual decode rates.
Loading uses cold files (`POSIX_FADV_DONTNEED`, zero resident pages checked
with `mincore`), C1/DFlash2/capacity 262144; memory uses C1/AR peak HIP
allocation. Commands, hashes and counts remain in [artifacts](artifacts/bench.json)
and the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).

## Component serving migration

October 19 refresh: main `270b5333` is merged into the component-cache PR.
Qualification was paused at the user’s request after the four 27B HTTP profiles
passed. Flash and timing/recovery checks below remain incomplete.
Q4/Q8 native adapter, DFlash2 state, serving-cache and target numerical checks
pass on the candidate. Restored component bytes and full logits match; checks
include FP16/FP32 KV, edits, rollback, images, sampled frontiers, cancellation,
disk compatibility, concurrent forks and cleanup. The shared byte-copy control
also covers registered/pageable memory, unaligned tails and peer graph capture.

Q4/Q8 AR/DFlash2 startup checks pass for RAM, disk and restart. RAM-only
serving does not fingerprint the artifacts; durable restart restores the same
reply. Loaded modes and actual draft counts are verified. Main's shared
backing, admission, retention and publication fixes are preserved, and 27B
uses the shared mapped snapshot copy kernels for private state and FP16 rows.
FP32's canonical head-major transfers retain their strided representation.

At readiness with four 32K sessions and a 16 GiB cache budget, the component
cache commits 2,270,560,256 bytes of backing before serving, as Flash's shared
policy does. Anonymous RSS is about 2.1 GiB above main's legacy 27B cache in
both modes; that reserve accounts for all but 6.6–11.6 MB of the observed
difference. This is a resident-memory cost of preparing captures in advance.
The startup timing observations do not establish readiness latency parity
because filesystem-cache histories were not matched.

Short-request main/PR/PR/main controls preserve every output and draft count.
DFlash2 passes the per-request timing limits against main. AR has an
inconclusive first-request queue flag. Against the merged branch without the
copy change, one initial AR snapshot outlier and token-gap flags in both modes
remain inconclusive. All four selected Q4/Q8 AR/DFlash2 HTTP runs
each pass all 23 stages and 290 recorded requests,
including long-context, conversation, image-count, edits, growth, depth,
concurrency, learned shared prefixes, state edges, cancellation and durable
restart/replay. The Q4 DFlash2 run observes 3,419 proposed and 2,016 accepted draft
tokens; Q8 DFlash2 also verifies actual draft execution. Flash shared-path controls,
deep/recovery workflows and complete latency qualification remain open.
See the [dated review record](artifacts/component-cache-review.json) for
commands, binary identities, every short-request observation and outstanding
checks. Earlier capacity and vision evidence does not qualify this revision.
The earlier cold-prefill versus decoded-history vision difference remains
unqualified; the wide prefill route restored by `dabc85cf` is retained.

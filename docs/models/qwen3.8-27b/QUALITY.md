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

Q4/Q8 AR and DFlash2 pass multi-image, sampled/greedy C4, cancellation,
generated-history continuation and disk-restart checks. Q8 also requires exact
cold/live/restored continuation logits. Encoder controls use official
Transformers operators over the **same converted BF16 GGUF**; they cannot detect
conversion errors. Flash-Next has a separate [vision limit](../qwen3.8-flash-next/QUALITY.md#vision).
Q4/Q8 image snapshots restored after cancellation inside the image encoder or
text prefill match all target logits exactly; token-limit endings and stripped
tool reasoning retain the compatible conversation prefix.

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

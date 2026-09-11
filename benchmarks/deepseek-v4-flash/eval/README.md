# DS4 quality reports

Independent operator checks, AR comparisons, and capability samples. These
are not complete official dataset scores. Capability requests use the real
`gufo serve` HTTP route.

## Official continuation comparison

This separate check feeds the matching hosted checkpoint's tokens into ordinary
AR decoding and measures their likelihood, greedy agreement, and matching
prefix. It covers 100 prompts / 2,313 tokens, plus five short/long smoke cases /
14 tokens. The stored API probabilities are saturated; these results do not
measure full-distribution parity.

Antirez's [release QA](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/QA_BEFORE_RELEASES.md#L1440)
reports the following for Flash 0731 Q2. His download script maps that release
to the target filename used here.

| 100-case run | Average NLL ↓ | First-token matches | Mean matching prefix |
| --- | ---: | ---: | ---: |
| Published ROCm | 0.398181736 | 56/100 | 5.170 |
| Published CUDA | 0.404714573 | 55/100 | 4.890 |
| Same-machine upstream control | 0.403401036 | 54/100 | 5.240 |
| Gufo | 0.399423334 | 56/100 | 5.470 |

First-token matches measure agreement with the hosted model, independently of
the question-answer grades below. The same-machine runs use the same GGUF and
pinned upstream source. Gufo repeats exactly across all 105 cases / 2,327
steps. Its 100-case NLL difference is −0.003978; a paired case bootstrap gives
a 95% interval of [−0.011932, +0.003730]. This finds no regression in this
sample and does not establish an improvement.

The five smoke cases produce all 14 expected greedy tokens in both engines,
but Gufo's average NLL is worse: **0.034363 versus 0.010908**. Three cases have
lower continuation likelihood, including the short reasoning and long memory
prompts. These differences remain visible even though greedy answers match.
The published QA does not pin the GGUF hash/build used for those scores; our
same-machine control does not reproduce its exact numbers.

The [matched 2K/4K, five-depth comparison](antirez-ds4-ar-comparison.json)
**fails four of 20 full-logit checks**, all immediately after prefill. Worst
RMSE is 1.3476 (limit 1.12), cosine is 0.96214 (minimum 0.979), and maximum
error is 5.8521 (limit 5). All post-decode vectors pass. Per-step greedy
agreement is 124–128/128; Gufo repeats exactly across all 1,280 choices and
20 logit vectors.

Changing prefill boundaries also changes both engines' numerical results.
Gufo's 2K/4K greedy agreement is 125–126/128 at nonzero depths; upstream's
12K prefill vectors also exceed these bounds when compared across call sizes.
These are differential alerts, not proof that Gufo is less correct. Antirez's
implementation is also unofficial and can contain errors. DSpark/AR equality
alone cannot detect shared errors either.

### Post-prefill formula audit

The [2026-09-11 audit](prefill-formula-audit.json) reproduces all four alerts.
**No model-quality improvement is demonstrated; production arithmetic is retained.**
All ten Gufo prefill vectors repeat exactly and match the retained pre-cleanup
vectors byte for byte. These are actual distribution differences: subtracting
mean logit shifts does not eliminate them. Both engines choose the same top
token at these four frontiers.

| Context / call size | RMSE | Cosine | Max logit error | Max probability difference |
| --- | ---: | ---: | ---: | ---: |
| 4K / 4K | 0.9691 | 0.98006 | 5.3386 | 5.90 percentage points |
| 12K / 4K | 1.1819 | 0.96935 | 5.4680 | 39.81 percentage points |
| 16K / 2K | 0.8128 | 0.98681 | 5.8521 | 11.63 percentage points |
| 16K / 4K | 1.3476 | 0.96214 | 5.8032 | 9.80 percentage points |

The official [0731 computation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L680)
projects raw FP32 HC activations, then applies RMS scaling. Gufo's fused
prefill follows this formula using the GGUF's F16 weights. The pinned
[Antirez ROCm path](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4.c#L29171)
normalizes first; its wide F16 GEMM also rounds the activations to F16.
At 12K, changing only the call size changes Antirez's probabilities by up to
40.58 percentage points, versus 6.00 for Gufo. This is evidence of sensitivity
in the comparator too; it does not establish official-model parity.

A temporary rounding experiment fails the independent official-formula oracle.
It passes three of four raw-logit checks, but **worsens** maximum probability
differences at 4K (5.90 → 20.70 percentage points) and 16K/4K (9.80 → 21.15).
The 12K raw-logit alert remains. The experiment is rejected. This precision
difference contributes to the discrepancy but does not explain every alert or
establish which model has better task accuracy.

The retained 16K layer trace shows post-attention RMSE growing from 0.000404
at layer 0 to 0.994 at layer 42. Its final logits match the current run.
This shows accumulated divergence, without isolating a single faulty operator.
Two cancellation cases reject extra F16 rounding. Two dense HC cases independently
check projection/RMS, the official Sinkhorn epsilon placement, weighted reduction,
and residual-matrix orientation in double precision; maximum error is 1.5e-6.
The larger-epsilon case exposes misplaced epsilons, and asymmetric data exposes
transposition. These checks join `ds4.projections`; no new executable is added.
Raw-logit limits remain unchanged. The four alerts remain open quality evidence,
not a reason to imitate an unofficial implementation.

The next investigation starts with the smallest failing frontier, 4K/4K:
capture real operator inputs and compare their outputs with the official
formulas using the same dequantized GGUF weights and explicit precision.
Locate the first unexplained deviation before comparing downstream layers.
Any demonstrated fix needs a captured-input regression, all four frontier
checks, repeatability, and the pinned official-continuation likelihood check.
An alert can also be explained by a proven approximation in the comparator;
agreement with Antirez or relaxed thresholds alone cannot close it.

The IQ2 projection check also independently unpacks weights and derives sign
parity on the CPU. It catches a corrupted sign lookup even when the scalar and
batched GPU kernels agree with each other.

The same official source specifies a DSpark history window of 128 target rows,
followed by the draft block seeded with the target's next token. The support KV
projection consumes injected target features directly. It does not add an
encoder row to the draft transformer. Sixteen GPU cases independently check
window boundaries and physical ring wrap, with stale slots poisoned to expose
out-of-window reads.

Qualification needs official operator checks, matched hosted continuations,
task-level accuracy, and DSpark preservation of target behavior. We have not
run the full official FP8 model locally or established full-distribution parity.
These finite samples cannot establish the absence of every quality issue.

```sh
nix develop -c tools/ds4/check.py reference --model "$MODEL" \
  --upstream /path/to/pinned-antirez-checkout --output /tmp/ds4-reference
```

## Greedy DSpark, 75 questions

Run date: 2026-09-10. Sequential requests, C1, context 32,768,
`temperature: 0`, and a 16,000-token completion limit. The reports retain
prompts, responses, reasoning, grades, token counts, artifact hashes, and source
identities. Timings are omitted: these are correctness runs.

| Run | Passed | Failed | Execution errors | Length finishes |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 53/75 | 22 | 0 | 9 |
| Candidate | TODO | TODO | TODO | TODO |

A length finish is reported separately even when the extracted answer passes.
The first 16 candidate cases repeated exactly twice: 13/16 correct, versus
14/16 in the baseline. The changed case, `aime2025-18`, also changes in ordinary
target decoding; the candidate DSpark and target responses match exactly.
The complete comparison is pending.

```sh
./result/bin/gufo eval --base-url http://127.0.0.1:8080/v1 \
  --questions 75 --greedy --output /tmp/ds4-quality.json
```

## Historical first-four sample

Run date: 2026-08-27. The evaluator omitted `temperature`, using server-default
sampling and thinking, and sent `max_completion_tokens: 16000`. These runs used
ordinary target decoding.

| Run | Passed | Failed | Execution errors | Length finishes | Completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Default](antirez-ds4-first4-default.json) | 4 | 0 | 0 | 0 | 939 |
| [Default repeat](antirez-ds4-first4-default-repeat.json) | 4 | 0 | 0 | 0 | 939 |

The extracted answer sequence was `B`, `C`, `70`, `C` in both runs. Complete
visible responses, reasoning content, grades, and run identities were
identical across the repeat. The server reported `cache_hit: false` and zero
cached tokens for every request, so no cache reuse occurred in this pair.

Endpoints and server configuration are sanitized in retained reports.

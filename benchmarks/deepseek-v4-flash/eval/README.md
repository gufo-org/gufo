# DS4 capability regression reports

These are Gufo regression samples from the pinned Antirez suite, not complete
official dataset scores. Requests use the real `gufo serve` HTTP route.

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

The official [0731 computation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L680)
casts HC activations to FP32, projects them, then applies RMS scaling.
It does not round normalized activations to F16. Gufo's fused prefill follows
that formulation with the F16 HC weights stored in this GGUF. Adding the extra
activation rounding moved the worst 16K comparison within the existing bounds,
but was rejected: agreement with Antirez did not justify that approximation.
Two maintained kernel cases check the official formula independently in double
precision and detect this extra rounding.

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

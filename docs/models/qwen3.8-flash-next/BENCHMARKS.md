# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user tables measured October 17, 2026, with PR #626 and the
validated sparse scheduling, MTP scratch reuse and vision allocation changes.
Concurrency retains the October 16 PR #622 capture; loading and memory retain
the October 16 `fdc21849` captures.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Gufo numerical regression checks pass; individual functional timing limits
remain documented in [QUALITY.md](QUALITY.md). Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

At 128K, Gufo reaches 1543.70 tok/s AR prefill and 1503.43 tok/s MTP prefill.
All 24 single-user rows retain the previous completion hashes and token/cache/
draft counts. Mixed MTP C2 in the retained concurrency capture alternates
between two draft-controller trajectories (about 54 and 60 tok/s, identical
text) on main and the PR alike.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1764.48 | 489.59 | +260.4% | 28.20 | 22.20 | +27.0% |
| 4,096 | 1694.37 | 455.24 | +272.2% | 28.21 | 21.21 | +33.0% |
| 8,192 | 1693.71 | 428.85 | +294.9% | 28.20 | 20.40 | +38.2% |
| 12,288 | 1692.24 | 400.21 | +322.8% | 28.23 | 19.64 | +43.7% |
| 16,384 | 1645.61 | 375.89 | +337.8% | 28.17 | 18.95 | +48.7% |
| 32,768 | 1661.24 | 301.31 | +451.3% | 28.13 | 16.54 | +70.1% |
| 65,536 | 1597.39 | 221.94 | +619.7% | 28.03 | 11.62 | +141.2% |
| 131,072 | 1543.70 | 144.78 | +966.2% | 27.78 | 7.98 | +248.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1723.45 | 468.76 | +267.7% | 35.58 | 31.73 | +12.1% | 73.80 | 48.12 | +53.4% |
| 4,096 | 1660.85 | 423.79 | +291.9% | 37.57 | 34.48 | +9.0% | 70.68 | 45.71 | +54.6% |
| 8,192 | 1659.31 | 394.98 | +320.1% | 36.85 | 32.20 | +14.4% | 58.11 | 44.31 | +31.1% |
| 12,288 | 1640.48 | 365.91 | +348.3% | 39.15 | 32.32 | +21.1% | 48.61 | 43.53 | +11.7% |
| 16,384 | 1632.92 | 341.98 | +377.5% | 40.59 | 32.07 | +26.6% | 59.99 | 43.66 | +37.4% |
| 32,768 | 1618.21 | 277.61 | +482.9% | 39.93 | 25.41 | +57.1% | 52.73 | 39.28 | +34.2% |
| 65,536 | 1562.75 | 208.04 | +651.2% | 39.18 | 19.65 | +99.4% | 53.08 | 27.75 | +91.3% |
| 131,072 | 1503.43 | 135.42 | +1010.2% | 39.82 | 14.63 | +172.2% | 54.73 | 20.96 | +161.1% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.27 | 22.34 | +26.5% |
| 2 | 50.68 | 37.08 | +36.7% |
| 4 | 85.65 | 54.82 | +56.2% |
| 6 | 107.53 | 65.34 | +64.6% |
| 8 | 125.08 | 68.79 | +81.8% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.66 | 31.77 | +12.2% | 73.17 | 46.93 | +55.9% |
| 2 | 54.35 | 42.79 | +27.0% | 97.55 | 52.57 | +85.6% |
| 4 | 86.42 | 48.78 | +77.2% | 157.76 | 48.23 | +227.1% |
| 6 | 108.58 | 52.59 | +106.5% | 176.70 | 48.51 | +264.3% |
| 8 | 125.47 | 61.92 | +102.6% | 194.29 | 58.13 | +234.2% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.27 | 117.83 | +671.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.75 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.73 | 85.31 | -6.0% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

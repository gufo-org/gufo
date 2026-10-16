# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user and concurrency tables measured October 16, 2026, with
PR #622 (compacted sparse-attention selections) rebased on `22e36769`; loading
and memory retain the same-day `fdc21849` captures.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Gufo numerical regression checks pass; individual functional timing limits
remain documented in [QUALITY.md](QUALITY.md). Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

PR #622 speeds up decode at depth: at 128K, tg improves 0.7% AR, 1.9% mixed
MTP and 2.1% repetitive MTP over the `fdc21849` rows, with identical completions
and draft counts. Prefill is unchanged by the PR; pp differences are run-level
variation. Mixed MTP C2 alternates between two draft-controller trajectories
(about 54 and 60 tok/s, identical text) on main and the PR alike.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1768.66 | 489.59 | +261.3% | 28.31 | 22.20 | +27.5% |
| 4,096 | 1697.96 | 455.24 | +273.0% | 28.27 | 21.21 | +33.3% |
| 8,192 | 1691.35 | 428.85 | +294.4% | 28.28 | 20.40 | +38.6% |
| 12,288 | 1672.13 | 400.21 | +317.8% | 28.25 | 19.64 | +43.8% |
| 16,384 | 1672.92 | 375.89 | +345.1% | 28.22 | 18.95 | +48.9% |
| 32,768 | 1643.71 | 301.31 | +445.5% | 28.10 | 16.54 | +69.9% |
| 65,536 | 1577.99 | 221.94 | +611.0% | 27.90 | 11.62 | +140.1% |
| 131,072 | 1517.44 | 144.78 | +948.1% | 27.49 | 7.98 | +244.5% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1722.29 | 468.76 | +267.4% | 35.71 | 31.73 | +12.5% | 73.44 | 48.12 | +52.6% |
| 4,096 | 1655.05 | 423.79 | +290.5% | 37.62 | 34.48 | +9.1% | 70.47 | 45.71 | +54.2% |
| 8,192 | 1641.90 | 394.98 | +315.7% | 36.85 | 32.20 | +14.4% | 58.12 | 44.31 | +31.2% |
| 12,288 | 1624.20 | 365.91 | +343.9% | 39.18 | 32.32 | +21.2% | 48.55 | 43.53 | +11.5% |
| 16,384 | 1611.73 | 341.98 | +371.3% | 40.53 | 32.07 | +26.4% | 59.96 | 43.66 | +37.3% |
| 32,768 | 1594.19 | 277.61 | +474.3% | 39.72 | 25.41 | +56.3% | 52.59 | 39.28 | +33.9% |
| 65,536 | 1531.42 | 208.04 | +636.1% | 38.99 | 19.65 | +98.4% | 52.68 | 27.75 | +89.8% |
| 131,072 | 1477.08 | 135.42 | +990.7% | 39.36 | 14.63 | +169.0% | 53.77 | 20.96 | +156.5% |
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

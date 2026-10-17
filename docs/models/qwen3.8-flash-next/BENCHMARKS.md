# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user and concurrency tables measured October 17, 2026, with PR #613 and the
validated prompt-lookup and PLE gather-scratch changes.
Loading and memory retain the October 16 `fdc21849` captures.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Gufo numerical regression checks pass; individual functional timing limits
remain documented in [QUALITY.md](QUALITY.md). Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

At 128K, Gufo reaches 1555.18 tok/s AR prefill and 1504.59 tok/s MTP prefill.
All 24 single-user rows match main `3389a585` in completion hashes and
prompt/cache counts; prompt lookup changes draft counts.
All 63 concurrent tg128 completions match the corresponding C1 AR control.
Prior mixed MTP C2 repeats show two draft-controller trajectories (about 54
and 60 tok/s, identical text), so its gain is not entirely attributable to lookup.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1787.54 | 489.59 | +265.1% | 28.26 | 22.20 | +27.3% |
| 4,096 | 1708.08 | 455.24 | +275.2% | 28.26 | 21.21 | +33.2% |
| 8,192 | 1716.12 | 428.85 | +300.2% | 28.23 | 20.40 | +38.4% |
| 12,288 | 1705.14 | 400.21 | +326.1% | 28.23 | 19.64 | +43.7% |
| 16,384 | 1694.13 | 375.89 | +350.7% | 28.19 | 18.95 | +48.8% |
| 32,768 | 1667.83 | 301.31 | +453.5% | 28.14 | 16.54 | +70.1% |
| 65,536 | 1606.04 | 221.94 | +623.6% | 28.07 | 11.62 | +141.6% |
| 131,072 | 1555.18 | 144.78 | +974.2% | 27.84 | 7.98 | +248.9% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1748.54 | 468.76 | +273.0% | 35.74 | 31.73 | +12.6% | 116.94 | 48.12 | +143.0% |
| 4,096 | 1681.08 | 423.79 | +296.7% | 37.65 | 34.48 | +9.2% | 116.02 | 45.71 | +153.8% |
| 8,192 | 1667.59 | 394.98 | +322.2% | 36.82 | 32.20 | +14.3% | 114.22 | 44.31 | +157.8% |
| 12,288 | 1662.09 | 365.91 | +354.2% | 39.20 | 32.32 | +21.3% | 105.53 | 43.53 | +142.4% |
| 16,384 | 1643.94 | 341.98 | +380.7% | 40.61 | 32.07 | +26.6% | 113.60 | 43.66 | +160.2% |
| 32,768 | 1625.87 | 277.61 | +485.7% | 39.97 | 25.41 | +57.3% | 105.69 | 39.28 | +169.1% |
| 65,536 | 1557.54 | 208.04 | +648.7% | 39.23 | 19.65 | +99.6% | 104.78 | 27.75 | +277.6% |
| 131,072 | 1504.59 | 135.42 | +1011.1% | 39.89 | 14.63 | +172.7% | 108.40 | 20.96 | +417.2% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.21 | 22.34 | +26.3% |
| 2 | 50.64 | 37.08 | +36.6% |
| 4 | 85.89 | 54.82 | +56.7% |
| 6 | 108.02 | 65.34 | +65.3% |
| 8 | 124.72 | 68.79 | +81.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.69 | 31.77 | +12.3% | 115.79 | 46.93 | +146.7% |
| 2 | 60.93 | 42.79 | +42.4% | 165.09 | 52.57 | +214.0% |
| 4 | 88.62 | 48.78 | +81.7% | 217.19 | 48.23 | +350.3% |
| 6 | 109.72 | 52.59 | +108.6% | 231.03 | 48.51 | +376.3% |
| 8 | 126.87 | 61.92 | +104.9% | 245.11 | 58.13 | +321.7% |
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

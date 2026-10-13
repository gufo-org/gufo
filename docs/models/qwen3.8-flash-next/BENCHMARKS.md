# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user AR and MTP: October 13, 2026, production Nix builds of
PR #571, every depth: AR d16K/d64K and MTP d64K use `ac51031a`, which adds
the short W8A8 tail path those prompts (2,051–2,058 tokens) take; the other
rows use `0951ea2f`. Other Gufo tables retain October 10's
production Nix review build `25cdb2a1` of PR #518.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Gufo numerical regression checks pass; two focused functional phases still
exceed the timing gate. Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1773.82 | 489.59 | +262.3% | 28.22 | 22.20 | +27.1% |
| 4,096 | 1694.96 | 455.24 | +272.3% | 28.20 | 21.21 | +33.0% |
| 8,192 | 1686.00 | 428.85 | +293.1% | 28.20 | 20.40 | +38.2% |
| 12,288 | 1669.01 | 400.21 | +317.0% | 28.18 | 19.64 | +43.5% |
| 16,384 | 1624.22 | 375.89 | +332.1% | 28.19 | 18.95 | +48.8% |
| 32,768 | 1648.69 | 301.31 | +447.2% | 28.05 | 16.54 | +69.6% |
| 65,536 | 1548.05 | 221.94 | +597.5% | 27.83 | 11.62 | +139.5% |
| 131,072 | 1529.63 | 144.78 | +956.5% | 27.24 | 7.98 | +241.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1689.43 | 468.76 | +260.4% | 33.29 | 31.73 | +4.9% | 65.21 | 48.12 | +35.5% |
| 4,096 | 1632.12 | 423.79 | +285.1% | 35.60 | 34.48 | +3.2% | 62.42 | 45.71 | +36.6% |
| 8,192 | 1622.93 | 394.98 | +310.9% | 35.11 | 32.20 | +9.0% | 53.04 | 44.31 | +19.7% |
| 12,288 | 1614.15 | 365.91 | +341.1% | 37.61 | 32.32 | +16.4% | 45.56 | 43.53 | +4.7% |
| 16,384 | 1604.17 | 341.98 | +369.1% | 38.18 | 32.07 | +19.1% | 54.59 | 43.66 | +25.0% |
| 32,768 | 1603.98 | 277.61 | +477.8% | 37.02 | 25.41 | +45.7% | 48.55 | 39.28 | +23.6% |
| 65,536 | 1560.11 | 208.04 | +649.9% | 32.60 | 19.65 | +65.9% | 48.18 | 27.75 | +73.6% |
| 131,072 | 1488.23 | 135.42 | +999.0% | 36.10 | 14.63 | +146.8% | 47.77 | 20.96 | +127.9% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.17 | 22.34 | +26.1% |
| 2 | 50.19 | 37.08 | +35.4% |
| 4 | 84.72 | 54.82 | +54.5% |
| 6 | 105.97 | 65.34 | +62.2% |
| 8 | 122.48 | 68.79 | +78.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.37 | 31.77 | +5.0% | 64.92 | 46.93 | +38.3% |
| 2 | 57.20 | 42.79 | +33.7% | 93.02 | 52.57 | +76.9% |
| 4 | 83.99 | 48.78 | +72.2% | 146.39 | 48.23 | +203.5% |
| 6 | 105.52 | 52.59 | +100.6% | 166.28 | 48.51 | +242.8% |
| 8 | 121.15 | 61.92 | +95.7% | 184.74 | 58.13 | +217.8% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 14.94 | 117.83 | +688.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 86.18 | 84.84 | -1.6% |
| 16K prefix, pp4096 + tg128 | 87.34 | 85.31 | -2.3% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

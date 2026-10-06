# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user pp/tg: October 6, 2026 (`4c5a00d2`); concurrency, loading,
memory and reference results: September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1639.03 | 489.59 | +234.8% | 25.96 | 22.20 | +16.9% |
| 4,096 | 1564.36 | 455.24 | +243.6% | 25.96 | 21.21 | +22.4% |
| 8,192 | 1557.47 | 428.85 | +263.2% | 25.95 | 20.40 | +27.2% |
| 12,288 | 1553.60 | 400.21 | +288.2% | 25.93 | 19.64 | +32.0% |
| 16,384 | 1553.97 | 375.89 | +313.4% | 25.89 | 18.95 | +36.6% |
| 32,768 | 1540.25 | 301.31 | +411.2% | 25.75 | 16.54 | +55.7% |
| 65,536 | 1459.69 | 221.94 | +557.7% | 25.50 | 11.62 | +119.4% |
| 131,072 | 1444.79 | 144.78 | +897.9% | 24.90 | 7.98 | +212.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1584.63 | 468.76 | +238.0% | 31.33 | 31.73 | -1.3% | 60.36 | 48.12 | +25.4% |
| 4,096 | 1528.90 | 423.79 | +260.8% | 32.83 | 34.48 | -4.8% | 57.61 | 45.71 | +26.0% |
| 8,192 | 1528.87 | 394.98 | +287.1% | 32.41 | 32.20 | +0.7% | 49.22 | 44.31 | +11.1% |
| 12,288 | 1511.31 | 365.91 | +313.0% | 34.72 | 32.32 | +7.4% | 42.17 | 43.53 | -3.1% |
| 16,384 | 1503.69 | 341.98 | +339.7% | 35.21 | 32.07 | +9.8% | 50.56 | 43.66 | +15.8% |
| 32,768 | 1501.78 | 277.61 | +441.0% | 34.21 | 25.41 | +34.6% | 44.99 | 39.28 | +14.5% |
| 65,536 | 1415.89 | 208.04 | +580.6% | 33.50 | 19.65 | +70.5% | 44.45 | 27.75 | +60.2% |
| 131,072 | 1403.96 | 135.42 | +936.7% | 33.10 | 14.63 | +126.2% | 44.34 | 20.96 | +111.5% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.85 | 22.34 | +15.7% |
| 2 | 45.70 | 37.08 | +23.2% |
| 4 | 76.29 | 54.82 | +39.2% |
| 6 | 95.66 | 65.34 | +46.4% |
| 8 | 108.67 | 68.79 | +58.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.09 | 31.77 | +1.0% | 59.20 | 46.93 | +26.1% |
| 2 | 51.68 | 42.79 | +20.8% | 93.16 | 52.57 | +77.2% |
| 4 | 75.92 | 48.78 | +55.6% | 127.91 | 48.23 | +165.2% |
| 6 | 89.18 | 52.59 | +69.6% | 142.14 | 48.51 | +193.0% |
| 8 | 106.47 | 61.92 | +71.9% | 157.22 | 58.13 | +170.5% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.45 | 117.83 | +662.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

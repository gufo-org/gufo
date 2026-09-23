# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. UD-Q4_K_XL target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Measurements from September 22, 2026. llama.cpp uses `b11069` for AR and
`6fcaa16f` for MTP.

Positive gain favors Gufo. **TODO** means unmeasured; **n/a** marks the recorded
reference memory limit.
[Quality and measurement details](EVALUATION.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1628.52 | 489.59 | +232.6% | 26.04 | 22.20 | +17.3% |
| 4,096 | 1523.39 | 455.24 | +234.6% | 25.98 | 21.21 | +22.5% |
| 8,192 | 1499.13 | 428.85 | +249.6% | 25.91 | 20.40 | +27.0% |
| 12,288 | 1477.98 | 400.21 | +269.3% | 25.46 | 19.64 | +29.6% |
| 16,384 | 1457.16 | 375.89 | +287.7% | 25.20 | 18.95 | +33.0% |
| 32,768 | 1421.93 | 301.31 | +371.9% | 24.33 | 16.54 | +47.1% |
| 65,536 | 1304.01 | 221.94 | +487.6% | 23.21 | 11.62 | +99.7% |
| 131,072 | 1292.02 | 144.78 | +792.4% | 21.93 | 7.98 | +174.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up. The reference exceeded available
memory at 64K/128K.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1605.30 | 468.76 | +242.5% | 32.18 | 31.73 | +1.4% | 59.39 | 48.12 | +23.4% |
| 4,096 | 1509.49 | 423.79 | +256.2% | 33.45 | 34.48 | -3.0% | 48.07 | 45.71 | +5.2% |
| 8,192 | 1472.04 | 394.98 | +272.7% | 33.32 | 32.20 | +3.5% | 50.19 | 44.31 | +13.3% |
| 12,288 | 1451.29 | 365.91 | +296.6% | 34.06 | 32.32 | +5.4% | 40.07 | 43.53 | -7.9% |
| 16,384 | 1429.19 | 341.98 | +317.9% | 33.09 | 32.07 | +3.2% | 50.57 | 43.66 | +15.8% |
| 32,768 | 1389.83 | 277.61 | +400.6% | 28.70 | 25.41 | +12.9% | 45.25 | 39.28 | +15.2% |
| 65,536 | 1202.92 | n/a | n/a | 28.69 | n/a | n/a | 37.23 | n/a | n/a |
| 131,072 | 1272.39 | n/a | n/a | 26.31 | n/a | n/a | 39.20 | n/a | n/a |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Repetitive workload, context 4096 per user, tg128. Throughput sums individual
request decode rates, excluding prefill and scheduling.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 27.05 | 22.82 | +18.5% |
| 2 | 46.23 | 41.19 | +12.2% |
| 4 | 76.55 | 66.37 | +15.3% |
| 6 | 96.12 | 79.58 | +20.8% |
| 8 | 110.51 | 86.38 | +27.9% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Mixed/repetitive workloads, context 4096 per user, tg128. Rates sum individual
request decode rates. No prompt-cache hits. The reference ran out of memory at C8.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.91 | 44.04 | +8.8% | 86.30 | 52.47 | +64.5% |
| 2 | 71.26 | 58.63 | +21.5% | 132.99 | 83.41 | +59.4% |
| 4 | 97.54 | 62.82 | +55.3% | 173.25 | 87.56 | +97.9% |
| 6 | 112.20 | 71.84 | +56.2% | 190.48 | 99.56 | +91.3% |
| 8 | 122.77 | n/a | n/a | 199.02 | n/a | n/a |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | TODO | TODO | TODO |
<!-- /bench -->

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

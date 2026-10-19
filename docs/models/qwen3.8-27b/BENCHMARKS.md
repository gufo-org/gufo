# Qwen27B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo versus llama.cpp
`b11069`, using the same Q4_K_XL / Q8_K_XL targets and Q4_K_M DFlash2 draft.
Gufo uses its adaptive controller. HTTP, greedy, thinking off.

Gufo DFlash2 rows were refreshed October 19, 2026 on `d5e90965` with the
local snapshot serialization cleanup. AR, loading and memory retain the
September 23 `628ed18e` measurements. All llama.cpp measurements are unchanged.
[Recurrent-state review](artifacts/recurrent-fork-review.json) records matched
main/PR checks and the remaining variable latency observations.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Qwen27B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 656.33 | 357.54 | +83.6% | 12.37 | 12.06 | +2.6% |
| 4,096 | 633.16 | 330.42 | +91.6% | 12.21 | 11.88 | +2.8% |
| 8,192 | 615.06 | 316.09 | +94.6% | 12.03 | 11.73 | +2.6% |
| 12,288 | 594.61 | 302.54 | +96.5% | 11.88 | 11.59 | +2.5% |
| 16,384 | 575.50 | 290.23 | +98.3% | 11.72 | 11.44 | +2.4% |
| 32,768 | 505.07 | 246.97 | +104.5% | 11.13 | 10.89 | +2.2% |
| 65,536 | 386.92 | 193.61 | +99.8% | 10.13 | 9.93 | +2.0% |
| 131,072 | 287.12 | 135.95 | +111.2% | 8.52 | 8.46 | +0.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q8 -->
| Qwen27B Q8 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 485.26 | 337.08 | +44.0% | 7.20 | 7.20 | +0.0% |
| 4,096 | 470.80 | 317.96 | +48.1% | 7.14 | 7.14 | +0.0% |
| 8,192 | 450.84 | 306.22 | +47.2% | 7.08 | 7.08 | +0.0% |
| 12,288 | 436.37 | 289.19 | +50.9% | 7.03 | 7.02 | +0.1% |
| 16,384 | 423.19 | 276.91 | +52.8% | 6.97 | 6.97 | +0.0% |
| 32,768 | 381.43 | 236.32 | +61.4% | 6.76 | 6.76 | +0.0% |
| 65,536 | 311.47 | 186.67 | +66.9% | 6.38 | 6.38 | +0.0% |
| 131,072 | 248.60 | 133.34 | +86.4% | 5.71 | 5.74 | -0.5% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

## Single user, DFlash2

pp is the highest measured rate per engine and depth across mixed/repetitive
text. Output-consistency controls and their dated scope are recorded in
the [quality details](QUALITY.md#meaning-of-exact).

<!-- bench:single-dflash2-q4 -->
| Qwen27B Q4 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 575.84 | 347.51 | +65.7% | 26.87 | 23.40 | +14.8% | 64.38 | 35.86 | +79.5% |
| 4,096 | 540.33 | 316.58 | +70.7% | 27.93 | 23.03 | +21.3% | 53.30 | 34.47 | +54.6% |
| 8,192 | 516.06 | 302.47 | +70.6% | 26.26 | 21.73 | +20.8% | 62.44 | 34.73 | +79.8% |
| 12,288 | 489.05 | 289.98 | +68.6% | 24.48 | 23.05 | +6.2% | 54.12 | 34.19 | +58.3% |
| 16,384 | 470.65 | 278.53 | +69.0% | 24.28 | 21.84 | +11.2% | 50.28 | 33.75 | +49.0% |
| 32,768 | 410.71 | 239.21 | +71.7% | 19.08 | 18.88 | +1.1% | 48.24 | 31.75 | +51.9% |
| 65,536 | 338.78 | 190.92 | +77.4% | 17.70 | 18.53 | -4.5% | 29.84 | 26.33 | +13.3% |
| 131,072 | 246.36 | 133.37 | +84.7% | 14.14 | 14.58 | -3.0% | 27.09 | 22.93 | +18.1% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4.svg)

---

<!-- bench:single-dflash2-q8 -->
| Qwen27B Q8 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 443.39 | 333.85 | +32.8% | 16.40 | 15.00 | +9.3% | 47.47 | 23.83 | +99.2% |
| 4,096 | 416.97 | 310.23 | +34.4% | 16.71 | 16.12 | +3.7% | 41.06 | 23.59 | +74.1% |
| 8,192 | 407.51 | 292.86 | +39.1% | 15.82 | 15.27 | +3.6% | 42.07 | 23.21 | +81.3% |
| 12,288 | 394.04 | 280.40 | +40.5% | 15.31 | 13.21 | +15.9% | 40.75 | 22.10 | +84.4% |
| 16,384 | 384.66 | 269.03 | +43.0% | 14.62 | 13.83 | +5.7% | 39.87 | 22.56 | +76.7% |
| 32,768 | 332.78 | 231.89 | +43.5% | 14.60 | 14.44 | +1.1% | 36.22 | 21.16 | +71.2% |
| 65,536 | 284.37 | 185.73 | +53.1% | 11.59 | 12.62 | -8.2% | 25.58 | 19.03 | +34.4% |
| 131,072 | 216.29 | 131.80 | +64.1% | 9.22 | 11.62 | -20.7% | 24.09 | 17.23 | +39.8% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q8.svg)

## Multiple users, autoregressive

Context 4096 per user, tg128. Throughput sums individual request decode rates,
excluding prefill and scheduling. One AR workload per concurrency.

<!-- bench:multi-ar-q4 -->
| Qwen27B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.42 | 12.20 | +1.8% |
| 2 | 23.84 | 22.58 | +5.6% |
| 4 | 44.23 | 38.94 | +13.6% |
| 6 | 60.99 | 47.45 | +28.5% |
| 8 | 74.46 | 50.14 | +48.5% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q8 -->
| Qwen27B Q8 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 7.22 | 7.24 | -0.3% |
| 2 | 14.45 | 13.71 | +5.4% |
| 4 | 27.81 | 25.32 | +9.8% |
| 6 | 39.96 | 34.64 | +15.4% |
| 8 | 51.62 | 41.57 | +24.2% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, DFlash2

Mixed prompts and repetitive output, context 4096 per user, up to 128 output
tokens. Rates sum individual request decode rates. No prompt-cache hits.

<!-- bench:multi-dflash2-q4 -->
| Qwen27B Q4 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.22 | 27.15 | +37.1% | 71.31 | 37.56 | +89.9% |
| 2 | 57.17 | 40.04 | +42.8% | 108.59 | 51.65 | +110.2% |
| 4 | 73.85 | 74.43 | -0.8% | 113.69 | 93.38 | +21.7% |
| 6 | 86.72 | 77.19 | +12.3% | 117.12 | 97.52 | +20.1% |
| 8 | 90.48 | 71.71 | +26.2% | 128.41 | 113.56 | +13.1% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q4.svg)

---

<!-- bench:multi-dflash2-q8 -->
| Qwen27B Q8 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.05 | 17.54 | +42.8% | 51.66 | 24.09 | +114.4% |
| 2 | 44.61 | 31.64 | +41.0% | 87.27 | 41.60 | +109.8% |
| 4 | 52.91 | 43.88 | +20.6% | 106.56 | 58.92 | +80.9% |
| 6 | 58.75 | 53.14 | +10.6% | 115.34 | 72.40 | +59.3% |
| 8 | 67.51 | 60.64 | +11.3% | 123.57 | 86.25 | +43.3% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q8.svg)

## Loading time

C1, context capacity 262144, DFlash2. Cold target/draft files to HTTP readiness.

<!-- bench:loading -->
| Qwen27B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 5.87 | 23.33 | +297.4% |
| Q8 | 8.17 | 36.47 | +346.4% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 262144, AR. Peak memory reported by HIP.

<!-- bench:memory-q4 -->
| Qwen27B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 37.63 | 36.19 | -3.8% |
| 16K prefix, pp4096 + tg128 | 39.88 | 36.62 | -8.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q8 -->
| Qwen27B Q8 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 50.58 | 48.60 | -3.9% |
| 16K prefix, pp4096 + tg128 | 52.83 | 49.02 | -7.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q8.svg)

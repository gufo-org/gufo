# Inkling 2 Mini benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory, ROCm 10.1. Unsloth
`UD-Q4_K_XL` (revision `8078d149`) with its Q8_0 DSpark drafter. HTTP, greedy,
thinking off (Inkling may still think briefly). Gufo `1d61903a` (November 26, 2026); llama.cpp
`4c92c5bfb` ROCm (November 24, 2026). Positive gain favors Gufo. **TODO**
means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method)

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Inkling 2 Mini UD-Q4_K_XL AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2295.93 | 346.56 | +562.5% | 47.42 | 21.95 | +116.0% |
| 4,096 | 2170.23 | 275.54 | +687.6% | 45.95 | 21.11 | +117.7% |
| 8,192 | 2104.73 | 236.43 | +790.2% | 44.49 | 20.26 | +119.6% |
| 12,288 | 2053.84 | 207.41 | +890.2% | 43.17 | 19.61 | +120.1% |
| 16,384 | 1996.69 | 185.51 | +976.3% | 42.00 | 18.76 | +123.9% |
| 32,768 | 1818.12 | 129.53 | +1303.6% | 37.95 | 15.88 | +139.0% |
| 65,536 | 1550.29 | 78.20 | +1882.5% | 31.91 | 12.11 | +163.5% |
| 131,072 | 1189.05 | 42.46 | +2700.4% | 24.32 | 7.74 | +214.2% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, DSpark

Shared pp, separate mixed/repetitive tg and gains. pp is the highest measured
rate per engine and depth across both text types. With thinking off Inkling
still thinks first, so the 128 timed tokens are its thought and the repetitive
task does not reach the copy. Output equals AR at every depth (reasoning
included). llama.cpp cannot load the DSpark drafter: N/A.

<!-- bench:single-dspark -->
| Inkling 2 Mini UD-Q4_K_XL DSpark<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2298.49 | N/A | N/A | 59.35 | N/A | N/A | 63.63 | N/A | N/A |
| 4,096 | 2158.32 | N/A | N/A | 57.55 | N/A | N/A | 56.05 | N/A | N/A |
| 8,192 | 2101.51 | N/A | N/A | 59.46 | N/A | N/A | 59.03 | N/A | N/A |
| 12,288 | 2045.97 | N/A | N/A | 63.26 | N/A | N/A | 57.14 | N/A | N/A |
| 16,384 | 1985.45 | N/A | N/A | 52.87 | N/A | N/A | 57.94 | N/A | N/A |
| 32,768 | 1812.56 | N/A | N/A | 54.10 | N/A | N/A | 50.10 | N/A | N/A |
| 65,536 | 1541.95 | N/A | N/A | 44.71 | N/A | N/A | 40.98 | N/A | N/A |
| 131,072 | 1181.88 | N/A | N/A | 35.96 | N/A | N/A | 31.51 | N/A | N/A |
<!-- /bench -->

![Single user, DSpark](artifacts/charts/single-dspark.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Prefill every session before timed decoding; sum individual request decode rates.

<!-- bench:multi-ar -->
| Inkling 2 Mini UD-Q4_K_XL AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 47.52 | 22.89 | +107.6% |
| 2 | 83.32 | 34.78 | +139.6% |
| 4 | 117.68 | 51.70 | +127.6% |
| 6 | 134.26 | 62.26 | +115.6% |
| 8 | 189.74 | 68.33 | +177.7% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, DSpark

Same pp2048 mixed/repetitive prompts as single-user d0, tg128. C1 directly
cross-checks that row. Prefill every session before timed decoding. Output
equals the isolated AR reference at every concurrency. llama.cpp: N/A, as
above.

<!-- bench:multi-dspark -->
| Inkling 2 Mini UD-Q4_K_XL DSpark<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 57.80 | N/A | N/A | 63.13 | N/A | N/A |
| 2 | 103.04 | N/A | N/A | 100.40 | N/A | N/A |
| 4 | 144.92 | N/A | N/A | 154.28 | N/A | N/A |
| 6 | 157.05 | N/A | N/A | 167.81 | N/A | N/A |
| 8 | 196.54 | N/A | N/A | 194.71 | N/A | N/A |
<!-- /bench -->

![Multiple users, DSpark](artifacts/charts/multi-dspark.svg)

## Loading time

C1, capacity 262,144, DSpark. Cold model files (`POSIX_FADV_DONTNEED`) to HTTP
readiness. llama.cpp: N/A (it cannot load the drafter).

<!-- bench:loading -->
| Inkling 2 Mini UD-Q4_K_XL<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 3.88 | N/A | N/A |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, capacity 133,121, AR, no projector. llama-server preallocates its KV
cache, so its footprint does not grow with the prefix.

<!-- bench:memory -->
| Inkling 2 Mini UD-Q4_K_XL AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 28.57 | 36.97 | +29.4% |
| 16K prefix, pp4096 + tg128 | 30.35 | 37.49 | +23.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## DSpark after a cache restore

Greedy, 400 tokens, WikiText-2 prose plus "Summarize the text above in five
bullet points", `--context 32768`; the second and third requests restore the
whole prompt from the RAM cache. Decode tok/s, Gufo `82645e0a` (checkpoints
hold the drafter's context) against `94ef8903` (a restore dropped it).

| Inkling Q4 DSpark<br>Prompt (tokens) | Fresh (tok/s) | Restored, before (tok/s) | Restored (tok/s) |
| ---: | ---: | ---: | ---: |
| 3,875 | 76.0 | 52.7 / 53.5 | 78.1 / 76.8 |
| 15,694 | 75.8 | 50.7 / 50.8 | 76.0 / 76.3 |
| 30,177 | 57.7 | 44.2 / 44.8 | 59.4 / 60.1 |

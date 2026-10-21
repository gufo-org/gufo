# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
The complete October 18, 2026 Gufo sweeps use `fec4759d` and the validated
host-memory and cache-stall fixes; the focused October 21 follow-up is separate.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

Gufo numerical regression checks pass; individual functional timing limits
remain documented in [QUALITY.md](QUALITY.md). Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

At 128K, Gufo reaches 1557.18 tok/s AR prefill and 1481.19 tok/s MTP prefill.
All 24 single-user rows match the previous qualified capture in completion
hashes and prompt/cache counts.
All 63 concurrent tg128 completions match the corresponding C1 AR control.
Repetitive MTP through 64K reports mean ± sample standard deviation across
the sweep and one focused repeat. Initial PP dips at 16K/32K/64K did not
repeat against current main; both observations remain included.
Prior mixed MTP C2 repeats show two draft-controller trajectories (about 54
and 60 tok/s, identical text), so its gain is not entirely attributable to lookup.

## Sampled decode follow-up, October 21

Focused production comparison against main `0c03748a`, candidate `1d3b8c68`,
with the same Nix toolchain, weights and ordered request histories. Model
defaults are used for sampled requests; concurrency is physical, and the C>1
rates sum the individual decode rates. These measurements qualify the decode
changes separately from the complete October 18 tables below. They do not
refresh the reference data.

| Workload | Main tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Copy existing code, default sampling, C1 | 109.59 | 112.30 |
| Generate new code, default sampling, C1 | 75.18 | 75.40 |
| Generate prose, default sampling, C1 | 39.74 | 39.76 |
| Thinking/code, default sampling, C1 | 51.93 | 52.05 |
| Greedy code control, C1 | 70.42 | 70.51 |
| Default-sampled copying, C2 aggregate | 132.93 | 135.45 |

These 128-token screens retain identical outputs and prompt/cache counts;
all 276 matched request/phase measurements pass the 5% and 3 ms gate. The
screens measure speed, not complete coding-task accuracy.

Earlier complete-workflow and mixed-cohort checks against main `da5d5a2b`,
before rebasing onto the merged small-prefill changes, remain separate:

| Workload | Main tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Extraction with thinking, 726 tokens | 82.52 | 82.76 |
| Generate and execute URI tests, 441 tokens | 78.54 | 78.79 |
| File-edit tool call at 32K, thinking off | 76.42 | 77.01 |
| File-edit tool call at 32K, thinking on | 72.40 | 72.47 |
| Mixed greedy/default-sampled copying, C2 | 137.63 | 139.50 |
| Mixed greedy/default-sampled copying, C4 | 200.22 | 202.94 |
| Mixed greedy/default-sampled copying, C6 | 218.78 | 221.54 |
| Mixed greedy/default-sampled copying, C8 | 237.34 | 240.87 |

All outputs in these matched captures agree. The extraction, URI and file-edit
fixtures check their actual results. A real Pi conversation also passes, with ten
successful tool calls and executed assertions; its unseeded run is correctness
evidence, not a matched speed claim.

Canonical greedy pp2048/tg128 spots on current main and candidate use the
original single-user context capacity and cache preparation:

| Mode | Cached depth | Main PP | Candidate PP | Main TG | Candidate TG |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 0 | 1793.17 | 1792.97 | 28.87 | 28.88 |
| AR | 32,768 | 1651.93 | 1650.85 | 28.74 | 28.77 |
| MTP mixed | 0 | 1724.11 | 1727.57 | 36.18 | 36.31 |
| MTP mixed | 32,768 | 1607.67 | 1605.39 | 40.56 | 40.51 |
| MTP repetitive | 0 | 1737.22 | 1733.71 | 118.61 | 120.25 |
| MTP repetitive | 32,768 | 1609.07 | 1609.65 | 113.61 | 115.07 |

Concurrent greedy TG: repetitive C2 166.15 → 170.96, repetitive C8
245.05 → 247.09, mixed C8 133.65 → 134.61 tok/s. All measured concurrent
completions match isolated AR controls.

Mixed C2 remains variable: the current pair is 62.17 → 55.30 tok/s and the
focused reverse-order control is 53.30 → 55.29. The pre-rebase pair was
62.17 → 55.36 and its reverse-order repeat was 53.30 → 64.03. Main proposes
93 versus 39 drafts per response across its current runs; the final candidate
proposes 56 in both. Both builds produce identical text. These observations do
**not** establish a C2 speedup or a confirmed slowdown. Individual cache,
queue and preparation timing flags remain visible in the
[quality record](QUALITY.md#exact-sampled-decode-october-21-2026); no blanket
timing pass is claimed.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1801.01 | 489.59 | +267.9% | 28.29 | 22.20 | +27.4% |
| 4,096 | 1727.95 | 455.24 | +279.6% | 28.27 | 21.21 | +33.3% |
| 8,192 | 1721.41 | 428.85 | +301.4% | 28.27 | 20.40 | +38.6% |
| 12,288 | 1710.64 | 400.21 | +327.4% | 28.21 | 19.64 | +43.6% |
| 16,384 | 1706.86 | 375.89 | +354.1% | 28.25 | 18.95 | +49.1% |
| 32,768 | 1672.60 | 301.31 | +455.1% | 28.18 | 16.54 | +70.4% |
| 65,536 | 1602.93 | 221.94 | +622.2% | 28.09 | 11.62 | +141.7% |
| 131,072 | 1557.18 | 144.78 | +975.5% | 27.85 | 7.98 | +249.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1734.96 ± 0.85 | 468.76 | +270.1% | 35.79 | 31.73 | +12.8% | 116.93 ± 0.10 | 48.12 | +143.0% |
| 4,096 | 1676.83 ± 1.73 | 423.79 | +295.7% | 37.64 | 34.48 | +9.2% | 116.41 ± 0.07 | 45.71 | +154.7% |
| 8,192 | 1669.91 | 394.98 | +322.8% | 36.91 | 32.20 | +14.6% | 114.08 ± 0.01 | 44.31 | +157.5% |
| 12,288 | 1660.54 ± 1.22 | 365.91 | +353.8% | 39.28 | 32.32 | +21.5% | 105.48 ± 0.15 | 43.53 | +142.3% |
| 16,384 | 1635.18 | 341.98 | +378.2% | 40.54 | 32.07 | +26.4% | 113.32 ± 0.23 | 43.66 | +159.6% |
| 32,768 | 1618.63 | 277.61 | +483.1% | 39.97 | 25.41 | +57.3% | 105.38 ± 0.17 | 39.28 | +168.3% |
| 65,536 | 1556.32 | 208.04 | +648.1% | 39.23 | 19.65 | +99.6% | 104.78 ± 0.21 | 27.75 | +277.6% |
| 131,072 | 1481.19 | 135.42 | +993.8% | 39.87 | 14.63 | +172.5% | 108.37 | 20.96 | +417.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.24 | 22.34 | +26.4% |
| 2 | 50.66 | 37.08 | +36.6% |
| 4 | 86.01 | 54.82 | +56.9% |
| 6 | 108.34 | 65.34 | +65.8% |
| 8 | 125.48 | 68.79 | +82.4% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.74 | 31.77 | +12.5% | 116.50 | 46.93 | +148.2% |
| 2 | 61.30 | 42.79 | +43.3% | 165.50 | 52.57 | +214.8% |
| 4 | 88.77 | 48.78 | +82.0% | 218.12 | 48.23 | +352.2% |
| 6 | 109.74 | 52.59 | +108.7% | 231.85 | 48.51 | +377.9% |
| 8 | 127.02 | 61.92 | +105.1% | 245.81 | 58.13 | +322.9% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 16.42 | 117.83 | +617.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.78 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.53 | 85.31 | -5.8% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

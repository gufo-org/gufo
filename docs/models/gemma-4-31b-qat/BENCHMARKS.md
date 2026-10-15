# Gemma 4 31B QAT benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-31B-it-qat-UD-Q4_K_XL` target (every projection Q4_0) and its Q4_0
QAT `gemma4-assistant` MTP drafter. Gufo drafts up to 15 tokens under its
calibrated length control, verifying the drafter's runner-up beside each
draft; llama.cpp drafts up to four. Gufo columns measured October 7, 2026;
llama.cpp columns September 28. HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B QAT AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 573.57 | 329.96 | +73.8% | 12.05 | 10.52 | +14.5% |
| 4,096 | 532.06 | 290.26 | +83.3% | 11.90 | 10.31 | +15.4% |
| 8,192 | 506.69 | 263.49 | +92.3% | 11.68 | 10.11 | +15.5% |
| 12,288 | 460.77 | 231.95 | +98.7% | 11.59 | 9.91 | +17.0% |
| 16,384 | 426.22 | 213.47 | +99.7% | 11.40 | 9.73 | +17.2% |
| 32,768 | 368.26 | 160.99 | +128.7% | 10.87 | 9.06 | +20.0% |
| 65,536 | 276.13 | 107.53 | +156.8% | 9.94 | 7.95 | +25.0% |
| 131,072 | 176.66 | 65.93 | +168.0% | 8.46 | 6.39 | +32.4% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.
Greedy texts differ between the engines, so accepted drafts per cycle differ
per depth (artifacts); llama.cpp's repetitive acceptance collapses past 32K.

<!-- bench:single-mtp -->
| Gemma 4 31B QAT MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 579.65 | 322.01 | +80.0% | 25.82 | 21.22 | +21.7% | 121.83 | 37.97 | +220.9% |
| 4,096 | 516.52 | 274.16 | +88.4% | 27.17 | 22.52 | +20.6% | 112.77 | 38.44 | +193.4% |
| 8,192 | 483.59 | 247.28 | +95.6% | 23.59 | 21.39 | +10.3% | 106.02 | 34.04 | +211.5% |
| 12,288 | 450.19 | 225.34 | +99.8% | 21.13 | 18.46 | +14.5% | 102.21 | 32.02 | +219.2% |
| 16,384 | 436.04 | 207.50 | +110.1% | 23.30 | 20.13 | +15.7% | 90.52 | 31.57 | +186.7% |
| 32,768 | 367.81 | 156.96 | +134.3% | 21.36 | 14.76 | +44.7% | 76.95 | 17.62 | +336.7% |
| 65,536 | 275.89 | 106.01 | +160.2% | 17.81 | 10.94 | +62.8% | 47.96 | 9.60 | +399.6% |
| 131,072 | 184.43 | 64.90 | +184.2% | 12.46 | 6.69 | +86.2% | 20.61 | 3.85 | +435.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Single user, sampled MTP

Temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05; a story-writing turn
after the cached prefix. Mean of five requests with seeds 1–5. Both engines
accept about as many drafts per request (70–78 vs 69–76), but Gufo's calibrated
draft length proposes fewer (160–190 vs 198–225; 43% vs 34% accepted), so less
of each cycle verifies drafts that are rejected.

<!-- bench:single-mtp-sampled -->
| Gemma 4 31B QAT MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 558.77 ± 22.78 | 317.38 ± 2.46 | +76.1% | 24.03 ± 1.89 | 19.84 ± 1.03 | +21.1% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| Gemma 4 31B QAT AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.04 | 10.50 | +14.7% |
| 2 | 21.69 | 18.99 | +14.2% |
| 4 | 37.66 | 32.10 | +17.3% |
| 6 | 47.86 | 40.95 | +16.9% |
| 8 | 57.82 | 45.30 | +27.6% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. As on the standard 31B, Gufo verifies all sessions'
drafts in one forward of at most 16 exact rows (16 / users − 1 drafts per
session), while llama.cpp verifies up to four drafts per user with
Q8_1-activation matrix kernels that scale further with rows.

<!-- bench:multi-mtp -->
| Gemma 4 31B QAT MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.69 | 24.36 | +5.5% | 120.32 | 39.61 | +203.8% |
| 2 | 42.58 | 38.18 | +11.5% | 129.65 | 60.58 | +114.0% |
| 4 | 67.74 | 37.44 | +80.9% | 124.04 | 66.07 | +87.7% |
| 6 | 79.58 | 49.68 | +60.2% | 136.89 | 87.51 | +56.4% |
| 8 | 82.71 | 61.50 | +34.5% | 105.86 | 100.93 | +4.9% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## MTP draft policy

`--draft-policy calibrated` became the default on September 28, 2026; the MTP
tables above use it. Relative A/B of the two policies on one development
build (`gpu-test` preset), two runs each in ABBA order with one server per
run; greedy texts are identical under both policies, sampled texts differ
because draft counts change the random draws. Depth is a shared earlier
conversation turn reused from the prompt cache.
[d0](artifacts/draft-policy-ab-d0k.json) · [8K](artifacts/draft-policy-ab-d8k.json) ·
`tools/gemma4/draft_policy_bench.py`

| Gemma 4 31B QAT MTP<br>Workload | d0 confidence (tok/s) | d0 calibrated (tok/s) | Gain | 8K confidence (tok/s) | 8K calibrated (tok/s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose, greedy (12 prompts) | 25.26 | 25.98 | +2.9% | 23.45 | 24.04 | +2.5% |
| Story, temperature 1 (12 seeds) | 22.11 | 22.48 | +1.7% | 20.82 | 21.17 | +1.7% |
| Code writing, greedy | 42.17 | 42.99 | +2.0% | 38.04 | 37.95 | -0.2% |
| Code writing, temperature 0.6 (12 seeds) | 30.82 | 32.36 | +5.0% | 27.89 | 29.04 | +4.1% |
| Code edit in context, greedy | 62.79 | 62.58 | -0.3% | 59.84 | 59.36 | -0.8% |
| Repetitive, greedy | 66.82 | 64.60 | -3.3% | 62.34 | 61.49 | -1.4% |

Prose and new code gain; verbatim copying is within about 1.5% (the QAT
d0 repetitive cell includes one slow calibrated run). Per-run values and
draft counts are in the artifacts.

## Loading time

C1, capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 31B QAT<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| QAT | 3.95 | 6.46 | +63.5% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR. Both engines allocate KV for the whole capacity;
device total minus free on this unified-memory APU. Gufo's figure includes its
prompt cache snapshots (see the
[standard 31B](../gemma-4-31b/BENCHMARKS.md#memory-occupation)); llama.cpp runs
with `--cache-ram 0`.

<!-- bench:memory -->
| Gemma 4 31B QAT AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 30.99 | 33.54 | +8.2% |
| 16K prefix, pp4096 + tg128 | 37.55 | 35.71 | -4.9% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

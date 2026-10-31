# Gemma 4 31B QAT benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth
`gemma-4-31B-it-qat-UD-Q4_K_XL` target (every projection Q4_0) and its Q4_0
QAT `gemma4-assistant` MTP drafter. Gufo drafts up to 15 tokens under its
calibrated length control, verifying the drafter's runner-up beside each
draft; llama.cpp drafts up to four. Gufo columns measured October 17, 2026 on a
reduced grid (depths to 32K, up to two users; the other Gufo cells are TODO);
llama.cpp columns September 28. HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069` (ROCm) reference.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B QAT AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 579.89 | 329.96 | +75.7% | 12.05 | 10.52 | +14.5% |
| 4,096 | 546.00 | 290.26 | +88.1% | 11.90 | 10.31 | +15.4% |
| 8,192 | TODO | 263.49 | TODO | TODO | 10.11 | TODO |
| 12,288 | TODO | 231.95 | TODO | TODO | 9.91 | TODO |
| 16,384 | 471.85 | 213.47 | +121.0% | 11.41 | 9.73 | +17.3% |
| 32,768 | 398.84 | 160.99 | +147.7% | 10.87 | 9.06 | +20.0% |
| 65,536 | TODO | 107.53 | TODO | TODO | 7.95 | TODO |
| 131,072 | TODO | 65.93 | TODO | TODO | 6.39 | TODO |
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
| 0 | 590.21 | 322.01 | +83.3% | 26.91 | 21.22 | +26.8% | 129.78 | 37.97 | +241.8% |
| 4,096 | 533.03 | 274.16 | +94.4% | 28.28 | 22.52 | +25.6% | 121.57 | 38.44 | +216.3% |
| 8,192 | TODO | 247.28 | TODO | TODO | 21.39 | TODO | TODO | 34.04 | TODO |
| 12,288 | TODO | 225.34 | TODO | TODO | 18.46 | TODO | TODO | 32.02 | TODO |
| 16,384 | 462.95 | 207.50 | +123.1% | 26.46 | 20.13 | +31.4% | 104.93 | 31.57 | +232.4% |
| 32,768 | 398.44 | 156.96 | +153.8% | 21.48 | 14.76 | +45.5% | 82.44 | 17.62 | +367.9% |
| 65,536 | TODO | 106.01 | TODO | TODO | 10.94 | TODO | TODO | 9.60 | TODO |
| 131,072 | TODO | 64.90 | TODO | TODO | 6.69 | TODO | TODO | 3.85 | TODO |
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
| 0 | 560.67 ± 11.13 | 317.38 ± 2.46 | +76.7% | 24.61 ± 1.55 | 19.84 ± 1.03 | +24.0% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| Gemma 4 31B QAT AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.05 | 10.50 | +14.8% |
| 2 | 21.66 | 18.99 | +14.1% |
| 4 | TODO | 32.10 | TODO |
| 6 | TODO | 40.95 | TODO |
| 8 | TODO | 45.30 | TODO |
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
| 1 | 26.61 | 24.36 | +9.2% | 127.04 | 39.61 | +220.7% |
| 2 | 43.92 | 38.18 | +15.0% | 135.01 | 60.58 | +122.9% |
| 4 | TODO | 37.44 | TODO | TODO | 66.07 | TODO |
| 6 | TODO | 49.68 | TODO | TODO | 87.51 | TODO |
| 8 | TODO | 61.50 | TODO | TODO | 100.93 | TODO |
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

C1, capacity 262144, MTP. Cold model files to HTTP readiness, with no vision
sidecar beside the target. With `mmproj-BF16.gguf` beside it Gufo also
prepares the vision encoder at load: about 0.7 s more.

<!-- bench:loading -->
| Gemma 4 31B QAT<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| QAT | 3.98 | 6.46 | +62.3% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR, no vision sidecar (with `mmproj-BF16.gguf` beside
the target Gufo holds 2.4 GiB more for the prepared vision encoder). Both
engines allocate KV for the whole capacity;
device total minus free on this unified-memory APU. Gufo's figure includes its
prompt cache snapshots (see the
[standard 31B](../gemma-4-31b/BENCHMARKS.md#memory-occupation)); llama.cpp runs
with `--cache-ram 0`.

<!-- bench:memory -->
| Gemma 4 31B QAT AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.05 | 33.54 | +8.0% |
| 16K prefix, pp4096 + tg128 | 37.61 | 35.71 | -5.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

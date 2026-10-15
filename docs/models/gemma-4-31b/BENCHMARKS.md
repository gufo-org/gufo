# Gemma 4 31B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` and
`UD-Q8_K_XL` targets and their `gemma4-assistant` Q8_0 MTP drafter. Gufo
drafts up to 15 tokens under its calibrated length control, verifying the
drafter's runner-up beside each draft; llama.cpp drafts up to four. The Q8
tables are a reduced grid (depths to 32K, up to two users). Gufo columns
measured October 7, 2026; llama.cpp columns September 26–30.
HTTP, greedy, thinking off. llama.cpp is the repository's pinned `b11069`
(ROCm) reference for AR and MTP.
Positive gain favors Gufo. **TODO** means unmeasured.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 544.93 | 300.91 | +81.1% | 10.95 | 9.80 | +11.7% |
| 4,096 | 506.28 | 264.62 | +91.3% | 10.83 | 9.61 | +12.7% |
| 8,192 | 464.59 | 239.48 | +94.0% | 10.65 | 9.43 | +12.9% |
| 12,288 | 433.79 | 215.96 | +100.9% | 10.58 | 9.26 | +14.3% |
| 16,384 | 411.52 | 202.00 | +103.7% | 10.41 | 9.11 | +14.3% |
| 32,768 | 357.65 | 154.12 | +132.1% | 9.98 | 8.51 | +17.3% |
| 65,536 | 267.93 | 104.28 | +156.9% | 9.18 | 7.53 | +21.9% |
| 131,072 | 174.45 | 64.56 | +170.2% | 7.90 | 6.12 | +29.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

---

<!-- bench:single-ar-q8 -->
| Gemma 4 31B Q8 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 515.74 | 309.27 | +66.8% | 6.35 | 5.84 | +8.7% |
| 4,096 | 471.15 | 271.60 | +73.5% | 6.31 | 5.77 | +9.4% |
| 16,384 | 401.85 | 203.60 | +97.4% | 6.17 | 5.59 | +10.4% |
| 32,768 | 347.71 | 155.38 | +123.8% | 6.01 | 5.36 | +12.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-mtp -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 553.10 | 294.20 | +88.0% | 26.32 | 19.08 | +37.9% | 110.83 | 31.11 | +256.3% |
| 4,096 | 502.88 | 253.33 | +98.5% | 23.79 | 20.32 | +17.1% | 102.40 | 32.28 | +217.2% |
| 8,192 | 465.55 | 231.63 | +101.0% | 23.72 | 18.58 | +27.7% | 108.66 | 28.74 | +278.1% |
| 12,288 | 433.35 | 211.21 | +105.2% | 23.80 | 18.14 | +31.2% | 95.83 | 28.37 | +237.8% |
| 16,384 | 421.40 | 195.22 | +115.9% | 22.63 | 15.95 | +41.9% | 81.93 | 27.19 | +201.3% |
| 32,768 | 357.74 | 150.59 | +137.6% | 20.67 | 14.89 | +38.8% | 79.19 | 21.08 | +275.7% |
| 65,536 | 269.39 | 102.73 | +162.2% | 17.65 | 10.56 | +67.1% | 58.69 | 14.85 | +295.2% |
| 131,072 | 181.73 | 63.79 | +184.9% | 13.06 | 6.92 | +88.7% | 42.88 | 4.09 | +948.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

---

<!-- bench:single-mtp-q8 -->
| Gemma 4 31B Q8 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 519.01 | 301.96 | +71.9% | 15.84 | 13.45 | +17.8% | 60.52 | 22.34 | +170.9% |
| 4,096 | 468.76 | 256.10 | +83.0% | 17.73 | 13.75 | +28.9% | 54.77 | 23.52 | +132.9% |
| 16,384 | 402.31 | 197.65 | +103.5% | 16.74 | 12.63 | +32.5% | 53.42 | 20.58 | +159.6% |
| 32,768 | 349.05 | 151.24 | +130.8% | 14.77 | 10.95 | +34.9% | 45.27 | 14.68 | +208.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q8.svg)

## Single user, sampled MTP

The sampler of a typical chat front end: temperature 1, top-k 64, top-p 0.95,
repeat penalty 1.05; a story-writing turn after the cached prefix (the prefix
turn itself is greedy). Mean of five requests with seeds 1–5; sampled text
differs between engines and runs, so acceptance varies more than in the
greedy tables.

<!-- bench:single-mtp-sampled -->
| Gemma 4 31B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 532.82 ± 21.90 | 291.19 ± 2.26 | +83.0% | 22.29 ± 1.04 | 17.77 ± 1.56 | +25.4% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled.svg)

---

<!-- bench:single-mtp-sampled-q8 -->
| Gemma 4 31B Q8 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 501.15 ± 25.14 | 294.38 ± 2.39 | +70.2% | 14.67 ± 1.46 | 12.28 ± 0.77 | +19.5% |
<!-- /bench -->

![Single user, sampled MTP](artifacts/charts/single-mtp-sampled-q8.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
Gufo decodes up to eight sessions in one forward: row-wise work runs once for
all of them, attention per session. Every session reads its own sliding-window
cache (16 MB per layer), about a fifth of an eight-user step.

<!-- bench:multi-ar -->
| Gemma 4 31B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 10.95 | 9.79 | +11.8% |
| 2 | 19.86 | 17.54 | +13.2% |
| 4 | 33.49 | 28.83 | +16.2% |
| 6 | 44.76 | 34.03 | +31.5% |
| 8 | 54.42 | 35.08 | +55.1% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

---

<!-- bench:multi-ar-q8 -->
| Gemma 4 31B Q8 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 6.36 | 5.84 | +8.9% |
| 2 | 11.91 | 11.04 | +7.9% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table. Gufo verifies all
sessions' drafts in one forward of at most 16 rows (16 / users − 1 drafts per
session), which keeps each row's arithmetic equal to its own session's decode;
llama.cpp verifies up to 4 drafts per user with Q8_1-activation matrix kernels,
which scale further with rows. Gufo's lead from AR does not carry past two
users.

<!-- bench:multi-mtp -->
| Gemma 4 31B Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.77 | 20.73 | +24.3% | 110.83 | 30.93 | +258.3% |
| 2 | 42.67 | 32.25 | +32.3% | 119.84 | 49.44 | +142.4% |
| 4 | 65.75 | 48.28 | +36.2% | 114.85 | 75.05 | +53.0% |
| 6 | 78.15 | 58.11 | +34.5% | 125.06 | 99.65 | +25.5% |
| 8 | 79.58 | 57.05 | +39.5% | 99.09 | 87.82 | +12.8% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

---

<!-- bench:multi-mtp-q8 -->
| Gemma 4 31B Q8 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.44 | 13.43 | +15.0% | 61.15 | 22.28 | +174.5% |
| 2 | 25.50 | 22.62 | +12.7% | 63.87 | 38.23 | +67.1% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q8.svg)


## MTP draft policy

`--draft-policy calibrated` became the default on September 28, 2026; the MTP
tables above use it. Relative A/B of the two policies on one development
build (`gpu-test` preset), two runs each in ABBA order with one server per
run; greedy texts are identical under both policies, sampled texts differ
because draft counts change the random draws. Depth is a shared earlier
conversation turn reused from the prompt cache.
[d0](artifacts/draft-policy-ab-d0k.json) · [8K](artifacts/draft-policy-ab-d8k.json) ·
`tools/gemma4/draft_policy_bench.py`

| Gemma 4 31B Q4 MTP<br>Workload | d0 confidence (tok/s) | d0 calibrated (tok/s) | Gain | 8K confidence (tok/s) | 8K calibrated (tok/s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose, greedy (12 prompts) | 22.81 | 23.05 | +1.0% | 20.90 | 21.39 | +2.4% |
| Story, temperature 1 (12 seeds) | 19.87 | 20.02 | +0.8% | 19.49 | 19.24 | -1.3% |
| Code writing, greedy | 35.28 | 36.16 | +2.5% | 30.09 | 31.15 | +3.5% |
| Code writing, temperature 0.6 (12 seeds) | 27.82 | 28.00 | +0.7% | 25.79 | 26.03 | +0.9% |
| Code edit in context, greedy | 54.50 | 54.38 | -0.2% | 52.76 | 51.94 | -1.5% |
| Repetitive, greedy | 57.38 | 57.03 | -0.6% | 55.27 | 55.06 | -0.4% |

Prose and new code gain; verbatim copying is within about 1.5%. Per-run
values and draft counts are in the artifacts.

## Image requests

Single user, AR, context 16384. Each image was resized once to a
multiple of 48 so both servers encode identical pixels into the same soft
tokens: Gufo at `--image-tokens 280` or `1120`, llama.cpp with its default
70–1120 range. llama.cpp needs `-b 2048 -ub 2048` here: with the default
512-token ubatch it aborts on 1,107-token images (non-causal image batches
must fit one ubatch). Cold: a fresh nonce precedes the image, so nothing is
reused; follow-up: the next user turn, reusing the image prefix. Median of
three warmed requests, 64 output tokens, greedy
([Gufo 280](artifacts/image-gufo-280.json), [Gufo 1120](artifacts/image-gufo-1120.json),
[llama.cpp](artifacts/image-reference.json); `tools/gemma4/image_bench.py`).

| Image | Budget | Prompt tokens | Gufo cold TTFT (s) | llama.cpp cold TTFT (s) | Gain | Gufo follow-up TTFT (s) | llama.cpp follow-up TTFT (s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chart 624×960 | 280 | 312 | 1.32 | 1.92 | +45.5% | 0.41 | 0.70 | +68.4% | 11.43 | 9.90 | +15.5% |
| Logo 768×768 | 280 | 309 | 1.30 | 1.85 | +42.4% | 0.41 | 0.72 | +75.9% | 11.43 | 9.54 | +19.8% |
| Chart 1296×1968 | 1120 | 1160 | 4.57 | 8.34 | +82.4% | 0.47 | 1.04 | +121.8% | 11.05 | 8.27 | +33.6% |
| Logo 1584×1584 | 1120 | 1140 | 4.34 | 8.35 | +92.4% | 0.48 | 1.10 | +126.3% | 11.04 | 7.96 | +38.7% |

Gain is llama.cpp time over Gufo time minus one (decode: Gufo over
llama.cpp). Gufo's vision encoder takes 162 ms for 260 soft tokens and
1,147 ms for 1,107; the rest of a cold request is ordinary prefill.

## Vulkan llama.cpp fork (single user)

A third-party baseline: the Strix Halo llama.cpp fork halo-box/strix-llama.cpp
`8c1c282ec` on Vulkan RADV with `GGML_VK_MMV_NO_SPLIT=1 -b 2048 -ub 512`, same
GGUF files, same driver workloads and four draft tokens (Gufo up to seven).
Hand-rendered from
[artifacts/fork](artifacts/fork); the Gufo cells here are from the
September 26 run (revision b3a13bc), not the refreshed tables above. At 128K with MTP its Vulkan queue timed out
(`Fence fallback timer expired on ring comp_1.1.0`); its repetitive MTP
workload was not run.

Greedy MTP texts differ between the engines, so accepted drafts per cycle
differ per depth; Gufo drafts up to seven tokens under its confidence-based
length control, the fork a fixed four. Gufo's cycle (draft plus verify) is
shorter at every measured depth: 117 vs 120 ms at d0, 118 vs 126 ms at 4K,
132 vs 165 ms at 32K and 159 vs 211 ms at 64K.

| Gemma 4 31B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 427.05 | 293.92 | +45.3% | 10.73 | 10.53 | +1.9% |
| 4,096 | 390.56 | 259.83 | +50.3% | 10.60 | 10.29 | +3.0% |
| 8,192 | 368.90 | 243.79 | +51.3% | 10.46 | 10.01 | +4.5% |
| 12,288 | 345.49 | 220.65 | +56.6% | 10.37 | 9.87 | +5.1% |
| 16,384 | 343.90 | 211.76 | +62.4% | 10.24 | 9.58 | +6.9% |
| 32,768 | 294.07 | 164.54 | +78.7% | 9.81 | 8.93 | +9.9% |
| 65,536 | 225.69 | 117.97 | +91.3% | 9.07 | 7.74 | +17.2% |
| 131,072 | 156.09 | 74.00 | +110.9% | 7.84 | 6.10 | +28.5% |

---

\* Gufo cells measured with the previous `--draft-policy confidence` default; see [MTP draft policy](#mtp-draft-policy) for the calibrated default at d0 and 8K.

| Gemma 4 31B Q4 MTP mixed<br>Depth (tokens) | Gufo pp (tok/s) | fork pp (tok/s) | Gain | Gufo tg (tok/s) | fork tg (tok/s) | Gain | Gufo / fork accepted per cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 418.34 | 286.71 | +45.9% | 23.19 | 21.71 | +6.8% | 1.72 / 1.61 |
| 4,096 | 394.89 | 253.75 | +55.6% | 23.49 | 23.17 | +1.4% | 1.78 / 1.91 |
| 8,192 | 373.51 | 234.69 | +59.2% | 22.25 | 20.84 | +6.8% | 1.67 / 1.78 |
| 12,288 | 356.43 | 216.64 | +64.5% | 21.53 | 17.49 | +23.1% | 1.61 / 1.37 |
| 16,384 | 347.03 | 201.12 | +72.5% | 21.65 | 19.42 | +11.5% | 1.72 / 1.72 |
| 32,768 | 298.45 | 159.66 | +86.9% | 21.00 | 16.47 | +27.5% | 1.78 / 1.72 |
| 65,536 | 229.88 | 114.11 | +101.5% | 15.83 | 12.67 | +24.9% | 1.51 / 1.67 |
| 131,072 | 161.09 | N/A | N/A | 12.73 | N/A | N/A | 1.51 / N/A |

## Loading time

C1, capacity 262144, MTP. Cold model files to HTTP readiness.

<!-- bench:loading -->
| Gemma 4 31B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 4.36 | 6.58 | +50.9% |
| Q8 | 7.04 | 10.27 | +45.9% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 133121 tokens, AR. Both engines allocate KV for the whole capacity.
The counter is device total minus free, which on this unified-memory APU
tracks system-wide use; Gufo (October 7) and llama.cpp (September 27) start
from the same 3.0 GiB idle baseline. Gufo's figure includes its prompt cache:
after the cold 16K prefix it holds about 7 GB of snapshots (a checkpoint
every 2,048 tokens of a cold prompt, the stable conversation boundary and the
complete prompt, sharing the position blocks they have in common), bounded by the snapshot cache and released under memory
pressure. llama.cpp runs with `--cache-ram 0`.
Gufo's global layers store V and only the rotated key dims (50 KiB per
token instead of 80).

<!-- bench:memory -->
| Gemma 4 31B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 32.43 | 35.47 | +9.4% |
| 16K prefix, pp4096 + tg128 | 39.00 | 37.64 | -3.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

---

<!-- bench:memory-q8 -->
| Gemma 4 31B Q8 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 47.50 | 50.79 | +6.9% |
| 16K prefix, pp4096 + tg128 | 54.06 | 52.96 | -2.0% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q8.svg)

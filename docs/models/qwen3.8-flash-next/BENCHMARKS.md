# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
All Gufo tables measured October 18, 2026, with `fec4759d` and the
validated host-memory and cache-stall fixes.
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

## Orca packed K16 loads — rejected trial

**These gains are not retained:** strict repeated HTTP phase checks fail.
Runtime code was restored to the rebased control; faster native prefill does
not override slower individual served phases.

October 21: main `f020bd3a` plus the existing mixed-quant implementation and
its dispatch integration fix (`14954fe2`) versus `f75eb7d9`. The trial groups
Q4_K/Q6_K code loads only at 2048 or more prompt rows; smaller batches and
Q5_0 keep their original reader. Same pinned Nix toolchain, Orca shards and
MTP draft; C1, depth zero, three repetitions, mean ± reported standard
deviation. PP and TG are separate tests; TG starts from 16 tokens.

| Mode | Control pp2048 (tok/s) | Trial pp2048 (tok/s) | PP gain | Control tg128 (tok/s) | Trial tg128 (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 385.00 ± 5.63 | 467.62 ± 3.03 | 21.5% | 17.01 ± 0.23 | 16.74 ± 0.52 |
| MTP, up to 7 drafts | 379.93 ± 1.16 | 460.05 ± 5.91 | 21.1% | 30.67 ± 0.30 | 30.64 ± 0.43 |

A separate pp2048/tg1 capture reduces mixed dense GPU time from 9.18 to
7.13 seconds and total kernel time from 11.10 to 9.05 seconds, with the
same logical dispatch/grid counts. Captures include startup/warmup;
whole-capture idle percentages are not inference utilization.

All 30,543,360 matched-history logits after a 3720-token prompt remain
byte-identical to the control. Nevertheless, repeated HTTP evidence with
contemporaneous unchanged controls confirms five AR and eight MTP phase
failures. The earlier unrestricted variant also failed qualification.
No decode speedup or production promotion is claimed. The service pin and
disabled state, and all UD-Q4_K_XL/llama.cpp benchmark cells, remain unchanged.
[Measurements, rejection and remaining limits](artifacts/mixed-prefill-k16.json).

## Orca Q4_K_M optimized prefill — local native capture

**Experimental: matched prior-output qualification has not passed.** These
are measured speeds, not a production-promotion claim.

October 20, `5f3c5b11` versus `40a68da4`, identical Orca shards and matching
MTP sidecar, pinned production Nix builds, C1, greedy, depth zero. Three
repetitions, mean ± reported standard deviation. Native `pp2048` and `tg128`
are **separate tests**: the TG test starts from 16 tokens, not the PP request.
These numbers do not compare Orca with the different Unsloth weights below.

| Mode | Vector baseline pp (tok/s) | Matrix-core pp (tok/s) | PP factor | Baseline tg (tok/s) | Candidate tg (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 142.59 ± 0.33 | **379.84 ± 1.53** | **2.66×** | 16.77 ± 0.36 | 16.91 ± 0.09 |
| MTP, up to 7 drafts | 142.59 ± 2.16 | **378.30 ± 1.83** | **2.65×** | 32.23 ± 0.44 | 30.16 ± 0.35 |

No decode speedup is claimed: MTP TG is 6.4% lower in this capture, and the
prefill arithmetic is not byte-compatible with the initial implementation.
See the explicit numerical comparison in [QUALITY.md](QUALITY.md).

A separate matched `pp2048/tg1` profile reduces mixed-projection GPU time from
27.46 to 9.19 seconds and total kernel time from 30.05 to 11.11 seconds.
The captures include model startup; their whole-capture idle percentages are
not serving utilization measurements. This is bounded local qualification,
not vision, deep-context or unquantized-model accuracy qualification.
[Commands, build identities, raw results and limitations](artifacts/mixed-prefill-wmma.json).
All existing UD-Q4_K_XL production and llama.cpp tables remain unchanged.

## Initial Orca Q4_K_M compatibility sample

October 20: one native pp2048/tg128 sample on the assertion-enabled development
build measures **143.94 pp / 16.61 tg tok/s AR** and **142.16 pp / 31.97 tg tok/s
MTP**. That initial build used eight-row vector tiles, before the optimized
mixed-quant prefill routes. These are observational smoke results, not a
matched performance qualification or a comparison with the different Unsloth
weights below. [Commands, identities and limits](artifacts/mixed-q4-k-m.json).
All existing production and llama.cpp tables remain unchanged.

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

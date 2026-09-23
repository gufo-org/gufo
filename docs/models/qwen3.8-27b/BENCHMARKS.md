# Qwen3.8 27B on Gufo

| | |
| --- | --- |
| Host | Linux x86-64, AMD `gfx1151`, 128 GB unified memory |
| Gufo | Focused C1 and concurrency results through `d0843504`, `nix build`; updated artifact rows record binary identities. Retained older controls are dated below |
| Targets | `unsloth/Qwen3.8-27B-GGUF` snapshot `4ca72078`: **UD-Q4_K_XL** (16.35 GiB), **UD-Q8_K_XL** (29.30 GiB). Historical memory Q8 rows use **UD-Q8_K_L** (26.12 GiB) and await replacement |
| Speculative | DFlash2 draft **Q4_K_M** with the **adaptive** controller |
| Reference | llama.cpp `llama-server` release `b11069` (`0.4.1-dev (build 11069)`), ROCm gfx1151, `LLAMA_HIP_UMA=ON`, from `flake.nix`, same GGUF files and same draft through `--spec-type draft-dflash` |
| Method | HTTP, greedy, thinking off, one warmed sample per point, isolated servers. Focused single-user/concurrency controls: 2026-09-23; retained older controls: 2026-09-21 |
| Gain | Gufo over llama.cpp, positive when Gufo is better |
| Quality gate | Full Q4/Q8 target replay, affected operators and generated-frontier continuation pass; [scope and independent-reference limits](EVALUATION.md) |
| Identities | [`artifacts/model-identities.json`](artifacts/model-identities.json) |
| Layout | [benchmark-model skill](../../../.agents/skills/benchmark-model/SKILL.md) |

PNG/JPEG image input uses the matching BF16 projector with AR or DFlash2;
native MTP is CLI-only. [Image usage and quality checks](README.md#images).
Unmeasured points are **TODO**; the reason is stated next to each table.

## Loading and continuation

The cold-file-cache loading table is **not measured**: the refresh host has
no privileged page-cache drop (`echo 3 > /proc/sys/vm/drop_caches`, no
`sudo`/`doas`), so the driver skipped the table on both targets and the
earlier hand-measured Gufo cells were retired rather than kept next to an
unmeasured reference. Rerun with `--drop-caches "<privileged command>"` on a
host that has one.

<!-- bench:loading -->
| Target | Gufo ready | llama.cpp ready | Gain |
| --- | ---: | ---: | ---: |
| Q4 | TODO | TODO | TODO |
| Q8 | TODO | TODO | TODO |
<!-- /bench -->

Warm-cache readiness during this refresh (not a cold measurement): Gufo
reported `load_completed` after 0.65 s (Q4, context 35456) and 1.23 s (Q4,
context 262144) once the GGUF was in the page cache. A 626-token Q4+DFlash2
prompt snapshot occupies **216 MiB**, independent of unused context
capacity; cancel/continue reused 661 tokens and prefilled 61 new tokens,
restoring the prompt snapshot in **3.13 ms** (2026-09-21 hand controls, not
a long-conversation latency distribution).

## Single user, autoregressive

Requested **pp2048 / tg128**, C1. Depth is a cached conversation prefix;
the measured turn appends synthetic paragraphs and requests ordinary prose.
The focused d0/d32K controls use context 36864, actual new-token counts
2010–2060 and a prefix of about 32K; exact counts/history are in the artifacts.
Q4 replays the pinned one-token prefix reply, Q8 an eight-token reply.
Unrefreshed Gufo depths are **TODO**. Retained Q4 reference-only rows are from
the earlier depth sweep; paired refreshed cells use the same input history.
Artifacts: `artifacts/single-ar-{q4,q8}-{gufo,reference}.json`.

<!-- bench:single-ar-q4 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 674.01 | 359.38 | +87.5% | 12.38 | 12.08 | +2.5% |
| 4,096 | TODO | 302.46 | TODO | TODO | 11.92 | TODO |
| 8,192 | TODO | 287.14 | TODO | TODO | 11.77 | TODO |
| 12,288 | TODO | 275.09 | TODO | TODO | 11.61 | TODO |
| 16,384 | TODO | 264.36 | TODO | TODO | 11.45 | TODO |
| 32,768 | 502.29 | 250.69 | +100.4% | 11.14 | 10.90 | +2.2% |
| 65,536 | TODO | 179.00 | TODO | TODO | 9.95 | TODO |
| 131,072 | TODO | 131.74 | TODO | TODO | 8.47 | TODO |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

<!-- bench:single-ar-q8 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 489.22 | 355.17 | +37.7% | 7.20 | 7.20 | +0.0% |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | 386.61 | 249.07 | +55.2% | 6.76 | 6.76 | +0.0% |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

Q4 AR reaches **12.38 tok/s at d0** and **11.14 at d32K**. Q8_K_XL
reaches **7.20 / 6.76 tok/s**, matching the pinned reference at those points.
These measurements supersede the earlier cold-conversation decode deficit.

## Single user, DFlash2

Same focused method as AR, **Q4_K_M draft**, adaptive controller.
llama.cpp uses `--spec-type draft-dflash` with its default draft parameters.
`accepted/step = accepted / (generated − accepted)`; emitted tokens per
verification step is this plus one. DFlash2 prefill includes feature capture
and draft context injection. Q4 d32K's paired reference cell remains **TODO**.
Q8 DFlash2 retains all 128 AR tokens at both depths, including when the
generated prefix is forked across four requests at d32K.
Artifacts: `artifacts/single-dflash2-{q4,q8}-{gufo,reference}.json`.

<!-- bench:single-dflash2-q4 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 619.74 | 344.56 | +79.9% | 30.66 | 27.16 | +12.9% | 2.20 | 1.91 |
| 4,096 | TODO | 298.19 | TODO | TODO | 23.03 | TODO | TODO | 1.51 |
| 8,192 | TODO | 284.44 | TODO | TODO | 21.76 | TODO | TODO | 1.42 |
| 12,288 | TODO | 272.58 | TODO | TODO | 22.66 | TODO | TODO | 1.61 |
| 16,384 | TODO | 260.16 | TODO | TODO | 21.87 | TODO | TODO | 1.51 |
| 32,768 | 476.85 | TODO | TODO | 21.91 | TODO | TODO | 1.78 | TODO |
| 65,536 | TODO | 177.90 | TODO | TODO | 18.32 | TODO | TODO | 1.51 |
| 131,072 | TODO | 126.64 | TODO | TODO | 14.16 | TODO | TODO | 1.37 |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4.svg)

Repetitive workload (**TODO**, not included in the focused depth controls):
same prefixes and depths, the measured turn asks the model to repeat the
passage word for word, so the output is fully predictable — the single-user
analogue of the `repetition` corpus below.

<!-- bench:single-dflash2-repetition-q4 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

<!-- bench:single-dflash2-q8 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 473.79 | 336.19 | +40.9% | 16.11 | 15.02 | +7.3% | 1.56 | 1.46 |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | 359.28 | 232.53 | +54.5% | 16.22 | 13.89 | +16.8% | 2.12 | 1.46 |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q8.svg)

At d32K, llama.cpp DFlash2 differs from its own AR output; the speed comparison
does not establish equivalent output. See [evaluation](EVALUATION.md#meaning-of-exact).

<!-- bench:single-dflash2-repetition-q8 -->
| Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain | Gufo accepted/step | llama.cpp accepted/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 4,096 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8,192 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 12,288 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 16,384 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 32,768 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 65,536 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 131,072 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- /bench -->

[Prompts, artifact identities and quality checks](EVALUATION.md).

## Multiple users, autoregressive

One AR workload per quantization and concurrency: `repetition_word`, exactly
128 output tokens. Context capacity is 4096 per user, greedy, thinking off,
one warmup and one measured repetition. Rates sum individual request decode
rates and average complete cohorts; prefill and scheduling are excluded.
Depth and batch size have separate controls; text categories do not trigger
additional AR performance sweeps.

Gufo Q4 C1 and both engines' Q8 C2/C4/C6 are from **2026-09-23**, with zero
cache hits. Other Q4 values retain **2026-09-21** controls; those Gufo rows
include prompt-cache reuse. Unmeasured Q8 cells remain **TODO**.
Artifacts: `artifacts/multi-ar-{q4,q8}-{gufo-ar,reference}.json`.

<!-- bench:multi-ar-q4 -->
| Users | Gufo AR | llama.cpp AR | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.42 | 12.20 | +1.8% |
| 2 | 23.04 | 22.46 | +2.6% |
| 4 | 41.97 | 38.32 | +9.5% |
| 6 | 56.96 | 44.74 | +27.3% |
| 8 | 67.94 | 46.02 | +47.6% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

<!-- bench:multi-ar-q8 -->
| Users | Gufo AR | llama.cpp AR | Gain |
| ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 2 | 14.45 | 13.78 | +4.9% |
| 4 | 27.81 | 25.35 | +9.7% |
| 6 | 39.96 | 34.72 | +15.1% |
| 8 | TODO | TODO | TODO |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, DFlash2

These tables run and compare **DFlash2 only**, with the Q4_K_M draft on both
servers (Gufo adaptive controller; llama.cpp `draft-dflash` defaults).
Context capacity is 4096 per user (`--sessions C --context 4096` on Gufo,
`-np C -c 4096·C` on llama.cpp), greedy, thinking off, up to 128 output tokens,
one warmup and one measured repetition on a fresh server at each concurrency.
Rates use the same sum of individual request decode rates as the AR tables.

The [corpus](artifacts/speculative-corpus.json) has two workloads:
`repetition` runs `repetition_word` on every user; `mixed` cycles through nine
cases. Its two-sentence summary ends early; rates use actual emitted counts.
Q8_K_XL rows are from **2026-09-23**. Q4 Gufo mixed C4/C6/C8 and repetition C8,
plus the C4 mixed reference, are refreshed; other Q4 cells retain **2026-09-21**
controls. Refreshed rows have zero cache hits; older Gufo rows used prompt-cache
reuse. Source dates and binary identities are recorded per artifact row.

Saved C1 AR hashes qualify each DFlash2 workload without repeating an AR
performance sweep. All refreshed Gufo completions match AR; reference-engine
agreement and its limits are in [evaluation](EVALUATION.md#meaning-of-exact).
Artifacts: `artifacts/multi-{mixed,repetition}-{q4,q8}-{gufo,reference}-dflash2.json`.

<!-- bench:multi-mixed-q4 -->
| Users | Gufo DFlash2 | llama.cpp DFlash2 | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 33.99 | 26.55 | +28.0% |
| 2 | 47.29 | 36.33 | +30.2% |
| 4 | 71.76 | 65.64 | +9.3% |
| 6 | 79.78 | 68.03 | +17.3% |
| 8 | 85.27 | 67.87 | +25.6% |
<!-- /bench -->

![Multiple users, DFlash2, mixed corpus](artifacts/charts/multi-mixed-q4.svg)

<!-- bench:multi-repetition-q4 -->
| Users | Gufo DFlash2 | llama.cpp DFlash2 | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 66.10 | 37.00 | +78.6% |
| 2 | 89.32 | 47.09 | +89.7% |
| 4 | 97.26 | 92.76 | +4.9% |
| 6 | 97.16 | 95.00 | +2.3% |
| 8 | 121.68 | 110.97 | +9.7% |
<!-- /bench -->

![Multiple users, DFlash2, repetition](artifacts/charts/multi-repetition-q4.svg)

<!-- bench:multi-mixed-q8 -->
| Users | Gufo DFlash2 | llama.cpp DFlash2 | Gain |
| ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 2 | 43.28 | 31.27 | +38.4% |
| 4 | 49.28 | 42.00 | +17.3% |
| 6 | 56.65 | 48.76 | +16.2% |
| 8 | 65.96 | 63.02 | +4.7% |
<!-- /bench -->

![Multiple users, DFlash2, mixed corpus](artifacts/charts/multi-mixed-q8.svg)

<!-- bench:multi-repetition-q8 -->
| Users | Gufo DFlash2 | llama.cpp DFlash2 | Gain |
| ---: | ---: | ---: | ---: |
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO |
| 8 | 116.90 | 86.53 | +35.1% |
<!-- /bench -->

![Multiple users, DFlash2, repetition](artifacts/charts/multi-repetition-q8.svg)

Earlier tg64 short-generation controls (2026-09-15/16, `prose_tides` and a
24-request mixed corpus) and the 2026-09-16 one-process-per-sweep tables
remain in Git history; they used different workloads or server lifecycles
and are not comparable with the tables above. Physical widths, output hashes
and acceptance counts are checked at every concurrency; C1 controls must
retain their performance. Controller and verification details:
[evaluation](EVALUATION.md).

## Memory

Retained **2026-09-21** measurements; Q8 uses the historical **Q8_K_L** file.
Current Q8_K_XL memory qualification is **TODO**.

Peak device-global HIP memory in use (`hipMemGetInfo` total − free, the
counter Gufo's loader logs as `gpu_device_used_mib`), sampled every 250 ms by
the driver while the request ran; idle baseline 2.38 GiB before either server
started. C1, context capacity 262144 on both servers, both autoregressive, no
draft and no projector loaded. llama.cpp preallocates its whole KV cache at
`-c`, so its footprint changes little with the prefix; Gufo's grows with the
retained prompt state. Gufo's loader reported 38208 (Q4) and 48209 (Q8) MiB
at readiness, which the sampled peaks reproduce.
Artifacts: `artifacts/memory-{q4,q8}-{gufo,reference}.json`.

<!-- bench:memory-q4 -->
| Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 37.99 | 36.72 | -3.3% |
| 16K prefix, pp4096 + tg128 | 39.89 | 37.42 | -6.2% |
<!-- /bench -->

![GPU-visible allocation](artifacts/charts/memory-q4.svg)

<!-- bench:memory-q8 -->
| Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 47.76 | 46.32 | -3.0% |
| 16K prefix, pp4096 + tg128 | 49.65 | 47.01 | -5.3% |
<!-- /bench -->

![GPU-visible allocation](artifacts/charts/memory-q8.svg)

Gufo uses 3–6% more device memory than llama.cpp at the same context
capacity, with the gap widening as the prefix grows. An earlier version of
this table sampled `rocm-smi` VRAM + GTT, which does not see Gufo's weight
mapping on unified memory; those numbers are superseded.

## Image encoder

Warm `mmproj-BF16.gguf` encoding, measured **2026-09-20** by hand; excludes
image preprocessing, first weight upload and language-model prefill. Q4 and
Q8 use the same projector. Embeddings are byte-identical to the prior
encoder; see [experiments](EXPERIMENTS.md). The driver does not automate
this table yet, so the 1024×1024 Gufo cell is the retained hand
measurement, the 256×256 cell and the llama.cpp `--mmproj` encode with the
same file remain **TODO**.

<!-- bench:image-encoder -->
| RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 64 | TODO | TODO | TODO |
| 1024×1024 | 1024 | 1252 | TODO | TODO |
<!-- /bench -->

![Image encoder](artifacts/charts/image-encoder.svg)

## Reproduce and maintain quality

```sh
nix build
nix develop -c python3 tools/qwen27b/check.py fast
Q4=/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
Q8=/path/to/Qwen3.8-27B-UD-Q8_K_XL.gguf
DRAFT=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
MMPROJ=/path/to/mmproj-BF16.gguf
FILES="--gguf q4=$Q4 --gguf q8=$Q8 --draft $DRAFT --mmproj $MMPROJ"
Q4_TABLES=single-ar-q4,single-dflash2-q4,memory-q4,multi-repetition-q4,multi-mixed-q4
Q8_TABLES=single-ar-q8,single-dflash2-q8,memory-q8,multi-repetition-q8,multi-mixed-q8
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b $FILES \
  run --target gufo --table $Q4_TABLES
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b $FILES \
  run --target reference --table $Q4_TABLES
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b $FILES \
  run --target gufo --table $Q8_TABLES
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b $FILES \
  run --target reference --table $Q8_TABLES
# Loading table, only on a host with a privileged page-cache drop:
#   ... run --target gufo --table loading --drop-caches "doas sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
nix develop -c python3 tools/bench/model-bench.py --model qwen3.8-27b render
```

The `Exact` column compares llama.cpp AR against Gufo AR C1; it is cross-engine
text agreement, not a correctness percentage. In the retained nine-case C1
corpus, each engine matches its own AR output with DFlash2 (9/9). llama.cpp
also changes some outputs across batch sizes. A separate pinned prose control
shows a llama.cpp AR/DFlash2 difference; Gufo retains agreement. Neither
cross-engine disagreement nor greedy agreement establishes official-model
parity. See [evaluation](EVALUATION.md) for counts and scope. The 2026-09-21 refresh took
34 min (Gufo Q4), 39 min (llama.cpp Q4), 41 min (Gufo Q8) and 47 min
(llama.cpp Q8) of driver wall time. `gufo bench` is greedy and C1 and is the
kernel-iteration tool; published comparison tables are measured over HTTP on
both sides. Add `--todo` to refresh only rows with `TODO` cells. DFlash2
prefill includes feature capture and draft context injection.

Start with `nix develop -c python3 tools/qwen27b/check.py fast`, then run the
[affected quality checks](EVALUATION.md) before publishing speed. Greedy
speculation must match AR IDs; sampled verification must preserve the target
distribution and reproduce seeded runs within the same configuration.
Independent original-target/MTP qualification: **TODO**.

Optimization targets: **depth-0 and deep-context (32K) generation, DFlash2
generation at C4–C8 on mixed prompts, and Q4/Q8 generation at C2–C8 with and
without DFlash2**. C1 must retain its performance. Track aggregate
throughput, per-user latency, physical batch width, output correctness and
seeded sampling at every concurrency level. Screen changes with short runs;
check retained changes across C1/2/4/6/8 before publishing speed.

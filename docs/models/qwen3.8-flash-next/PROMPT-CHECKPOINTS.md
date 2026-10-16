# Prompt checkpoints

Qualification of the Flash-Next prompt-checkpoint path, measured 2026-10-06 on
Strix Halo `gfx1151` against `main` at `e03bb911` (v0.8.0) and the
pre-regression `f797b5b` used for the September tables. Same pinned Nix
toolchain, Unsloth UD-Q4_K_XL target and shared Q8_0 MTP sidecar. The branch
was then rebased onto `4ec92f2d`, whose new commits change tool-call syntax,
answer whitespace and functional reporting only; on that build, hosted checks
and MTP long-context, cache-edits and cache-concurrency correctness pass again.

**Prefill is 1.3–30.4% faster than current `main` at every measured AR and
MTP depth and within −2.5% to +5.9% of `f797b5b`.** AR decode is unchanged;
MTP decode differs only with adaptive draft acceptance. At d0, a run order
balanced across binaries measures the candidate level with `f797b5b`. All 14
selected functional jobs pass correctness. Remaining timing flags are explained
below; none is averaged away or hidden.

## Implementation

- Flash-Next captures a text prompt's stable boundary inside the final prefill
  pass: GDN recurrent state in both kernel routes, convolution/PLE history,
  kept hidden rows, boundary logits and token/position metadata. The final
  prefill pass takes up to 128 tokens beyond the normal 2,048-token chunk,
  which absorbs a short assistant suffix and any short prompt tail. Attention queries keep the boundary's grouping. Other
  models, image prompts, disk captures and intermediate/shared checkpoints keep
  the split path.
- Snapshots copy mutable recurrent, convolution and PLE state (about 111 MB in
  AR, 119 MB with MTP) into private device storage. Committed K/V and pooled
  rows stay borrowed from the live session while it appends; only rows that a
  rewind, reset or destruction would overwrite are copied, once per set of
  consumers. Byte readers and the disk tier materialize the unchanged
  version-16 payload; RAM admission charges the full logical payload.
- Same-session restore keeps the live prefix without a K/V upload.
  Cross-session restore copies device buffers directly and reuses only rows
  whose lineage proves them identical.
- Cache acquisition prefers an available snapshot's source slot; a busy source
  falls back to LRU. Restore keeps recorded graphs and clears unrecorded warmup
  shapes, so a restored decode never compiles a graph on its first step.
- Snapshot storage comes from one private HIP memory pool per executor.
  Allocation and release never synchronize peer inference streams.
- The scheduler now decodes a request as soon as its post-first-token capture
  finishes, before another bounded prefill chunk of a peer. Before this change
  the request waited one further chunk.

## Prefill and decode

HTTP `model-bench`, pp2048/tg128, thinking off, three repetitions per depth.
Values are medians of the per-request server rates. Rates shift by a few
percent with run order and time, so each table alternates the binary order and
reports it.

Single user, AR, two passes (candidate, main, original; then reversed), six
samples per binary and depth:

| Depth | `f797b5b` pp | `main` pp | Candidate pp | vs `main` | vs `f797b5b` | `f797b5b` tg | `main` tg | Candidate tg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1373 | 1315 | 1342 | +2.0% | -2.3% | 25.95 | 26.00 | 26.05 |
| 16,384 | 1196 | 1230 | 1246 | +1.3% | +4.2% | 25.80 | 25.90 | 25.90 |
| 32,768 | 1231 | 1112 | 1218 | +9.5% | -1.0% | 25.70 | 25.80 | 25.80 |
| 65,536 | 1094 | 1078 | 1132 | +5.0% | +3.5% | 25.40 | 25.50 | 25.50 |
| 131,072 | 1054 | 890 | 1117 | +25.5% | +5.9% | 24.80 | 24.95 | 24.90 |

Single user, MTP, two passes (original, main, candidate; then reversed), six
samples per binary and depth. Mixed text:

| Depth | `f797b5b` pp | `main` pp | Candidate pp | vs `main` | vs `f797b5b` | `f797b5b` tg | `main` tg | Candidate tg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1368 | 1277 | 1348 | +5.6% | -1.5% | 32.25 | 31.35 | 31.30 |
| 16,384 | 1186 | 1174 | 1212 | +3.2% | +2.3% | 35.80 | 34.50 | 34.60 |
| 32,768 | 1202 | 1083 | 1185 | +9.4% | -1.4% | 33.70 | 31.10 | 31.10 |
| 65,536 | 1097 | 1035 | 1101 | +6.3% | +0.4% | 31.40 | 33.15 | 32.40 |
| 131,072 | 1075 | 863 | 1071 | +24.1% | -0.4% | 32.90 | 33.90 | 33.90 |

Repetitive text:

| Depth | `f797b5b` pp | `main` pp | Candidate pp | vs `main` | vs `f797b5b` | `f797b5b` tg | `main` tg | Candidate tg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1366 | 1287 | 1333 | +3.6% | -2.5% | 58.30 | 57.15 | 56.90 |
| 16,384 | 1218 | 1183 | 1222 | +3.3% | +0.3% | 47.85 | 47.80 | 47.90 |
| 32,768 | 1208 | 1092 | 1179 | +8.0% | -2.4% | 44.80 | 45.15 | 45.05 |
| 65,536 | 1102 | 1049 | 1120 | +6.8% | +1.7% | 45.85 | 46.20 | 46.25 |
| 131,072 | 1126 | 847 | 1104 | +30.4% | -2.0% | 45.20 | 45.75 | 45.30 |

MTP decode stays within 2.3% of `main` at every depth, mostly within 1%.
Where `f797b5b` differs, for example 35.80 against 34.50 mixed at 16K, `main`
shares the candidate's value, so this change does not cause the difference.

Multiple users, pp2048 then tg128, context 4,096 per user, two repetitions.
AR decode, tok/s:

| Users | `f797b5b` | `main` | Candidate |
| ---: | ---: | ---: | ---: |
| 1 | 25.97 | 26.07 | 26.05 |
| 2 | 45.98 | 45.92 | 46.05 |
| 4 | 75.99 | 76.00 | 76.09 |
| 6 | 93.67 | 93.69 | 93.84 |
| 8 | 104.63 | 104.67 | 104.70 |

MTP decode, tok/s, one repetition after warmup:

| Users | `f797b5b` mixed | `main` mixed | Candidate mixed | `f797b5b` repetitive | `main` repetitive | Candidate repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.23 | 31.41 | 31.32 | 58.80 | 59.51 | 59.63 |
| 2 | 51.58 | 47.47 | 51.24 | 90.90 | 89.64 | 91.81 |
| 4 | 75.27 | 75.58 | 74.99 | 114.26 | 125.05 | 125.10 |
| 6 | 92.42 | 93.24 | 93.04 | 127.11 | 132.14 | 132.04 |
| 8 | 102.15 | 103.72 | 103.26 | 145.98 | 151.23 | 145.75 |

Every multi-user output matches its C1 and isolated AR completion, and every
MTP run executed drafts. Rate differences follow adaptive draft acceptance:
for example 48 of 102 drafts accepted at C2 mixed on `main` against 82 of 146
on the candidate, and 720 of 760 against 688 of 744 at C8 repetitive.

A separate d0 run used three prompts (2040/2032/2053 tokens) per binary in
candidate, main, original, original, main, candidate order, with the build
before the scheduler change, which does not run during a lone prefill.
Per-prompt means: candidate 1469.9/1462.7/1373.4, original
1458.6/1467.4/1356.9, main 1349.3/1381.0/1306.7 tok/s.

The unchanged `f797b5b` binary measures 10–16% below its published
September d0 value on this day, so every prefill value here is lower than the
published tables. The [benchmark tables](BENCHMARKS.md) are therefore not
refreshed by this change: a refresh needs a same-day reference run as well.

## Functional qualification

`tests/functional/run.py`, context 32,768, four sessions unless noted,
17 GiB RAM cache. Main baselines were recorded with the same harness and
order. Every MTP job executed real drafts; AR jobs executed none.

| Mode | Suite | Correctness | Timing | Flags / measurements |
| --- | --- | --- | --- | ---: |
| MTP | long-context | pass | pass | 0 / 124 |
| MTP | cache | pass | inconclusive | 5 / 479 |
| MTP | cache-edits | pass | pass | 0 / 178 |
| MTP | cache-growth | pass | inconclusive | 3 / 785 |
| MTP | cache-depth | pass | inconclusive | 5 / 240 |
| MTP, one session | cache-depth | pass | pass | 0 / 240 |
| MTP | cache-rotation | pass | pass | 0 / 406 |
| MTP, one session | cache-rotation | pass | inconclusive | 10 / 406 |
| MTP | cache-concurrency | pass | inconclusive | 68 / 767 |
| MTP | cache-shared-prefix | pass | inconclusive | 6 / 271 |
| AR | cache | pass | inconclusive | 9 / 479 |
| AR | cache-edits | pass | pass | 0 / 178 |
| AR | cache-growth | pass | inconclusive | 1 / 785 |
| AR | cache-depth | pass | inconclusive | 4 / 240 |

Summed over each job's requests, candidate wall time is 0.8–8.4% lower than
main in every job. Capture time is 62–87% lower, except 16–20% in the `cache`
suite, whose disk captures keep the copying path. For example,
cache-concurrency falls from 261.1 s to 246.6 s and its captures from 26.7 s
to 7.1 s.

### Timing flags

- **Growth stalls.** Alternating main/candidate cache-growth runs show
  intermittent 240–400 ms decode gaps and 25–90% single-request prefill
  slowdowns in both binaries: the first main run had 17 measurements more than
  10% slower than the second. The stall is not specific to this branch. On the
  same history, candidate prefill totals 80.1–80.2 s against 82.4–83.0 s on
  main.
- **Capture resumption.** Before the scheduler change, a request whose
  post-first-token capture finished during a peer's bounded chunk waited for
  that chunk and one more, about 1.2 s. The change removed the second wait:
  the largest gap fell from 1,237 ms to 710 ms. A gap of one bounded chunk
  remains. Main pays its boundary capture before the first token instead, so
  the comparison moves time from TTFT to the first inter-token gap; for
  example one fan-out request's TTFT falls from 4,976 to 1,737 ms and its wall
  time from 5,844 to 2,968 ms.
- **Cross-slot restores.** Restoring into a slot whose rows other checkpoints
  still borrow first protects those rows. Each fresh device allocation commits
  pages at about 25 GB/s (4.5 ms per 111 MB); ROCm's pool does not reuse
  physical pages even with a release threshold. These restores take 9–13 ms
  instead of 4–6 ms, and 28–30 ms for the three shared-prefix `long_tasks`
  restores. A preallocated block reserve would remove this cost and is not
  part of this change.
- **Small shifts.** A few requests move by 3–23 ms: cold-control admission is
  about 4 ms slower in MTP cache-growth, and 24-token prefills in the cache
  suite's tool histories are 12–23 ms slower. Their cause is not established.
  Depth branches can change decode cohort width between runs, so their decode
  times are not directly comparable.

## Other checks

- Native snapshot test on the final source: exact AR/MTP in-pass checkpoints,
  borrowed-row rewind, branch, reader, reset and destruction, and the
  retention guard (786,432 and 849,920 additional device bytes for a short
  prefix in AR and MTP, both exact). The full session test passes, including
  C2/C4/C6/C8 batches, prefill chunk boundaries and sampled MTP cache replay.
- Hosted CPU/repository checks and the formatter pass. The new scheduler test
  fails without the change and passed 20 consecutive runs with it.
- The scheduler change is shared. Qwen3.8-27B Q4/Q8 cache-concurrency in AR
  and DFlash2 (drafts executed) passes correctness in all four profiles. Summed
  per-request maximum inter-token gaps fall in each (16.3→15.0, 13.8→11.1,
  21.1→19.7 and 18.4→15.0 s); summed wall time changes by −1.4% to +0.5%.
  Per-request timing remains inconclusive. DeepSeek weights were not available
  locally, so DeepSeek was not run.
- Pi 0.87.0 literal-protocol replay against the final build, MTP, four
  sessions: pass. The write call stored the literal `<|im_end|>` and
  `<|im_start|>` strings, the read call returned the file unchanged and the
  bash assertions printed `ASSERTIONS PASSED`; no tool errors. Turns two to
  four reused 1,890, 2,297 and 2,620 cached tokens, with drafts executed.

## Allocator decision

The executor-wide snapshot pool is retained on design grounds; the final runs
did not repeat a per-session-pool control. Per-session pools created one HIP
stream and pool per session and tied the executor's in-pass scratch to the
first capturing session. Measured allocation cost is per byte (about 25 GB/s)
with or without a release threshold, so sharing the pool does not change it.

## Observations outside this change

During a functional run, `ksoftirqd` threads used roughly seven cores, almost
all in tasklet softirqs. The n-gram table is read with `O_DIRECT` from a
dm-crypt volume on this machine, a likely source that was not confirmed. It
applies to every binary and was not investigated further.

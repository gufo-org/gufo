# Experiments

Measurements that decide between [option A and option C](options.md). All of
them run on unmodified main binaries. The drivers and analysis scripts are in
[scripts/](scripts/README.md), together with how to replay the analyses from
the committed token arrays and logs in [results/](results/), and how to rerun
the measurements.

## Scope (agreed 2026-10-07)

- **Models:** Flash-Next and Qwen 27B. DeepSeek V4 is out of scope: its
  compressed KV leaves little to share.
- **Workloads:** all four in [E2](#e2-workload-traces).
- **Restarts:** surviving a restart is essential, so the disk tier is measured
  as a first-class feature.
- **Disk space:** the test host has about 76 GB free (92% used). Disk cache
  budgets stay small, each run's cache directory is deleted afterwards, and
  `df -h /` is checked before every run. A `cache` suite run alone leaves
  about 8 GiB.

## E1. Snapshot size model

**Question.** How large are a checkpoint's fixed part and its per-token part,
per model and configuration? This bounds what sharing KV can save.

**Method.** Capture checkpoints at several depths (for example 2k, 8k, 32k,
64k and 128k tokens) and read the `live_checkpoint bytes=` and disk
`payload_bytes=` log fields. Fit fixed bytes + bytes per token. Vary the
settings that may change the fixed part: speculative mode (MTP or DFlash2
state), `--context`, and thinking.

**Models.** Flash-Next UD-Q4_K_XL and 27B UD-Q4_K_XL.

## E2. Workload traces

**Question.** What does real use look like for the cache?

**Method.** Run the server with `--trace PATH` (it records the rendered
prompts) and keep the logs (they record cache events). Workloads:

| Id | Workload | Why |
| --- | --- | --- |
| W1 | One coding-agent session past 100k tokens (`tests/functional/pi_agent.py` replay) | Long linear run, the #275 shape |
| W2 | An agent whose subagents share a long prefix | Sharing across conversations |
| W3 | Several users or conversations alternating on few session slots | Multi-user chat; the #275 comment scenario |
| W4 | W1 with a server restart in the middle | Disk tier value |

Start with Flash-Next, then 27B, at the default RAM budget and a disk budget
that fits the free space.

## E3. Missed reuse

**Question.** How much prefill does today's cache fail to avoid, and why? This
is the upper bound for any redesign.

**Method.** For each request in the traces, compare the reused tokens with the
longest prefix the prompt shares with anything earlier in the trace
(re-tokenized offline). Convert the gap to prefill seconds using the measured
prefill rate at that depth. Attribute each gap to one of:
- RAM eviction;
- disk eviction (`lru` or `superseded`);
- checkpoint spacing (grid or `min_step`);
- a checkpoint skipped for capacity;
- a prompt changed upstream, which is a template problem no cache design
  fixes.

## E4. Overhead of the current design

**Question.** What does the cache cost on the request path today?

**Method.** From the logs:
- capture and snapshot time per request, including captures that overlap a
  disk write;
- disk bytes written per hour and `write_ms`;
- restore time.

## E5. Simulation

**Question.** How much would option C save on the same traces?

**Method.** Replay the token traces under the same budgets through a small
simulator with two policies:
- (a) today's rules: full-size accounting and the current eviction ranks;
- (b) chunked: shared 2,048-token KV chunks plus fixed state per checkpoint.

Report prefill seconds saved, bytes written and checkpoints retained. Trust
(b) only after (a) reproduces the reuse measured in E3.

## Decision rule (proposal)

Redesign if the chunked policy saves at least about 20% of prefill time on the
agent traces, or materially cuts capture and write overhead. Otherwise keep
the current design and merge #409. The threshold is still to be agreed.

## Results

### E1. Snapshot size model (2026-10-07)

Disk `payload_bytes` against checkpoint tokens, main-equivalent build,
`--sessions 1`. Every configuration fits fixed + per-token exactly (largest
residual under 0.01%) **(measured)**:

| Configuration | Fixed | Per token | At 100k tokens |
| --- | ---: | ---: | ---: |
| Flash-Next, AR, `--context 260000` | 113.5 MiB | 25.35 KB | 2.65 GB |
| Flash-Next, AR, `--context 32768` | 113.5 MiB | 25.35 KB | 2.65 GB |
| Flash-Next, MTP, `--context 260000` | 113.8 MiB | 27.46 KB | 2.87 GB |
| 27B, AR, `--context 256000` | 152.4 MiB | 65.54 KB (64 KiB) | 6.71 GB |
| 27B, AR, `--context 32768` | 152.4 MiB | 65.54 KB | 6.71 GB |
| 27B, DFlash2, `--context 256000` | 232.4 MiB | 65.54 KB | 6.80 GB |

- **Context size doesn't matter.** Checkpoint size depends only on the token
  position, not on `--context`.
- **Speculative drafting adds state.** MTP adds 2.1 KB per token; DFlash2 adds
  80 MiB of fixed state.
- **27B is hybrid too.** It has a 152 MiB fixed part, but its per-token cost
  is 2.6× Flash-Next's, so sharing KV would save proportionally more on 27B.
- **Automatic RAM budgets are small.** With one session loaded they were
  12.9–17.5 GB for Flash-Next and 33–34 GB for 27B. At 100k tokens that is
  about 4–6 Flash-Next checkpoints or 5 for 27B.
- **Deep checkpoints don't reach disk.** The automatic staging limit (3.4 GiB
  with Flash-Next loaded) is below one Flash-Next checkpoint at 131k tokens,
  and two 27B checkpoints at 64k exceed even 6 GiB. Today's defaults skip
  persisting those deep checkpoints (`reason=staging_capacity`).
- **The RAM cache can refuse the newest checkpoint.** At 127k tokens it
  refused the prompt checkpoint (`reason=byte_capacity`, 3.34 GB against a
  14.6 GB budget), because the request's own earlier checkpoints already
  filled the budget.

Scripts: `e1_snapshot_size.py`, `run_e1.sh`, `fit_e1.py`.

### E2. Workload traces (2026-10-07)

Production-like servers (the llama-swap command lines plus the disk tier):
- Flash-Next: MTP, `--sessions 2`, `--context 260000`.
- 27B: Q8_K_XL, DFlash2, `--sessions 2`, `--context 256000`.

Common settings: automatic RAM and staging budgets, a 16 GiB disk budget, and
`--trace`. W1 is a real Pi session grown through tool results by
`tests/functional/agent_long.py` (thinking `low`). W2–W4 are synthetic
(`workloads.py`, thinking off, real model replies).

Automatic budgets the servers chose **(measured)**:

| Model | RAM cache budget | Staging budget | Fits at 100k tokens |
| --- | ---: | ---: | --- |
| Flash-Next (2 sessions) | 8.9–9.2 GB | 2.23–2.31 GB | 3 checkpoints in RAM; nothing past ~75k tokens reaches disk |
| 27B (2 sessions) | 23.0–23.3 GB | 5.74–5.84 GB | 3 checkpoints in RAM; nothing past ~75k tokens reaches disk |

### E3. Missed reuse

"Ideal" is the longest prefix each prompt shares with any earlier prompt plus
its output. Prompts were re-tokenized from the trace with the model's own
vocabulary. Re-tokenized text is a few tokens shorter than the server's count,
so positions are rescaled per request. Gaps under 256 tokens are ignored: they
are the re-rendered assistant reply, not a capacity issue.

| Workload | Requests | Ideal reuse | Actual reuse | Largest misses |
| --- | ---: | ---: | ---: | --- |
| fn W1 agent to 149k | 36 | 95.4% | 95.4% | none |
| fn W2 subagents | 36 | 89.1% | 85.7% | parent after forks: 20.6k tokens, ~17 s |
| fn W3 multi-user | 60 | 87.9% | 87.6% | 1 request, 991 tokens |
| fn W4 restart | 33 | 89.7% | 88.5% | subagent after restart, 6.6k tokens; agent restore 3.7k short |
| 27B W1 agent to 149k | 36 | 95.4% | 95.4% | none |
| 27B W2 subagents | 36 | 89.0% | 85.6% | parent after forks: 20.4k tokens, ~60 s |
| 27B W3 multi-user | 60 | 89.3% | 88.8% | 2 small requests |
| 27B W4 restart | 33 | 89.7% | 87.7% | agent after restart: 13.1k tokens, ~37 s; subagent 6.6k tokens, ~22 s |

Causes found **(measured, from logs and source)**:

1. **Disk is ignored after any RAM hit.** It is consulted only when RAM has no
   hit at all (`src/cli/serve/text_model_runner.cpp:1877`). In W2 the
   parent's turn after its forks matched only the 4,661-token system prompt in
   RAM while disk held its 24,599-token checkpoint, so 22,558 tokens were
   re-prefilled.
2. **The RAM cache refuses new checkpoints when full.** An incoming checkpoint
   may only evict entries of equal or lower rank (`MaxRemovalPriority`,
   `src/cli/serve/continuation_cache.cpp:32`, used at `:759`). Refusals
   (`event=snapshot action=skipped reason=byte_capacity`) per run: 19–42.
3. **Deep checkpoints never reach disk.** A full file larger than the
   automatic staging budget is skipped (`reason=staging_capacity`): 71 skips
   in fn W1 and 85 in 27B W1. The deepest persisted checkpoint stops at about
   75k tokens for both models.
4. **No checkpoint at a shared boundary after a restart.** Disk holds prompt
   boundaries but not grid checkpoints, so a new subagent sharing a 6.6k-token
   system prompt reused nothing.

### E4. Overhead of the current design

From the server logs (`analyze_e4.py`) **(measured)**:

| Run | Live captures | Capture time, total / max | Disk writes | Written | Staging skips | RAM refusals |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| fn W1 | 35 | 1.1 s / 158 ms | 6 | 7.1 GB | 71 | 39 |
| fn W2 | 13 | 0.5 s / 110 ms | 25 | 16.3 GB | 8 | 22 |
| fn W3 | 18 | 0.9 s / 246 ms | 24 | 13.7 GB | 9 | 41 |
| fn W4 | 23 | 0.8 s / 117 ms | 26 | 23.2 GB | 29 | 27 |
| 27B W1 | 35 | 5.9 s / 399 ms | 6 | 16.5 GB | 85 | 39 |
| 27B W2 | 8 | 2.6 s / 611 ms | 25 | 37.9 GB | 3 | 19 |
| 27B W3 | 24 | 5.8 s / 634 ms | 25 | 31.7 GB | 7 | 42 |
| 27B W4 | 25 | 7.3 s / 635 ms | 31 | 70.9 GB | 16 | 31 |

- **27B copies its whole state per capture.** At 149k tokens a 10.0 GB
  capture took 235–265 ms, every turn.
- **Flash-Next captures are cheaper** because KV is borrowed (#445).

### E5. Simulation

`simulate_e5.py` replays each tokenized trace under the budgets the server
actually used. Its variants share capture points, lookup, eviction ranks and
disk spacing:

| Variant | Accounting | Disk lookup |
| --- | --- | --- |
| full | Full copy per checkpoint (today) | Only when RAM misses (today) |
| full + disk fix | Full copy per checkpoint | Whenever disk holds a longer prefix |
| chunked | 2,048-token KV chunks counted once; staging holds only new bytes | Only when RAM misses |
| chunked + disk fix | As chunked | Whenever disk holds a longer prefix |

**Validation.** The "full" variant reproduces the server's actual reuse: within
64 tokens on 34/36, 53/60, 30/33, 36/36, 34/36, 55/60, 28/33 and 36/36
requests, and within 0.5% of total reused tokens for every run.

Reused tokens:

| Workload | Actual | full | full + disk fix | chunked | chunked + disk fix |
| --- | ---: | ---: | ---: | ---: | ---: |
| fn W1 | 3,088,096 | 3,088,096 | 3,088,096 | 3,088,096 | 3,088,096 |
| fn W2 | 553,620 | 553,658 | 573,599 | 573,606 | 573,606 |
| fn W3 | 745,248 | 743,755 | 743,755 | 743,755 | 743,755 |
| fn W4 | 868,875 | 872,701 | 872,701 | 872,701 | 872,701 |
| 27B W1 | 3,074,904 | 3,074,904 | 3,074,904 | 3,074,904 | 3,074,904 |
| 27B W2 | 552,915 | 553,206 | 572,917 | 572,924 | 572,924 |
| 27B W3 | 853,387 | 853,943 | 853,943 | 853,943 | 853,943 |
| 27B W4 | 869,495 | 869,489 | 869,489 | 882,556 | 882,556 |

Accounted bytes (simulated, full → chunked):

| Workload | RAM checkpoint bytes | Disk bytes written |
| --- | --- | --- |
| fn W1 | 275.5 → 19.4 GB | 7.0 → 5.7 GB |
| fn W2 | 48.6 → 12.8 GB | 16.3 → 5.2 GB |
| fn W3 | 80.0 → 24.5 GB | 13.5 → 6.3 GB |
| fn W4 | 85.2 → 16.9 GB | 26.7 → 6.9 GB |
| 27B W1 | 650.7 → 41.9 GB | 16.5 → 13.2 GB |
| 27B W2 | 112.5 → 28.0 GB | 37.8 → 11.7 GB |
| 27B W3 | 194.7 → 51.7 GB | 31.2 → 13.5 GB |
| 27B W4 | 194.5 → 36.7 GB | 70.9 → 15.9 GB |

For Flash-Next the RAM column is accounting only (KV is already borrowed); for
27B it is real copy volume. Disk bytes for W1 are low today only because most
deep writes were skipped.

**Simulated restarts during the W1 agent sessions.** A restart is inserted
before request N; the RAM cache empties and disk persists:

| Model | Restart at | Today restores | Chunked restores | Extra prefill today |
| --- | ---: | ---: | ---: | ---: |
| Flash-Next | 60,844 | 60,560 | 60,560 | none |
| Flash-Next | 104,938 | 75,275 | 104,810 | ~29.5k tokens, ~25 s |
| Flash-Next | 149,145 | 75,275 | 134,388 | ~59k tokens, ~50 s |
| 27B | 60,622 | 60,396 | 60,396 | none |
| 27B | 104,505 | 75,026 | 104,386 | ~29k tokens, ~2 min |
| 27B | 148,523 | 75,026 | 133,842 | ~59k tokens, ~4.3 min |

Prefill at 90–150k depth ran at 1,140–1,180 tokens/s on Flash-Next and
196–250 tokens/s on 27B.

### Micro-benchmarks (2026-10-07)

These back the [cost model](cost-model.md) **(measured)**:

- **GPU copies** ([copybench.hip](scripts/copybench.hip), 56 MiB–10 GiB):
  device to device 104–110 GB/s; pinned host either way about 85 GB/s;
  pageable 61–83 GB/s. A 10 GiB device copy takes 103 ms. The real 27B capture
  of the same 10 GB took 235 ms.
- **Disk** ([diskbench.py](scripts/diskbench.py), NVMe under dm-crypt):
  - write + fsync 0.58–0.60 GB/s at every size from 56 MiB to 4 GiB;
  - cold read 0.97–1.15 GB/s;
  - 72 files of 56 MiB versus one 4.2 GB file: write 8.0 s vs 7.1 s, cold
    read 4.18 s vs 3.69 s (+13% each).
- **Fixed-state captures already exist.** Flash-Next captures 2.2–5.4 ms at
  1.8k–149k tokens, because its KV is borrowed (#445). 27B captures 13.7 ms at
  1.8k tokens, where its snapshot (0.36 GB) is mostly fixed state.
- **Prefill model** ([fit_prefill.py](scripts/fit_prefill.py)), fitted on 150
  requests per model. Median error 9.8% (Flash-Next) and 3.9% (27B).
- **Restoring from chunks** ([chunkcopy.hip](scripts/chunkcopy.hip)): 8 GiB
  moved as many device copies on one stream, in scattered order.

  | Piece size | GB/s |
  | ---: | ---: |
  | One 8 GiB copy | 103 |
  | 64 MiB | 109 |
  | 4 MiB | 106 |
  | 1 MiB | 88 |
  | 256 KiB | 56 |
  | 64 KiB | 23 |

  A 2,048-token chunk split per attention layer and K/V is estimated at about
  1–4 MiB per piece for these models, so assembling a session from chunks
  keeps device-copy bandwidth.
- **Physical vs accounted RAM** (W1 logs):
  - Flash-Next accounted 7.1–8.9 GB of retained checkpoints, while available
    memory fell only 4–6 GB including the live session. Borrowed KV is
    physically shared, which supports unique-bytes accounting.
  - 27B accounted 16.5–22.3 GB for a 14–20 GB fall, consistent with real full
    copies.

### E7. Today, Phase 0 and the hybrid, in seconds (simulated)

[simulate_e7.py](scripts/simulate_e7.py) replays every trace through four
variants, with the budgets each server actually chose:

| Variant | Changes from today |
| --- | --- |
| Today | — |
| Phase 0 | Disk consulted whenever it holds a longer prefix; bounded, streamed writes **and** restores, so no staging limit in either direction; Flash-Next RAM counted by unique bytes. Today both directions are limited by staging (`ReadImage` loads whole files), so streaming writes alone is not enough |
| Hybrid | Shared KV chunks in RAM and on disk, small checkpoints, one index |
| Hybrid + dense | Also a RAM checkpoint at every message boundary, and the end of the system prompt persisted |

Prefill time is the fitted model applied to each request's uncached tokens.
For runs without a simulated restart, simulated "today" matches the measured
prefill time within 1–8%.

| Run | Measured prefill | Today | Phase 0 | Hybrid | Hybrid + dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 130 s | 137 s | 137 s | 137 s | 137 s |
| Flash-Next W1, restart at 60.8k | — | 137 s | 137 s | 137 s | 137 s |
| Flash-Next W1, restart at 104.9k | — | 161 s | 137 s | 137 s | 137 s |
| Flash-Next W1, restart at 149.1k | — | 185 s | 137 s | 137 s | 137 s |
| Flash-Next W2 | 85 s | 87 s | 71 s | 71 s | 71 s |
| Flash-Next W3 | 108 s | 108 s | 108 s | 108 s | 107 s |
| Flash-Next W4 | 108 s | 100 s | 100 s | 100 s | 95 s |
| 27B W1 | 589 s | 596 s | 596 s | 596 s | 596 s |
| 27B W1, restart at 60.6k | — | 597 s | 597 s | 597 s | 597 s |
| 27B W1, restart at 104.5k | — | 718 s | 597 s | 597 s | 597 s |
| 27B W1, restart at 148.5k | — | 855 s | 597 s | 597 s | 597 s |
| 27B W2 | 293 s | 292 s | 235 s | 235 s | 233 s |
| 27B W3 | 352 s | 348 s | 348 s | 348 s | 345 s |
| 27B W4 | 403 s | 400 s | 400 s | 364 s | 346 s |

Bytes written to disk, and modelled 27B capture time:

| Run | 27B capture, today → hybrid | Disk: today | Phase 0 | Hybrid | Hybrid + dense |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Next W1 | — | 7.0 GB | 24.1 GB | 5.7 GB | 5.7 GB |
| Flash-Next W2 | — | 16.3 GB | 15.5 GB | 5.2 GB | 5.7 GB |
| Flash-Next W3 | — | 13.5 GB | 13.5 GB | 6.3 GB | 6.2 GB |
| Flash-Next W4 | — | 26.7 GB | 29.0 GB | 6.9 GB | 6.8 GB |
| 27B W1 | 18.2 → 0.8 s | 16.5 GB | 56.8 GB | 13.2 GB | 13.2 GB |
| 27B W2 | 3.1 → 0.5 s | 37.8 GB | 36.0 GB | 11.7 GB | 13.0 GB |
| 27B W3 | 5.5 → 1.0 s | 31.2 GB | 31.2 GB | 13.5 GB | 13.1 GB |
| 27B W4 | 5.5 → 0.6 s | 70.9 GB | 70.9 GB | 15.9 GB | 15.6 GB |

Reading E7:

- **Phase 0 recovers the in-session miss** (W2: −16 s Flash-Next, −57 s 27B)
  and the restart losses (−24 to −48 s Flash-Next, −121 to −258 s 27B).
- **Phase 0 pays in disk writes:** 24–57 GB on the agent runs against 6–13 GB
  for the hybrid. With a 16 GiB disk budget it also keeps fewer checkpoints;
  27B W4 is where that shows (400 s vs 364 s for the hybrid).
- **Dense checkpoints are worth it only with the hybrid:** W4 −5 s Flash-Next,
  −18 s 27B, for almost no extra disk.
- **The capture saving is real only on 27B.** The modelled 18 s on W1 is about
  0.5 s per turn at long context.

### E6. Today's system at concurrency 4

The production-like servers were rerun with `--sessions 4`, a 131,072-token
context and four parallel clients ([run_e6.sh](scripts/run_e6.sh)). Workloads:
W3 with 10 users and 100 requests, and W2 with four workers.

| Run | RAM budget | Staging | Ideal reuse | Actual reuse | RAM refusals | Disk written | Disk write time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B W3, concurrency 4 | 20.2 GB | 5.05 GB | 86.7% | 86.1% | 92 | 33.7 GB | 903 s |
| 27B W2, concurrency 4 | 20.1 GB | 5.03 GB | 89.2% | 89.0% | 22 | 36.1 GB | 97 s |
| Flash-Next W3, concurrency 4 | 8.5 GB | 2.12 GB | 87.5% | 87.1% | 77 | 16.2 GB | 31 s |
| Flash-Next W2, concurrency 4 | 9.0 GB | 2.25 GB | 89.0% | 88.9% | 27 | 14.5 GB | 29 s |

E7 on the concurrency-4 traces (four sessions):

| Run | Measured prefill | Today (sim) | Phase 0 | Hybrid | Hybrid + dense | 27B capture, today → hybrid | Disk, today → hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 27B W3, concurrency 4 | 478 s | 501 s | 501 s | 501 s | 498 s | 6.6 → 1.7 s | 33.6 → 20.1 GB |
| 27B W2, concurrency 4 | 230 s | 229 s | 229 s | 229 s | 227 s | 3.0 → 0.5 s | 36.1 → 11.4 GB |
| Flash-Next W3, concurrency 4 | 159 s | 158 s | 158 s | 158 s | 157 s | — | 15.9 → 8.7 GB |
| Flash-Next W2, concurrency 4 | 72 s | 71 s | 71 s | 71 s | 70 s | — | 15.4 → 5.3 GB |

- **The W2 miss disappears with four sessions.** The parent's live state stays
  resident while its forks run, so the disk-lookup defect never triggers.
  More sessions hide the defect rather than fix it.
- **Reuse stays near ideal.** As at concurrency 2, the hybrid's gain is in
  bytes written and capture time, not reuse.
- **One disk write stalled for 782 s.** In 27B W3, a 1.67 GB checkpoint
  (21,694 tokens) took 782 s to write, finishing at 22:52:29; another took
  38 s. Together they account for most of the 903 s total. Excluding them,
  27B writes ran at 0.27–0.43 GB/s (median 0.36), in line with the
  0.44 GB/s model. The cause is not yet known. Candidates: serializing from
  the device while four sessions keep the GPU busy, or lock contention in the
  writer. It is worth its own investigation.
- **The hybrid writes 40–70% less** at concurrency 4, which also shortens
  the time the writer competes with inference.

### E8. Simulator replaying production behaviour (simulated, revision 2)

Two reviews pointed out where E5/E7 departed from production.
[simulate_e8.py](scripts/simulate_e8.py) revision 2 models:

- **Concurrency:** requests replay as events at their real times. Each starts
  when its generation started (trace timestamp), captures when its prefill
  ends (received + time to first token) and finishes at received + duration.
  Live frontiers belong to sessions and are lost when another conversation
  takes the session.
- **The capture sequence of `text_model_runner.cpp`:**
  - on a cache hit, a frozen copy of the reused frontier (continuation);
  - the stable boundary (continuation);
  - the complete prompt as a retry copy;
  - grid points as history, learned divergence points as branch points.

  Only these prompt-path snapshots are committed. Prompt plus output stays as
  the live frontier, not as a retained snapshot (revision 1 got this wrong).
- **Admission:** a new checkpoint may evict only entries at or below its rank
  ceiling (`MaxRemovalPriority`: retry 0, history 1, continuation and branch
  point 3), lowest rank first. The entry being restored from is protected, and
  RemovalPriority's retry, history, covered-continuation and branch-point
  ranks apply.
- **Disk:** one serial writer at 0.44 GB/s. Spacing and staging are checked
  when the writer reaches a job, and an entry becomes restorable only when its
  write completes. A restart is graceful (the queue drains, as on SIGTERM) or,
  with `--abrupt`, drops queued and in-flight writes. Revision 1 published
  entries when their write started.
- **Lineage-aware chunks:** a request inherits its restore source's chunks
  only for the rows it restored; rows it computed form new chunks.
  Independent computations of equal tokens are never deduplicated.

[summarize_e8.py](scripts/summarize_e8.py) produces the tables:

| Run | Measured prefill | Today | Phase 0 | Hybrid | Hybrid + dense | Reuse, simulated today vs actual | Requests within 64 tokens | Refusals: server / simulated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 130 s | 137 s | 137 s | 137 s | 137 s | +0.0% | 36/36 | 39 / 64 |
| Flash-Next W2 | 85 s | 88 s | 72 s | 72 s | 71 s | -0.1% | 35/36 | 22 / 26 |
| Flash-Next W3 | 108 s | 108 s | 107 s | 107 s | 107 s | -0.1% | 58/60 | 41 / 47 |
| Flash-Next W4 | 108 s | 100 s | 100 s | 100 s | 95 s | +0.4% | 28/33 | 27 / 26 |
| Flash-Next W2-C4 | 72 s | 72 s | 72 s | 72 s | 70 s | -0.2% | 34/36 | 27 / 24 |
| Flash-Next W3-C4 | 159 s | 157 s | 157 s | 157 s | 157 s | -0.1% | 99/100 | 77 / 81 |
| 27B W1 | 589 s | 596 s | 596 s | 596 s | 596 s | +0.0% | 36/36 | 39 / 64 |
| 27B W2 | 293 s | 293 s | 236 s | 236 s | 233 s | -0.0% | 32/36 | 19 / 21 |
| 27B W3 | 352 s | 347 s | 347 s | 347 s | 347 s | +0.1% | 56/60 | 42 / 48 |
| 27B W4 | 403 s | 430 s | 430 s | 394 s | 376 s | -1.3% | 27/33 | 31 / 26 |
| 27B W2-C4 | 230 s | 231 s | 231 s | 231 s | 227 s | -0.2% | 28/36 | 22 / 25 |
| 27B W3-C4 | 478 s | 497 s | 497 s | 497 s | 497 s | -0.1% | 97/100 | 92 / 97 |

| Model | Restart before request | Shutdown | Today | Phase 0 | Hybrid | Restored after restart: today / Phase 0 / hybrid |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Flash-Next | 12 | graceful | 137 s | 137 s | 137 s | 60,560 / 60,560 / 60,560 |
| Flash-Next | 12 | abrupt | 137 s | 137 s | 137 s | 60,560 / 60,560 / 60,560 |
| Flash-Next | 20 | graceful | 161 s | 137 s | 137 s | 75,275 / 104,810 / 104,810 |
| Flash-Next | 20 | abrupt | 161 s | 149 s | 137 s | 75,275 / 90,034 / 104,810 |
| Flash-Next | 28 | graceful | 185 s | 137 s | 137 s | 75,275 / 134,388 / 134,388 |
| Flash-Next | 28 | abrupt | 185 s | 149 s | 137 s | 75,275 / 119,579 / 134,388 |
| 27B | 12 | graceful | 597 s | 597 s | 597 s | 60,396 / 60,396 / 60,396 |
| 27B | 12 | abrupt | 649 s | 649 s | 597 s | 45,676 / 45,676 / 60,396 |
| 27B | 20 | graceful | 718 s | 597 s | 597 s | 75,026 / 104,386 / 104,386 |
| 27B | 20 | abrupt | 718 s | 659 s | 597 s | 75,026 / 89,697 / 104,386 |
| 27B | 28 | graceful | 855 s | 597 s | 597 s | 75,026 / 133,842 / 133,842 |
| 27B | 28 | abrupt | 855 s | 667 s | 597 s | 75,026 / 119,098 / 133,842 |

| Run | Disk written: server | Today (sim) | Phase 0 | Hybrid | Hybrid + dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next W1 | 7.1 GB | 7.0 GB | 24.1 GB | 5.7 GB | 5.7 GB |
| Flash-Next W2 | 16.3 GB | 16.6 GB | 15.8 GB | 5.3 GB | 5.7 GB |
| Flash-Next W3 | 13.7 GB | 13.5 GB | 13.5 GB | 6.3 GB | 6.3 GB |
| Flash-Next W4 | 23.2 GB | 26.7 GB | 29.0 GB | 7.1 GB | 6.8 GB |
| Flash-Next W2-C4 | 14.5 GB | 15.6 GB | 15.6 GB | 5.4 GB | 5.8 GB |
| Flash-Next W3-C4 | 16.2 GB | 15.9 GB | 15.9 GB | 8.7 GB | 8.7 GB |
| 27B W1 | 16.5 GB | 16.5 GB | 56.8 GB | 13.2 GB | 13.2 GB |
| 27B W2 | 37.9 GB | 38.4 GB | 36.5 GB | 12.0 GB | 13.0 GB |
| 27B W3 | 31.7 GB | 31.5 GB | 31.5 GB | 13.5 GB | 13.4 GB |
| 27B W4 | 70.9 GB | 71.4 GB | 71.4 GB | 17.2 GB | 16.0 GB |
| 27B W2-C4 | 36.1 GB | 36.6 GB | 36.6 GB | 11.6 GB | 12.6 GB |
| 27B W3-C4 | 33.7 GB | 33.6 GB | 33.6 GB | 20.1 GB | 20.0 GB |

Reading E8:

- **Fidelity.**
  - Actual reuse is reproduced within 0.4% on 11 of 12 runs (27B W4 −1.3%),
    with 28–36 of 36 and 97–99 of 100 requests within 64 tokens.
  - Prefill time within 0–7%, disk writes within about 15%.
  - RAM refusals match on 10 of 12 runs; the W1 runs over-count (64 vs 39).
  - The set of retained checkpoints cannot be verified, because the server
    logs removals only when the entry limit forces them.
- **Phase 0 and the hybrid recover the same reuse** in most runs: the W2 miss
  (Flash-Next 88 → 72 s, 27B 293 → 236 s) and graceful restarts at depth.
- **The hybrid does better than Phase 0 in three cases:**
  - *Abrupt restarts:* Phase 0 loses queued full-file writes. Flash-Next at
    149k: 149 s vs 137 s. 27B: 649–667 s vs 597 s.
  - *27B W4:* 394 s vs 430 s, because more checkpoints fit the 16 GiB disk
    budget.
  - *Dense checkpoints:* 27B W4 376 s, Flash-Next W4 95 s instead of 100 s.
- **Disk writes:**
  - the hybrid, with lineage chunks, writes 1.7–4.2× less than today on the
    multi-conversation runs, and 1.2× less on W1, where today skips its deep
    writes;
  - Phase 0 writes 3.4× more than today on the agent runs.

### Summary so far

All numbers in this section are simulated (E8 revision 2) unless marked measured.

1. **Reuse within a running server** is not the problem at concurrency 2.
   The one large in-session miss comes from a defect (a RAM hit hides a longer
   disk hit) that Phase 0 fixes.
2. **Restarts are where today's design fails at depth.** The disk tier stops
   at ~75k tokens (measured); a deeper restart costs ~25–50 s on Flash-Next
   and ~2–4 min on 27B. Phase 0 fixes it only if both writes and restores are
   streamed. Phase 0 writes 2–4× more and fits fewer checkpoints on disk.
3. **Cost per turn:** the hybrid cuts 27B captures from 100–240 ms to ~14 ms
   per checkpoint, disk writes 2.5–4.5×, and multiplies the checkpoints a RAM
   budget holds by 10–20× at long context.
4. **Concurrency 4 doesn't change the picture (E6).** Reuse stays within
   0.1–0.6 points of ideal, and more sessions hide the W2 defect rather than
   fix it. Most of the 27B disk-write time came from one unexplained 782 s
   write, so these runs don't show whether the writer is a bottleneck. The
   hybrid writes 40–70% less.
5. **Phase 0 first.** It recovers most of the measured reuse losses: the W2
   miss and graceful restarts at depth. Losses it does not recover:
   - abrupt restarts, where queued full-file writes are lost (27B at 149k:
     667 s vs 597 s for the hybrid);
   - 27B W4: the hybrid fits more checkpoints in the 16 GiB disk budget,
     394 s vs 430 s of prefill;
   - the subagent created after a restart: only dense hybrid checkpoints
     persist the system-prompt boundary.

   The hybrid adds those cases plus efficiency: disk writes, 27B capture
   time and RAM capacity.

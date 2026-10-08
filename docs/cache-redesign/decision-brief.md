# Decision brief

A one-page summary of the evidence for choosing the continuation cache's
direction. Details: [experiments](experiments.md), [cost model](cost-model.md),
[hybrid design](hybrid-design.md).

## The question

Keep today's design (a full snapshot per checkpoint, in RAM and on disk) and
fix its defects, or move to the hybrid design: shared KV chunks, small
checkpoints holding only fixed state, and one index over RAM and disk?

## What the evidence says

Measured on Flash-Next and 27B with production-like settings, on real agent
sessions to 149k tokens, subagents, multi-user chat and restarts:

| Finding | Evidence |
| --- | --- |
| Checkpoint size is fixed state + per-token KV, exactly | Flash-Next with MTP: 113.8 MiB + 27.46 KB/token. 27B: 152 MiB (+80 MiB with DFlash2) + 64 KiB/token |
| Automatic budgets are small at long context | Two sessions: RAM 9.2 GB (Flash-Next) or 23 GB (27B) holds 2–3 checkpoints of a 100k conversation |
| Long checkpoints never reach disk | Automatic staging (2.3 / 5.8 GB) is smaller than a full checkpoint past ~75k tokens: 71 and 85 skipped writes on the agent runs |
| A RAM hit hides a longer disk hit | Disk is consulted only on a RAM miss; one W2 turn re-prefilled 22.6k tokens (24 s Flash-Next, ~60 s 27B) |
| A full RAM cache refuses checkpoints | 19–42 refusals per run, because a new checkpoint may evict only lower-ranked entries |
| 27B captures copy everything | 14 ms at 1.8k tokens up to 235 ms at 149k (10 GB). Flash-Next already captures only fixed state: 2–5 ms at any depth |
| In-session reuse is near ideal otherwise | Actual vs ideal reuse within 0.3–1.2 points except W2 and the restarts |
| The simulators track reality | E7 reproduces actual reuse within 0.5% and prefill time within 0–8% on every run; E8 (production admission) within 0.5% except one run at −2.7% |
| Concurrency 4 changes little | Reuse within 0.1–0.6 points of ideal; more sessions hide the W2 defect; the 27B disk writer is busy half the time; the hybrid writes 40–70% less |

## Options compared (simulated with E7/E8, plus the cost model)

| | Today | Phase 0 (fixes, same format) | Hybrid (Phases 1–2) |
| --- | --- | --- | --- |
| W2 miss after forks | 22.6k tokens re-prefilled | Fixed | Fixed |
| Restart at 105k / 149k, Flash-Next | +24 s / +48 s prefill | Fixed, if writes **and** restores are streamed | Fixed |
| Restart at 105k / 149k, 27B | +2.0 / +4.3 min prefill | Fixed, same condition | Fixed |
| 27B W4: restart with mixed conversations, 16 GiB disk | 399 s prefill | 399 s | 363 s (more checkpoints fit) |
| Subagent created after a restart | System prompt re-prefilled | Same | Recovered with dense checkpoints (−5 s Flash-Next, −18 s 27B) |
| Disk written, agent runs | 7–17 GB (deep writes skipped) | 24–57 GB | 6–13 GB |
| Disk time per checkpoint at 100k | 6.5 s Flash-Next, 15 s 27B (if staged) | Same, streamed | 0.4 s / 0.9 s |
| 27B capture per checkpoint at 100k / 149k | 181 / 236 ms | Same | ~14 ms |
| Checkpoints of one 100k conversation in RAM | 3 | 3 (27B), more for Flash-Next | 43–52 |
| Dense checkpoints (message boundaries, shared system prompt) | Unaffordable | Unaffordable | Affordable; W4 −5 s Flash-Next, −18 s 27B |
| Effort | — | Small: lookup rule, streamed writes, accounting | Large: model interface split for each model, chunk pool, new disk format |
| Risk | Known defects | Low | Medium: new store, format, eviction rules |

At concurrency 2 and 4 alike, the hybrid adds efficiency, not reuse that
Phase 0 cannot reach.

## Recommendation

1. **Do Phase 0 now, independently of the redesign.** It fixes most of the
   measured reuse losses (the W2 miss and the deep restarts), with small
   changes:
   - consult disk whenever it holds a longer prefix than RAM;
   - bounded, streamed transfers in both directions. Today writes stage the
     whole snapshot and `ReadImage` loads the whole file, both limited by
     staging, so streaming writes alone would persist checkpoints that still
     could not be restored;
   - count Flash-Next RAM checkpoints by unique bytes, with exact reservations
     for borrowed rows instead of treating them as free;
   - let new checkpoints evict lower-value entries instead of being refused
     (not simulated).
2. **Decide on the hybrid by the features you need**:

   | If you need | Then |
   | --- | --- |
   | Restarts at 100k+ with low disk wear | Hybrid: Phase 0 writes 3–10 GB per deep checkpoint |
   | 27B at long context with lower time to first token | Hybrid: ~90–220 ms less per checkpoint |
   | Many conversations, subagents or edit points kept warm | Hybrid: 10–20× more checkpoints per RAM budget |
   | Mostly Flash-Next, a few conversations, rare restarts | Phase 0 is enough |

3. **If the hybrid goes ahead, build in these requirements from the start**
   (see the [hybrid design](hybrid-design.md)):
   - chunks shared only within one lineage, never by token prefix alone,
     because equal tokens can produce different KV bytes;
   - reserved spill capacity for borrowed rows;
   - a model interface listing each state component and its valid position;
   - bounded transfers both ways;
   - chunks published before manifests, pinning, orphan recovery, and one
     writer per directory;
   - compaction off until measured restores justify its extra writes;
   - checkpoint density bounded by records and budget.

4. **Phase it:**
   - Phase 1: model interface split and the RAM chunk pool. Flash-Next is
     mostly there already.
   - Phase 2: the disk tier on the same chunks, with background compaction.
   - Phase 3 (optional): paged KV.

## Evidence quality

- **Measured:** checkpoint sizes, budgets, refusals, staging skips, capture
  and restore times, disk rates, actual reuse and prefill time, the
  micro-benchmarks.
- **Simulated:** everything for Phase 0 and the hybrid. Two simulators (E7,
  and E8 with production admission and asynchronous disk) agree on every
  conclusion, but refusals are under-reproduced (server 19–92 per run, E8
  0–36), and on 27B W3 at concurrency 4 E8 overestimates today's prefill by
  18%.
- **Unexplained:** one 1.67 GB disk write took 782 s at concurrency 4.

## Features to agree on

- Restart survival at long context, and acceptable disk wear per day.
- Several server processes sharing one cache directory.
- Edit and rewind points inside a conversation, or only its latest state.
- Subagents and shared system prompts as a primary workload.
- Output equality after a restore: greedy only, or sampled too.
- Model priority: Flash-Next vs 27B.
- Typical concurrency (2, 4, 8) and context length.

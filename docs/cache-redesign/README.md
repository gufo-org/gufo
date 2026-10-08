# Continuation cache redesign

Research started 2026-10-07. Nothing in this directory changes server
behaviour; it records the question, the evidence and the decision.

**Question.** Every cached checkpoint today is a full, self-contained copy of
the model state, in RAM and on disk. Should gufo move to a design that stores
shared KV once and keeps only per-checkpoint state separately, and if so, how
far should it go?

| Document | Contents |
| --- | --- |
| [Decision brief](decision-brief.md) | One-page summary of the evidence, options and recommendation |
| [Current design](current-design.md) | What the continuation cache does today, with source references and known costs |
| [Other engines](external-engines.md) | llama.cpp, vLLM, SGLang, LMCache and ds4, with links |
| [Options](options.md) | Candidate designs, pros and cons, diffs and keyframes, and the features still to decide |
| [Hybrid design](hybrid-design.md) | Option E in detail, with worked examples |
| [Cache package](cache-package.md) | Shared library ownership, model adapters, lifecycle contracts and migration |
| [Implementation plan](implementation-PLAN.MD) | Phased tasks, pending decisions, validation gates and review sequence |
| [Cost model](cost-model.md) | Per-operation costs, today vs hybrid, from measured constants |
| [Experiments](experiments.md) | Measurements needed to decide, and their results |

## Status

1. Write down the discussion and external references. Done.
2. Run the experiments in [experiments.md](experiments.md). Done for
   Flash-Next and 27B: E1–E7, the micro-benchmarks and the
   [cost model](cost-model.md). See the [decision brief](decision-brief.md).
3. Agree on the required features (see [Options](options.md)).
4. Choose a design, or keep the current one.

## Decision log

### 2026-10-07

**#275 and #409.** #275 reports that one long conversation fills the disk
budget and evicts every other conversation. A plan was posted on the issue
([comment](https://github.com/gufo-org/gufo/issues/275#issuecomment-6041775357)):
close it when [#409](https://github.com/gufo-org/gufo/pull/409) merges, and
decline eager K=2 supersession, a per-conversation byte cap, per-lineage
logging and a checkpoint-interval flag. The fixed 2,048-token disk spacing from
#348 is a good performance tradeoff.

#409 evicts covered intermediate disk checkpoints before falling back to LRU
and logs them as `reason=superseded`. It was rebased onto main `b39c530e`, and
review fixes were pushed to the contributor's branch (head `223e8515`):
learned shared-prefix checkpoints are never treated as covered, the eviction
scan walks the prefix tree by node instead of by token, and the log-line table
gained the `superseded` row. **Its merge is on hold**: if the redesign goes
ahead, chunk reference counting would replace its policy.

**Conclusions from the discussion.**

- Each checkpoint is a full snapshot of the model state at one position. A
  Flash-Next RAM checkpoint borrows the append-only KV rows of its live
  session (#445), but the RAM budget still counts its full size. Disk files are
  always full copies.
- No design document explains why disk files are full copies. The code
  explains it: snapshots are opaque to the cache, each file is verified,
  published and evicted on its own, and several processes may share one
  directory.
- Hashing the prompt at each turn would not add anything. The existing token
  prefix tree already finds the longest stored prefix exactly. The gain of a
  "diff" design comes from storing shared KV once.
- Recurrent state cannot be diffed. Any design keeps a full copy of it for
  every position it can restore.
- llama.cpp uses the diff idea only inside a live slot. Its cache across
  conversations stores full copies and deletes prompts contained in a newer
  one.
- vLLM, SGLang and LMCache share KV in blocks or radix-tree nodes and keep
  recurrent state as sparse checkpoints. Most of their machinery (per-token
  paging, host and remote tiers, prefetch policies, elastic pools, session
  IDs) addresses a scale gufo does not have: at most 8 concurrent requests,
  one machine, and unified memory.
- The smallest useful candidate: KV in 2,048-token chunks shared across
  checkpoints, a full copy of fixed state per checkpoint, the existing prefix
  tree and eviction ranks, and budgets that count each chunk once.
- The gain grows with context length. For Flash-Next, a checkpoint 2,048
  tokens past its parent costs about 2× less than a full copy at 8k tokens
  and about 24× less at 145k.

**Experiments, first round** ([results](experiments.md#summary-so-far)):

- Scope agreed: Flash-Next and 27B, four workloads (long agent, subagents,
  multi-user chat, restart), surviving restarts is essential, and disk space
  is tight.
- Checkpoint size is exactly linear in position. Flash-Next with MTP:
  113.8 MiB + 27.46 KB per token. 27B: 152 MiB + 64 KiB per token, plus
  80 MiB with DFlash2.
- Defects found in today's design:
  - a RAM hit of any length hides a longer disk hit;
  - a full RAM cache refuses new checkpoints;
  - automatic staging keeps checkpoints past ~75k tokens off disk;
  - 27B captures copy the whole state (235–635 ms).
- A simulator of today's cache reproduces the server's actual reuse within
  0.5% on every run. On the same traces, chunked KV:
  - adds no in-session reuse beyond a one-line disk-lookup fix;
  - keeps restarts cheap at depth: today a restart past ~75k tokens costs
    ~25–50 s on Flash-Next and ~2–5 min on 27B;
  - cuts 27B RAM copies 4–15× and disk writes 2.5–4.5×.

**Design discussion.**

- Keyframes plus diffs work if each diff carries the recurrent state at its
  end, and keyframes are built by background compaction on disk (see
  [Options](options.md#diffs-and-keyframes)).
- Recurrent state at a divergence point cannot be rebuilt from stored KV
  rows. The way to shrink the leftover prefill is cheap, dense checkpoints
  (see [example 7](hybrid-design.md#example-7-why-kv-rows-cannot-rebuild-recurrent-state)).
- The preferred direction is the [hybrid design](hybrid-design.md): live
  sessions, a shared KV chunk pool, small checkpoints, one index for RAM and
  disk. It would be phased: cheap fixes on today's design first (Phase 0),
  then the RAM pool, then disk, with paged KV optional.

**Overnight.** The [decision brief](decision-brief.md) covers:
- micro-benchmarks (copy bandwidth, chunk copies, disk);
- the [cost model](cost-model.md);
- E7, which simulates Phase 0 and the hybrid on every trace;
- E6, today's system at concurrency 4.

### 2026-10-08

A review of these documents raised points that are now addressed:

- **KV provenance.** Equal tokens can produce different KV bytes, so chunks
  are shared only within one lineage, never by token prefix alone
  ([hybrid design](hybrid-design.md#chunk-identity-and-provenance)).
- **Borrowed rows** need reserved spill capacity, not zero accounting.
- **Bounded transfers in both directions.** Today restores are limited by
  staging too: `ReadImage` loads the whole file. Phase 0 is corrected
  accordingly.
- **Crash consistency, pinning, orphan recovery and one writer per
  directory,** for the disk tier.
- **Compaction** is excluded from write volumes, so start without it.
  **Checkpoint density** is bounded by the record limit and the budget.
- **The model interface** lists each state component and its valid position.
- **Package architecture.** A dedicated common cache library owns slot
  leasing, shared chunks, budgets and disk persistence. Model-family adapters
  own state layouts and device operations, with explicit preservation,
  capture and restore contracts; see [Cache package](cache-package.md).
- **E8 simulator.** Revision 2, after a second review, replays requests
  concurrently with production's capture sequence and admission. Disk entries
  become restorable only when their write completes, restarts are graceful
  or abrupt, and chunks follow lineage. It reproduces actual reuse within
  0.4% on 11 of 12 runs. It shows the hybrid ahead of Phase 0 after abrupt
  restarts, on 27B W4 and with dense checkpoints.
- **Phase 0 does not fix everything:** abrupt restarts, 27B W4 and the
  subagent created after a restart favour the hybrid.
- **A 782 s disk-write stall** found at concurrency 4; cause unknown.
- **Reproducibility:** configurable script paths, committed token arrays and
  server logs (force-added: the repository ignores `*.log`), run metadata,
  and measured vs simulated labels.

**Next.** Discuss the required features, then choose.

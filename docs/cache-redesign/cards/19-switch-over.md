# 19 · Switch serving to the new cache

**Milestone:** Switch-over · **Depends on:** 01, 07, 09, 12, 13, 14, 15, 16,
17, 18, D1, D4 · **Size:** L (a stack of PRs merged together) ·
**Affects:** every continuation mode; cache directory format and ownership
(breaking) · **Status:** proposed

## Goal

Serving uses the new cache for every continuation mode, in RAM and on disk,
and the legacy cache is deleted in the same change (D1).

## Scope

- **Wiring:** `TextRunnerPool` builds the package and each mode's adapter;
  the request flow (acquire, capture at policy boundaries, commit) moves to
  the new API.
- **Options** per D4: `--cache-ram-bytes`, `--cache-disk`,
  `--cache-disk-bytes`, `--cache-disk-staging-bytes`.
- **Per-request cache-work reporting** through `--trace` and the server
  metrics, with a versioned schema:
  - reuse source (`none`, `live`, `ram`, `disk`), restored and prefilled
    tokens;
  - deepest candidate per tier and the selected boundary with its reason;
  - restore time, capture count, bytes and time;
  - admission, refusal and eviction events with rank and reason;
  - gauges for retained unique bytes, reservations and staging.

  No prompt text or token values.
- **Deletion:**
  - `ContinuationCache` and `ContinuationDiskStore`, with their tests;
  - the runner snapshot hooks the adapters replace: `Snapshot`,
    `SnapshotForPersistence`, `RestoreOrFork`, `SnapshotPayloadBytes`,
    `PersistentSnapshotPayloadBytes`, `SerializePersistentSnapshot`,
    `StreamPersistentSnapshot`, `RestorePersistentSnapshot`;
  - model snapshot classes and mechanisms the adapters replace, including
    Flash-Next's borrowed snapshots.
- **Docs:** `docs/KV-CACHE.md`, `docs/SERVER.md`, `docs/CLI.md`
  (single-process directories, cold rebuild after upgrade, new staging
  meaning), and the RFC status line.
- PR title marked breaking (`!`) per D4.

## Keeping it reviewable

- Build it as a stack: reporting schema, serving wiring, deletion, docs. Each
  PR in the stack is reviewed alone; they merge together, because there is no
  coexistence (D1).
- Start the branch as soon as card 14 lands and run it locally for
  Flash-Next (D1). Add the other modes as their adapters land.

## Required runs

Correctness runs on the candidate; timings use matched `main` controls.

- Every converted mode and profile: `cache-growth`, `cache-depth`,
  `cache-edits`, `cache-shared-prefix`, `cache-rotation`, `cache-concurrency`,
  `state-edges`, `long-context`, `cache`, `cache-bridge` (explicit), and the
  card 01 scenarios.
- Reassignment: B's TTFT and each decoding peer's per-token latency for idle,
  zero-idle and mid-spill reassignment.
- Restarts: graceful and abrupt, including a 100k+ checkpoint larger than the
  staging budget where capacity permits.
- Numerical quality checks and the standard speed benchmark per mode.
- A `tests/functional/pi_agent.py` replay of a long agent session; inspect the
  actual tool results and drafts.

## Step baseline

Per mode and suite, per request: prefilled and restored tokens, TTFT,
capture and restore time; peak RAM by ledger category; disk bytes and write
volume; time to readiness and the first restored request's TTFT. Compare the
card 01 scenarios with their `main` results. All of this feeds card 20.

## Done when

- [ ] Enable card 01's opt-in model workloads in routine functional
  qualification after they pass on the new cache, using the required separate
  RAM-pressure and disk/restart settings. Until then, keep them outside
  `--suite all`; the CPU harness/gate checks stay active.
- [ ] Correctness passes for every mode; restored state passes the numerical
  checks; no partial state executes after a crash, corruption or cancellation.
- [ ] Per-request 5% / 3 ms timing gates against matched `main` controls hold,
  or confirmed regressions are fixed; noisy results stay marked inconclusive.
- [ ] No reference to the deleted types remains.

## Review focus

- Nothing still reachable is deleted: check each removed hook across all
  runners.
- No per-request regression hidden by an average.
- Report separately the effect of charging unique bytes (card 07) and the
  storage effects.

## RFC

[RAM delivery](../RFC.md#ram-delivery-resume-conversations-across-execution-slots) ·
[Disk delivery](../RFC.md#disk-delivery-resume-retained-conversations-after-restart) ·
[Acceptance contract](../RFC.md#acceptance-contract) ·
[Compatibility and rollout](../RFC.md#compatibility-and-rollout)

## Review notes

## Results: Flash-Next draft integration, October 10, 2026

PR #543 is the early Flash-Next branch permitted by D1, not the final
all-model switch-over. The follow-up starts from `1cb96824`, with matched
production controls at its main base `665fc182`, the pinned Nix toolchain,
Strix Halo gfx1151, UD-Q4_K_XL target and shared Q8_0 MTP weights.

The integration checks exposed and fixed these defects:

- Streamed publication now resolves the store's attested checksum when a
  later checkpoint inherits an already published chunk. It still verifies
  the file and rejects corruption or conflicting nonzero checksum claims.
- Capture admits its complete backing plan before copying. A late size-class
  refusal therefore cannot repeatedly copy the whole checkpoint while
  reclaiming earlier records. Preparation pins inherited chunks and failed
  admission leaves history unchanged.
- All private components attest the target checkpoint boundary, including
  predictor-private state. Draft append rows and metadata retain the actual
  lagging draft frontier. The component-state ABI changes to 2.
- The typed in-process model API can use RAM caching without file
  fingerprints, with identity scoped to its immutable model lifetime.
  Durable caching still requires validated artifact fingerprints.
- Messages growth controls explicitly match the request's thinking mode.
  Cache-edit retries use the existing strict, logged capacity-refusal
  exception for at most nine assistant-opening tokens.

Peak total/RAM/category ledger gauges are exposed in `component-cache-v1`.
They describe cache allocations, including free committed pool backing;
model memory and process RSS are separate. Publication logs include logical
payload bytes for the oversized-staging check. Optional disk pieces yield
while an execution lease is active.

| Check | Follow-up status |
| --- | --- |
| Hosted PR contract suite | All 57 tests pass |
| Shared C++ formatting | All 567 files pass |
| AR/MTP component adapter | Transfer guards, 2047/8203 boundaries, edits, failed restores and image restoration pass |
| Serving EOS | Scalar/concurrent MTP, exact zero-prefill replay and per-request ignore-EOS policy pass |
| Serving sampling | All 25 strategies pass AR/MTP replay, C2 interleaving and short output budgets; actual drafts checked |
| Independent MTP numerical audit | Full-width normalization, split projections, attention, recursive carry and Q8 head pass |
| Core RAM functional histories | R7 passes growth, depth, edits, shared-prefix, rotation, concurrency, state-edges, long-context and cancellation in AR/MTP |
| Disk lifecycle | R8 AR and R11 MTP each pass all four cases |
| Disk reload and spacing | R7 misses recent queued histories; R8 restores all requested histories but drops the first spacing boundary; R11 passes the full focused cache suite in AR/MTP |
| RAM pressure | R11 AR fails five reuse assertions; unpushed R13 allocator reduces this to three; main fails six |
| Bridge, long Pi replay and standard speed benchmark | Pending; matched standard RAM table measured, with an unresolved RAM regression |

Raw logs, production binaries, source patches and hashes are retained outside
Git at `/home/mixer/gufo-qualification/pr543-followup`. The initial measured
candidate was `candidate-r3-gufo`, SHA-256
`b2b271a70620e91e5d95e78fea8563f78c4f76b9b59796dfcbb0e68991b86eec`.
The core run uses C2, context 32768, 8 GiB cache RAM, 8 GiB disk and 1 GiB
staging, selecting growth, depth, edits, shared-prefix, rotation, concurrency,
state-edges, long-context and cache explicitly for both AR and MTP.

Borrowed-row capture and idle spill are not wired into serving yet. These
checks do not establish reassignment latency, capacity-row qualification or
the final per-request 5% / 3 ms timing gate. The PR remains draft.

### Backing assignment follow-up

The `candidate-r3-gufo` AR run passed all 54 growth requests, then exposed
77.65 seconds of snapshot work on a cold 23,521-token depth request. The
retained CPU profile shows exception unwinding in `ReserveBacking`: every
assignment restarted its scan at block zero, repeatedly trying occupied
blocks. That matrix was interrupted, with its observations retained.

The pool now resumes its scan after the last successful assignment. A relaxed
atomic cursor is only a hint; ledger ownership still arbitrates races. The
128-block fill/wraparound host-allocation bound fails before this change and
passes afterwards. Existing allocation/failure/persistence-pin/concurrent-owner
checks and the AR/MTP adapter restoration tests also pass.

The restarted matrix uses `candidate-r4-gufo`, SHA-256
`618c9cbe3baf4d9e3e77a05217436ceb4c965090377f7bea392d2e786584d001`.
Main is still `665fc182` after checking the remote. Both production builds now
have external RSS/high-water and global HIP/UMA memory sampling for a matched
RAM comparison. The retained R4 results below precede the next correction.

### R4 matched RAM observations

The complete C2/32768 core histories used the same 8 GiB RAM, 8 GiB disk
and 1 GiB staging settings. These are observations from the failing R4
candidate, not qualification of the subsequent fixes.

| Mode | Main process RSS high-water (GiB) | R4 process RSS high-water (GiB) | Difference (GiB) | Main global HIP peak (GiB) | R4 global HIP peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 8.640 | 9.385 | +0.746 | 94.181 | 92.412 |
| MTP | 8.744 | 9.372 | +0.628 | 97.588 | 95.494 |

RSS is sampled through `/proc/<pid>/status`, retaining kernel `VmHWM`;
completed-request logs independently retain RSS throughout each history.
HIP is device-global `hipMemGetInfo` on UMA, sampled every 0.5 seconds. Do
not add HIP and RSS: their accounting can overlap. The main AR sampler began
after the first suite because its initial status parser rejected a hexadecimal
field; its RSS high-water remains complete, while its initial HIP observations
are missing. The other three runs have samples from startup.

Precommitting the candidate's pool changes the memory curve substantially:
growth alone peaks at 9504 MiB completed-request RSS versus 6790 MiB on main
in AR, while later main histories allocate more memory. After restart, the
candidate also immediately recommits its pool. The whole-run peak difference
therefore does not describe each request or startup. The retained
`memory-analysis-r4.json` includes all 202 matched AR and 176 matched MTP
completed-request RSS observations. Request coverage differs because failed
suites stopped early; hit rates across these unequal sets are not comparable.

The candidate's global RAM ledger peaks at 7733701160 bytes in AR and
7736221724 bytes in MTP (about 7.20 GiB), with a separate 1073741824-byte
staging allocation. Free backing, assigned/materialized rows, private state,
tails and metadata are charged within that RAM total. Category high-waters
occur at different times and must not be summed.

R4 AR passes growth, depth, shared-prefix, rotation and concurrency. MTP
fails several reuse histories. Both modes expose strict replay failures;
main also fails history-edit reuse and long-context replay. No clean timing
qualification is claimed. All original reports and failed comparisons remain
in the artifact directory.

The next allocator correction selects backing by component identity. A short
KV tail must use its component's full chunk class, rather than the smallest
class fitting its current bytes: that could consume a scarce predictor-private
class and make later MTP checkpoints impossible despite ample free backing.
With that correction, the focused RAM-only MTP run passes growth, depth and
history edits. State-edge and long-context exact retries still fall back by
seven tokens after those histories fill the private pools. Fresh 8 GiB RAM
checks of state edges and long-context replay pass in both modes; MTP proposes
426 drafts and accepts 333 in that run. This isolates the accumulated-pressure
failure from a numerical restoration defect.

### Capture batching and capacity planning

`candidate-r6-gufo` (SHA-256
`bceaef2cb4399e8edc5dda88d81b395183af055bcc314290684ff6d64928ae2d`)
coalesces adjacent chunks from the same physical allocation and batches four
independent transfers. Capture admits all buffers before submission and waits
before publication; restore waits before validation or execution. Failed
batches drain before buffers or stream leases are released. Occupied backing
blocks are skipped before allocating an assignment token.

R6 passes growth, depth, shared-prefix, rotation, concurrency, cancellation and
disk spacing in both modes. Edits and exact retries still fail after
accumulated pressure, and the subsequent disk reload checks fail. These
reports remain failed. The first AR growth captures take 22.51/14.85/14.95 ms
versus main's 37.36/31.73/28.62 ms; deeper capture and restart phases still
have timing flags. These individual observations do not establish a clean
timing gate.

The pool planner now counts compressed rows in pools rather than tokens,
reserves row capacity for configured concurrency and partial checkpoint
tails, and assigns the remaining feasible budget to private frontiers. It
keeps the same startup RAM limit. Independent CPU geometry checks cover AR
and MTP at 2/8 GiB, C2/32K, C8/4K and C1/100K+ capacities. A delayed-transfer
test verifies that a failed capture leaves the preceding published boundary
restorable; existing branch, restart and corruption checks also pass.

The revised production build is `candidate-r7-gufo`, SHA-256
`ae09ee54a37ef2ea1e6595709445428bce45ce560fb76c8f5f64b2a98739e432`.
Its AR/MTP model sampling and EOS replay checks pass, including all 25
strategies, concurrent interleaving and short output budgets. The complete RAM-side functional histories pass in both modes at the original
8 GiB limit. The four disk-reload checks fail because the bounded persistence
queue retains old histories while later requests keep the writer yielding.
The overall reports remain failed; timings are not qualified.


### R7 RAM and hit-rate observations

The R7 core run uses the same settings and immutable main controls as R4.
All original failures remain in the artifact directory.

| Mode | Main RSS high-water (GiB) | R7 RSS high-water (GiB) | Difference (GiB) | Main global HIP peak (GiB) | R7 global HIP peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 8.640 | 9.401 | +0.761 | 94.181 | 92.422 |
| MTP | 8.744 | 9.398 | +0.655 | 97.588 | 95.515 |

RAM-ledger peaks are 7736760420 bytes in AR and 7737002784 bytes in MTP
(about 7.21 GiB), plus the separate 1 GiB staging allocation. The same
RSS/HIP overlap and main AR startup-sampling limitation described above apply.
The main/candidate reports have unequal whole-run request coverage, so their
overall hit rates must not be compared. In the identical growth history,
24 of 30 normal requests hit in each mode on both builds: 80% request hit
rate and about 78.65% of prompt tokens reused. Cold controls are excluded.

### Fresh matched RAM benchmark

A separate fresh standard `memory` table compares main `665fc182` with
`candidate-r8-gufo`, SHA-256
`0ab38a58aed6715d92adc92649f152f40366c099be2a2e176c4818a701f1e86e`.
Both use AR, C1, context 133121, 8 GiB cache RAM, no disk/staging, the same
weights, pinned production toolchain and standard HTTP harness. Source
manifests, commands, raw requests, external startup samples and per-workload
HIP polling are retained in `main-standard-r8-memory` and
`candidate-standard-r8-memory`.

| Memory measurement | Main (GiB) | R8 (GiB) | Difference (GiB) |
| --- | ---: | ---: | ---: |
| Process RSS high-water across the table | 0.824 | 8.258 | +7.434 |
| Global HIP peak, pp2048/tg128 at depth 0 | 86.18 | 92.83 | +6.65 |
| Global HIP peak, pp4096/tg128 after a 16K prefix | 87.34 | 92.83 | +5.49 |

The new cache commits its backing pool at startup. Main grows allocations
with cache history. Consequently the smaller RSS difference in the long core
history does **not** imply lower memory use on lighter workloads. Process RSS
and global HIP memory overlap on UMA and must not be added. These measurements
qualify the memory observation only; they do not establish performance gates.

### Persistence follow-up

The queue now chooses pending writes by submission order, keeps the first
pending boundary of a history, and coalesces later pending ancestors into the
latest descendant. At capacity, eligible old pending jobs can yield to recent
work; learned pending checkpoints are protected from ordinary replacements.
Active writes are never replaced. Superseded callbacks run outside queue and
serving locks, and old source pins are released before new pins are admitted.
The serving queue allows eight jobs under the existing RAM pin limit.

Independent tests cover first/latest retention beside an unrelated active
writer, learned checkpoints, failed pin admission, callback reentrancy,
strict peak pin limits and exact restored recurrent state. The existing
streaming and serving lifecycle CPU targets pass, including first-prompt
publication after a RAM grid and learned near-tail disk branch recovery. Per-request reports expose
cumulative global queue counts, source-capacity pin gauges, write bytes and
I/O costs; source capacities can overlap between jobs, while ledger pin bytes
count physical allocations once.

R8's focused AR/MTP histories pass all eight request-level cache checks,
including each disk hit and exact cold control, but the aggregate spacing
assertion fails because its first boundary was coalesced away. R9 protects the first pending anchor, but exposed a separate spacing defect:
an intermediate grid consumed the first complete turn's disk allowance.
R10 keeps grid captures in RAM and persists semantic prompt/learned boundaries;
its first and long turns are correctly spaced, but a near-tail learned branch
is missing after restart. R11 retains disk shared-prefix configuration and
learns that near-tail boundary despite the ordinary spacing gate. A dedicated
CPU restart test checks exact recurrent state; the full focused AR cache run
now passes in AR and MTP. Duplicate submission rejection releases only
its own serving handle, preserving an accepted job with the same checkpoint.
The R8 standalone AR lifecycle passes all four cases, including a checkpoint
larger than 1 MiB staging and SIGKILL during an unfinished write. No partial
checkpoint is executed after restart. Pressure, bridge, MTP lifecycle,
100K+ durability, actual long Pi tool sessions and standard timing qualification
remain required before calling this draft complete.


The R11 AR cache comparison retains all 479 per-request/phase observations:
452 pass and 27 need confirmation under the unchanged 5% / 3 ms gates.
Disk restore is the main flagged phase (for example, 119.84 ms on main versus
268.75 ms on R11 for the final spacing history). One pair is insufficient
for confirmation; no clean timing gate is claimed. The lifecycle runner can
now require a measured 100K seed/publication/restore depth instead of inferring
it from background line count. Its existing 12 CPU workload/gate tests pass.

### CRC and pressure follow-up

Native CPU profiles of matched 13K disk restores identify CRC-64 as 64.10%
of R11 CPU samples. The GPU profiler fails during startup with duplicate
`gflags` registration; that failure and the successful native profiles remain
in the artifact directory. R12 processes eight bytes per checksum update,
with the same CRC-64/ECMA and streaming seed semantics. An independent bitwise
oracle covers alignments, lengths, seeds and split updates; all six focused
CPU disk/cache targets and all 57 hosted PR checks pass. Focused AR and MTP
HTTP cache histories also pass. Source and evidence:
`/home/mixer/gufo-qualification/pr543-followup/cache-crc-review.json`.

AR disk restore observations improve from 238–526 ms on R11 to 65–164 ms
on R12, versus 89–180 ms on main. All 479 observations are retained: 476
pass and three need confirmation. MTP disk restore improves too, but its
overall comparison remains failed due to changed cache-work counts after
cancellation, with 99 timing flags needing investigation. This is not an
overall performance pass. The R11 MTP lifecycle passes all four cases.

The separate R11 AR pressure history at 2 GiB fails five reuse assertions,
while every generated answer matches its cold control. Private-state pools
fill while row backing remains free; the fixed class allocation strands
capacity. The pressure report remains failed. The subsequent main control under
the same history also fails six assertions. Its timing is excluded because a
build overlapped that control. This does not waive candidate reuse assertions.

The measured main control remains `665fc182`; current main `d0f7c3a0` differs
only in server/cache documentation and has identical production sources,
toolchain lock and functional harness. The R8 matched RAM observations above
remain applicable to the allocation plan, which R12 does not change.

### RAM regression and rejected working-set provisioning

**RAM qualification remains failed.** The RFC's 83.2% retained-payload estimate
compares identical checkpoint boundaries for 27B DFlash2, which copies whole
snapshots. It explicitly excludes live storage, alignment, staging and temporary
publication, and says Flash Next already shares physical RAM. It is not evidence
of a Flash Next RAM saving. The existing Flash integration still commits roughly
7.2 GiB of backing at startup with an 8 GiB cache ceiling and eagerly copies rows
into that backing. Main borrows live rows and delays commitment of their
preservation storage. Pool reservations are correctly charged once, but correct
accounting does not make this extra physical storage disappear.

R13, an unpushed allocator experiment, lets private state, rows and tails share
64 KiB pages rather than stranding capacity in separate component classes.
Independent grouped-charge/fault tests, the GPU backing tests, all 57 hosted
checks and shared formatting pass. The 2 GiB AR pressure history still fails
three strict reuse assertions: compaction, continuation after volatile input and
parent resume. It is not a pressure or RAM-saving pass.

R14 tested a smaller initial backing pool, growing it only when all execution
leases were idle before admission. Capture itself continued to use committed
pages. The frozen experiment used the same standard memory table, C1 AR,
context 133121, 8 GiB RAM ceiling, no disk/staging, weights and production
toolchain as the R8 controls. A fresh main run reproduced both earlier main
global HIP peaks exactly. All five requests match prompt, prefill, cached and
generated token counts, including 16242 cached and 4077 prefilled tokens in the
deep continuation. Identical retained checkpoint sets have not been established.

| Measurement | Fresh main (GiB) | R14 experiment (GiB) | Difference (GiB) |
| --- | ---: | ---: | ---: |
| Process RSS high-water across the table | 0.824 | 3.791 | +2.967 |
| Global HIP peak, pp2048/tg128 at depth 0 | 86.18 | 86.73 | +0.55 |
| Global HIP peak, pp4096/tg128 after a 16K prefix | 87.34 | 88.35 | +1.01 |

These are overlapping measurements on UMA, not additive RAM categories. The
global counter includes other HIP users; idle is 2.3767 GiB in all four retained
memory runs. R14 reduces the earlier candidate footprint but still uses more
memory than main. Admission-time commitment also produces queue times of
122.5, 22.8, 333.9 and 107.1 ms, versus 0.9, 0.8, 1.8 and 1.9 ms in the fresh
main control. All four fail the unchanged 5% and 3 ms per-phase gates.

**R14 is rejected and its code has been removed from the working tree.** Its
binary, complete patch, successful focused CPU checks, failed timing comparison
and raw memory samples remain retained. It was never pushed. The remaining R13
allocator work is separate and unpushed. Source manifests, hashes, commands,
memory observations and individual queue gates in
`/home/mixer/gufo-qualification/pr543-followup/cache-ram-review.json` make the
experiment reviewable without presenting it as the current PR.

Borrowed-row capture and idle spill still need serving integration and real
reassignment qualification. They can reduce eager capture work, but the RFC
requires their spill backing to be committed; borrowing alone cannot remove
that reservation. Qualification must account for both the live allocation and
retained backing, compare useful retained boundaries as well as matched requests,
and pass latency and reuse gates. No physical RAM reduction versus main is
established, and the PR remains draft.


### Borrowed serving rows and idle spill, October 11, 2026

The follow-up merges main `3323e7c7`, retaining its device-headroom snapshot
coverage and text-only image rejection. Flash-Next serving now borrows full
completed device append chunks against assigned, precommitted backing. Host
token rows, private state and partial tails are copied. Mutation guards preserve
borrowers before overwrite, reset, restore or release. Each idle slot spills in
bounded 1 MiB pieces on independent pooled streams, yielding before each piece
when a serving request holds an execution lease.

A row pin now carries the immutable native byte view captured during exclusive
execution. RAM restoration keeps all source pins until its asynchronous loads
settle, including failure drain. It can read a busy peer's completed prefix
without acquiring that peer's execution lease or forcing preservation; device
loads accept both host backing and pinned device views. Adjacent compatible
views still coalesce into bounded transfer batches.

Disk jobs pin assigned backing at submission and wait for preservation before
filesystem publication, staging or stream acquisition. The queue never captures
live state. (Revised for #573: Flash-Next jobs now read rows still in a slot
through the same per-piece row pin RAM restore uses, because idle preservation
only runs while no request is busy. In a six-session run, queued jobs waited
minutes for it, held 1.8 of 1.9 GB of cache RAM and published nothing, so RAM
captures failed and later turns missed.) Cancellation releases its pins while retaining the original rows.
An idle transfer failure fails the optional disk job rather than hanging its
drain; a later foreground preservation can retry the retained rows. Shutdown
drains durable jobs, then stops every idle worker before destroying slots.

Five focused CPU cache/serving targets and all 57 hosted PR checks pass. The
serving fixture restores from a source still holding its execution lease and
checks that the source remains borrowed until overwrite. Streamed-store tests
cover assigned pin accounting, foreground publication while another job waits,
cancellation and failed idle spill with a successful foreground retry. Shared
formatting checks all 568 C++ files; the pinned production release build passes.
The real GPU adapter guard checks pass in AR and MTP, including complete
byte-for-byte checkpoint restoration through pinned native views at 2,047 and
8,203 tokens, plus short mutation, reset, restore and release cases.

Request metrics expose copied logical payload bytes, borrowed backing capacity
and cumulative global idle/foreground spill bytes and waits. These distinguish
capture work from assigned storage and subsequent preservation. The backing
pool's eager physical commitment is unchanged: no Flash-Next physical RAM saving
is claimed. HTTP histories and matched latency, pressure, deep disk/restart,
bridge, Pi and standard-speed qualification remain open. This supersedes the
earlier statement that borrowed rows and idle spill were unwired; it does not
complete card 19 or make the PR ready to merge.

Commands, immutable binaries, patches, source/harness hashes and raw results,
including the initial expanded test's context-configuration failure, are retained
at `/home/mixer/gufo-qualification/pr543-borrowed-2026-10-11`.


The frozen `36cacfb6` serving candidate passes RAM `cache-growth` (54
responses) and `cache-depth` (16 responses) in each of AR and MTP, at C2,
32768 context and 8 GiB RAM with disk disabled. Loaded-mode logs attest `off`
and `mtp`; normalized Chat observations record zero AR drafts and 1,229 MTP
drafts. Messages cases also pass their strict prompt/prefill/reuse controls.
These are correctness observations, recorded without matched timing controls;
they do not qualify the remaining workload matrix or a latency gain.

Review of cold assignment then found that serving still picked the first idle
slot even when another slot had completed preservation. It now prefers an
already preserved slot, matching the package's assignment policy. A deterministic
CPU test makes idle streams unavailable and verifies that an unrelated request
performs zero foreground preservation while the preceding history remains
borrowed. Shared Chat/Responses/Messages timing objects now also carry the
versioned component-cache metrics. Both focused serving/protocol targets and
all 57 hosted checks pass after these changes; the production build passes.
The earlier HTTP observations remain attached to the immutable pre-selection
candidate. The revised candidate still needs its affected HTTP histories and
fresh matched main controls.

The matched core-history run freezes `3064a01e` (the later `8ea0d99c` changes
only assertion formatting) and main `3323e7c7`, using the same pinned toolchain,
model, C2, 32768 context, 8 GiB RAM, sampling and suite order. Growth, depth,
shared-prefix, rotation and concurrency functional checks pass in both modes
on both builds. The candidate also passes all three MTP edited-history cases,
reusing 2048 tokens in each, with 1506 actual MTP drafts over the complete run.
Main fails edited-history reuse in both modes, and the candidate's AR
older-tool edit loses its retained boundary and fails. The comparator rejects
these failed runs; they establish no matched latency gain. Every raw request,
failed report and diagnostic comparison remains retained alongside the binaries.

The AR failure exposes stranded free capacity between fixed backing classes:
the candidate can refuse captures while reporting more than 5 GiB of free
backing. Capture/prefill timing flags also require investigation. Flash serving
disabled its existing coherent in-pass checkpoint capability, adding a separate
five-token prefill tail. Restoring that capability through the component adapter
is under qualification, including an independent native boundary serialization
oracle and transfer-drain failure checks. No result is claimed for that pending
change.

[Issue #542's PR reproduction](https://github.com/gufo-org/gufo/issues/542#issuecomment-6105542426)
adds a deep pressure/restart gate: with a roughly 174K-token history and a RAM
budget smaller than its full checkpoint, an earlier PR build retained only the
system prefix while main eventually resumed the deep disk checkpoint. That
external observation is preserved and needs a matched local reproduction;
passing the shorter RAM histories does not qualify it.

The in-pass fix is frozen locally as `131db71c`. Four focused CPU targets,
all 57 hosted checks, the shared formatter and the production release build
pass. The new real GPU oracle passes AR/MTP at 2047, 8192 and 8203 tokens:
predicted positions and every frozen byte match a native boundary restore,
completed-live and restored logits are exact, and sampled continuations match
with actual MTP drafts. Reentrant execution/reset/invalidation is rejected;
failed consumers drain transfers before the native scratch arena is reused.
The existing native snapshot suite also passes, including device-headroom
fallback, source reset, borrowed/serialized replay and checkpoint retention.

A fresh growth-only AR comparison has exact output and prompt/cache/prefill
work, but timing remains inconclusive at the unchanged 5% / 3 ms per-phase
gate. The extra five-token pass is removed; capture and other timing flags
remain retained. A separate CPU profile of frozen `3064a01e` identifies
retained-byte accounting, linear chunk lookup and buffer clearing as request
overhead. The resulting bookkeeping changes are not yet qualified.

The MTP growth-only run also passes all 54 functional responses, with 1298
actual drafts. Its final Messages retry prefills zero tokens where main's
capacity-refused retry prefills seven; the strict comparator rejects that work
difference and retains the timing flags. This is not a matched speed result.

The bookkeeping follow-up caches retained-byte totals across recency-only
updates, resolves earlier component peers once per component, and orders
history entries for binary chunk lookup. Private capacities and shared-chunk
identity are unchanged. Six focused CPU targets, all 57 hosted checks,
formatting and the production build pass, including removal while a reader
keeps the checkpoint pinned and sharing after unordered-descriptor restoration
and pruning. Its profile and HTTP timing qualification remain pending.

### Shared backing pages and deep disk qualification, October 11, 2026

The Flash-Next pool now commits one set of 64 KiB pages shared by private
state, rows and tails. A payload takes a contiguous page group with one ledger
owner; slack is charged. This replaces fixed per-component classes, which could
refuse captures while more than 5 GiB of other classes stayed free. The
change ports the previously unpushed R13 allocator onto `8ada5989`, including
borrowed rows. Ten focused CPU targets (ledger, backing, checkpoint, slot,
retention, disk, streaming, component serving, runner, layout) and the shared
formatter pass. The candidate binary SHA-256 is
`fdc2878760419f544b9baf04e23bf5677400dc4cf43c5a2dc881f5bfb2c12c76`.

**Deep disk delivery, AR, context 133760, one session, 8 GiB RAM.** The
standalone lifecycle runner seeds about 102K tokens, appends to about 113K and
requires a published checkpoint of at least 100K tokens.

| Case | Main `3323e7c7` | Candidate |
| --- | --- | --- |
| restart | Fails: no deep publication | Passes: restores 113,451 tokens from disk, 7 prefilled |
| tiered | Fails: no deep publication | Passes: deeper disk boundary selected |
| oversized (1 MiB staging) | Fails: no deep publication | Passes: 3,007,420,756-byte checkpoint published |
| crash (SIGKILL at fsync) | Fails: no published seed | Passes: only the published 102,231 boundary restores |

Main skips each deep write with `reason=staging_capacity`: the 2.71 GB and
3.00 GB images exceed its 2 GiB staging. The candidate streams them in bounded
pieces. After restart, each 113K restore takes 1.89–1.99 s and TTFT is
2.16–2.27 s; the matched 113K cold controls take 75.8–77.2 s TTFT. Every restored answer
matches its cold control.

Disk payload sharing is measured on the same history. Each initial stage
publishes 102K and 113K checkpoints with 5.73 GB (5.34 GiB) of logical
full-copy payload; the server processes physically wrote 3.71–3.79 GiB,
29–31% less. The peak cache directory size was 5.14 GiB. Main wrote no deep
checkpoint, so this is not a matched main write-volume comparison. Process
RSS high-water is about 10.3 GiB, including the 8 GiB committed pool.

**Core RAM histories, C2, context 32768, 8 GiB RAM.** Growth, depth,
shared-prefix, rotation and concurrency pass in AR and MTP. History edits fail
in both modes on the first older-tool edit: 111 AR and 113 MTP captures are
refused. The cause is retention rank, not page stranding. History grids may
evict only retry/history ranks, while earlier suites leave rank-2
continuations and rank-3 branch points that never age out. A continuation then
evicts its own covered 2,048-token grid, the edit's only usable boundary. Main
fails the same suite in both modes with the same inherited ranks. With fixed
classes, MTP passed by chance of capacity; it no longer does. The RFC keeps
eviction-priority changes in separately measured review.

**RAM pressure, AR, 2 GiB.** Three reuse assertions fail: compacted history,
child fork and parent resume. The fixed-class candidate failed five and main
six on the same history. This is a partial improvement, not a pass.

**Capture clearing.** Private capture zero-filled each complete destination
on the host before its copy: about 132 MB per AR checkpoint. A 132 MB
device-to-host copy alone takes 1.6–2.1 ms into coherent, non-coherent or
default pinned memory. Capture now clears only bytes no source piece covers;
those still serialize as zeros. The default, in-pass and guard adapter GPU
oracles pass in AR and MTP, including every private byte at 2,047 and 8,203
tokens and every frozen in-pass byte. Temporary phase timers in one AR growth
run put `Capture` itself at 4.1 ms median (11.4 ms p90, 112 captures). The
reported `cache_snapshot_ms` median falls from 28.6 to 21.4 ms; it is the wall
time of an asynchronous post-token capture that overlaps decode, not added
request latency by itself.

**Latency against main, identical work only.** Of the five passing core suites,
137 AR and 140 MTP requests have identical prompt/reuse/prefill work. The
candidate still regresses user-visible phases; the per-request 5% / 3 ms
gate fails:

| Revision | Mode | Queue Δ median / p90 (ms) | TTFT Δ median / p90 (ms) | Mean inter-token Δ median / p90 (ms) |
| --- | --- | ---: | ---: | ---: |
| `3064a01e` (fixed classes) | AR | +7.1 / +22.3 | +226.7 / +763.7 | +4.2 / +15.6 |
| shared pages + gap clearing | AR | +10.6 / +37.6 | +104.6 / +517.9 | +5.5 / +13.1 |
| `3064a01e` (fixed classes) | MTP | +6.2 / +17.9 | +121.3 / +330.2 | +5.9 / +20.5 |
| shared pages + gap clearing | MTP | +11.0 / +38.4 | +12.1 / +54.5 | +6.5 / +23.7 |

Main's medians are 0.8/3.5 ms queue and 35.6/56.9 ms mean inter-token gap
(AR/MTP). These are single runs with no alternating controls, so
individual flags are not confirmed. The decode and queue regressions predate
this follow-up.

**Guard, reassignment and batch follow-up.** Three causes were found and fixed:

- Every mutation guard walked the slot's whole borrowed-row registry, locking
  one weak handle per entry. This ran for every component on every decode
  step. Entries now keep their component and range. Each component also keeps
  an upper bound of its borrowed ends, so appends past it skip the registry.
  The bound rises before a range is registered and clears only after
  reassignment has preserved every range.
- An experiment overlapped reassignment preservation on up to eight pooled
  streams. It barely moved queue time (+10.6 to +10.3 ms median). It also
  broke the bounded-cancellation contract of `cache_idle_spill_test`, which
  hosted CI caught on `d9fa9dfb`–`4c638417`. `b060a2f0` restores bounded
  synchronous pieces.
- Capture and restore waited after every four transfers. Batches now reach 16;
  when the shared stream pool is momentarily empty, a batch settles its own
  transfers and retries instead of failing.

The ten focused CPU targets and the adapter guard GPU oracle pass in AR and
MTP. A fresh unchanged-main control shows that the earlier main runs had
drifted: fresh main AR prefill is 75 ms slower at the median than in the
morning. The table therefore compares against fresh main controls of the same
core suites, on identical prompt/reuse/prefill work only:

| Candidate | Mode | Queue Δ | TTFT Δ | Prefill Δ | Decode Δ | Mean inter-token Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| before follow-up (`79e24777`) | AR | +9.3 / +29.3 | +3.1 / +37.4 | −9.2 / +18.0 | +8.1 / +27.0 | +4.7 / +12.8 |
| this follow-up | AR | +3.3 / +11.7 | −4.3 / +30.6 | −7.2 / +16.4 | +0.5 / +1.5 | +1.5 / +6.8 |
| before follow-up (`79e24777`) | MTP | +11.5 / +43.3 | +28.8 / +136.7 | +12.3 / +76.8 | +8.4 / +24.8 | +2.1 / +20.5 |
| this follow-up | MTP | +4.3 / +20.3 | +25.5 / +196.1 | +16.7 / +167.4 | −0.2 / +11.5 | +0.1 / +7.6 |

Values are median / p90 in milliseconds over 137 AR and 140 MTP requests.
Strict 5% / 3 ms flags against fresh main fall as follows:

| Phase | AR before | AR after | MTP before | MTP after |
| --- | ---: | ---: | ---: | ---: |
| Decode | 85 | 1 | 70 | 27 |
| Queue | 104 | 71 | 107 | 82 |
| Maximum gap | 81 | 54 | 63 | 42 |
| TTFT | 32 | 17 | 44 | 39 |
| Wall | 27 | 12 | 42 | 38 |

The remaining queue excess is about 3–4 ms at the median. In the
concurrency suite, roughly half of it is reassignment wait, and the rest is
not attributed. The final code (`b060a2f0`, SHA-256
`6025a4299c74f74603aab59c2f31b4d194cb828871fc1c0b28ca3b27c8ac5076`) remeasures
within noise of the table. Medians: AR queue +3.2, TTFT −0.5, decode +0.6 and
mean inter-token +1.9 ms, with 2 decode flags. MTP queue +4.6, TTFT +20.5,
decode −0.9 and mean inter-token +0.1 ms. MTP's largest TTFT flags are concurrency `long_tasks`
requests: +409 and +632 ms on about 5–6.6 s. The long cold MTP prefills are
about 2% slower (310–370 ms on 16 s), within the 5% gate. The gate is still
not met. The final candidate binary SHA-256 is
`69b5343618643ef168b73ae49f37a2c3dac94c32b4a27ad02ca55a537471932d`.

A same-slot RAM restore still reloads rows that are already resident. For
example, a 2,927-token restore copies 206 MB, of which 74 MB are those rows.
Such requests show 4.8–6.6 ms of queue time, versus 0.6–1.0 ms for live
reuse. Restoring in place, by keeping resident rows and loading only private
state, needs an adapter protocol change and is not implemented.

**Further functional checks on the final candidate.**

- AR `long-context`, `state-edges` and the full `cache` suite pass (cancelled
  text/thinking/tool histories, disk restart and spacing), at C2, context
  32768 and 8 GiB.
- AR `cache-bridge` passes at a 2 GiB RAM budget.
- The 100K MTP lifecycle passes restart, tiered, oversized (3,248,448,596-byte
  checkpoint through 1 MiB staging) and crash. The 113K disk restores take
  2.03–2.13 s, with TTFT 2.24–2.42 s; the crash case restores only the
  published 102,231 boundary.
- A real Pi 0.87.0 session (`pi_agent.py`, AR, one session, context 133760,
  about 102K tokens of background context; cases tools, edit and bugfix) passes
  all three cases with nine tool calls and no tool errors. The inspected tool
  results are actual file writes/reads, a passing edited-code run (`ok`) and
  shell output. After the cold first turn (64.8 s TTFT), each of the 11
  follow-ups prefills only 19–129 tokens, with 173–354 ms TTFT. Pi 0.87.0 was
  installed from `@earendil-works/pi-coding-agent`, because the system Pi is
  0.99.2 and the harness pins 0.87.0.
- MTP passes `long-context`, `state-edges`, the full `cache` suite and
  `cache-bridge`. AR and MTP both pass `cache-transforms`.
- The MTP Pi replay passes on its second attempt. It records 465 actual drafts
  (372 accepted); follow-ups prefill 19–478 tokens with 178–651 ms TTFT.
  Its only tool error is the bugfix task's intended first run of the failing
  test, which the agent then fixes (`ok`). The first attempt is retained: Pi
  sent no request within 300 s, and the server log shows none received.
- Standard single-user benchmark (`tools/bench/model-bench.py`, pp2048/tg128)
  run in the order main, candidate, main again. Main drifted between its own
  runs (AR d0 prefill 1694.6, then 1594.2 tok/s), so the result is
  inconclusive. Against the later main run, candidate prefill is −3.4%/−4.4%
  at depth 0, +0.6%/−1.2% at 32K and −1.3%/−2.7% at 128K (AR/MTP), and decode
  differs by at most 0.4%. That is inside the 5% gate but consistently lower
  at depth 0; it needs repeated alternating runs.

| Depth | Mode | main pp/tg | candidate pp/tg | main again pp/tg |
| ---: | --- | ---: | ---: | ---: |
| 0 | AR | 1694.6 / 28.25 | 1540.7 / 28.20 | 1594.2 / 28.23 |
| 32,768 | AR | 1500.5 / 28.14 | 1413.3 / 28.07 | 1404.2 / 28.11 |
| 131,072 | AR | 1347.4 / 27.15 | 1284.4 / 27.09 | 1301.5 / 27.10 |
| 0 | MTP | 1568.1 / 33.87 | 1454.6 / 33.45 | 1521.8 / 33.75 |
| 32,768 | MTP | 1400.8 / 35.66 | 1350.5 / 35.18 | 1366.3 / 35.65 |
| 131,072 | MTP | 1298.8 / 36.64 | 1257.4 / 36.29 | 1292.3 / 36.64 |

Not yet qualified: retention-rank starvation; Issue #542's history deeper than
the RAM pool (rows must fit committed backing before publication); the
remaining queue and concurrency TTFT flags; the inconclusive depth-0 prefill
benchmark; reassignment latency;
the standard speed benchmark; a lower physical RAM footprint. Evidence,
commands and binaries are retained at
`/home/mixer/gufo-qualification/pr543-claude-2026-10-11`. The gap-clearing
candidate binary SHA-256 is
`6c1cc173fe0c1c193a444c5021151081b49de3bb760fd70bbbf756c8e0262409`.

### Review follow-up: MTP startup budget (October 11)

The reported `6 × 131072`, MTP, seven drafts, 2 GiB RAM configuration fails
on `05f6ec2a` on this host too. Before allocating slots, serving subtracts
31,616,116,728 bytes of worst-case state from a 26,797,985,792-byte host
ceiling and passes zero to `BackingPlan`. The state claim includes lazy
image embeddings and rollback capacity; it is not the startup footprint.

Serving now creates the adapter's execution slots first, observes remaining
host RAM, lowers the ledger's initialization limits, then commits its backing
pages. Automatic sizing uses the post-slot half-available observation;
explicit requests remain capped by the post-slot ceiling. The ledger refuses
limit increases or reductions below admitted metadata/reservations/pins.

The two focused CPU regression targets pass, as do all 57 hosted contracts
and the shared formatter. Production GPU load and 48-token generation pass
for MTP at `6 × 131072 / 2 GiB`, `3 × 262144 / automatic`, and the
`1 × 32768 / automatic` control. Actual drafts are required and observed.
This establishes startup/generation, not deep durability or a cache timing
pass. The initial load-only client used an incorrect served model name and
received HTTP 404; those attempts are retained separately and excluded from
generation evidence.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`base-load-c6.log`, `diagnose-load-c6.log`, `load-generation.json`,
`*-generation.log`, `check-pr-budget.log`.
Production binary SHA-256:
`74b29924fa41b618829897bd2c7585057e99b9940a891c87e8b2f113602a4f02`.


### Review follow-up: checkpoints exceeding RAM (October 11)

Semantic checkpoints can now persist when completed append rows exceed the
committed RAM pool. Disk-only capture copies private state, partial tails and
movable host token rows into admitted backing; fixed native append allocations
are observed under the existing mutation guards. Each staging piece pins its
source until the transfer completes. Mutation/reassignment retires unread
observations and drains only current readers, so an abandoned job cannot
publish partial state or force an entire-history foreground spill. Observations
cannot enter retained RAM payloads and carry no payload backing obligation.

Published full chunks may be inherited only from a validated disk restore or
a completed capture on the same uninterrupted execution branch. The ancestor
publication is pinned, its immutable lineage is retained, and private state and
tails always get new files. Token-prefix equality alone does not establish
chunk provenance. CPU regressions cover mutation while a reader is active,
abandoned publication, a history larger than the physical pool, two restarts,
full-chunk sharing and partial-tail promotion.

The opt-in lifecycle runner adds `--require-over-ram`: publication must exceed
the server's actual post-slot RAM budget, rather than merely exceed staging or
appear in a queue. Flash MTP, three sessions, context 262144, requested RAM
4,593,364,992 bytes and 1 MiB staging passes oversized publication/restart with
actual RAM 4,419,084,288 bytes. It publishes a 5,247,536,596-byte checkpoint,
restores 186,253 tokens and prefills seven. All three cold controls match and
actual MTP drafts execute. The four focused CPU checks and all 57 hosted
contracts pass; the shared formatter passes.

This is a durability correctness result, not a latency pass: disk restore is
18,631 ms. The earlier run restored the same deep boundary but its second cold
control timed out at 180 seconds; it remains a failed qualification run, with
a lower effective RAM budget. The successful repeat does not establish that
the lower-budget stall is fixed. Crash recovery and transfer latency remain
follow-up work.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`deep-oversized-mtp-final/report.json`, `deep-oversized-mtp/report.json`,
`deep-cpu-tail-final.log`, `deep-focused-final.log`, `deep-pr-contract.log`,
`deep-format-final.log`. Candidate production binary SHA-256:
`14333e03b1c8bc83667d5d9270ba4d47c88b0cf65120b60c374e368589d72971`.


### Review follow-up: grouped disk restores (October 11)

Streamed restore gathers adjacent, individually validated row files in its
existing admitted staging allocation, then submits one contiguous native
transfer. Large files retain bounded streaming; private state settles before
staging is reused. Checksum or identity failures invalidate the whole restore.
A new regression checks odd staging alignment, partial tails, next-append state
and corruption after an earlier group transferred. The lifecycle runner now
records a configurable staging budget for restart, tiered and crash controls.

The two focused CPU targets, all 57 hosted contracts, shared formatter and
12 lifecycle/workload harness tests pass. Native GPU AR/MTP adapter oracles
pass for in-pass frozen bytes, guards, restoration, earlier edits and images.
A single-seed production MTP restart at C1/context 262144, 8 GiB RAM and
6 GiB staging restores 175,032 tokens and prefills seven, matching the seed's
ALPHA answer. It uses 130 transfers and takes 6,861 ms: filesystem reads
4,753 ms, checksums 1,817 ms, transfer waits 121 ms. This is a diagnostic
sample, not a matched latency pass; the main control and disk profiling follow.

The earlier append timing attempt used an 8 GiB disk/queue limit: an active
4.94 GB write prevented admission of the 5.25 GB deeper checkpoint. It was
interrupted while awaiting that unavailable publication and remains recorded
as unqualified. The earlier SIGKILL test reached the correct published-only
restore, but a competing GPU-test build OOM-killed its final cold control;
the kernel evidence is retained and an isolated rerun is required.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`group-deep-seed-timing`, `group-deep-timing`, `deep-crash-mtp`,
`deep-crash-build-oom.log`, `group-cpu4.log`, `group-pr-contract.log`,
`group-format-final.log`, `group-harness.log`, `group-adapter-gpu.log`.
Production binary SHA-256:
`d3adf7c18714505e4ec924613143a589557c4140f5f67065ee62b99f95ea64fc`.


### Review follow-up: native disk checksums (October 11)

CRC-64/ECMA updates use four carry-less multiplication lanes on supporting
x86-64 CPUs, retaining the existing polynomial, streaming seeds and v2 disk
format. The table implementation remains the fallback. Bounded filesystem
read-ahead hints cover at most 8 MiB and 32 future chunk files; hints never
validate a dependency or allocate another payload buffer. Actual reads retain
identity checks and independent checksum validation before restored execution.

Independent bitwise checks cover three seeds, sixteen alignments, ragged sizes
and streaming splits. Integration tests cover short reads, EINTR, a multi-piece
payload and corruption after a read-ahead hint. Four focused CPU checks,
all 57 repository contracts and the shared formatter pass.

The same single-seed MTP history (C1/context 262144, 8 GiB RAM, 6 GiB staging)
restores 175,032 tokens and prefills seven, with the same ALPHA answer and actual
drafts. Restore falls from 6,861 ms to 4,299 ms; checksum time falls from
1,817 ms to 883 ms. Read-ahead alone took 7,195 ms and is not claimed as a
latency gain. The matched main restore is 3,141 ms: the remaining 37% regression
fails the strict timing gate. These runs omit cold controls and establish a
matched timing observation, not full lifecycle correctness. No inference
benchmark or numerical-quality claim changes.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`fold-deep-seed-timing`, `prefetch-deep-seed-timing`, `main-deep-seed-timing`,
`fold-cpu.log`, `fold-pr-contract.log`, `fold-format-final.log`.
Production binary SHA-256:
`66d1c04b485fac662bab7e4f41d80dc82dba0ae83207589755157b30aff10aeb`.


### Review follow-up: uncovered history retention (October 11)

A shorter continuation or branch checkpoint no longer makes a deeper history
redundant merely because it shares the history's leading prefix. A history is
covered only when its complete tokens prefix a compatible continuation. This
corrects the reverse-prefix defect described in #553 within the new module;
legacy model caches are unchanged.

The new regression fails on the previous implementation, then verifies both
continuation/branch peers, actual record-pressure eviction and the remaining
four-token reusable history. It also checks genuinely deeper coverage and input
identity isolation. All 57 contracts and the shared formatter pass. Production
AR/MTP growth, depth, shared-prefix, rotation, concurrency and edit replays are
running separately. This rank correction does not establish that grid admission
starvation or constrained-RAM pressure is fixed.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`rank-red.log`, `rank-pr.log`, `rank-format-final.log`, `rank-functional-summary.json`.


### Review follow-up: open and closed cache issue audit (October 11)

The issue audit (`review-543-issues.md` and `issue-audit-classification.json`
in `/home/mixer/gufo-qualification/pr543-review-2026-10-11`) records 106 cache
search matches and
160 unique issues after expanding to checkpoint, prefill and continuation
(139 closed, 21 open). Fifty directly related assessments name the evidence and
its limits; other matches remain outside checkpoint qualification. Closed state
is not treated as proof that a historical reproduction is fixed. Unsupported
features, renderer defects, legacy-model behavior and unfinished performance
checks remain explicit. Production qualification of the new edit-admission
policy and the deep side-traffic regression continues separately.


### Review follow-up: edit-boundary admission (October 11)

[Card 21](21-edit-boundary-admission.md) records an explicit change from the
original retention-policy port: history byte admission can reclaim old families,
and each execution lease protects its admitted histories from later captures
in that turn. Retry ranks, full-record optional guards and LRU within ranks
remain unchanged. Protection ends with the lease.

The actual four-block serving regression fails without protection, then restores
both edited boundaries with independent cold-state agreement after the fix.
All 57 repository contracts and shared formatting pass. Production C2/context
32768/RAM 8 GiB passes growth, depth, shared prefix, rotation, concurrency and
edits in both AR and MTP, with the same accumulated history that failed edits
on the prior policy. Last-user/older-tool/rewind edits restore 2048 tokens and
prefill 1199/1656/1199; cold outputs match. MTP proposes 1506 drafts, accepting
871. The constrained 2 GiB pressure replay and affected matched timings remain
open; this is an edit-reuse correctness result, not a latency qualification.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`grid-functional-summary.json`, `grid-core-{off,mtp}`, `grid-protection-red.log`,
`grid-protection-green.log`, `grid-pr-final.log`, `grid-format-after-control.log`.
Production binary SHA-256:
`def206c3bb6a37af4cb6eaed2a914209252ca9781d1321cfb4c45f47062442f7`.

### Review follow-up: small-budget retention (October 12)

At 2 GiB the four `cache-pressure` reuse failures on `def3763e` came from
retention policy, not capture correctness: four grid captures per long turn,
rank order removing a stale conversation's shared boundary before tiny
checkpoints, LRU letting short requests remove an idle conversation, and one-use
requests displacing a reused conversation. [Card 22](22-small-budget-retention.md)
records the bounded grid density and byte-pressure order that fix them, their
regressions, the AR/MTP reuse results and the still-unqualified timing.

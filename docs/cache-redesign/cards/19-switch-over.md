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
| Disk lifecycle | R8 AR passes all four cases; R11 MTP running |
| Disk reload and spacing | R7 misses recent queued histories; R8 restores all requested histories but drops the first spacing boundary; R11 passes the full focused cache suite in AR/MTP |
| RAM pressure, bridge, long Pi replay and standard speed benchmark | Pending; fresh matched standard RAM table measured |

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

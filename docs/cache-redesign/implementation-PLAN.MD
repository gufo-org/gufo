# Hybrid cache implementation plan

Status: planned; implementation has not started. This document is the
execution checklist for the [hybrid design](hybrid-design.md) and
[cache package architecture](cache-package.md). The existing measurements
and simulator results are [research evidence](experiments.md), not evidence
that the proposed implementation has passed its acceptance gates.

## Scope and boundaries

- Build a dedicated CMake library in `src/cache/`. It owns slot leasing,
  exact-prefix lookup, checkpoint policy, chunk ownership, reservations,
  RAM/disk accounting, bounded persistence and recovery.
- Keep state layouts, numerical contracts and device operations under
  `src/models/<family>/cache_adapter.*`. Use one adapter per state layout or
  model family; variants may share it. Register adapters without model-name
  switches in the generic package or a dynamic plugin requirement.
- First implement hybrid storage for Flash-Next in AR/MTP modes and 27B in
  AR/DFlash2 modes. Preserve existing behavior and advertised capabilities
  for every other model family through the extraction and migration.
- Keep inference scheduling, batching, sampling and kernels with their
  existing owners. The cache leases slots; the adapter creates and manages
  the concrete model state.
- Start with 2,048-token chunks, private partial tails, existing eviction
  ranks and a bounded checkpoint count. Execution-slot allocations remain
  separate from retained-cache budgets.
- Use one disk writer per directory and bounded transfers in both directions.
  Keep disk compaction and optional paged KV deferred until measured results
  justify them. Cross-slot restores initially copy into contiguous sessions.

## Decisions to record before the affected milestone

| Decision | Required before | Deliverable |
| --- | --- | --- |
| Implementation base revision and exact baseline environment | M1 | Revision, toolchain, model/sidecar identities, options, harness revision and cache history |
| Complete state inventory for each initial model/mode | M3 | Component table, valid positions, byte layouts and every mutation path |
| Adapter operations, ownership, completion and failure semantics | M1 for legacy state; M3 for components | Concrete C++ interface and lifecycle invariants |
| Output/numerical equality requirements after restore | M3 | Model/mode acceptance contract consistent with existing quality tests |
| Initial checkpoint selection and density policy | M3 | Stable/learned/message/grid boundaries, priorities and record/byte limits |
| Legacy disk-file handling and format versioning | M4 | Explicit read/migration or isolated-version policy, compatibility and cleanup rules |
| Qualification workloads and resource settings | M1, extended at M3/M4 | Focused suites, context depths, concurrency, RAM/disk/staging budgets and retained artifacts |

Do not block package extraction on disk-format or future paging decisions.
Resolve each decision before code that depends on it changes behavior. Record
the chosen policy and its rationale here or in the linked architecture docs.

## M0: establish contracts and the baseline

- [ ] Select and record the implementation base; keep documentation-only
  commits distinguishable from measured binaries.
- [ ] Inspect existing `ContinuationCache`, `TextModelRunner`, the disk store
  and each model's capture/restore implementation. Identify dependencies that
  must move out of serving code without introducing a second cache.
- [ ] Specify the legacy adapter interface used during extraction: state
  creation/reset/destruction, capture, restore, persistence, compatibility,
  resource claims, cancellation and transfer completion.
- [ ] Define slot-lease, checkpoint, reservation, reader-pin and completion
  lifetimes, including shutdown and failed-operation cleanup.
- [ ] Establish matched baseline evidence for affected workflows. Keep the
  existing numerical quality checks and standard speed benchmark. Retain
  actual loaded modes, draft execution, cache work and per-request/phase
  timings rather than relying on configured mode names.

**Exit gate:** the ownership/API specification is concrete enough to implement
and the baseline can be reproduced with the same inputs and cache history.

## M1: extract the common package without changing behavior

- [ ] Add the CMake-owned library and public contracts under `src/cache/`.
  Serving and adapters depend on it; it does not depend on HTTP,
  `TextModelRunner`, model headers or model names.
- [ ] Move common slot lifecycle, lookup, checkpoint policy, budgets, events
  and persistence infrastructure into the package. Remove the disk store's
  direct dependency on the serving runner interface.
- [ ] Wrap existing full snapshots in legacy adapters. Preserve lookup rules,
  byte charging, record limits, disk format, source-slot preferences and live
  continuation fast paths for this milestone.
- [ ] Give every existing model family an explicit capability declaration.
  Preserve currently supported reuse; unsupported continuation remains an
  ordinary execution path. Do not force image/video denoising into a KV API.
- [ ] Update existing CPU tests and add dependency/contract checks where
  useful. Use a fake adapter for deterministic ownership and failure cases.

**Exit gate:** affected CPU and real-model workflows retain their existing
behavior, expected reuse and timing gates. The common package builds without
model-specific dependencies. No storage saving is claimed from extraction.

## M2: Phase 0 improvements, independently reviewable

- [ ] Consult disk when it has a longer compatible prefix than the available
  RAM checkpoint; validate stable-prefix fallback requirements before use.
- [ ] Stream legacy full-snapshot writes and restores within the staging
  budget. Include queued captures, active transfers and scratch buffers in
  reservations; do not materialize an entire deep snapshot as a fallback.
- [ ] Replace Flash-Next's full-payload admission charging with exact unique
  byte accounting and reserved spills. Express this through the common
  ownership/accounting contract that the chunk pool will also use.
- [ ] Test a shorter RAM hit beside a longer disk hit, a checkpoint larger
  than staging, transfer cancellation/failure, and concurrent publication.
- [ ] Keep additional eviction-policy changes in separate commits with their
  own expected retention outcomes; preserve the documented rank rules until
  a policy change has been explicitly specified and tested.

**Exit gate:** the focused lookup, bounded-transfer and accounting cases pass
on actual code. Recoverable cache failures preserve inference behavior. These
changes can be reviewed and qualified independently of the new disk format.

## M3: Phase 1 component adapters and the RAM chunk pool

- [ ] Complete the Flash-Next and 27B inventories. For every component record
  kind, shape/precision, byte count, valid range/position, backing owner and
  mutation paths. Include recurrent/convolution state, target and draft KV,
  pooled rows, rings, residual/kept state and required positional metadata.
- [ ] Implement component descriptors and bounded transfer operations.
  Target, draft and pooled lengths are explicit; never derive them all from
  the prompt token count. Separate append-only ranges from private state.
- [ ] Add immutable chunks, provenance, checkpoint manifests, private tails
  and reader pins. Share rows only through an actual inherited computation
  or verified byte identity. Independent equal-token prefills stay separate.
- [ ] Implement transactional admission: reserve fixed state, private tails,
  unique spill capacity and transfer buffers before allocation or mutation;
  convert reservations to backing allocations without double charging.
- [ ] Enforce preservation before overwrite on reset, rewind, restore, slot
  reassignment, destruction and speculative rollback. Cover mutations caused
  by ordinary model execution as well as cache operations.
- [ ] Publish RAM checkpoints only after coherent capture completion. Restore
  every required component into a protected destination; invalidate partial
  destinations on failure. Never mix KV and fixed state from different origins.
- [ ] Implement bounded checkpoint density using the chosen boundary policy,
  rank ordering and record/byte limits. Refused captures remain cache skips.
- [ ] Adapt Flash-Next and 27B, retaining their existing live fast paths.
  Keep unrelated model adapters working through the legacy contract.

**Exit gate:** component round trips and numerical checks pass in AR and
speculative modes. Slot churn and memory-pressure cases prove reservations
cover required spills. Removing one checkpoint cannot free chunks needed by
another checkpoint or reader. Measured RAM/capture savings are reported for
equivalent retained boundaries, including private tails and fixed state.

## M4: Phase 2 chunked disk storage and a unified index

- [ ] Choose and document legacy-file handling before switching writers.
  Version manifests and component layouts; reject incompatible descriptors
  without silently interpreting old snapshot bytes as the new format.
- [ ] Use one compatible-prefix index across RAM and disk. Track component
  availability and restore missing ranges from disk through bounded buffers.
- [ ] Persist fixed state, private tails and missing chunks. Complete and
  fsync dependencies before atomic manifest publication; sync the directory
  as needed to make publication durable. Queued writes remain non-durable.
- [ ] Pin dependencies for writers and readers. Account for staging and
  in-flight work, and clean up reservations after cancellation, failure and
  shutdown. Keep metadata locks out of device copies, disk I/O and fsync.
- [ ] Enforce one writer per directory; implement checksums, corrupt-entry
  rejection, orphan recovery, reference-based retirement and budget eviction.
- [ ] Inject crashes before/after dependency writes and manifest publication.
  Check graceful and abrupt restart, missing/corrupt files, failed writes,
  concurrent restore/eviction and incomplete manifests.
- [ ] Qualify a focused 100k+ checkpoint round trip exceeding the staging
  buffer, using retained agent history. Record peak staging, physical bytes,
  write volume, durable frontier and restore latency. Keep compaction off.

**Exit gate:** every visible durable manifest has valid surviving
dependencies; incomplete work never becomes a restore hit. Deep checkpoints
can be saved and restored within configured limits. Disk savings and restart
benefits are measured on the implemented system, including failure paths.

## M5: qualify the first release and preserve model coverage

- [ ] Run the focused workflow matrix below on candidate production binaries.
  For common behavior changes, use the four profiles and applicable modes
  documented in [functional tests](../../tests/functional/README.md). Narrow
  model-only changes to affected profiles/modes.
- [ ] Check other model families' advertised capabilities and existing
  execution paths after integration. Additional component adapters need their
  own inventories, numerical contracts and focused qualification.
- [ ] Replay affected coding-agent workflows with `pi_agent.py`; retain its
  sessions and inspect actual tool results. Use the normal speculative options
  and verify that drafts actually ran.
- [ ] Compare matched affected timing controls against main. Apply existing
  per-request/phase gates of 5% and 3 ms; fix confirmed regressions before
  publishing. Keep noisy results visibly inconclusive and investigate them.
- [ ] Retain numerical quality checks and the standard speed benchmark.
  Publish observed memory/copy/disk/latency results, residual costs and limits.
- [ ] Update user/developer documentation, event descriptions, persistence
  compatibility policy and this checklist to reflect completed behavior.

**Exit gate:** required correctness, resource accounting, persistence,
compatibility and performance checks pass with retained evidence. Optional
paged KV, compaction and cross-process sharing are separate later projects.

## Focused validation matrix

Select suites from the code diff and milestone; do not run a full model sweep
for every edit. Include `long-context` and `cache` for continuation changes.
Reuse baseline evidence only when revision, toolchain, harness, weights,
settings and prior cache history match. Missing-model skips are not passes.

| Workflow | Required outcome | Existing coverage / extension |
| --- | --- | --- |
| Exact live continuation and unchanged retry | Preserve the live frontier and expected prefill work; retain a valid fallback under pressure | `cache-growth`, `cache-depth`, `long-context` |
| Earlier edits and shortened history | Restore an exact valid boundary; never truncate recurrent state to an arbitrary common prefix | `cache-edits`, `cache-depth` |
| Shared-prefix branches and independent identical prompts | Inherited rows share provenance; independent computations remain separate; learned boundaries advance reuse | `cache-shared-prefix`, `cache-concurrency`, adapter/pool tests |
| More conversations than slots | Preserve required borrowed rows before overwrite; restore coherent state into a different slot | `cache-rotation`, `cache-depth`, pool failure tests |
| Tight budgets and checkpoint eviction | Unique bytes/reservations stay bounded; last-reference retirement releases storage; capture refusal does not fail requests | CPU cache tests, `cache-rotation`, measured resource cases |
| AR and speculative state | Correct target/draft positions, actual draft execution and numerical state after restore/rollback | Model-local snapshot tests, `state-edges`, existing quality tests |
| Image/non-token identity | Compatible boundaries reuse; changed inputs cannot falsely hit | `cache`, affected image suites when adapter supports them |
| RAM/disk selection and restart | Choose the valid longer prefix; restore only durable state; transfer deep snapshots within staging limits | `cache`, disk-spacing tests, deep-history and crash-injection extensions |
| Cancellation, peer progress and shutdown | Release leases/pins/reservations; invalidate partial destinations; avoid holding metadata locks over blocking work | CPU runner/store tests, `cache-concurrency`, fault-injection extensions |

Use the smallest affected CMake/CTest targets during iteration and the
repository's required formatting check before C++ commits. Keep real-model
functional and agent tests outside hosted model CI. Compare correctness on the
candidate; use main for matched timing controls and suspected regressions,
not a duplicate full correctness run.

## Review sequence and completion record

Keep interface/extraction, lookup fixes, bounded legacy transfers, accounting,
RAM pool, each initial model adapter and disk format/store in focused commits
or PRs. Every behavioral change names its affected invariants and validation
evidence. Do not combine kernel paging with the storage redesign.

For each milestone, record the implementation commit, selected tests and
model/mode, commands/environment, artifact locations, measured results and
remaining limitations. Mark a milestone complete only after its exit gate
passes; simulated results and partial test runs do not complete it.

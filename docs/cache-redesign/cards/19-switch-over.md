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
| End-to-end functional and matched timing matrix | Running; no performance qualification yet |
| Pressure, bridge, lifecycle, long Pi replay and standard speed benchmark | Pending |

Raw logs, production binaries, source patches and hashes are retained outside
Git at `/home/mixer/gufo-qualification/pr543-followup`. The current measured
candidate is `candidate-r3-gufo`, SHA-256
`b2b271a70620e91e5d95e78fea8563f78c4f76b9b59796dfcbb0e68991b86eec`.
The core run uses C2, context 32768, 8 GiB cache RAM, 8 GiB disk and 1 GiB
staging, selecting growth, depth, edits, shared-prefix, rotation, concurrency,
state-edges, long-context and cache explicitly for both AR and MTP.

Borrowed-row capture and idle spill are not wired into serving yet. These
checks do not establish reassignment latency, capacity-row qualification or
the final per-request 5% / 3 ms timing gate. The PR remains draft.

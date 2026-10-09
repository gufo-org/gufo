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


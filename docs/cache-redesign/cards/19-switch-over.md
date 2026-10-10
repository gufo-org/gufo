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

### Minimum pass length at capture boundaries

Context: this note comes from an out-of-tree model port that uses exact
recurrent checkpoints.

Some models select kernels by the size of a prefill chunk. One example is a port
of Ling-3.0-flash (bailingmoe3). It has 35 KDA linear-attention layers and 7 MLA
layers. In the port, the model has two numeric families:

- The decode family: a chunk of 1 to 8 tokens uses decode kernels. The kernels
  process each token as a single-token step.
- The prefill family: a chunk of more than 8 tokens uses prefill kernels.

In the port, every split inside one family gives bit-identical results. Between
the two families, the results differ. The difference is about 1e-3 per layer.
The whole-model logits differ by up to about 2e-2 relative L2. The top-1 token is
usually the same. In some cases it differs.

A recurrent checkpoint needs an exact pass end at its boundary (RFC.md:467).
Assume two boundaries are 1 to 8 tokens apart. The tokens between them then run
in the decode family. A cold run of the same prompt runs these tokens in the
prefill family. A boundary that is 1 to 8 tokens before the prompt end causes
the same effect. A cached request and an uncached request with the same prompt
then give different logits. Sometimes they give different greedy tokens.

`PlanCaptures` (`src/cache/retention.cpp:21-84`) does not prevent this
condition. These are its rules at `9024101b`:

- A grid point needs `position - r.reused >= 2048` and
  `r.prompt - position > 128` (lines 66-67). It moves down to a pass end that is
  within 128 tokens and at least 2048 tokens past `r.reused` (lines 62-65).
- A learned point needs `r.common - r.reused >= 512` and
  `r.prompt - r.common > 64` (lines 71-72). It moves down to a planned capture
  that is within 64 tokens. It looks only at planned captures at or below its
  own position (lines 75-79).
- Required boundaries `r.reused`, `r.stable` and `r.prompt` never move
  (lines 54-56, RFC.md:472).
- No rule sets a minimum distance between two boundaries.

The RFC already has one such rule. It skips grid points within 128 tokens of the
prompt end because they split the final pass (RFC.md:474-478). This request
makes that gap per model. It applies the gap between any two boundaries. The
change lands in `PlanCaptures` (card 07) and the `Capabilities` record
(card 02). Card 19 is where capture at policy boundaries is wired in.

This case is reachable with the fake adapter and its default pass plan. Use
`prompt=5000`, `reused=0`, `common=4090` and `learn=true`. The grid is
`(5000 - 1) / 2048 = 2`, so `count = 2`. The grid points are 2048 and 4096.
The learned point 4090 sees only the planned capture 2048. That capture is 2042
tokens below it. The point does not move. The result is `2048 4090 4096 5000`.
The pass from 4090 to 4096 has 6 tokens. A required `r.stable` at 4996 gives a
4-token final pass in the same way.

The RFC lists identical output as a non-goal (RFC.md:90-91): "Guaranteeing
identical sampled output between a fresh computation and a restored prefix
computed with a different floating-point execution shape." This note does not
ask to change that non-goal. It asks for an opt-in capability. The capability
does not promise identical output for every model.

The port enforces one rule itself: `min_checkpoint_tail_tokens`, at least 9
tokens after each planned checkpoint. The port added this limit after a
bit-exact cache test found a 4-token tail defect.

Request: add a per-model capability, with default 0, for example
`min_pass_rows` in the `Capabilities` record. Models that set 0 do not change.
Alternatively, add an adapter query that filters the optional candidates after
`PlanCaptures` builds them. For a value N above 0:

1. Skip or merge an optional boundary that is closer than N tokens to another
   boundary.
2. Skip or merge an optional boundary that is closer than N tokens to the prompt
   end.
3. Prefer to keep a learned point and drop the grid point near it. A grid point
   has no follower that needs its exact position. Drop an optional point that
   is within N tokens of a required boundary.
4. Keep required boundaries exact. The caller that sets `r.stable` is
   responsible for its distance to `r.prompt`.
5. Optionally, let lookup prefer a checkpoint that leaves at least N tokens to
   replay. This relates to card 13.

This agrees with the RFC guidance to avoid small split passes (RFC.md:471-472):
"Prefer candidates aligned with planned prefill passes; coalesce nearby optional
boundaries." A 6-token pass also adds a synchronization point for little work
(RFC.md:468-469).

Benefits:

- Cached and uncached runs stay identical for such models.
- Cache tests can compare a cached run to a cold run bit for bit. In the port,
  this comparison found a snapshot capture race and the tail defect.

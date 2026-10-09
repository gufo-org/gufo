# 09 · Idle spill and reassignment

**Milestone:** Common package · **Depends on:** 06, 08 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** agreed

## Goal

Preserve borrowed rows in the background while a slot is idle, so that giving
the slot to another request rarely waits for copies.

## Scope

- **Worker:**
  - picks idle slots whose borrowed rows are still needed by retained
    checkpoints;
  - pins the source rows and slot generation;
  - copies bounded pieces into committed backing on a leased stream;
  - publishes each completed chunk under the mutation guard.
- **Request arriving mid-spill:**
  - the worker stops at a piece boundary;
  - the foreground completes the remaining required ranges, or retires
    eligible checkpoints;
  - a partially copied chunk is never published.
- **Slot choice:** prefer a slot whose preservation is complete when assigning
  an unrelated request.
- **Metrics:** residual spill wait at reassignment, idle versus foreground
  bytes, copy time.
- No eager copy of every possible checkpoint when a slot goes idle.

## Not in this PR

Real-request latency gates (card 19 runs them).

## Test first

With the fake adapter and controllable completions:

- idle, zero-idle and mid-spill reassignment all end with correct checkpoints;
- cancellation during spill releases pins;
- no partial chunk is ever visible.

## Step baseline

With card 08's backing: time to preserve 1 GiB and 6 GiB of borrowed rows
when idle, and the residual wait for zero-idle and mid-spill reassignment.

## Done when

- [ ] Tests above pass, including under ThreadSanitizer.

## Review focus

- Worker priority: how it yields to model work on the same device.
- What "eligible to retire" means when a foreground request is waiting.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [SGLang HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design): write-through and write-back between device, host and storage tiers.
- [Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) (USENIX ATC 2024): asynchronous saving that does not block inference.
- [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/abs/2312.05516) (EuroSys 2025): moving conversation state between GPU and CPU tiers.

## RFC

[Idle spill and independent transfer streams](../RFC.md#idle-spill-and-independent-transfer-streams)

## Review notes


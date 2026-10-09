# 08 · Committed backing pool and transfer streams

**Milestone:** Common package · **Depends on:** 03, D4 · **Size:** M ·
**Affects:** startup memory once wired in (card 19) · **Status:** agreed

## Goal

Provide the two device-side primitives the cache needs: backing memory whose
pages are committed before requests run, and one stream per in-flight transfer.

## Scope

- **Backing pool** in `src/core/hip/`:
  - committed at initialization, sized per D4;
  - fixed block size;
  - released blocks are reused;
  - never refilled while peers execute.
- **Stream and event pool:** each in-flight transfer leases its own stream and
  completion event, and returns it only after completion.
- **Bounded piece copy helper:** device to host and back, in pieces of a
  configured size.
- Implement the abstract `Stream` and `Completion` from card 02.
- Ledger integration: pool blocks appear as "committed backing" in card 03.

## Not in this PR

Using the pool for spill (card 09) or disk (card 12).

## Test first

- GPU test: after initialization, the request path performs no page-commit
  allocation. Count allocations through the pool.
- Microbenchmark: two concurrent copies on independent streams versus one
  shared stream.

## Step baseline

- Startup commit time per GiB (estimate: about 0.34 s for 8 GiB at ~25 GB/s).
- Copy throughput by piece size, device to host and back.
- Concurrent copies on independent streams versus one shared stream.

## Done when

- [ ] GPU test passes in `gpu-test`.

## Review focus

- Allocation call: pinned host memory versus other options on unified memory.
- Pool block size versus chunk size per model.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) (ROCm): pinned host memory, streams and events.
- [Flash-Next prompt checkpoints](../../models/qwen3.8-flash-next/PROMPT-CHECKPOINTS.md) (in this repository): the allocator decision and stall observations from the Flash-Next prompt checkpoint work.
- [Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) (USENIX ATC 2024): overlapping KV transfers with computation.

## RFC

[Budget accounting and preservation before mutation](../RFC.md#budget-accounting-and-preservation-before-mutation) ·
[Idle spill and independent transfer streams](../RFC.md#idle-spill-and-independent-transfer-streams)

## Review notes


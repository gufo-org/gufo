# 03 · Resource ledger and reservations

**Milestone:** Common package · **Depends on:** 02 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** agreed

## Goal

Account for every byte the cache is responsible for, charging each physical
allocation exactly once, with reservations that either commit or roll back.

## Scope

- Categories:
  - committed backing, split into free, assigned to a spill reservation, and
    materialized chunk;
  - checkpoint private state and tails;
  - metadata;
  - transfer staging;
  - bytes pinned by queued persistence jobs.
- Reservation lifecycle: reserve, then convert (reservation becomes owned
  backing without a second charge) or release. Failure leaves the ledger
  unchanged.
- Peak tracking and a snapshot of all categories for the per-request
  reporting added in card 19.
- One short lock. No I/O and no device waits while holding it.
- Fault injection at every reserve and convert step.

## Not in this PR

Physical allocation (card 08). Chunk ownership (card 04). Budget-driven
eviction decisions (card 07).

## Test first

Property tests over random operation sequences:

- the total never exceeds the budget;
- shared references never double-charge;
- an injected failure leaves every category unchanged;
- teardown returns every category to zero.

## Step baseline

Cost of reserve, convert and release, and lock hold time, under contention
from eight threads (microbenchmark).

## Done when

- [ ] Tests above pass under the `gpu-test` assertions build and with sanitizers
  in the CPU presets.

## Review focus

- Do the categories match the RFC's budget model, including "borrowed rows
  cannot count as free"?
- Is it clear who calls reserve and who calls convert?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, SOSP 2023): block reference counting and copy-on-write accounting.
- [vLLM automatic prefix caching, v0.22.1](https://docs.vllm.ai/en/v0.22.1/design/prefix_caching/): block pool, free queue and reference counts for shared prefix blocks.

## RFC

[Budget accounting and preservation before mutation](../RFC.md#budget-accounting-and-preservation-before-mutation)

## Review notes


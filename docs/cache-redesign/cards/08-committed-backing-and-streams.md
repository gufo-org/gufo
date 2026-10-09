# 08 · Committed backing pool and transfer streams

**Milestone:** Common package · **Depends on:** 03, D4 · **Size:** M ·
**Affects:** startup memory once wired in (card 19) · **Status:** done

## Goal

Provide the two device-side primitives the cache needs: backing memory whose
pages are committed before requests run, and one stream per in-flight transfer.

## Scope

- **Backing pool** in `src/core/hip/`:
  - committed at initialization, sized per D4;
  - fixed block size;
  - released blocks are reused;
  - never refilled while peers execute.
  - checkpoint-private state and private tails use the same precommitted pool:
    card 03's `ReserveBacking` moves the entire block's charge from free backing
    into private state/tail (including capacity padding), then `Convert` marks
    successful capture. The categories partition physical bytes; do not add
    private pool bytes again to the original backing charge. Reserve separate
    metadata headroom within D4's RAM budget before sizing the pool.
  - block assignment may race: `bad_alloc` from `ReserveBacking` means retry
    another committed block or decline capture. No request-time private-state,
    tail or spill page commitment is allowed when the pool is exhausted.
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

- [x] GPU test passes in `gpu-test`.

## Implementation record

[`committed_backing.hpp`](../../../src/core/hip/committed_backing.hpp) and
[`committed_backing.cpp`](../../../src/core/hip/committed_backing.cpp) provide
`CommittedBackingPool`. Initialization reserves metadata headroom and every
whole block through the ledger, allocates one coherent pinned host slab with
`hipHostMalloc`, and zeroes all bytes before converting the backing charges.
Blocks are fixed page multiples, chosen by the caller to fit the model's chunk
and private-state layouts. The pool uses its explicit share of D4's RAM budget;
remaining metadata for indexes/checkpoints is budgeted separately. Pool metadata
requires at least the state size plus 1 KiB per block for ledger/borrower/pin
bookkeeping and allocator headroom; actual retained owner counts remain bounded
by the store. Unused budget below one block is left uncharged.

`TryAcquire` tries each committed block with `ReserveBacking`, including after
assignment races or `bad_alloc`, and declines capture when none is available.
`BackingBlock::Convert` marks successful capture. Its whole capacity moves
between free backing, assigned spill, private state and private tail, without a
second physical charge. Persistence pins retain both the ledger ownership and
the slab. Blocks/pins may outlive the facade; move assignment releases old
ownership before old storage. Failed initialization frees payload pages before
rolling back their reservations. There is no refill path.

Pinned host memory supplies a stable host buffer for later disk I/O and HIP
asynchronous copies on UMA. Explicit `hipHostMallocCoherent` avoids an environment
variable changing coherence; startup `memset` commits every page. This follows
the [HIP memory contract](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management/host_memory.html).
Anonymous mapped memory plus registration and device-only allocation remain
unselected alternatives, rather than request-time fallbacks. This PR contains
original code and adds no third-party implementation.

[`transfer_pool.hpp`](../../../src/core/hip/transfer_pool.hpp) and
[`transfer_pool.cpp`](../../../src/core/hip/transfer_pool.cpp) implement cache
`Stream` and `CompletionSignal` with precreated nonblocking streams and
timing-disabled events. Each lease accepts one completion. A stream and its
completion share ownership; the pair returns only after both handles release
it and work drains. Destroying either handle settles its work. Query/wait/copy/
record failures remain latched, and failed pairs are quarantined. Pool facade
destruction does not invalidate outstanding leases. `Copy` queues bounded
device-to-host or host-to-device pieces and records one final event. Callers
must freeze sources and retain buffers through completion; sources on other
streams require caller ordering before submission.

The HIP implementation is a separate `gufo_cache_hip` library. The common cache
package keeps its CPU-only dependency boundary. These primitives remain
disconnected from production serving until card 19; spill/disk use belongs to
cards 09/12.

## Results

Measured on 2026-10-09, Linux 7.2.9, Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`), based on main `51ceefc1`, with the unchanged `flake.lock` toolchain:
GCC 15.3.0, HIP Clang 22.0.0, ROCm 7.2.3 (HIP runtime 70253211).

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target cache_backing_test cache_ledger_test --parallel 4
nix develop -c ctest --preset gpu-full -R '^cache_(backing|ledger|boundary)_test$' --output-on-failure
nix develop -c cmake --preset release -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset release --target cache_backing_bench --parallel 4
nix develop -c build/release/cache_backing_bench
```

All three focused tests pass. [`backing_test.cpp`](../../../tests/cache/backing_test.cpp)
intercepts HIP allocation, registration, stream/event creation and copy calls:
request borrowing, conversion and transfer submission allocate no payload
pages or HIP streams/events. Host ownership bookkeeping is allowed. It checks
accounting, cancellation, initialization rollback, competing borrowers, pin
lifetimes after facade destruction, unaligned guarded byte-exact round trips,
piece bounds, repeat waits, lease reuse, and injected copy/event failures.
A deliberately blocked stream stays unready while an independent peer finishes.
Host allocation failure during lease creation cannot synchronize or release an
unclaimed stream slot. Ledger and package-boundary checks also pass in the
CPU-only preset.

The production release benchmark [`backing_bench.cpp`](../../../tests/cache/backing_bench.cpp)
committed **1 GiB in 55.45 ms** with one payload allocation, including admission,
allocation, zeroing and conversion. HIP runtime initialization precedes the
timer. This is 55.45 ms/GiB; the RFC's ~0.34 s/8 GiB is an estimate from a
different allocation experiment, not a matched control. Larger pool sizes have
not been measured here.

Two disjoint 128 MiB regions, each with a 13-byte partial tail, exceed MALL
capacity. The table reports median wall time for both copies and their completion
waits, with one warmup and seven retained samples per case; verification runs
outside the timer on both regions every iteration. GB/s uses total payload
bytes and decimal GB. The shared-stream control queues both regions FIFO;
independent streams each receive one region. Acquisition and device allocation
are outside the copy timer. Every raw sample is retained in
[`08-gfx1151.csv`](../measurements/08-gfx1151.csv).

| Piece | D2H shared ms / GB/s | D2H independent ms / GB/s | H2D shared ms / GB/s | H2D independent ms / GB/s |
| --- | ---: | ---: | ---: | ---: |
| 64 KiB | 22.792 / 11.778 | 22.744 / 11.802 | 21.977 / 12.215 | 21.998 / 12.203 |
| 1 MiB | 5.561 / 48.275 | 5.431 / 49.427 | 4.389 / 61.167 | 4.364 / 61.510 |
| 4 MiB | 4.768 / 56.298 | 4.707 / 57.028 | 3.522 / 76.227 | 3.496 / 76.786 |
| 16 MiB | 3.853 / 69.665 | 3.882 / 69.148 | 3.299 / 81.361 | 3.273 / 82.021 |

Independent streams have similar aggregate throughput here; this is a step
baseline, not evidence of a serving speedup. Their benefit is independent
completion without shared FIFO blocking. Small pieces impose substantial
submission overhead. Card 12 will select its transfer-buffer default, and card
19 will qualify startup and request timing when these primitives are wired in.

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

The first fresh-context PR review reproduced a lease-creation allocation race:
cleanup of an unclaimed lease could unlock and release a slot already acquired
by another thread. Leases now claim their slot only after both host allocations
succeed, and unclaimed destruction performs no HIP or pool operation. A focused
allocation-failure regression check covers that path.

# 09 · Idle spill and reassignment

**Milestone:** Common package · **Depends on:** 06, 08 · **Size:** L ·
**Affects:** nothing at runtime until card 19 · **Status:** done

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

## Implementation record

[`slot.hpp`](../../../src/cache/slot.hpp) and
[`slot.cpp`](../../../src/cache/slot.cpp) extend `LeasedSlot` with an optional
`IdleSpillConfig`. Configured slots own one stop-aware worker. Successful lease
release wakes it; it selects only still-referenced borrowed ranges, pins their
source and generation, leases a fresh stream, and settles one bounded piece.
Unreferenced registry entries are discarded. The worker sleeps when no ranges
remain and performs no capture or backing allocation.

Each range tracks completed rows internally. Readers continue using pinned
source rows until the entire range finishes. Reservation conversion and backing
publication share the existing row mutex with reader/persistence admission;
no partial chunk or tail becomes readable. Copies, stream admission, callbacks
and completion waits run outside metadata locks. Submitted pieces drain before
their stream and source pins release, including failures and cancellation.

Acquisition claims the slot before waiting for an active piece, preventing the
worker from submitting another. Live reuse keeps unfinished ranges borrowed;
subsequent guarded mutation completes only the required ranges. Cold
reassignment finishes remaining pieces and old source-reader waits through a
cancellable preparation step before nonthrowing reset, without recopying
finished pieces. Cancelled or failed preparation releases its claim without
resetting the source, including while waiting for stream admission.
`StopIdleSpill` drains the current piece and leaves unfinished ranges available
for foreground preservation; facade destruction stops the worker while an
outstanding lease can still retain the slot.

`AcquireSlot` tries preservation-complete idle slots first, then all available
slots, tolerating busy races without swallowing other errors. `SpillMetrics`
records successful idle/foreground bytes and copy wall time, stream admission
wait, idle failures, and residual cold-reassignment wait (active-piece drain,
remaining preservation and old reader waits; slot recreation is excluded).

Scheduler admission is explicit through `device_idle`, checked before every
idle piece. A false result yields to peer model work; a submitted piece is
already bounded and must drain. Stream exhaustion defers idle work and retries
foreground admission; permanent stream failure must throw through the supplied
factory. Copy/conversion/admission exceptions suspend the idle worker until the
next lease release. The foreground keeps card 06's contracts: ordinary mutation
and cold reassignment refuse on failure, while actual nonthrowing release
retires failed ranges and drains
their old source readers. Merely waiting for reassignment does **not** make any
checkpoint eligible for policy eviction. This card adds no policy retirement;
all still-needed ranges are preserved during successful reassignment. Card 19
will connect scheduler admission and qualify peer/request latency.

[`BackingBlock::Borrow`](../../../src/core/hip/committed_backing.cpp) consumes an
unconverted assigned block into a row handle, retaining the precommitted slab
and its whole-capacity charge. Its owner aliases the payload address for reads;
capacity padding remains charged. The pool facade may be destroyed while those
rows are retained. No third-party implementation was reused.

## Validation

[`idle_spill_test.cpp`](../../../tests/cache/idle_spill_test.cpp) uses the fake
adapter and permit-controlled completions for idle, zero-idle and mid-spill
reassignment, byte-exact retained chunks/tails, cancellation during worker drain
and foreground stream admission (including cold reassignment), cold
cancellation during copying and source-reader waits, source-reader draining, conversion/copy
failures, scheduler priority, stream-pool exhaustion, slot preference and
concurrent persistence/read admission during publication. Each fixture's final
ledger total must return to zero. The test belongs to the hosted PR target.
[`backing_test.cpp`](../../../tests/cache/backing_test.cpp) additionally verifies
borrowed payloads survive pool-facade destruction, consume no payload/stream
allocation during borrowing, and keep padding charged.

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset pr --parallel 4
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer --target cache_idle_spill_test cache_slot_test --parallel 4
nix develop -c ctest --preset cpu-sanitizer -R '^cache_(idle_spill|slot)_test$' --output-on-failure
nix develop -c cmake -S . -B build/cache-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=OFF -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
nix develop -c cmake --build build/cache-tsan --target cache_idle_spill_test cache_slot_test --parallel 4
nix develop -c ctest --test-dir build/cache-tsan -R '^cache_(idle_spill|slot)_test$' --output-on-failure
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target cache_backing_test --parallel 4
nix develop -c ctest --preset gpu-full -R '^cache_backing_test$' --output-on-failure
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
```

All 50 hosted PR checks pass. Final focused CPU cache checks pass (11 tests),
as do the slot/idle-spill ASan/UBSan and ThreadSanitizer tests, the GPU backing
test, the shared formatting check (537 C++ files), and documentation validation
(86 Markdown files, 456 local links/anchors before adding measurement links).
This validates the inactive common package, without claiming model numerics or
serving latency qualification.

## Results

Measured 2026-10-09 on Linux 7.2.9, Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`), implementation `d201c87f` based on main `e6c850d3`, with unchanged
`flake.lock`: GCC 15.3.0, HIP Clang 22.0.0, ROCm 7.2.3, HIP runtime 70253211.

```sh
nix develop -c cmake --preset release -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset release --target cache_idle_spill_bench --parallel 4
nix develop -c build/release/cache_idle_spill_bench
```

[`idle_spill_bench.cpp`](../../../tests/cache/idle_spill_bench.cpp) runs the
actual worker, mutation guard, committed backing pool and HIP transfer leases
through a byte-row adapter. Source allocation/initialization, backing commitment,
stream creation and borrow registration precede the timer. Blocks are 128 MiB,
pieces 4 MiB, with one stream/event pair returned after each settled completion.
Every iteration checks every byte of every retained block outside timing and
checks the pool still has only its single startup payload allocation.

One warmup and five measured iterations per size/history are retained in
[`09-gfx1151.csv`](../measurements/09-gfx1151.csv); the table reports medians in
milliseconds. The idle total runs from lease commit through fully preserved
reassignment. Residual wait comes from cold acquisition and includes any active
piece drain, remaining copies and old readers; recreation is excluded. The
zero-idle history denies all background work. The mid-spill history stops idle
admission after exactly half the payload settles, then immediately reassigns;
it measures halfway preservation, without artificially blocking DMA. Unit tests
separately control arrivals while a piece's completion is blocked.

| Borrowed rows | Idle total ms | Residual after idle ms | Zero-idle residual ms | Half-spilled residual ms |
| --- | ---: | ---: | ---: | ---: |
| 1 GiB | 17.345 | 0.003 | 17.271 | 7.556 |
| 6 GiB | 96.663 | 0.005 | 94.403 | 46.443 |

Idle/foreground bytes are respectively 1/0, 0/1 and 0.5/0.5 GiB for the 1 GiB
histories, and 6/0, 0/6 and 3/3 GiB for 6 GiB. Median successful idle copy wall
time is 16.971/94.605 ms for fully idle 1/6 GiB; zero-idle foreground copy time
is 17.185/93.892 ms. Stream admission contributes 0.014–0.031 ms (1 GiB) and
0.080–0.122 ms (6 GiB) across these histories. Raw data includes both copy-time
categories, total and residual timing, and stream wait for every sample.

The earlier preliminary run is retained in
[`09-initial-gfx1151.csv`](../measurements/09-initial-gfx1151.csv). It overlapped
a hosted build and preceded the final no-work sleep refinement; it is excluded
from the table. The final run had no concurrent compilation. These are step
baselines rather than matched serving speedups; card 19 still must gate real
requests and decoding peers under idle, zero-idle and active-copy reassignment.

## Done when

- [x] Tests above pass, including under ThreadSanitizer.

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


### Independent review follow-up

Round one, on `9b85d77e`, reproduced cancellation being ignored during cold
stream admission: acquisition stayed blocked after stop, then invalidated the
old frontier once a stream arrived. Cold acquisition now prepares all remaining
ranges with cancellable preservation before entering nonthrowing invalidation.
Cancellation/refusal drops the claim while retaining the original generation
and execution. Actual lease teardown still drains and may retire failed ranges.
Regression tests cancel cold admission, an in-flight foreground piece, and old
source-reader waits, checking that the original live frontier can be reacquired.

The finding and response are recorded in
[PR #505](https://github.com/gufo-org/gufo/pull/505).

The fixed standalone cancellation reproducer returns promptly after stop,
returns no cancelled lease, and confirms the original live frontier survives.
The expanded slot/idle-spill checks pass in CPU, ASan/UBSan and ThreadSanitizer;
GPU backing, formatting and docs checks pass too.

Remeasured the same release benchmark on `3c3c3e3b`, with the same device,
toolchain, sizes, piece/block configuration, histories and five samples after
one warmup. Every byte passes verification. Results in milliseconds:

| Borrowed rows | Idle total ms | Residual after idle ms | Zero-idle residual ms | Half-spilled residual ms |
| --- | ---: | ---: | ---: | ---: |
| 1 GiB | 17.647 | 0.002 | 17.025 | 7.610 |
| 6 GiB | 95.427 | 0.007 | 93.696 | 45.955 |

Raw samples are retained in
[`09-review1-gfx1151.csv`](../measurements/09-review1-gfx1151.csv), alongside
the original baseline. This confirms the successful copy/reassignment paths
after the cancellation fix; it is not a matched model latency comparison.

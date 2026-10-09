# 03 · Resource ledger and reservations

**Milestone:** Common package · **Depends on:** 02 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** done

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
- Peak tracking, reporting windows and a snapshot of all categories for the per-request
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

### Implementation record

[`src/cache/ledger.hpp`](../../../src/cache/ledger.hpp) defines admission
reservations, shared charges, persistence pins, limits and snapshots. The
implementation uses one mutex for counter/state changes; host bookkeeping is
allocated before acquiring it. No payload allocation, I/O, callbacks or device
waits happen under the lock. It is part of `gufo_cache`, with the reviewed
`Threads::Threads` dependency, and is not linked into serving.

`ResourceLedger::Reserve` admits a new allocation against total and RAM or
staging limits. Its category includes pending bytes; `reserved_bytes` reports
that subset separately. The allocator/capture owner calls `Convert` only after
its work succeeds. Conversion consumes the reservation without another charge;
destruction cancels it. Failure preserves the reservation and the complete
resource snapshot, including peaks. Callers charge allocation capacity rather
than logical payload size, and admit owner/index metadata explicitly. This is
an accounting contract, not a physical allocator or process-memory tracker;
live slots, weights and execution scratch remain outside the retained-cache
ledger, as specified in the RFC.

The pool first reserves and converts a free backing allocation after committing
its pages. `ResourceCharge::ReserveBacking` reclassifies an entire free block
as `kBackingAssigned` for borrowed spill rows, `kPrivateState`, or `kPrivateTail`.
The owner converts after capture/preservation succeeds: spill becomes
materialized; private state/tails stay in their respective categories. No
transition increases total or RAM bytes. Private captures and tails therefore
work even when committed blocks fill the RAM budget, with no request-path page
commitment and no second charge for carved-out private bytes. Card 08 supplies
physical blocks and reserves metadata headroom before sizing its pool.

The pool may race/retry candidates: `bad_alloc` on an already assigned block
means try another committed block or decline capture. Invalid handles instead
throw `logic_error`. The last reservation/owner/pin returns the whole block to
free while the pool retains its original charge. Without that pool reference,
the final reference releases the physical charge. Assignment and admission are
distinct: `kBackingAssigned` is borrowed backing's physical category, whereas
`reserved_bytes` is the not-yet-converted subset in any admission category.

Charges and pins can be shared without duplicating accounting. Persistence
pins report unique pinned bytes as an independently bounded overlay, rather
than adding them to the physical total. Only retained payload is eligible:
assigned spill, materialized chunks and private state/tails. Staging and metadata
retain their charges directly for job lifetimes and stay bounded by their own
admission categories; they do not consume the retained-payload pin limit.
Queued jobs may pin assigned spill or borrowed-tail backing before
materialization; cancellation cannot recycle that backing until all pins
release it. New allocation reservations and incomplete private-state captures
cannot be pinned before conversion. Tokens account for lifetimes; their owners must also retain the
actual buffers and finish accesses before releasing them. Shared ledger state
survives destruction of the public ledger object until the final token ends.

`SnapshotAndResetPeaks` atomically reports the completed global window and
starts the next window's peaks at current charges/pins. Carried-over allocations
are therefore present in the new peak. Overlapping request attribution remains
card 19's responsibility; resetting this global window alone does not attribute
physical bytes to one request. Category and step counts derive from enum
sentinels.

`cache_ledger_test` is included in the hosted `check-pr` target. Its independent
operation model covers 64 deterministic seeds × 2,000 operations, including
shared charges/pins, pending and converted backing, cancellation, injected
reserve/convert/pin failures, and zero current bytes after teardown. Additional
cases cover separate RAM/staging/pin limits, SIZE_MAX overflow, moves replacing
live reservations/charges, lifetime after ledger destruction, eight-thread
shared pin activity, and races in which exactly one of eight contenders can
admit an allocation or assign one backing block.

Validation on Linux x86-64 Strix Halo, based on `2dd8b111`, with the flake.lock
toolchain (GCC 15.3.0, CMake 4.3.4, Python 3.14.6):

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-test -R '^cache_(ledger|adapter(_lifecycle)?|boundary)_test$' --output-on-failure
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(ledger|adapter(_lifecycle)?|boundary)_test$' --output-on-failure
cmake --preset gpu-test
cmake --build --preset gpu-test --target cache_ledger_test --parallel 4
ctest --preset gpu-full -R '^cache_(ledger|boundary)_test$' --output-on-failure
cmake --preset release -DGUFO_BUILD_TOOLS=ON
cmake --build --preset release --target gufo_cache cache_ledger_bench --parallel 4
build/release/src/cache/cache_ledger_bench
```

The four focused cache tests passed in CPU and ASan/UBSan builds; the ledger
and boundary tests passed in `gpu-test`. The last additions to the randomized
pin model and admission races passed in all three presets. The release library
build, shared formatting check, focused `clang-tidy`, documentation links and
diff whitespace check passed. Nix development shells used `inputsFrom` the
production package plus clang-tools and Python; the GPU/tools configuration
also required the pinned rocprofiler-sdk. The initial test-first compile failed
on the missing ledger header. No model or device-transfer quality claim is made.

The [initial baseline](../measurements/03-resource-ledger.json), before the review
follow-up below, has source and
binary hashes, machine/build details and every measured result. One unpinned
run measured 20,000 cycles per thread, with 64-byte charges. External operation
timers include host bookkeeping and mutex contention. The library uses release
optimization; internal lock clocks are disabled for operation measurements.

| Operation | Metadata allocation: eight-thread mean / p95 | Spill block: eight-thread mean / p95 | Instrumented mean lock hold: metadata / spill |
| --- | --- | --- | --- |
| Reserve | 907 / 2,965 ns | 935 / 3,367 ns | 44 / 42 ns |
| Convert | 712 / 2,715 ns | 836 / 3,236 ns | 39 / 39 ns |
| Release | 736 / 2,746 ns | 799 / 3,166 ns | 35 / 37 ns |

Lock hold time was measured in separate instrumented runs. It includes the
ending clock read but excludes the subsequent statistics update and unlock;
wait time includes the acquiring clock read. Instrumentation increased the
eight-thread metadata cycle wall time from 47.87 to 92.07 ms and spill cycles
from 52.85 to 82.36 ms. Those instrumented operation timings are retained but
are not the uninstrumented baseline. Single-thread results and maximum hold
times are also retained. These are step measurements, without an inference
timing gate or a comparable RFC microbenchmark.

### Review follow-up

The committed-private-memory path, assignment contention, pin scope, assertion
builds, naming/counts and peak-window findings are addressed above and in card
08. New focused tests cover private/tail assignment when committed backing
fills the entire RAM budget, reserve/convert rollback in each category,
pending borrowed-tail pins, rejection of staging/metadata pins, and resetting
peaks with live allocations and pins. The independent random model now assigns
blocks to all three destinations.

The four cache tests pass in CPU and ASan/UBSan presets; ledger and boundary
tests pass in `gpu-test`. Compile commands confirm the ledger itself receives
`-UNDEBUG` in all three test presets, after `-DNDEBUG` where present. Release
retains `-DNDEBUG` without `-UNDEBUG`. Formatting, focused static analysis,
documentation links and whitespace checks pass.

The [review baseline](../measurements/03-resource-ledger-review.json) uses the
release library and benchmark, built with the same pinned toolchain and
20,000-cycle methodology. The original run is retained separately. These
single-run CPU measurements describe the revised implementation; they are not
a matched performance-gain comparison.

| Allocation/category | Eight-thread mean reserve / convert / release (ns) | Instrumented mean lock hold for each operation (ns) |
| --- | --- | --- |
| Metadata admission | 905 / 730 / 739 | 56 / 48 / 44 |
| Spill block assignment | 782 / 742 / 707 | 40 / 37 / 36 |
| Private-state block assignment | 943 / 863 / 871 | 43 / 38 / 37 |
| Private-tail block assignment | 877 / 810 / 814 | 47 / 41 / 40 |

Raw records include single-thread cases, percentiles, lock wait/max hold and
instrumentation overhead. Eight-thread metadata wall time was 48.43 ms without
internal clocks versus 113.39 ms with them; spill was 45.91 versus 80.27 ms,
private state 54.77 versus 81.56 ms, and tails 50.89 versus 90.77 ms.

## Done when

- [x] Tests above pass under the `gpu-test` assertions build and with sanitizers
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

# 06 · Slot leases and the mutation guard

**Milestone:** Common package · **Depends on:** 03, 04 · **Size:** L (split
candidate: leases, then guard) · **Affects:** nothing at runtime ·
**Status:** done

## Goal

Lease execution slots to requests, and make every overwrite of borrowed rows
safe and transactional.

## Scope

- **Lease lifecycle:** acquire, reuse the live frontier or restore, execute,
  publish checkpoints or commit, release; invalidate on any failure.
- **Slot generation counter:** bumped on every reassignment and reset, and
  checked by any pin on borrowed rows.
- **Mutation guard** (`MutationGuard::BeforeOverwrite`), in this order:
  1. find the checkpoints and readers that need the rows;
  2. preserve them into reserved backing, or retire eligible checkpoints;
  3. wait for transfers and reader pins on those rows;
  4. allow the overwrite.

  Preservation is synchronous here; card 09 moves most of it to idle time.
- **Paths covered:** prefill, decode, rewind, speculative rollback, reset,
  reassignment and destruction.
- **Cancellation:** at every step it releases leases, pins and reservations.
- **Lock scope:** metadata changes under a short lock; copies and waits
  outside it.

## Not in this PR

Device copies (the fake adapter copies host memory). Idle spill (card 09).

## Test first

With the fake adapter:

- each mutation path preserves needed rows before overwrite;
- an injected allocation or copy failure before mutation leaves published
  checkpoints valid and nothing overwritten;
- cancellation at each step leaks nothing (ledger back to zero);
- an adapter path that forgets the guard is detected in assertion builds by a
  generation mismatch on read.

## Step baseline

Guard cost per mutation when no checkpoint needs the rows (the common fast
path), and lease acquire and release cost.

## Implementation record

[`src/cache/slot.hpp`](../../../src/cache/slot.hpp) and
[`slot.cpp`](../../../src/cache/slot.cpp) add exclusive move-only leases,
generation-checked live reuse, reset/reassignment and failure cleanup. Commit
releases a successful lease; abandoning or cancelling it invalidates the live
frontier. A busy failed slot is destroyed, draining adapter transfers, and is
recreated lazily. The lease retains the slot/guard if its facade is destroyed.
Adapter and preservation-stream lifetimes must cover all outstanding leases.
Adapters call `AfterReset` exactly once after successful invalidation, including
empty resets. Slot IDs must be unique within the runner.

Borrowed checkpoint payloads now require registration through a lease. The
unmanaged description constructor is removed. Each retained row range owns an
assigned backing reservation and a stable synchronized handle. Preservation
copies into this existing backing, settles the completion, converts its ledger
reservation and redirects new readers to backing. Existing source reader and
persistence handles still prevent overwrite until they release. Row pins must
cover source transfers through completion on every stream. Chunk pin copies
share their accounting and retain source protection even during preservation.
Private tails use the same guard; private recurrent state remains committed.

Copies and reader waits run outside slot and ledger locks. Cancellation before
mutation refuses the overwrite and unwinds leases/pins/reservations. Completed
preservations may remain committed after a later failure; no source rows are
overwritten and all previously published checkpoints remain readable. Release
cannot fail: if preservation fails it retires the affected range, drains old
source pins, and allows reclamation. Checkpoints referencing retired rows become
invalid and are excluded from prefix selection; inherited history refuses them.
Retention policy decides eligible retirement in card 07. This card preserves
all needed rows during ordinary mutation and retires only on release failure.

[`tests/cache/slot_test.cpp`](../../../tests/cache/slot_test.cpp) covers prefill,
decode, shorter restore/rewind/rollback, reset, reassignment, destruction, real
fake-adapter restore/continuation, allocation/copy/conversion failure, cancellation
before borrowing/during copying/while waiting, retirement, stale generations,
source reads on another delayed stream and eight-thread lease contention. A
subprocess verifies the assertion when an adapter bumps generation without
preserving borrowers. Teardown checks the ledger returns to zero. The test is
part of the hosted PR target. This package remains inactive in production;
model adapters, device copies and idle spill remain later cards.

## Results

Initial step baseline on Linux 7.2.9 x86-64, AMD Ryzen AI MAX+ 395, using
flake.lock-pinned GCC 15.3.0 and the `cpu-test` RelWithDebInfo build with tools:

| Measurement (100,000 iterations) | ns/op |
| --- | ---: |
| Live lease acquire, commit and release | 14.27 |
| Empty guard via fake adapter | 4.12 |
| No-op guard via identical fake adapter fixture | 2.54 |
| Incremental empty guard overhead | 1.57 |

These are single CPU microbenchmark observations from
[`slot_bench.cpp`](../../../tests/cache/slot_bench.cpp), without borrowers or
contention; they do not qualify GPU transfers, model numerics or serving speed.
The baseline predates review follow-ups; remeasurement is recorded below if the
implementation changes. Commands and ignored artifact directories:

```sh
nix develop -c cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset cpu-test --target cache_slot_test cache_slot_bench cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test cache_prefix_index_test --parallel 4
nix develop -c ctest --preset cpu-test -R '^cache_(slot|checkpoint|adapter|adapter_lifecycle|ledger|prefix_index|boundary)_test$' --output-on-failure
build/cpu-test/tests/cache/cache_slot_bench
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer --target cache_slot_test cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test cache_prefix_index_test --parallel 4
nix develop -c ctest --preset cpu-sanitizer -R '^cache_(slot|checkpoint|adapter|adapter_lifecycle|ledger|prefix_index|boundary)_test$' --output-on-failure
nix develop -c cmake -S . -B build/cache-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=OFF -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
nix develop -c cmake --build build/cache-tsan --target cache_slot_test --parallel 4
nix develop -c ctest --test-dir build/cache-tsan -R '^cache_slot_test$' --output-on-failure
```

All 48 hosted PR checks and the production release cache library passed. The
shared formatting and tracked-source documentation checks passed (86 Markdown
files, 438 local links/anchors); the workspace-wide documentation scan encounters
unrelated links in ignored `hrx-system` files. All seven focused CPU checks
passed in ordinary and ASan/UBSan builds; the slot
check including concurrent leases passed under ThreadSanitizer. Per-test reports
remain under each build directory's `Testing/Temporary/`. This is common-package
validation; no serving behavior or model quality qualification is claimed.

## Done when

- [x] Tests above pass, including under ThreadSanitizer for concurrent leases.

## Review focus

- The transactional sequence: what happens if step 3 waits on a pin held by a
  slow restore?
- What is the right split if this is too big for one review?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): how checkpoints are created and restored around slot reuse.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, SOSP 2023): copy-on-write before overwriting shared blocks.

## RFC

[Budget accounting and preservation before mutation](../RFC.md#budget-accounting-and-preservation-before-mutation) ·
[Capture and borrowing](../RFC.md#capture-and-borrowing)

## Review notes

### Independent review follow-up

Round one identified a race between reservation conversion and persistence-pin
admission. An independent ThreadSanitizer reproducer reported concurrent access
to the reservation's shared-pointer handle. Conversion and publication now share
the row mutex with persistence admission; copies and completion waits remain
outside it. Added contention tests use the two-step chunk persistence-pin path,
copy existing reader/persistence pins during mutation, and fail each preservation
conversion after earlier ranges have succeeded. Row metadata also charges the
borrowed-handle object. Ordinary, ASan/UBSan and ThreadSanitizer slot checks pass.

Rounds two and three independently reviewed `db7d31c2` without conversation
history and reported no additional actionable findings. Round three repeated
focused CPU, ASan/UBSan and ThreadSanitizer checks. The final local hosted PR
suite passed all 48 checks; the release cache library and tracked-source
documentation check also passed. Repeating the same microbenchmark on `db7d31c2`
recorded 14.21 ns/op for live acquire/commit/release, 3.78 ns/op for the empty
guard and 2.51 ns/op for the no-op fixture (1.27 ns/op incremental). Both baseline
runs are retained above; neither is an inference performance qualification.

Review comments and the fix response are retained in
[PR #501](https://github.com/gufo-org/gufo/pull/501). The independent race
reproducer rebuilt against the fixed ThreadSanitizer library passed with
`TSAN_OPTIONS=halt_on_error=1`, reporting zero persistence-admission errors.
Its source/report remain in `/tmp/gufo-pr501-review-persistence.cpp` and
`/tmp/gufo-pr501-review-persistence-tsan.log`; the final local hosted report is
`/tmp/gufo-card06-final-pr.log`.

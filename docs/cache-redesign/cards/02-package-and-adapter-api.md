# 02 · `src/cache/` package, adapter API and fake adapter

**Milestone:** Common package · **Depends on:** — · **Size:** L (contracts and fake adapter) ·
**Affects:** nothing at runtime (new library, not linked into serving yet) ·
**Status:** done

## Goal

Create the common package with its public contracts and a deterministic fake
adapter. No cache behavior yet. This card settles the RFC's open item
"concrete adapter API, completion signals and mutation guards".

## Scope

- CMake static library `gufo_cache` under `src/cache/`, with its tests
  registered in the CPU test presets.
- Headers: identifiers, component descriptors and positions, capabilities,
  compatibility and input identity (port the `ContinuationInputPrefix`
  semantics from the current cache), adapter, completion, transfer stream,
  events.
- Fake adapter:
  - two append-row components that end at different positions (target and
    draft);
  - private "recurrent" state built as a hash chain of tokens, so a wrong
    restore is detectable;
  - injectable allocation and transfer failures, and controllable completion
    delays.
- Boundary check: rejects direct/transitive serving, model or HIP includes and
  unreviewed CMake link dependencies. It runs with the `pr` preset.

## API sketch

This summarizes the public contract. The headers in `src/cache/` define the
complete lifetime and failure rules.

```cpp
namespace gufo::cache {

enum class ComponentKind : std::uint8_t {
  kAppendRows,    // immutable once below the frontier; shareable
  kPrivateState,  // copied whole at each checkpoint
};

struct ComponentDescriptor {
  ComponentId id;                 // stable within one layout version
  std::uint32_t layout_version;
  ComponentKind kind;
  std::size_t row_bytes;          // kAppendRows
  Rows rows_per_chunk;            // kAppendRows: sharing granularity
  std::size_t state_bytes;        // kPrivateState
};

struct ComponentPosition {
  ComponentId id;
  Rows valid_rows;                // target and draft KV may differ
};

// The adapter calls the ordinary guard on the host before queuing writes.
// Preservation completes across all streams before it returns. Release cannot
// refuse: it preserves or retires borrowers synchronously without throwing.
class MutationGuard {
 public:
  virtual void BeforeOverwrite(ComponentId, Rows first, Rows end) = 0;
  virtual void BeforeRelease(ComponentId, Rows first, Rows end) noexcept = 0;
};

class Adapter {
 public:
  virtual Capabilities GetCapabilities() const = 0;
  virtual std::span<const ComponentDescriptor> Components() const = 0;
  virtual Identity CompatibilityIdentity() const = 0;

  virtual std::unique_ptr<Slot> CreateSlot(MutationGuard&) = 0;
  virtual std::vector<ComponentPosition> Positions(const Slot&) const = 0;
  virtual void BeginRestore(Slot&, std::span<const ComponentPosition>) = 0;

  // Bounded transfers. The cache owns buffers and streams; the adapter only
  // knows the layout. Each call returns a completion the cache waits on.
  virtual Completion CapturePrivate(const Slot&, ComponentId,
                                    std::span<std::byte> dst, Stream&) = 0;
  virtual Completion CopyRowsOut(const Slot&, ComponentId, Rows first,
                                 Rows end, std::span<std::byte> dst,
                                 Stream&) = 0;
  virtual Completion CopyRowsIn(Slot&, ComponentId, Rows first, Rows end,
                                std::span<const std::byte> src, Stream&) = 0;
  virtual Completion LoadPrivate(Slot&, ComponentId,
                                 std::span<const std::byte> src, Stream&) = 0;

  // Failed loads or validation latch until successful Invalidate, even if a
  // completion's failed result is discarded. No content hashing here.
  // Outside an active restore, throw logic_error without changing the slot.
  virtual bool Validate(Slot&, std::span<const ComponentPosition>) = 0;
  // Refuses pending transfers; success leaves a valid empty slot for prefill.
  virtual bool Invalidate(Slot&) noexcept = 0;
};

}  // namespace gufo::cache
```

Later cards extend it: card 07 adds a prefill pass-plan query, card 10 adds a
stable private-state encoding version.

## Mapping from today's runner interface

| `TextModelRunner` today | New cache |
| --- | --- |
| `CreateState` | `Adapter::CreateSlot`; serving keeps owning execution |
| `Snapshot`, `SnapshotPayloadBytes`, `SnapshotForPersistence` | `Positions` + `CapturePrivate`; rows stay borrowed until mutation |
| `RestoreOrFork` | Cache-driven `BeginRestore` + `CopyRowsIn` + `LoadPrivate` + `Validate` |
| `PersistentSnapshotPayloadBytes`, `SerializePersistentSnapshot`, `StreamPersistentSnapshot` | The store streams chunks and private state (cards 10–12) |
| `RestorePersistentSnapshot` | The same restore path, fed by streamed pieces |
| `CheckpointPosition` | `Positions` |
| `PrefillCheckpointBytes`, `PrefillThrough` | Stay in the runner; the in-pass boundary calls `CapturePrivate` (card 07) |
| `PreparePrefixReuse` | Runner framing preparation stays; exact component shortening belongs to `BeginRestore` |
| `ContinuationSnapshot::PrefersState` | Removed; provenance tells the cache which slot still holds rows |

## Not in this PR

Ledger, chunks, lookup, leases and policy (cards 03–07). No serving change.

## Test first

Fake-adapter contract tests: positions per component, capture and restore
round trip through the adapter directly, injected failure surfaces as an error.

## Step baseline

None: this card adds contracts only, no executable behavior worth measuring.

### Implementation record

`src/cache/` now builds as the independent static library `gufo_cache` in the
CPU and gfx1151 release presets. Public headers define typed component/slot
identifiers, descriptors, independent component positions, capabilities,
compatibility and supplemental input identity, slots, mutation guards,
bounded transfers, completions, streams and events. It has no link to serving.

The cache supplies a guard that outlives model-owned slots. Ordinary mutation
uses `BeforeOverwrite` on the host and may refuse by throwing; every preservation
copy and reader wait completes across all streams before it returns. Release
uses a separate `BeforeRelease` method that cannot throw or refuse. Real
preservation policy, pins and leases still belong to card 06.

`BeginRestore` prepares a whole restore while the old slot is still readable:
it checks/allocates capacity and guards all replacement ranges before disabling
execution or submitting loads. It handles exact shorter frontiers; the cache
does not truncate recurrent state, and the runner's `PreparePrefixReuse` is not
responsible for truncating restored components. Row pieces can arrive in any
order on different streams after preparation. Overlapping writes require caller
ordering.

`Completion` owns one signal, is move-only and drains outstanding work on
destruction or replacement. `Wait` settles all buffer accesses even on failure
and retains the result. Callers retain slots, streams and immutable source
buffers until completion. Adapters latch every failed load, including submission
failures, independently of the handle. `Validate` checks positions and completed
loads during an active restore; any false result latches failure until successful
`Invalidate`. Repeating validation with other positions cannot recover a failed
slot. Calling `Validate` without an active `BeginRestore`, including a second
call after successful validation, throws `std::logic_error` without changing the
slot or its execution state.

`Invalidate` refuses without changing a slot with pending reads or loads. After
they settle, it releases borrowers, clears failures and leaves a valid empty
slot ready for cold prefill. Destruction drains tracked reads and loads before
release. The fake refuses execution or restore preparation during pending reads,
so a delayed capture retains its original coherent boundary.

Supplemental input identities preserve the current cache's inclusive boundary
semantics: before an appended image, earlier checkpoints keep their original
identity. The explicit `InputIdentity` constructor takes the prompt length and
checks nondecreasing boundaries within it, including the legacy first-match
behavior at equal boundaries. Queries cannot exceed the prompt length. The
unchecked lookup helper is internal. Token-prefix matching arrives with card 05.

The fake adapter is test-only in `tests/cache/`. Its target and draft row arrays
advance independently. Its continued execution depends on both restored rows
and private hash chains; tests compare continued state with controls to detect
corrupted target, draft or private bytes. `Validate` does not inspect their
contents, matching what real adapters can guarantee. Allocation, transfer and
submission failures are injectable; transfer injection is consumed only after
successful submission. Reentrant waits on a running stream callback fail
immediately instead of spinning.
`FakeStream` supplies FIFO ordering and manually delayed completion without
threads, clocks or device dependencies.

Validation on Linux x86-64, based on `bf4f811f`, with the flake.lock-pinned GCC
15.3.0, CMake 4.3.4 and Python 3.14.6. That base's source tree is identical to
merged main `94a4fa8d` (card 01, PR #490), on which this PR is based:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-test -R '^cache_(adapter(_lifecycle)?|boundary)_test$' --output-on-failure
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(adapter(_lifecycle)?|boundary)_test$' --output-on-failure
cmake --preset release
cmake --build --preset release --target gufo_cache --parallel 4
cmake --build --preset pr --parallel 4
```

The focused adapter contracts cover independent positions, bounded and delayed
round trips, continued execution with corrupted checkpoint controls,
missing/duplicate components, latched failures, discarded completion errors,
rejected validation outside a restore without changing rows or execution,
preservation before queueing, cold fallback, exact shorter restores, reversed
piece arrival, pending capture lifetimes, allocation/submission/transfer failure,
reentrant waits and bounded image identity. The boundary check follows project
includes across header/inline/source suffixes, rejects serving/model/HIP/ROCm
dependencies (including system library headers) and checks generated CMake link
metadata. Comment removal preserves quoted literals and ignores raw-string
contents. Only the reviewed CPU
platform target `Threads::Threads` may be linked; arbitrary relay targets cannot
hide transitive model/HIP dependencies. All three tests run in the hosted `pr`
target. The unused event-header test include and redundant library flags were
removed; event contracts remain reserved for later policy/observability cards.

At review revision `93240037`, all 44 hosted PR tests and the release library
build passed. The validation-misuse follow-up passed all three focused tests in
ordinary and ASan/UBSan builds. Its regression test failed before the fix. The
shared formatting check, focused static analysis, documentation link check and
diff whitespace check passed. Nix shells used the flake's pinned inputs: a
CPU-only dependency shell for hosted checks and `inputsFrom` the
production package for release configuration, without building reference tools.

The test-first compilation failed on the missing cache headers before their
implementation. No GPU numerics, model workload or performance qualification
is claimed by these CPU fixtures; those remain with the adapter and switch-over
cards. Build artifacts stay in the ignored `build/` directories.

## Done when

- [x] `gufo_cache` builds in the `cpu-test` and `release` presets.
- [x] The boundary check runs in the `pr` preset.

## Review focus

- **Guard direction.** The cache hands a guard to the slot and the adapter
  calls it. The alternative is the cache asking the adapter "what will the
  next step overwrite?". The guard is harder to forget on rollback paths.
- **What a `Slot` is.** Today the execution state is `QwenTextRunnerState`
  and similar types in the serving backend. The adapter should own the
  model-level session instead (Qwen arena, Flash-Next session), with the
  runner state holding it. Confirm this split.
- **Completion and streams are abstract in `src/cache/`.** HIP implementations
  live in `src/core/hip/` (card 08), so CPU tests run without a GPU.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Unified Radix Cache: one tree for hybrid model prefix caching](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache) (SGLang): validating a candidate boundary against every required state component.
- [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/): how separate state groups with different layouts share one manager.
- [LMCache hybrid model support](https://docs.lmcache.ai/mp/hybrid_models.html): what an external store needs from a hybrid-model adapter.
- [Hybrid Models Meet SGLang: More than Full Attention](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/): recurrent-state checkpoints as part of a prefix cache.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules) ·
[Two-layer code structure](../RFC.md#two-layer-code-structure)

## Review notes

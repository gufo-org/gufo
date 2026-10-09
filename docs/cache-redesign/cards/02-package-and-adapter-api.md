# 02 · `src/cache/` package, adapter API and fake adapter

**Milestone:** Common package · **Depends on:** — · **Size:** L (contracts and fake adapter) ·
**Affects:** nothing at runtime (new library, not linked into serving yet) ·
**Status:** in progress

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
- Boundary check: a test that fails if `src/cache/` includes `cli/`, `models/`
  or `text_model_runner.hpp`. It runs with the `pr` preset.

## API sketch

This is the part to review. Names are placeholders.

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

// Handed to each slot by the cache. The adapter calls it before any
// operation that overwrites append-only rows: prefill, decode, rewind,
// speculative rollback, reset, destruction.
class MutationGuard {
 public:
  virtual void BeforeOverwrite(ComponentId, Rows first, Rows end) = 0;
};

class Adapter {
 public:
  virtual Capabilities capabilities() const = 0;
  virtual std::span<const ComponentDescriptor> Components() const = 0;
  virtual Identity CompatibilityIdentity() const = 0;

  virtual std::unique_ptr<Slot> CreateSlot(MutationGuard&) = 0;
  virtual std::vector<ComponentPosition> Positions(const Slot&) const = 0;

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

  // After every component is loaded. A false result invalidates the slot.
  virtual bool Validate(Slot&, std::span<const ComponentPosition>) = 0;
  virtual void Invalidate(Slot&) noexcept = 0;
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
| `RestoreOrFork` | Cache-driven `CopyRowsIn` + `LoadPrivate` + `Validate` |
| `PersistentSnapshotPayloadBytes`, `SerializePersistentSnapshot`, `StreamPersistentSnapshot` | The store streams chunks and private state (cards 10–12) |
| `RestorePersistentSnapshot` | The same restore path, fed by streamed pieces |
| `CheckpointPosition` | `Positions` |
| `PrefillCheckpointBytes`, `PrefillThrough` | Stay in the runner; the in-pass boundary calls `CapturePrivate` (card 07) |
| `PreparePrefixReuse` | Stays in the runner; audited per adapter |
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

The concrete API retains the sketch's guard direction: the cache supplies a
guard, and model-owned slots call it before changing or releasing append rows.
Guard refusal may throw before ordinary mutation. Invalidation and destruction
require nonthrowing preservation/retirement; the guard outlives its slots.
Real preservation, pins and leases still belong to card 06.

`Completion` owns one signal, is move-only and drains outstanding work on
destruction or replacement. `Wait` settles all buffer accesses even on failure
and retains the result. Callers retain slots, streams and immutable source
buffers until completion. Loads invalidate execution immediately; successful
`Validate` requires every component at its own recorded position. All slot
transfers must settle before validation, invalidation or destruction.

Supplemental input identities preserve the current cache's inclusive boundary
semantics: before an appended image, earlier checkpoints keep their original
identity. `InputIdentity` owns and checks ordered boundaries. Token-prefix
matching remains separate and arrives with card 05.

The fake adapter is test-only in `tests/cache/`. Its target and draft row arrays
advance independently, and its private hash-chain state detects mismatched
target or draft rows. It injects slot allocation and transfer failures.
`FakeStream` supplies FIFO ordering and manually delayed completion without
threads, clocks or device dependencies.

Validation on Linux x86-64, based on `bf4f811f`, with the flake.lock-pinned GCC
15.3.0, CMake 4.3.4 and Python 3.14.6. That base's source tree is identical to
merged main `94a4fa8d` (card 01, PR #490), on which this PR is based:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target cache_adapter_test --parallel 4
ctest --preset cpu-test -R '^cache_(adapter|boundary)_test$' --output-on-failure
cmake --preset release
cmake --build --preset release --target gufo_cache --parallel 4
cmake --build --preset pr --parallel 4
```

The focused adapter contracts cover independent positions, bounded and delayed
round trips, continued execution after restore, corrupted target/draft rows,
missing/duplicate components, pending or failed loads, preservation refusal,
allocation/transfer failure, completion lifetime/moves and image boundaries.
The boundary check rejects serving/model includes, including nested paths and
continued preprocessor lines, and tests its own rejection cases. Both new tests
are registered in the hosted `pr` target.

Both focused tests and all 43 hosted PR tests passed; the release library
build passed. The shared formatting check, focused static analysis, documentation
link check and diff whitespace check passed. Nix shells used the flake's pinned
inputs: a CPU-only dependency shell for hosted checks and `inputsFrom` the
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

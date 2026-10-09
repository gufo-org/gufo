# 02 · `src/cache/` package, adapter API and fake adapter

**Milestone:** Common package · **Depends on:** — · **Size:** M ·
**Affects:** nothing at runtime (new library, not linked into serving yet) ·
**Status:** agreed

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

## Done when

- [ ] `gufo_cache` builds in the `cpu-test` and `release` presets.
- [ ] The boundary check runs in the `pr` preset.

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


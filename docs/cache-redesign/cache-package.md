# Cache package and model adapters

The proposed hybrid cache should be a dedicated CMake library under
`src/cache/`, with adapters under each model's own directory. The common
package owns retention and storage policy; each adapter describes its model's
state and performs the operations needed to capture or restore it. This is an
implementation plan for the [hybrid design](hybrid-design.md), not an
implemented API.

## Ownership boundary

| Common cache package | Model adapter |
| --- | --- |
| Slot leasing, availability, cancellation cleanup and reuse policy | Creates, resets and destroys the actual model execution state |
| Exact-prefix lookup, input identity matching and checkpoint selection | Supplies model compatibility identity and boundary-specific input identity |
| Checkpoint admission, density, eviction and RAM/disk budgets | Describes valid state components and their actual byte sizes |
| Chunk ownership, provenance, reference counts and reserved spill capacity | Exposes append-only ranges and captures private mutable state |
| Bounded persistence queues, manifests, checksums, atomic publication and recovery | Encodes and decodes model-specific state through bounded transfers |
| Transfer reservations, completion tracking and cache metrics | Performs device-specific copies and restores a coherent execution state |

The serving scheduler continues to decide which requests execute and when to
run prefill or decode. The cache leases their execution slots and chooses
retained state. It does not take over batching, sampling or inference kernels.
The adapter manages the model's concrete allocations and device operations.

The package must not depend on HTTP, `TextModelRunner`, model headers or model
names. Serving and model adapters depend on its public contracts. Reusable
HIP transfer primitives can remain in `src/core/hip/`; model-specific layouts
and numerical contracts remain with their model.

## Component contract

An opaque full snapshot is insufficient for shared storage. An adapter
describes the components of a coherent checkpoint at a requested boundary:

- A stable component identifier, layout/version, element representation and
  byte-size calculation.
- Its kind: append-only rows that may be chunked, or private mutable state
  that must be copied for this checkpoint. Rings and recurrent buffers belong
  to the latter unless the adapter provides a stronger contract.
- Its valid range and position. Target KV, draft KV and pooled rows can have
  different lengths; the common package must not infer all lengths from the
  prompt token count.
- An opaque storage handle and bounded transfer operations, with an explicit
  completion and lifetime contract. A handle is not permission to read an
  arbitrary device pointer from the common package.

The generic pool groups shareable rows into chunks and records which
computation produced them. It owns inherited provenance across captures and
restores; adapters must not establish sharing merely from equal tokens. Fixed
state and private partial tails remain associated with their exact checkpoint.
Restoring that checkpoint restores every required component together.

The adapter also supplies capabilities and compatibility identity. Identity
covers the model/weights, state ABI, precision, tokenizer/template, context
policy and relevant adapters or speculative configuration. Supplemental input
identity covers images and other non-token inputs at the appropriate prefix
boundaries. A cache instance is bound to its compatible model descriptor;
matching tokens do not permit reuse across incompatible models or layouts.

## Required lifecycle contracts

1. **Before mutation.** Every operation that may overwrite borrowed rows,
   including rewind, reset, slot reassignment and destruction, must pass
   through the preservation contract. The adapter identifies the affected
   ranges; the cache reserves backing bytes and preserves required rows, or
   drops eligible checkpoints and their references before mutation proceeds.
   Existing spill reservations and active reader pins must remain valid.
2. **Capture completion.** The adapter establishes a coherent model boundary
   and freezes required mutable state. Borrowed source ranges stay protected
   until transfer completion. RAM readers see the checkpoint only once it is
   coherent and ready; a queued disk write is not durable publication.
3. **Restore completion.** The cache selects and pins an exact compatible
   checkpoint, leases a destination slot and protects that slot's borrowed
   rows before overwrite. The adapter loads all components and validates
   their positions. The slot becomes executable only after restore finishes.
   On partial failure or cancellation it is invalidated before reuse.
4. **Persistence and retirement.** The worker streams immutable components
   within the staging budget, durably writes dependencies before publishing
   the manifest, and releases pins/reservations when work completes or fails.
   Eviction frees a physical chunk only after its last owner and reader leave.

These contracts must cover ordinary prefill/decode and speculative rollback,
not just cache-triggered restores. Borrowed rows cannot silently become
mutable through another executor path. A lifetime guard or equivalent
ownership mechanism should enforce the rule rather than an optional callback.

Cache metadata locks must not be held across model execution, device copies,
filesystem writes or fsync. The adapter provides completion signals for the
affected state; other slots can continue executing. An admission or capture
failure skips the cache optimisation without failing inference. A failed
restore invalidates its partial destination and permits a cold retry before
generation resumes.

## Module layout and model coverage

The intended layout is:

```text
src/cache/                              # common CMake library and contracts
src/models/qwen/cache_adapter.*         # Qwen state layouts, including 27B
src/models/qwen38_flash_next/cache_adapter.*
src/models/deepseek_v4_flash/cache_adapter.*
```

Use one adapter implementation per state layout or model family, with
configuration for variants that share that layout. Adding a model should not
require a model-name switch inside `src/cache/`. The loader/runner constructs
the matching adapter and passes it to the common package; this does not
require dynamically loaded plugins.

Every existing model family needs an explicit capability declaration. During
migration, preserve each model's existing advertised capabilities. Families
without reusable continuation state declare it unsupported and use ordinary
execution; image/video denoising state must not be forced into an
autoregressive KV contract. Audio or multimodal adapters expose continuation
only when they can supply coherent state and complete input identity.

## Migration and validation

The existing [ContinuationCache](../../src/cli/serve/continuation_cache.hpp)
already separates policy from opaque model state, and
[TextModelRunner](../../src/cli/serve/text_model_runner.hpp) supplies capture,
restore and persistence operations. The
[disk store](../../src/cli/serve/continuation_disk_store.hpp) still depends on
the runner interface. Evolve these seams instead of building a second cache
alongside them:

1. Extract the shared lifecycle/policy contracts into the CMake package.
   Adapt the existing full-snapshot implementations and serving integration
   without changing reuse, budgets or disk semantics. Keep the inexpensive
   Phase 0 fixes independently reviewable.
2. In Phase 1, add component-aware adapters for Flash-Next and 27B, then move
   shared chunk ownership, exact reservations and checkpoint accounting into
   the common pool. Preserve coherent fixed state and existing live fast paths.
3. In Phase 2, persist the same component/chunk representation with one index
   for RAM and disk. Version the format and compatibility descriptor; define
   explicit handling of legacy full-snapshot files rather than interpreting
   them as chunk manifests. Keep compaction disabled initially as proposed.
4. Migrate other supported layouts through the same contract. Keep optional
   paged KV in Phase 3; extracting a package must not require kernel paging.

CPU tests of the common package can use a small fake adapter to exercise slot
churn, reservations, sharing, reader pins, cancellation, failed transfers and
crash publication. Model-local tests must verify actual component boundaries,
target/draft positions, restored numerical state and persistence compatibility.
Replay affected long-context, continuation, branch and restart workflows on
real models, retaining per-request cache work and latency measurements. The
package extraction itself must preserve behavior before storage changes are
evaluated.

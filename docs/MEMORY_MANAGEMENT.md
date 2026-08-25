# Unified Memory Management

Status: design draft, 2026-08-11

## Purpose

Strix Halo exposes a large physically unified LPDDR5X memory system to the CPU,
GPU, and NPU. Physical unification does not imply that every allocation has the
same virtual address, access permissions, cache behavior, residency, or
performance on every agent.

One broker in `strix` owns global memory policy. HIP, XDNA2, model
implementations, KV cache, graphs, persistence, and future modalities request
memory from that broker instead of maintaining unrelated capacity estimates.

## Goals

- Keep one resident quantized weight copy when CPU, GPU, and NPU can consume it
  efficiently and coherently.
- Prevent page faults and allocation from entering timed inference paths.
- Protect the OS, desktop, driver recovery, and request rollback reserves.
- Make every duplicate, staging buffer, graph, KV page, and NPU context visible
  in accounting.
- Degrade from shared GPU/NPU execution to explicit copies or GPU-only serving
  without corrupting state.
- Reject or queue work before an allocation failure becomes a partially
  committed request.

## Allocation Broker

The broker exposes backend-neutral handles:

```cpp
enum class MemoryClass : uint32_t {
    ImmutableWeight,
    BackendProgram,
    PersistentState,
    KvPage,
    ProvisionalKv,
    GraphStable,
    SharedActivation,
    TransientScratch,
    IoStaging,
    HostMetadata,
};

struct AllocationHandle {
    AllocationId id;
    MemoryClass memory_class;
    uint64_t size;
    uint64_t alignment;
    AgentMask visible_to;
    CachePolicy cache_policy;
    ResidencyPolicy residency;
    Generation generation;
};
```

Backends import an `AllocationHandle` and create their own typed view. An
arbitrary host or HIP pointer is never passed to the NPU merely because the
machine uses unified memory.

The broker records:

- Allocation origin and owning memory pool.
- CPU, GPU, and NPU access permissions.
- Virtual addresses or imported handles per agent.
- Cache and coherency policy.
- Current and required residency.
- Last writer and completion token.
- Logical owner, reference count, and generation.
- Pinned, movable, evictable, or reconstructable state.
- Current committed and peak bytes by memory class.

## Interoperability Tiers

The API evidence and validation matrix for GPU/NPU sharing are recorded in
[XDNA2 GPU/NPU Interoperability Research](NPU_RESEARCH.md).

Runtime probing selects one tier for each allocation class.

### Tier A: Direct shared allocation

One allocation is legally and efficiently accessible by CPU, GPU, and NPU.
Explicit producer completion plus required cache maintenance establishes
visibility without copying payload bytes.

The first candidate path is an XRT-owned BO exported as a DRM PRIME/dma-buf FD
and imported into HIP external memory. The exact `amdxdna`/XRT to gfx1151/HIP
combination is a measured platform capability, not an assumed API guarantee.

Tier A is the required mode for:

- One-copy shared weights.
- GPU/NPU split execution of one operator.
- Zero-copy request-stage handoff.

Tier A is enabled only after probes validate access, explicit ordering, cache
visibility, sustained concurrency, and measured bandwidth. Immutable
simultaneous reads may be promoted separately. Concurrent writes, including
disjoint ranges, remain disabled until a dedicated safety gate proves their
whole-BO fencing and cache behavior.

### Tier B: Shared source with backend views

One canonical allocation or file mapping exists, but one backend requires a
losslessly repacked or copied resident view.

The duplicate:

- Is derived without changing quantized values.
- Has an explicit owner and checksum relationship.
- Counts against the global budget.
- Is created at model load or controlled warmup, not during inference.
- Is evicted before canonical weights.

GPU/NPU split execution is allowed only when the participating buffers satisfy
the route's visibility contract.

### Tier C: Isolated backend allocations

GPU and NPU require separate buffers and explicit asynchronous copies.

In this tier:

- Copies appear as scheduler work and in route costs.
- No operator is advertised as zero-copy.
- Latency-critical decode defaults to the GPU.
- The NPU may still run independent request stages or permanently owned model
  regions when memory permits.

### Tier D: GPU-only

The NPU is unavailable, incompatible, unstable, or not beneficial. The server
remains ready with GPU-only capabilities when the selected model supports it.

## Global Budget

At startup, determine:

```text
physical memory
- current OS and desktop use
- configured OS safety reserve
- driver and backend recovery reserve
- model weights
- backend programs and contexts
- graph-stable allocations
- persistent model state
- minimum KV service reserve
- persistence staging reserve
= discretionary request capacity
```

Do not derive capacity from physical RAM alone. The broker periodically samples
system pressure and backend-reported allocation state.

The default safety reserve is configurable but cannot be reduced below a
platform-tested minimum without an unsafe-development flag.

## NPU Context Accounting

The `amdxdna` driver may create workload contexts, assign a subset of AIE
columns, and allocate driver-managed host buffers. These bytes and resources
must be included even when they are not returned by the Strix allocator.

Track:

- Workload context count.
- Assigned and requested AIE columns.
- Driver internal, external, and shared allocation statistics.
- Overlay and instruction-buffer cost.
- Queue and command-buffer memory.
- Firmware and context recreation after suspend or reset.

Route tables are keyed by the columns actually available to the workload, not
only the physical 4 x 8 array.

## Weight Loading

The load sequence is:

1. Memory-map and validate the model index and manifests.
2. Reserve the complete destination budget transactionally.
3. Allocate immutable weight planes with final alignment.
4. Read and verify bounded shard regions.
5. Copy or map them into the selected interoperability tier.
6. Prefault or explicitly touch pages required by the serving path.
7. Create optional backend views.
8. Run checksums, kernel smoke tests, and model warmup.
9. Publish the model alias only after every required allocation is ready.

Source mappings are released when they no longer improve startup, reload, or
recovery behavior. Avoid retaining a second accidental copy through the page
cache without measuring its cost.

Weights are immutable after publication.

## Residency and Page Faults

The runtime must measure and control first-touch and migration behavior:

- Timed inference does not depend on demand paging a model for the first time.
- Weight and graph pages are warmed before readiness.
- KV and scratch pools are committed before admission.
- Host metadata remains CPU-local where device visibility is unnecessary.
- `mlock`, huge pages, managed memory, and explicit host registration are
  opt-in strategies promoted only by measured end-to-end results.
- A technique that improves a microbenchmark but increases system pressure or
  harms NPU/GPU sharing is rejected.

## KV and Request Reservations

Admission produces a reservation record containing:

```text
minimum committed pages
maximum requested pages
provisional speculative pages
model-specific recurrent state
graph slot
scratch ceiling
```

The server supports one of two explicit policies:

- Guaranteed: reserve enough capacity for the declared maximum request before
  admitting it.
- Controlled overcommit: admit against a configured overcommit ratio with
  defined pause, eviction, or failure policy.

Controlled overcommit is never implicit. The response metadata and operational
metrics expose which policy is active.

No model transition starts unless all memory needed to commit that transition
is already reserved.

## Scratch and Graph Memory

- Scratch arenas are sized by retained shape profiles.
- Each dispatch receives a bounded slice with a generation-tagged lifetime.
- HIP graph pointers remain stable for the graph lifetime.
- NPU command, instruction, and shared-activation buffers remain stable until
  the completion token retires.
- Shape growth that exceeds a scratch class falls back or queues; it does not
  allocate from a device callback.

## Pressure States

The broker publishes:

```text
NORMAL
CONSTRAINED
CRITICAL
RECOVERY
```

Under `CONSTRAINED`:

- Stop creating optional backend repacks.
- Reduce prefill and speculative concurrency.
- Evict unreferenced prefix-cache entries.
- Delay background dump, restore, image, and video work.

Under `CRITICAL`:

- Stop new admissions.
- Reclaim cancelled and completed state.
- Release reconstructable caches and inactive backend programs.
- Preserve active canonical request state and the recovery reserve.

Under `RECOVERY`:

- Quarantine allocations associated with an incomplete backend reset.
- Recreate backend contexts and imported views.
- Revalidate shared-buffer visibility before restoring heterogeneous routes.

## Multi-Model Residency

One executable may support multiple model kinds, but not every configured model
must be resident simultaneously.

The model manager:

- Computes the full load budget before changing residency.
- Drains or rejects requests for an alias being replaced.
- Loads a candidate model into unpublished state.
- Atomically switches the alias after validation.
- Keeps the previous model only when the rollback budget was reserved.
- Never unloads weights referenced by active sessions or snapshots being
  created.

Model eviction is an explicit operational policy, not an incidental result of
allocation failure.

## Future Modalities

Audio, image, and video implementations use the same broker for reservations
and pressure feedback. Their internal formats remain implementation-specific.

Large diffusion or video scratch reservations:

- Must be known before a generation step begins.
- Default to batch or background service classes.
- Yield optional caches before active LLM KV.
- Cannot force the LLM backend to relocate stable graph or weight pointers.

## Metrics

Expose current and peak bytes by:

- Memory class.
- Model alias and model kind.
- Canonical versus duplicate backend view.
- Active, shared-prefix, provisional, and cold KV.
- GPU, NPU, driver-internal, host, and file-backed ownership.

Also expose:

- Reservation failures.
- Pressure-state transitions.
- Page faults during ready-state inference.
- Backend import and shared-visibility failures.
- Bytes copied because Tier A was unavailable.
- Model load, warmup, and alias-swap memory peaks.

## Tests

- CPU/GPU/NPU shared-allocation visibility and ordering.
- Disjoint concurrent GPU/NPU writes.
- Explicit-copy fallback accounting.
- Startup capacity calculation with mocked OS and driver pressure.
- No timed-path allocation or first-touch page fault.
- Guaranteed and controlled-overcommit admission behavior.
- Model load rollback after every allocation stage.
- Alias swap with active and draining requests.
- Pressure-state reclamation ordering.
- Backend reset and suspend/resume allocation reconstruction.
- Allocation generation and stale-handle rejection.
- Integer-overflow and malformed-size rejection.
- Long-running allocation/reclamation returns to baseline.

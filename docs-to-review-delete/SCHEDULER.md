# Continuous Batching and Request Scheduling

Status: design draft, 2026-08-11

## Purpose

The scheduler should provide vLLM-class continuous request handling without
forcing a lone user through a throughput-oriented batch path.

The central rule is:

> Single-request execution is a first-class optimized route. Batching is added
> when other ready work exists; the scheduler does not wait merely to create a
> larger batch.

The scheduler owns admission, request state, model state, KV state,
speculative transactions, device routing, cancellation, and reclamation.

## Request State Machine

```text
RECEIVED
  -> TOKENIZING
  -> ADMITTED
  -> PREFILLING
  -> DECODING
  -> FINISHED

Any active state
  -> CANCELLED
  -> RECLAIMING
  -> CLOSED

Any active state
  -> FAILED
  -> RECLAIMING
  -> CLOSED
```

Speculative work is a transaction nested inside `DECODING`; it does not create
a separate public request lifecycle.

Canonical state changes only at defined commit points.

## Work Classes

Each scheduling tick selects bounded work items:

| Work class | Action | Commit |
| --- | --- | --- |
| `RECLAIM` | Free finished, failed, or cancelled state | Slots and pages released |
| `ADMIT` | Reserve request slot and minimum KV budget | Request becomes runnable |
| `PREFILL_CHUNK` | Process prompt rows | Prompt cursor and KV advance |
| `DECODE_STEP` | Generate one token per selected request | Token and model state advance |
| `VERIFY_STEP` | Verify speculative rows | Provisional target state created |
| `ACCEPT_COMMIT` | Select and commit accepted speculative state | Tokens, KV, and cursors advance |
| `DUMP_SESSION` | Persist a cold session snapshot | Snapshot manifest committed |
| `RESTORE_SESSION` | Load a persisted session | Session becomes runnable |

`RECLAIM` always has priority because stale state reduces capacity.

## Execution Layers

```text
HTTP request adapters
        |
Admission controller
        |
Logical request scheduler
        |
Batch and shape planner
        |
GPU/NPU route planner
        |
Model implementation execution
```

The logical scheduler does not know kernel names. The compiled-in model
implementation publishes capabilities and costs for physical shape buckets.

## Single-Request Fast Lane

When exactly one decode-ready request exists:

- Dispatch immediately.
- Use the dedicated physical width 1 graph or eager route.
- Do not wait for a batching timeout.
- Do not pad to width 2, 4, or 8.
- Do not run NPU tensor splitting unless a retained single-request profile
  proves a latency win.
- Stream the token as soon as sampling commits it.

The direct route remains loaded and graph-ready even under multi-request load.
When load falls back to one request, the next tick returns to it without
recapturing resources.

## Continuous Batching

Logical active requests are packed into supported physical widths:

```text
1, 2, 4, 8, 16, 32, ...
```

The model implementation may expose different widths by operation and backend.

The planner chooses the smallest efficient bucket that can hold the selected
rows. It records:

- Logical rows.
- Physical rows.
- Active mask.
- Padding fraction.
- Per-row request and session IDs.
- Per-row position and KV page table.
- Output destinations.

Requests may join or leave between model steps. They do not join in the middle
of a committed model transition.

## Prefill Chunking

Long prompts are divided into bounded chunks so they cannot monopolize the
GPU, NPU, or memory controller.

Policy inputs:

- Decode-ready request count and latency.
- Prompt length and remaining rows.
- Model prefill shape efficiency.
- NPU availability.
- KV page availability.
- Current memory-bandwidth pressure.

Under interactive load:

- Protect decode deadlines.
- Run short prefill chunks between decode steps.
- Prefer the NPU for compute-efficient chunks when GPU decode remains within
  latency policy.

Under throughput load:

- Combine compatible prompt chunks.
- Increase physical prefill width.
- Use GPU/NPU split routes where retained profiles justify them.

Chunk boundaries are scheduler decisions and must not change model semantics.

## Fairness and Priorities

Each request has:

- Arrival time.
- Service class.
- Deadline or latency target.
- Prompt tokens processed.
- Output tokens generated.
- Accumulated device time.

Initial service classes:

```text
INTERACTIVE
DEFAULT
BATCH
BACKGROUND
```

Use deficit-based service accounting rather than strict FIFO. A long prompt or
long generation cannot indefinitely starve shorter requests.

Interactive decode receives a bounded latency reservation. Background snapshot
and restore work yields under device or memory pressure.

## Admission Control

Admission is based on resources, not only request count:

- Model and adapter residency.
- Minimum KV pages.
- Maximum context and requested output.
- Request slots.
- Graph-stable state slots.
- Speculative draft and provisional-state capacity.
- Estimated prefill work.
- Queue token budget.

Reject with `429` or `503` before allocating partial session state when the
request cannot be admitted within policy.

The scheduler may queue an otherwise valid request if:

- Its deadline permits waiting.
- The queue is within configured token and count bounds.
- No model load or restore failure is already known.

## Shape Planning

The cost model compares:

```text
estimated wall =
    kernel wall
  + padding cost
  + graph/eager overhead
  + synchronization
  + activation conversion
  + device transfer
  + reduction
```

Inputs come from retained benchmarks generated for the exact:

- Model revision.
- Quantized artifact ID.
- Driver and ROCm versions.
- GPU and NPU compiler artifacts.
- Power and memory configuration.

Unknown shapes use conservative routes and are measured offline, not during an
untrusted production request.

## GPU/NPU Scheduling

The planner may issue independent GPU and NPU work during one tick:

```text
GPU: latency-critical decode batch
NPU: prompt prefill or draft batch
```

It may also issue one heterogeneous operator when the model implementation
exposes an approved split route.

GPU decode is protected by:

- Maximum concurrent NPU DMA pressure.
- Inter-token latency feedback.
- A bandwidth saturation state.
- Preemption at scheduler work boundaries.

NPU work is not launched when it is likely to delay a protected GPU deadline
more than the policy permits.

## Speculative Scheduling

Speculative decoding uses a provider-neutral transaction:

```text
PROPOSE
VERIFY
ACCEPT
COMMIT
UPDATE_CURSORS
```

The provider may be:

- Attached MTP/NextN head.
- Dense draft model.
- DSpark support model.
- Future tree or multi-candidate provider.

The target remains authoritative.

The scheduler may combine verification rows from several requests when:

- They use the same model implementation and target quantization.
- Their speculative topology is compatible.
- KV and recurrent state remain isolated.
- The combined batch is expected to improve full-cycle economics.

Acceptance, selected-state copies, KV transactions, and cursor updates occur
before generated tokens are exposed to clients.

## Cancellation

Cancellation is observed at work boundaries:

- Before dispatch.
- After device completion but before commit.
- After commit but before the next work item.

If cancellation arrives during an uninterruptible kernel:

- Let the kernel retire.
- Discard provisional outputs if no commit occurred.
- Reclaim state.
- Stop streaming immediately.

No cancelled request may remain referenced by an active physical row in a later
tick.

## Multi-Model and Future Modalities

The resource manager supports multiple package classes:

```text
LLM
EMBEDDING
STT
TTS
IMAGE
VIDEO
```

Common scheduling concepts:

- Admission.
- Priority and deadline.
- Resource reservations.
- Device work items.
- Cancellation.
- Streaming outputs.

Execution remains implementation-specific. Diffusion steps, audio frames, and
LLM tokens do not share a generic hot operator graph.

Large image or video jobs default to `BATCH` or `BACKGROUND` and yield to
interactive LLM decode unless explicitly configured otherwise.

## Metrics

- Admission and queue delay.
- TTFT and inter-token latency by service class.
- Logical and physical batch widths.
- Padding and inactive-row ratio.
- Work items per scheduling tick.
- Decode deadline misses.
- Prefill chunk sizes.
- GPU and NPU utilization and route decisions.
- Bandwidth-pressure state.
- Cancellation-to-reclamation delay.
- Speculative rows, acceptance, and tokens per cycle.
- Snapshot and restore queue depth.

## Tests

- Single request never waits for batch formation.
- Direct width 1 equals the batched width 1 oracle.
- Requests join and leave without row-state leakage.
- Mixed prompt lengths and shrinking decode batches.
- Long prefill under continuous decode load.
- Fairness and starvation bounds.
- Queue saturation and admission errors.
- Cancellation at every commit boundary.
- GPU/NPU concurrent scheduling and fallback.
- Speculative transaction rollback and commit.
- Graph cache shape selection.
- No memory growth after repeated admission/reclamation.
- HTTP streaming order under scheduler backpressure.

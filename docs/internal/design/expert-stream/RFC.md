# RFC: expert streaming for extensions

Status: an experimental extension API (0.3.38), discussed in #1286. Written
November 2026.
Two extensions use it: DeepSeek V4.1 Flash, where the store came from, and
Gemma 4 26B-A4B, which tests the API on a second model.

This document explains how the expert store works and why its API is shaped
the way it is. The header,
[`store.hpp`](../../../../src/core/hip/expert_stream/store.hpp) (included
as `<gufo/ext/experimental/expert_stream.hpp>`), documents each call; the trace format is in
[`tools/expert_stream/`](../../../../tools/expert_stream/README.md).

## The problem

In a mixture-of-experts (MoE) layer, each token uses only `top_k` of the
layer's **experts**: 6 of 384 in DeepSeek V4.1 Flash, 8 of 128 in Gemma 4
26B-A4B. A model whose experts don't fit in memory can keep its other weights
resident and hold the experts in a bounded cache, reading misses from disk.
That works because routing is skewed: a small working set of experts takes
most routes, and it changes slowly within a conversation.

As of November 2026, DeepSeek V4.1 Flash `UD-IQ2_XXS` has 146 GiB of expert
weights. On a 128 GB Strix Halo it runs with a 100 GiB expert cache, and 98.4%
of a coding-agent session's routes hit it.

The **expert store** (`gufo::expert_stream::Store`) does this one job for any
model:

- it holds a cache of experts in device memory;
- it reads the misses from the model's files;
- it tells the model's kernels where each expert is.

It knows nothing about the model. Tensor names, quantization formats, which
experts matter and how much memory to leave stay with the extension.

## How it works

At load, the extension describes where each expert's bytes are (`Layout`) and
how the cache should behave (`Policy`, which also ranks the experts to cache
first), and creates the store; nothing is cached yet. Once serving has
created the sessions and the prompt cache, it tells the store how much memory
it may keep (`Fit`), and the store fills that once, from the ranking.

For each MoE layer, after the router has chosen the experts:

1. The extension calls `Prepare` with the routes (device `[tokens][top_k]`
   expert ids) and its stream.
2. The store waits until the routes exist, copies them to the host and counts
   the experts the batch uses.
3. Hits stay where they are, in their cache **slot**.
4. Misses are read from the model's files into the **stage**, a device buffer
   of expert rows. Eight reader threads read with direct I/O into pinned
   buffers and upload them on their own streams. Misses that neighbour each
   other in the file merge into reads of up to 32 MiB.
5. The store returns one device **table** per piece, `[experts] → address`:
   the slot for a hit, the stage row for a miss. The extension's kernels read
   each expert through the table.
6. Misses worth keeping are **admitted**: copied from the stage into a slot,
   evicting a colder expert when the cache is full.

A hit costs a table lookup. A miss costs a disk read, an upload and, when
admitted, one device-to-device copy.

Two refinements hide part of the read time:

- **Overlap.** When a batch both hits and misses, `Prepare` can call back the
  extension with separate hit and miss row lists, so it runs the hit rows'
  kernels while the misses are read and the miss rows after. Every row keeps
  its single-launch arithmetic, so outputs are bit-identical. DeepSeek uses
  this for decode and speculative verification.
- **Prefetch.** The extension can predict the next layer's experts, for
  example by running the next layer's router on this layer's input, and pass
  the prediction to the store (`Predict`). A store thread reads the predicted
  misses into a prefetch stage, only while no demand read runs. The next
  `Prepare` of that layer adopts what it actually routes; a wrong prediction
  costs bandwidth, not latency.

### The cache

Quantizations mix formats per layer, so experts of different layers can have
different sizes: DeepSeek's `UD-IQ2_XXS` has seven. Slots are grouped in
**size classes**, one per distinct size, and each class grows in 128 MiB
chunks. An expert takes a slot of its own class or a larger one.

Eviction is by **frequency class**, after llama.cpp's ClassLRU. Each cached
expert counts its uses, decaying with a half-life in tokens of its layer's
clock. Its class is `floor(log2(uses))`, capped, and the victim is the expert
with the largest `age / 2^class`: an expert used twice as often may stay
unused twice as long. On DeepSeek this replaced a decayed score and read 6%
fewer bytes on an agent replay, with identical outputs.

A long prompt touches most experts once and would flush the decode working
set. Once the cache is full, a batch of more than 16 tokens admits only the
experts its last 16 tokens route and those it routes at least
max(4, tokens / 32) times. A prompt chunk also advances its layer's clock by
at most 16 tokens, so one long prompt doesn't age the whole working set.

## The API, and why it is shaped this way

### `Layout`: experts where they already are on disk

A GGUF stores each layer's experts as one tensor per projection, with the
expert dimension outermost: expert `e` is `bytes` long at
`offset + e * bytes`. A `Piece` describes exactly that (`PlatformFile`,
offset, bytes per expert), so the store reads the model's own files without
conversion. An expert has one piece per tensor its kernels take: gate, up and
down for DeepSeek; fused gate-up and down for Gemma.

- **Pieces, not one blob per expert.** Kernels take separate tensors per
  projection, and pieces let the store merge reads of neighbouring experts
  within a projection. A slot holds all of an expert's pieces back to back,
  so an expert is cached whole or not at all.
- **`max_batch_tokens`**: the largest batch one `Prepare` routes; it sizes
  the hit and miss row lists.
- **`table_experts`**: a fixed table width across builds with fewer experts.
  DeepSeek has pruned builds whose kernels index a 384-wide table.
- **`read_margin_bytes`**: kernels that read whole k-iterations run past an
  expert's last row (Gemma's by up to 4 KiB). The store allocates the slots
  and stages, so it has to keep that much readable after each.

### `Prepare` and the tables: no copy for a hit

Kernels read experts through one table per piece instead of a base pointer
and a stride. Gathering the routed experts into a contiguous buffer instead
would copy every hit on every token, which costs more than the reads the
cache saves.

The price is that each extension adapts its expert kernels to take a table.
The extension API has no shared MoE kernel, so the change stays in the
extension. For Gemma it was about 110 lines over its routed GEMV and GEMM
kernels: one optional pointer, the same arithmetic. With resident weights
the kernels run as before.

`Prepare` takes the caller's stream. It reads the routes once the work queued
on that stream before the call has run, and queues its table copies and
admissions there, so kernels queued after it see them. The default, the null
stream, keeps the legacy-stream ordering DeepSeek uses. Gemma runs on
non-blocking streams, which the legacy stream doesn't order against, so it
passes its own. With `routes_ready`, an event recorded when the routes are
written, later work keeps the device busy while the store reads them.

### `Policy`: the cache's behaviour

`Policy` holds every number that decides how the cache sizes itself, evicts
and admits. They depend on a model's routing and workload, so an extension
sets them, and can tune them with the trace tools, without changing Gufo.
The defaults are DeepSeek V4.1 Flash's measured values. Gemma keeps them and
sets only the limit, the trace path and the log tag.

| Field | Default | Effect |
| --- | --- | --- |
| `cache_limit_bytes` | 0 | Hard cap on the cache; 0 leaves the size to serving's grant |
| `live_floor_bytes` | 1 GiB | The cache grows only while this much stays free |
| `ranking`, `ranking_source` | empty | Experts `Fit` caches first, most wanted first, and the ranking's name in the log |
| `use_half_life` | 2,048 tokens | Decay of an expert's use count, in tokens of its layer's clock |
| `use_cap` | 4 | Most uses one `Prepare` counts per expert |
| `max_class` | 7 | Highest frequency class: protection of at most 2^7 |
| `clock_rows` | 16 | Most tokens one `Prepare` advances its layer's clock by |
| `seed_uses` | 0.5 | Starting weight of seeded experts relative to their ranking |
| `recent_rows` | 16 | Rows whose experts a large batch admits once the cache is full |
| `sweep_tokens` | 1,024 | Smallest batch that adopts a prefill-scale ("wide") prefetch |
| `demand_stage_rows` | 64 | Experts the demand stage holds |
| `prefetch_stage_rows` | 96 | Experts the prefetch stage holds, in two halves for alternating layers |
| `trace_path`, `log_tag` | | Routing trace and log prefix |

`seed_uses` is low on purpose: seeded experts serve a cold prompt, and
decode's real working set replaces them. A cold batch with more misses than
`demand_stage_rows` reads the experts it admits straight into their slots.

### The ranking: start warm

An empty cache makes the first requests pay for every expert. `Fit` reads
experts in the order of `Policy::ranking` (`Ranked`: layer, expert, weight),
then the rest interleaved across layers, as far as the grant and free memory
allow. The ranking is model knowledge, so the extension supplies it:
DeepSeek ships one measured on real traffic. Gemma passes none and starts
with every expert when the cache holds them all, which took its full-cache
decode from 88 to 104 tok/s (resident: 106).

The cache is filled once, at its final size. An earlier version also seeded
at load, before the grant, into a guessed share of memory: it read part of
the cache twice, and on DeepSeek it took the device memory the prompt cache's
backing needed next, which then committed 711 of 948 MB. Before `Fit`
nothing is cached and misses read into the stage; a host without serving
(DeepSeek's `gufo prompt` and `bench`) calls `Fit` itself once its session
exists.

### `Fit` and the claim: serving owns the budget

On Strix Halo device memory is host memory, so the expert cache, the prompt
cache and session state share the same RAM. Only serving sees all three, so
serving decides the split. A runner declares its part in
`TextRunnerResourceClaim::expert_cache`:

```cpp
struct TextRunnerExpertCacheClaim {
  expert_stream::Store* store;
  std::size_t headroom_bytes;  // left free after the expert cache
  std::optional<std::size_t> prompt_cache_share_per_session_bytes;
};
```

Once the sessions and the prompt cache's backing exist, serving gives the
prompt cache its budget (`--cache-ram-bytes`, or the share times the
sessions), then grants the store what it holds plus `HostAvailableBytes()`,
less the headroom and the prompt-cache budget beyond the backing already
committed, calls `Store::Fit(grant)` and logs both budgets
(`expert_cache_grant`). The store allocates its prefetch stage from the
grant and fills the rest from its ranking.

- **A fixed share, not a ratio.** What the prompt cache needs follows the
  workload (sessions, context, prefix reuse), not the machine's size. A ratio
  would shrink the expert cache exactly where every GiB of it matters.
- **The prompt cache holds what its backing commits.** While experts stream,
  the RAM prompt cache keeps what its backing commits at startup, which can
  be less than its budget: on DeepSeek at two sessions, 0.47 of 2 GiB,
  because the backing grows only while 4 GiB of device heap stay free.
  Reserving the rest from the expert cache was tried and left the memory
  idle. Longer prompts are restored from the disk cache instead.
- **Before the claim**, DeepSeek did this arithmetic in its own constants and
  a backing hook. A second streaming extension would have copied both.
- **Host pressure is measured against the store's floor.** Serving gives
  prompt-cache memory back at idle while MemAvailable is below its 4 GiB host
  reserve, assuming Gufo stays well above it, so that only another process
  can push it there. A streamed model fills the host down to the store's
  floor (`live_floor_bytes`, 1 GiB) on purpose. Against the reserve, DeepSeek
  read its own steady state as pressure: the prompt cache lost its backing
  as the server started listening, and a 53-request agent replay prefilled
  in 200-205 s instead of 35 s, with 22 replies different. Keeping the reserve free would have cost the expert cache 4 GiB.
  So while experts stream, the floor replaces the reserve, and below it the
  expert cache gives memory back first:
  - `Store::Release(bytes)` frees the newest whole chunks, drops the experts
    in them without reading them, and holds the cache there; the prompt cache
    gives memory only if that is not enough. Losing 3 GB of experts out of
    100 GiB costs less than dropping cached prompts.
  - `Store::Restore()` lets the cache regrow to its grant once what it gave
    back is free again (the floor, `ReleasedBytes()` and 256 MiB available);
    misses refill it.
  - New requests are refused below half the floor.

  On DeepSeek, with another process taking memory at idle down to 0.67 GiB,
  the next idle tick released 2.7 GB of experts in 108 ms and left the prompt
  cache alone; once the process exited, the cache was restored and the same
  request gave the same reply. Down to 0.21 GiB available, a first version
  took 67 s: it moved the experts it kept out of the freed chunks, and the
  kernel had swapped part of the cache out, so the moves read it back.
  Dropping them instead took 3.3 s with swap; keeping the cache in device
  pages (below), 0.8-1.3 s for 3.0-3.1 GB at 0.25-0.29 GiB available.
  `Release` runs on the idle scheduler, so requests wait for it.

### Residence: device pages

On Strix Halo the cache is host RAM either way; what matters is whether the
kernel may swap it. Managed memory (`hipMallocManaged`) is pageable, and
under pressure the kernel swapped cache pages out, which made every later
hit on them a page fault and the release above slow. The store therefore
maps the cache from 2 MiB device pages (`hipMemCreate`) where the device
supports virtual memory management and its heap covers host memory
(`mem::HeapCoversHost`), and falls back to managed memory elsewhere. Device
pages are not swapped: in the pressure run above, none of the server's
95-135 MiB in swap was cache. They are also what lending through
`mem::PageLender` needs (Decisions).

- **Packed slots.** Each size class's slots lie back to back in one
  reserved address range, mapped a page at a time as the class grows, so a
  slot may cross a page and a chunk leaves no part page unused.
- **Two limits.** The cache grows only while both the host and the device
  heap have room above the floor.
- **Cost.** The kernel keeps about 330 MiB for the 51,000 pages of a
  101 GiB cache, against about 200 MiB of page tables for managed memory.
  At an equal cache size, DeepSeek wrote the same expert trace on both and
  decoded within 0.3%.

### Prefetch: `Predict`, `Prefetch`, `Expect`, `PredictFlush`

- `Predict` takes predicted ids in caller-owned pinned memory and an event
  that signals when they are written. A store thread submits the read when the
  event lands, so the extension never waits for its own prediction.
- `Prefetch` does the same with host ids.
- `Expect` names the next `Prepare`'s layer when an evaluation doesn't start
  after the last layer.
- `PredictFlush` waits until the store has taken every prediction, so the
  extension can release their buffers.

DeepSeek's streamed build depends on them. It predicts the next layer during
the current layer's reads, and for single tokens only when the current layer
waited on reads: misses cluster in neighbouring layers, and a prediction with
no stall to hide costs more than it saves. A sweep of the next layer's
uncached experts without a router read 3.6 times the demand volume on
DeepSeek and was rejected. Only DeepSeek has shaped these calls, so they may
change. Gemma doesn't prefetch yet.

**The window is one layer.** When `Prepare(L)` starts, `Prefetch` and
`Predict` accept only layer `L + 1`; any other layer returns false.
Prefetched bytes go to a stage of two halves chosen by layer parity: while
layer L's kernels read their misses from one half, a prediction for L + 1
fills the other. A prediction for L + 2 would overwrite the half L is still
reading. The reader pool matches: one background job and one parked job.

One layer of lead is enough for DeepSeek, but likely not for other shapes.
The GGUF headers of two other mixture-of-experts models show why:

| | Qwen3.8 Flash-Next Q8_0 | GLM 5.3 Flash UD-Q4_K_XL |
|---|---|---|
| MoE layers, experts, active | 48, 512, 10 | 42 + MTP, 288, 8 |
| Bytes per expert-layer | 4.98 MiB | 14.5–17.6 MiB |
| Routed experts | 119.5 GiB | 176.8 GiB |

Both store their experts as `Layout` already describes: three stacked
tensors per layer, each expert one contiguous slice, with sizes varying by
layer and tensors split across files. Flash-Next's layers are narrow
(hidden size 2560), so one layer's compute is probably shorter than one
expert read, and prefetch would need to start 2–4 layers ahead (an estimate,
not measured). Predictions made once per token, such as the previous token's
routes or a draft model's routing, cover every layer at once.

The signatures can stay: both calls already take the target layer. A
`Policy` field would set the window W, accepting `L + 1 … L + W`, with W = 1
keeping today's behaviour. Two ways to support it:

- **More stage regions.** W + 1 regions in a ring and one job per target
  layer. A wrong prediction still evicts nothing, but staging grows with W:
  each region holds `prefetch_stage_rows` experts, 478 MiB on Flash-Next and
  1.6 GiB on GLM, taken from the cache.
- **Prefetch into the cache.** Predicted experts are admitted to cache slots
  in the lowest frequency class, so a wrong prediction is evicted first and a
  right one is a plain hit. No staging, so W can span the model. `Prepare`
  must wait on an expert whose read is still running, as it already waits on
  adopted stage rows.

The second covers both a few layers of lead and per-token predictions;
keeping the two-region path for W = 1 leaves DeepSeek's measured behaviour
unchanged. `Expect`, `PredictFlush` and `wide` stay as they are. No core
model is planned to stream (#1286), so the window stays one layer until a
model needs more.

### Reads and platforms

Files and reads go through `src/core/platform/`. The store reopens each
`Piece`'s file with `ReopenRandomReadFile` and reads with `ReadAt`, like Qwen
Flash Next's n-gram table: direct `O_DIRECT` reads on Linux, with buffered
reads as the fallback, and cached overlapped reads on Windows. As of November
2026 it is tested on Linux on Strix Halo (`gfx1151`) only; Windows is built
in CI, not run.

`AttachTails` splits every expert read between the model file and a second
file, normally on another drive, holding the end of each expert, so both
drives read at once. The extension builds the file; the store checks its
header and samples its bytes against the layout before using it. On btrfs the
file must be NOCOW, or direct reads of it aren't direct.

### Trace and offline tools

`Policy::trace_path` writes a versioned binary trace: the layout, the policy,
the seeded cache, then each `Prepare`'s routes, hits and timing, and the
extension's predictions (`TracePrediction`). `tools/expert_stream/` replays
it. `trace_compare.py` checks that two runs made the same cache decisions,
and `prefetch_sim.py` and `evict_sim.py` compare policies offline before a
live A/B. `GetStats` and `Log` report uses, hits, bytes read, stall time and
cache size.

## Evidence

DeepSeek V4.1 Flash, streamed `UD-IQ2_XXS` with DSpark and vision on a 128 GB
Strix Halo, replaying a 53-request coding-agent session with a fresh server
per arm (the extension's experiments E124 to E126):

- Each step (store made model-neutral, moved into Gufo, budget owned by
  serving, reads moved to the platform layer) wrote the same expert trace as
  before it: 57,480 prepares and 838,500 predictions, the same hit on every
  route.
- Replies and draft counts are identical in every arm.
- Decode time stayed within noise. The platform-layer step, in A B B A
  order: 155.6 and 156.1 s against 152.6 and 156.3 s, the first arm run first
  after idle.

Gemma 4 26B-A4B fits in memory, so it only tests the API: different tensors,
its own streams, kernels that read past an expert. About 240 lines of model
integration plus the kernel change. Twelve prompts of 256 greedy tokens with
MTP, a fresh server per arm; replies and draft counts are identical to
resident in every arm:

| Quant | Arm | Decode tok/s | Prefill tok/s |
| --- | --- | ---: | ---: |
| UD-Q4_K_XL | resident | 106.3 | 555 |
| UD-Q4_K_XL | every expert cached, seeded in 3 s | 103.9 | 541 |
| UD-Q4_K_XL | 8 / 4 / 2 GiB cache | 43–48 / 17 / 10 | |
| UD-Q8_K_XL | resident | 87.9 | 406 |
| UD-Q8_K_XL | every expert cached, seeded in 6 s | 86.2 | 399 |
| UD-Q8_K_XL | 16 / 8 GiB cache | 53–55 / 17 | |

The partial caches fall off steeply because Gemma doesn't prefetch.

## Delivery

One pull request with two change records: the store (with its HIP test, the
caller-stream contract and `tools/expert_stream/`), and the serving claim
with host-pressure handling. On its own the store would fill the host to its
reserve, which #1275's idle relief reads as pressure, so the two land
together. The extensions raise their `api_version` once it merges.

## Decisions (#1286)

- **Lending memory to Gufo** goes through `mem::PageLender` on the device's
  `PagePool`, as the RAM cache does, and not through an unmap:
  `hipMemUnmap` waits for in-flight GPU work like a free. Cache chunks on
  pool pages keep their own mapping while the pool maps a lent page a second
  time, with no change to `Layout`, `Prepare` or the tables. The first gain
  is a larger cache, borrowing the sessions' promised but unwritten state
  (#1251). Still to solve: `Lend` runs beside `Prepare`, so it may give only
  pages no handed-out table points to and must evict in place; and while
  serving, the pool makes lender pages only above a floor of about 12 GiB, so
  the store would commit its pages at `Fit`. Release to other processes stays
  idle-only, and pass scratch (#1265) waits until wider passes are shown to
  help DeepSeek. Not part of this change.
- **Option names.** `--stream-experts` and `--expert-cache-bytes` stay in the
  extensions.
- **Core models.** No core model is planned to stream, so prefetch keeps its
  one-layer window.

## Open questions

- **Refitting at runtime.** `Fit` runs once at startup. Should a change to
  the prompt-cache budget at runtime refit the expert cache?

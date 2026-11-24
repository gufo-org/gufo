# RFC: expert streaming for extensions

Status: experimental extension API, discussed in #1286. November 2026.

This document describes the expert store, an experimental extension API that
runs mixture-of-experts models whose routed experts do not fit in memory, and
explains the decisions behind its shape. Each call is documented in the
header, [`store.hpp`](../../../../src/core/hip/expert_stream/store.hpp)
(included as `<gufo/ext/experimental/expert_stream.hpp>`); the trace format
is in [`tools/expert_stream/`](../../../../tools/expert_stream/README.md).

Contents:

- [Context](#context), [problem](#problem-statement) and
  [goals](#goals-and-non-goals).
- [Glossary](#glossary).
- [Design](#design): lifecycle, layout, tables, cache, memory budget, host
  pressure, prefetch, reads over two drives and tracing.
- [Alternatives considered](#alternatives-considered).
- [Compatibility](#compatibility-and-delivery), [decisions](#decisions-1286)
  and [open questions](#open-questions).

## Context

In a mixture-of-experts (MoE) layer each token uses only `top_k` of the
layer's experts. The routed experts are most of such a model's weights, so a
model can keep its other weights resident and hold the experts in a bounded
cache, reading misses from disk. This works because routing is skewed: a
small working set of experts takes most routes and changes slowly within a
conversation.

The store started inside the DeepSeek V4.1 Flash extension, where it runs the
model on a 128 GB Strix Halo, and was made model-neutral before moving into
Gufo. Gemma 4 26B-A4B is a second user: its model fits in memory, but its
different tensors, streams and kernels tested that the API carries no
assumption from the first model.

## Problem statement

Without Gufo support, each extension that streams experts reimplements the
same machinery: reading expert slices from the model files, a cache with
eviction and admission, background prefetch, and a memory budget. The first
three are model-independent. The fourth cannot be done correctly inside an
extension: on unified-memory devices the expert cache, the prompt cache and
session state share the same RAM, and only serving knows the prompt cache's
budget, what its backing has committed, and when the host is under pressure.
An extension sizing its cache alone either guesses a reserve that sits idle
or takes memory serving needs next.

## Goals and non-goals

Goals:

- One store for any model: the model describes where its experts are; the
  store never learns tensor names or quantization formats.
- A cache hit costs a table lookup, not a copy.
- Experts are read from the model's own files, without conversion.
- Serving decides how host memory is split between the expert cache and the
  prompt cache, and the expert cache gives memory back first under pressure.
- Every behavioural constant is set by the extension, and tunable offline
  from a routing trace, without changing Gufo.

Non-goals:

- A stable API. The header is experimental and may change with its users.
- Shared MoE kernels. Each extension adapts its own kernels to the tables.
- Streaming for core models. None is planned (#1286).
- Lending cache memory to Gufo's page pool (see [Decisions](#decisions-1286)).

## Glossary

- **Expert**: one routed expert of one layer, made of one or more pieces.
- **Piece**: the bytes of one expert in one tensor (for example the up
  projection), at a fixed stride in the model file.
- **Slot**: cache space holding all pieces of one expert back to back.
- **Size class**: slots of one expert size; quantizations that mix formats
  per layer give several.
- **Stage**: device rows that receive experts read on a miss or prefetched.
- **Table**: per piece, a device array mapping expert id to the address the
  kernels read: a slot for a hit, a stage row for a miss.
- **Grant**: the bytes serving lets the store keep, given at `Fit`.
- **Floor**: free host memory below which the cache stops growing and,
  while serving, gives memory back.
- **Ranking**: the experts to cache first, supplied by the extension.
- **Tails file**: a copy of the end of every expert, normally on a second
  drive, so each miss reads from both drives at once.

## Design

### Lifecycle

1. At load the extension describes its experts (`Layout`) and the cache's
   behaviour and ranking (`Policy`), and calls `Store::Create`. Nothing is
   cached yet; misses read into the stage.
2. Optionally, `AttachTails` splits every expert read across two drives.
3. Once the sessions and the prompt cache's backing exist, serving calls
   `Fit(grant)` through the runner's claim. The store takes its prefetch
   stage from the grant and fills the rest once, from the ranking. A host
   without serving (a CLI prompt or benchmark) calls `Fit` itself after
   loading.
4. Each MoE layer calls `Prepare` after its router; prefetch calls run
   alongside.
5. Under host pressure serving calls `Release`, and `Restore` once memory is
   free again.

### `Layout`: experts where they are on disk

A GGUF stores each layer's experts as one tensor per projection with the
expert dimension outermost: expert `e` is `bytes` long at `offset + e *
bytes`. A `Piece` is exactly that (`PlatformFile`, offset, bytes per expert),
so the store reads the model's files in place. An expert has one piece per
tensor its kernels take (gate, up and down; or fused gate-up and down).

- **Pieces, not one blob per expert**, because kernels take one tensor per
  projection, and because reads of neighbouring experts merge within a
  piece. A slot holds all of an expert's pieces, so an expert is cached whole
  or not at all.
- `max_batch_tokens` bounds one `Prepare` and sizes its row lists.
- `table_experts` fixes the table width, for builds with fewer experts whose
  kernels index a wider table.
- `read_margin_bytes` keeps memory readable past an expert's end, for kernels
  that read whole k-iterations past the last row. The store allocates slots
  and stages, so only it can guarantee that margin.

### `Prepare` and the tables: no copy for a hit

`Prepare(layer, routes, …)` takes the routes (device `[tokens][top_k]` ids),
reads the misses, and returns one table per piece. Kernels read each expert
through the table instead of a base pointer and a stride, so hits are used
where they are cached.

1. The store waits for the routes, copies them to the host and counts the
   experts the batch uses.
2. Misses are read into the demand stage by a pool of reader threads with
   direct I/O into pinned buffers; adjacent misses merge into larger reads.
3. The tables point hits at their slots and misses at their stage rows.
4. Misses worth keeping are admitted: copied from the stage into a slot.

Two options hide read time. With an `overlap` callback the extension runs
the hit rows' kernels while the misses are read and the miss rows after;
every row keeps its arithmetic, so outputs are bit-identical. And a cold
batch with more misses than the demand stage holds reads the experts it
admits straight into their slots.

`Prepare` runs on the caller's stream: it reads the routes once the work
queued before the call has run, and queues its table updates and admission
copies there, so kernels queued after it see them. The null stream keeps
legacy-stream ordering; an extension on non-blocking streams passes its own.
With `routes_ready`, an event recorded when the routes are written, later
work keeps the device busy while the store reads them.

The cost of tables is that each extension adapts its expert kernels to take
one optional table pointer; with resident weights they run as before.

### The cache

**Size classes.** Each distinct expert size is a class. An expert takes a
free slot of its class, or of a larger one once its class cannot grow.

**Residence.** The cache is mapped from 2 MiB device pages (`hipMemCreate`)
where the device supports virtual memory management and its heap covers host
memory (`mem::HeapCoversHost`), and from managed memory elsewhere. On a
unified-memory device both are host RAM; the difference is that the kernel
may swap managed pages, turning later hits into page faults and making a
release under pressure read swapped pages back. Device pages are not
swapped, and they are what lending through `mem::PageLender` needs. Each
class's slots lie back to back in one reserved address range, mapped in
128 MiB chunks as the class grows, so slots may cross pages and no chunk
leaves a part page unused. The cache grows only while both the host and the
device heap stay above the floor.

Releasing and regrowing reuses addresses. Until its work has been idle, the
device keeps reaching the page an address had before it was unmapped
(`mem::RemapBarrier`), so when a class maps new pages where it released
others, the store settles the barrier before any table points there, about
0.5 ms on the first regrowth after a release; on destruction it leaves its
addresses clean for their next owner. Its reader threads upload on their
own streams, so they bracket that work with `mem::DeviceWorkScope`, and a
barrier settling elsewhere waits for them.

**Eviction** is by frequency class, after llama.cpp's ClassLRU. Each cached
expert counts its uses, decaying with a half-life measured in tokens of its
layer's clock; its class is `floor(log2(uses))`, capped. The victim is the
expert with the largest `age / 2^class`, so an expert used twice as often may
stay unused twice as long. A batch's own hits are never victims.

**Admission.** A long prompt touches most experts once and would flush the
decode working set. Once the cache is full, a large batch admits only the
experts its last rows route and those it routes unusually often, and a batch
advances its layer's clock by a bounded number of tokens.

### `Policy`

Every number that decides how the cache sizes itself, evicts and admits
depends on a model's routing and workload, so the extension sets it. The
defaults are the values measured on DeepSeek V4.1 Flash.

| Field | Default | Effect |
| --- | --- | --- |
| `cache_limit_bytes` | 0 | Hard cap on the cache; 0 leaves the size to the grant |
| `live_floor_bytes` | 1 GiB | The floor |
| `ranking`, `ranking_source` | empty | Experts `Fit` caches first, and the ranking's name in the log |
| `use_half_life` | 2,048 tokens | Decay of an expert's use count |
| `use_cap` | 4 | Most uses one `Prepare` counts per expert |
| `max_class` | 7 | Highest frequency class |
| `clock_rows` | 16 | Most tokens one `Prepare` advances its layer's clock by |
| `seed_uses` | 0.5 | Starting uses of ranked experts, relative to their weight |
| `recent_rows` | 16 | Rows whose experts a large batch admits once the cache is full |
| `sweep_tokens` | 1,024 | Smallest batch that adopts a prefill-scale prefetch |
| `demand_stage_rows` | 64 | Experts the demand stage holds |
| `prefetch_stage_rows` | 96 | Experts the prefetch stage holds, in two halves |
| `trace_path`, `log_tag` | | Routing trace and log prefix |

`seed_uses` is low so that experts cached from the ranking give way to the
real working set.

### Filling the cache: the ranking at `Fit`

An empty cache makes the first requests pay for every expert, so `Fit` fills
the cache: first the experts of `Policy::ranking` (layer, expert, weight) in
order, then every other expert, interleaved across layers, as far as the
grant and free memory reach. The ranking is model knowledge (from routing
histograms, for instance), so the extension supplies it; with none, the
interleaved order fills a cache that holds every expert completely.

The cache is filled once, at its final size, and nothing is cached before
`Fit`. Filling at load, before serving knows the budget, would have to guess
a share of memory: it reads part of the cache twice, and it can take device
memory the prompt cache's backing needs next.

### The budget: the serving claim and `Fit`

A runner declares its expert cache in `TextRunnerResourceClaim::expert_cache`:

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
less the headroom and the prompt-cache budget its backing has not committed
yet. It calls `Store::Fit(grant)` and logs both budgets
(`expert_cache_grant`).

- **A fixed share per session, not a ratio.** What the prompt cache needs
  follows the workload (sessions, context, prefix reuse), not the machine's
  size; a ratio would shrink the expert cache where it matters most.
- **The prompt cache keeps what its backing commits at startup.** Growing the
  backing later needs the device heap's dispatch reserve free, which a fitted
  expert cache does not leave; longer prompts are restored from the disk
  cache.

### Host pressure: `Release` and `Restore`

Serving gives prompt-cache memory back at idle while available memory is
below its host reserve, assuming Gufo itself stays well above it. A streamed
model fills the host down to the store's floor on purpose, so measured
against the reserve, its own steady state would read as pressure and empty
the prompt cache at startup. While experts stream, serving therefore measures
pressure against the floor (`Store::FloorBytes`), and below it the expert
cache gives memory back first:

- `Release(bytes)` drops any background prefetch, frees the newest whole
  chunks and the rows misses borrowed when no slot could be lent, and holds
  the cache there, even when it freed nothing. Each size class keeps the experts eviction ranks highest: those in the
  freed chunks move into the slots of the lowest, as a shrinking `Fit` does.
  The moves are device copies of a few GB at most, since the device pages
  are never swapped; in the managed fallback they can read swapped memory
  back. Losing a few GB of cold experts costs less than dropping cached
  prompts; the prompt cache gives memory only if that is not enough.
- `Restore()` lets the cache regrow to its grant once what it released is
  free again (the floor, `ReleasedBytes()` and 256 MiB); misses refill it.
  Room under the process's cgroup counts as available memory too.
- New requests are refused below half the floor.

`Release` runs on the idle scheduler, so no `Prepare` runs beside it.

### Prefetch: `Predict`, `Prefetch`, `Expect`, `PredictFlush`

The extension can predict the next layer's experts, for example by running
the next layer's router on this layer's input. A store thread reads the
predicted misses into the prefetch stage, only while no demand read runs, and
the next `Prepare` of that layer adopts what it actually routes; a wrong
prediction costs bandwidth, not latency.

- `Predict` takes ids in caller-owned pinned memory and an event signalled
  when they are written, so the extension never waits for its own
  prediction. `Prefetch` takes host ids.
- `wide` marks a prefill-scale set, which only a batch of at least
  `sweep_tokens` adopts.
- `Expect` names the next `Prepare`'s layer when an evaluation does not start
  after the last one; `PredictFlush` waits until every prediction was taken,
  so the extension can free its buffers.

The window is one layer: once `Prepare(L)` starts, only layer `L + 1` is
accepted, and after the last layer only layer 0. The prefetch stage has two
halves, which consecutive windows alternate (counting windows, not layers,
so an odd layer count does not give the last layer and the first the same
half): a prediction for `L + 1` fills one half while layer `L`'s kernels
read the other. Only one extension has shaped these calls, so they are the part of the
API most likely to change.

### Reads and platforms

The store reopens each piece's file through `src/core/platform/`
(`ReopenRandomReadFile`, `ReadAt`): direct reads on Linux with buffered reads
as the fallback, and overlapped reads on Windows. Reader threads take read
tasks in file and offset order, and adjacent misses of a piece merge into
larger reads.

### Two drives: `AttachTails`

Once the cache holds the working set, what is left of a miss is read
bandwidth, and one drive's bandwidth bounds it. A second drive helps only if
both read for every miss. Placing whole layers on the second drive leaves
one drive idle during each layer's reads; alternating whole experts
balances the drives only on average, and a layer with few misses often reads
from just one.

So the store splits every piece read. The **head**, from the piece's start
to a cut point, comes from the model file as before; the **tail**, from the
cut to the end, comes from a tails file on the second drive. The two reads of
each piece are queued next to each other, so the reader threads keep both
drives busy and a miss completes at close to the two drives' combined
bandwidth. Each read is still megabytes long, large enough to run a drive at
full speed, although neighbouring experts are no longer merged into one read.

- **A copy, not a conversion.** The tails file holds the model's bytes
  unchanged, and the model file stays complete: outputs do not change, and
  without the tails file reads simply come from the model file.
- **Where to cut.** The file records the share of each piece read from the
  model file. An even split suits two equal drives; a faster second drive
  takes a longer tail.
- **Aligned for direct reads.** Cuts are multiples of 4 KiB and each tail
  is padded to 4 KiB, so every head and tail read starts on the alignment
  direct I/O needs, and an expert's tails lie next to each other on the
  drive. On btrfs the file must be NOCOW, or reads of it are not direct.
- **Checked before use.** `AttachTails` compares the header with the layout
  (counts, each piece's size, offset and model file size), checks the file's
  length, and compares the tails of a few experts with the model's bytes.
  On any mismatch it logs a warning and reads stay on the model file. It is
  called before `Fit`, so filling the cache uses both drives.

The file has an 8 KiB header (magic `GUFOETL1`, version 1) followed, per
layer, by one record per expert, so expert `e` of a layer is at the layer's
first record plus `e` times its record size.

### Trace and offline tools

`Policy::trace_path` writes a versioned binary trace: the layout, the policy,
the cache after `Fit`, then each `Prepare`'s routes, hits and timing, and the
extension's predictions (`TracePrediction`). In `tools/expert_stream/`,
`trace_compare.py` checks that two runs made the same cache decisions, and
`evict_sim.py` and `prefetch_sim.py` compare policies offline before a live
comparison. `GetStats` and `Log` report uses, hits, bytes read, stalls and
the cache size.

## Alternatives considered

- **Gathering routed experts into a contiguous buffer.** Kernels would keep
  their base-pointer form, but every hit would be copied on every token,
  costing more than the reads the cache saves. Rejected for tables.
- **One blob per expert in `Layout`.** Simpler, but kernels take one tensor
  per projection, and it prevents merging reads of neighbouring experts
  within a projection. Rejected for pieces.
- **The extension sizes its own cache.** It would need constants for the
  prompt cache's budget and a hook into its backing, copied into every
  streaming extension, and still could not see host pressure. Rejected for
  the claim.
- **Reserving the full prompt-cache budget from the expert cache.** The
  backing cannot grow past what it commits at startup, so the reserved memory
  stayed idle. Rejected.
- **Filling the cache at load, then topping it up at `Fit`.** Starts warm
  even without serving, but the load-time share is a guess: it took device
  memory the prompt cache's backing needed and made the final cache size
  vary between starts. Rejected for a single fill at `Fit`.
- **Measuring host pressure against the host reserve while experts stream.**
  Keeping that reserve free would cost the cache several GiB, and not
  keeping it empties the prompt cache at startup. Rejected for the floor.
- **Managed memory for the cache.** Swappable: under pressure, hits become
  page faults and releases read swapped pages. Kept only as the fallback
  where device pages are unavailable.
- **Prefetching the next layer's uncached experts without a prediction.**
  Read several times the demand volume. Rejected for router predictions.
- **Spreading experts over two drives by placement, or striping them with
  RAID 0.** Placement leaves one drive idle whenever a layer's misses fall on
  the other. RAID 0 splits every read, but needs both drives reformatted into
  one array; a tails file uses free space on any second drive and leaves the
  model file untouched. Rejected for split reads.

## Compatibility and delivery

The header is experimental and carries no compatibility promise; extensions
declare the API version they need and raise it as the store changes. The
option names (`--stream-experts`, `--expert-cache-bytes`) belong to the
extensions. The store is tested on Linux on Strix Halo (`gfx1151`); the
Windows read path builds through the platform layer but is not run.

One pull request carries two change records: the store (with its HIP test,
the caller-stream contract and `tools/expert_stream/`), and the serving claim
with host-pressure handling. They land together because on its own the store
fills the host below the reserve that the idle relief (#1275) reads as
pressure.

## Decisions (#1286)

- **Lending memory to Gufo** goes through `mem::PageLender` on the device's
  `PagePool`, as the RAM cache does, and not through an unmap:
  `hipMemUnmap` waits for in-flight GPU work like a free. Cache chunks on
  pool pages keep their own mapping while the pool maps a lent page a second
  time, with no change to `Layout`, `Prepare` or the tables. Still to solve:
  `Lend` runs beside `Prepare`, so it may give only pages no handed-out table
  points to, and the pool makes lender pages only above a floor, so the store
  would commit its pages at `Fit`. Release to other processes stays
  idle-only. Not part of this change.
- **Option names** stay in the extensions.
- **Core models.** None is planned to stream, so prefetch keeps its
  one-layer window.

## Open questions

- **Who writes the tails file.** The format belongs to Gufo, but its only
  writer is the DeepSeek extension's packer, which reads that model's GGUF
  shards. A model-neutral writer in `tools/expert_stream/` needs only the
  layout (each piece's file, offset and size); should Gufo ship one?
- **Refitting at runtime.** `Fit` runs once at startup. Should a change to the
  prompt-cache budget at runtime refit the expert cache?
- **A wider prefetch window.** One layer of lead suits models whose layer
  compute is longer than an expert read. Narrow layers, or predictions made
  once per token (the previous token's routes, a draft model's routing),
  would need a window of `W` layers, set in `Policy` with `W = 1` as today;
  `Prefetch` and `Predict` already take the target layer. Two designs:
  - **More stage regions**: `W + 1` regions in a ring. A wrong prediction
    still evicts nothing, but staging grows with `W` and is taken from the
    cache.
  - **Prefetch into the cache**: predicted experts are admitted in the lowest
    frequency class, so a wrong prediction is evicted first and a right one is
    a plain hit, with no staging. `Prepare` waits on an expert whose read is
    still running, as it already waits on adopted stage rows.

  The second covers both cases; keeping the two-half stage for `W = 1` leaves
  today's behaviour unchanged.

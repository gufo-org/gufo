# State memory

Session state that grows with the context (K/V rows and the like) keeps one
device address for its full size but takes physical memory only for the rows
it writes. This page explains how that works and the rules for code that uses
it. The package is `src/mem` (HIP-free) with its HIP backend in
`src/core/hip/state_*`; extensions reach both through
`gufo/ext/experimental/state_memory.hpp`.

## Regions

A **region** (`mem::Region`) is one buffer, reserved as a virtual address range
for its whole size and split into equal **segments** that grow from their first
row: one segment for a K/V buffer per layer, or one per plane, layer (and head)
for a layer-major buffer such as Qwen 27B's. `Ensure(rows)` maps 2 MiB pages so
rows `[0, rows)` of every segment are backed; `MapAhead(rows)` asks the page
pool's thread to do the same in the background. The address never changes, so
captured graphs and kernel arguments stay valid and growth copies nothing.

Rules for a model using regions:

- Call `Ensure` before every write, at one choke point per model, cache
  restores included. A restore that skips it faults the GPU.
- Back a tile of rows past the frontier (the models use 2,048): attention reads
  are tile-rounded.
- Map a prompt pass and more ahead with `MapAhead` (the models use 8,192 rows),
  so the scheduler normally finds its rows mapped and makes no HIP call.
- Never destroy a region beside running GPU work: unmapping waits for every
  queue on the device. Regions end with their sessions.

Without virtual memory management (`hipDeviceAttributeVirtualMemoryManagementSupported`),
a region allocates its full size with `hipMalloc` and `Ensure` only checks
bounds.

## Page pool

One pool per device (`mem::PagePool`) creates the pages, never frees them while
serving, and maps ahead for regions. When regions are reserved, the pool
creates up to 8 GiB of pages ahead (the expected working set, at load, before
memory fragments), then keeps 256-512 free pages. It never holds more free
pages than regions may still map.

Creating a 2 MiB page with a few GiB free forces compaction, and compaction
that touches GPU-visible pages evicts all of the process's GPU queues. So
while serving, the pool makes pages only from free 2 MiB blocks
(`/proc/buddyinfo`) above the **free floor** (see the RAM cache section).
Below it, it borrows pages from the RAM cache first. It makes pages by reclaim
only when there is nothing to borrow and writers would otherwise make them on
their own threads, and never while `MemAvailable` is below the **background
floor** (one sixteenth of RAM, 4-8 GiB).

A page recycled from a released region is zeroed before reuse, as new pages
are: tile reads masked past a frontier must never meet NaN patterns.

## Ledger

The **state ledger** (`mem::StateLedger`) counts bytes **promised** to regions
at their full size, bytes **mapped**, and pages **pooled**. The **outstanding**
bytes are promised minus mapped minus pooled. Every capacity decision that reads
free memory subtracts them: `mem::MemGetInfo` replaces `hipMemGetInfo`, and the
host snapshot budget and admission queries use `mem::LessOutstanding`. Budgets
therefore see the memory they saw when sessions allocated their whole context
at creation: the RAM cache grant, admission, snapshot slots and copy widths are
unchanged, the unwritten rows stay free for the kernel, and no other owner can
take memory a session may still grow into.

Logging the device's raw usage keeps `hipMemGetInfo`. The load log reports the
ledger as `state_reserved_mib`, `state_mapped_mib`, `state_pooled_mib` and
`state_outstanding_mib`.

## RAM cache on state pages

The RAM cache keeps its payloads in pool pages, not in pinned host memory
(`mem::LendableRange`, `src/core/hip/state_lendable_range.cpp`). Each page is
mapped for the GPU, exported as a dma-buf and mapped for the CPU at the same
address, so one pointer serves memcpy, kernels and copies. As of November 2026
on gfx1151, the CPU reads these pages at 28 GB/s and writes at 26 GB/s (as
malloc memory), and GPU copies into them run at 107 GB/s, against 51 GB/s for
registered host pages. Each page is faulted in for the CPU when it is
committed: the driver fills page tables on fault, and a first touch on a
capture path cost up to 10 ms.

The cache's **own grant** is what it would get without borrowing. On top of
it, the cache may **borrow** the outstanding state, up to its cap. It commits
2 GiB while loading and then grows as captures need it, as before. A segment
takes state pages only from free 2 MiB blocks (`/proc/buddyinfo`) above the
**free floor** (`mem::HostFreeFloorBytes()`: the background floor plus 4 GiB),
so no cache page is made by compaction. A segment that cannot take
state pages that way takes pinned host pages instead, within the cache's own
grant, as before: page cache left by other files can hold free pages below
the floor for a whole run, and the cache must still grow as far as without
state pages. Only state pages are lent. When a region needs pages and
none can be made that way, the pool thread **lends** whole free cache pages to
it instead: it evicts checkpoints by the cache's pressure order until enough
2 MiB pages are free, then maps them into the region (cleared first). The
cache keeps its own mapping of a lent page, unused, and the page returns to it
when the region releases it. Lending is a map: it never unmaps (which waits
for the device), unregisters (about 100 ms per 128 MiB, with a device drain)
or creates a page under pressure.

The free floor is about free pages, not `MemAvailable`, which also counts page
cache. As of November 2026, in scenario B (four Flash-Next agents with the disk
tier), the disk tier's writes took 7-8 GB of free pages as page cache in the
first round. With the cache filled down to `MemAvailable` minus the background
floor, free pages fell to 2-9 GB. Creating 2 MiB pages then compacted, and the
kernel's page migration evicted the GPU queues for 1-6.5 s per run. With the
free floor, free pages stayed at 10-13 GB and the queues lost 3 ms.

Borrowed bytes stand for the promise they cover, so the ledger does not count
them as outstanding a second time: every other budget sees what it saw before
the cache borrowed. Pages within the own grant count as ordinary usage, as
before.

The load log reports `borrowable_bytes` and `own_grant_bytes` on the cache's
`configured` event and `backing=state_pages` on `backing_prepared`. Without
virtual memory management, exportable pages or a CPU reservation of the range
(Windows today), the cache keeps pinned host pages and does not borrow.

## Results

As of November 2026 on gfx1151 (ROCm 10.1), Flash-Next with MTP at 4 × 262,144
tokens, against main built the same way:

| | main | state memory |
| --- | --- | --- |
| Load | 17.4 s | 13.6 s |
| `MemAvailable` after load | 7.0 GB | 24.8 GB |
| GPU-queue eviction, scenario B | 3.75-5.5 s | 0 ms |
| Compaction / direct-reclaim stalls, scenario B | 9-19 / 65-78 | 0 / 0 |
| Return round, scenario B | 3.02-3.10 s | 1.91-3.03 s |
| Return round from disk, scenario B | 1.85-2.11 s | 1.71-2.05 s |

**Scenario B** is four concurrent Flash-Next agents with the disk tier: cold
first turns, return turns and a return after a restart (ranges over four runs
of each, on two days). The return round gains only when the cache can borrow
free 2 MiB blocks; on a host with none above the free floor it matches main.
Greedy outputs are byte-identical to main for Flash-Next, Qwen 27B with DFlash2
and DeepSeek V4 Flash with DSpark, including a 66.6K-token RAM restore;
`gufo bench` is within 0.6%. Growing the cache beside decode does not disturb
it: committing a 128 MiB segment on another thread (8.5 ms of state pages,
20 ms of pinned host pages) left a graph-replay loop's p99 unchanged over 40
segments each.

## Users

| Model | State in regions | Choke points |
| --- | --- | --- |
| Qwen3.8 Flash-Next | Trunk and MTP K/V, one region per buffer | `Executor::EnsureKv`: forward, batch, MTP, snapshot and cache restore |
| Qwen 27B | Attention K/V planes, one region per session | `QwenGpuArena::EnsureKvRows`: decode, batch, prefill chunk, verification commit, snapshot and cache restore |
| DeepSeek V4 Flash | Compressed attention, mirror and indexer rows (outside the managed-memory policy) | `rocm_graph_reserve_compressed_rows` grows them in place |

## Host buffers

GPU-visible host buffers (`hip::HostAllocations`) are anonymous pages, pinned
long-term and registered with HIP, not `hipHostMalloc` memory. Compaction keeps
trying to migrate pinned `hipHostMalloc` pages; each failed attempt invalidates
their GPU mapping and evicts every GPU queue of the process (a probe froze the
GPU for 45-109 s). Pinned anonymous pages are never isolated for migration.
Never register, unregister or free pinned host memory while serving under
memory pressure: each costs 60-547 ms of queue evictions per 2 GiB.

## Restore streams

The cache restores into sessions on a pool of two high-priority streams and
writes to disk on a stream of its own. On the shared transfer streams, a
restore's copies used to wait behind other sessions' captures and persistence.

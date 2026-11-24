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
or where device pages are not host RAM (`platform::DevicePagesUseHostMemory`:
Windows, whose HIP has virtual memory management but takes pages from the
dedicated carve-out, so holding outstanding state back from host budgets would
shrink them for nothing), a region allocates its full size with `hipMalloc`,
zeroed like new pages, and `Ensure` only checks bounds.

## Declaring session state

A model declares the state one session holds (`mem::StateDeclaration`,
`gufo/ext/state.hpp`) and the core allocates it (`mem::SessionState`). Each
buffer has a **kind**:

| Kind | Holds | Memory |
| --- | --- | --- |
| `kContext` | K/V, compressed K/V, MTP K/V, depth-sized scratch | A region charged to the session's account, `tokens_per_row` tokens per row |
| `kWhole` | Small row buffers checkpoints share (Flash-Next pooled keys) | A region backed whole at creation |
| `kFixed` | Recurrent and convolution state, control blocks, draft rows | A zeroed `hip::AllocateDevice` allocation |

The declaration's margin is the account's: the tokens the model backs past
its frontier. The same declaration gives the runner's claim, so the bytes a
claim counts cannot drift from what a session allocates: `Bytes()` for
`per_request_state_bytes` and `ContextBytes()` for
`demand_mapped_state_bytes` where `mem::DemandMapped()`. `StateBytes::kPages`
counts what the ledger charges; Flash-Next and Qwen 27B claim `kSize`, as they
did before declarations, and DeepSeek V4 Flash `kPages`.

What a declaration leaves to the model: state it allocates lazily or swaps
(Flash-Next rollback rows, Qwen 27B's two recurrent buffers, DeepSeek V4
Flash's raw window and compressor state), whose bytes the model adds to its
claim, and every `Ensure`, `MapAhead`, `PrepareWrite` and `Restore` call,
which stay at the model's write choke point.

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
host snapshot budget (`ext::HostMemorySample::outstanding`) and admission
queries (`mem::LessOutstanding`) leave them out. Budgets therefore see the
memory they saw when sessions allocated their whole context at creation: the
RAM cache grant, admission, snapshot slots and copy widths are unchanged, the
unwritten rows stay free for the kernel, and no other owner can take memory a
session may still grow into.

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

It also does not borrow where the GPU heap is smaller than host memory less
its background floor (`mem::HeapCoversHost`). State pages are charged to that
heap (on Linux, the `amdgpu.gttsize` limit) and pinned host pages are not.
With a 100 GiB limit on a 128 GiB host, borrowing took 2 GiB at load and grew
the cache until Flash-Next's graph instantiation ran out of heap (18 MB free);
the same host on main kept 14 GB free. Install's
[memory settings](../install.md#memory-settings) set the limit to cover RAM.

## Shared pages

Rows that grow with the context are append-only: once written, a row never
changes. A page full of written rows can therefore serve several users at
once (`mem::SharedPage`, `src/core/hip/state_shared_page.hpp`):

- the session that wrote it, which keeps appending to the last page;
- the RAM cache, which retains the rows for a checkpoint;
- other sessions that restore that checkpoint.

**The checkpoint keeps the page, not a copy.** When a capture retains rows
that live in a region, the cache maps their page a second time in the
**park** (`mem::PagePark`), an address range of its own, and the checkpoint's
payload points there. The capture copies nothing and commits no cache
backing; the ledger charges the payload to the cache budget as it charges a
copy. The park address stays valid whatever the session does next, so these
rows need no borrow, guard or preservation copy.

**A session takes a page of its own only when it writes shared rows.**
Before every write the model calls
`Region::PrepareWrite(first_row, end_row, kept_rows)`. With no shared page in
the region it costs one atomic read. Otherwise each page of those rows that
someone else uses is replaced: by a copy when it holds rows the session still
needs (the rows before an append or a rewind; every row when rewriting in
place), by a cleared page otherwise. The copy and the clear run in
`Region::SettleRemaps`, after the [remap barrier](#remapped-addresses).

**A restore maps the pages.** `Adapter::BeginSharedRestore` gets the park
addresses of the checkpoint's chunks. For every page they fill completely,
`Region::Restore(plan)` maps that page into the session; the cache loads only
the rest (the page the checkpoint ends in, tails, private state). Pages the
session already maps stay as they are.

**A session given to another conversation copies nothing.** It keeps the old
conversation's shared pages mapped past what the new one writes, as it keeps
its own pages, and replaces them page by page as its writes reach them. If
the old conversation returns to that session, its restore finds most pages in
place.

Rules:

- Call `PrepareWrite` at the choke point where you call `Ensure`, with the
  rows the write covers and the rows it keeps. A missing call lets a write
  land in a page a checkpoint still reads.
- Call `Restore` before a cache restore loads rows, with an empty plan when
  there is nothing to map: it releases the pages the loads would overwrite.
- After `PrepareWrite` or `Restore` on every region of the write or the
  restore, call `SettleRemaps` once, before the device uses those rows. It
  costs two atomic reads when no page was remapped.
- All three unmap pages or wait for the device: call them on the thread that
  runs the model, between its launches. Mapping never waits.
- A released park address keeps its mapping until `PagePark::Sweep`, which
  the cache calls before prefill steps and while idle
  (`ComponentCacheResources::settle`).
- A region shares pages when each page belongs to one segment (it has one
  segment, or its segments start on a page), for an adapter that sets
  `Capabilities::shared_restore`. A payload is shared only when it lies
  within one page. Everything else is copied or borrowed as before.

The state ledger counts a shared page once: as mapped while any session maps
it, else as cache memory while a checkpoint holds it. A session's account is
charged for every page it maps, shared or not.

Pages are shared where the cache may hold device pages
(`StateLedger::LendingEnabled()`: demand-mapped regions and a GPU heap that
covers host memory). Elsewhere the park does not exist and rows are copied or
borrowed as before.

Measured on gfx1151 in November 2026 (`hipMemMap`, `hipMemSetAccess` and
`hipMemUnmap` on 2 MiB pages; evidence in
[state-memory-shared-pages](state-memory-shared-pages.md)):

| | Cost |
| --- | --- |
| Map a page a second time | 2.1-2.6 us idle, 18-21 us beside running kernels; never waits |
| Make a mapped range accessible (mandatory) | 2.9 us per page, one call per range |
| Unmap a range, idle | 1.2 us per page in one call |
| Unmap, one call per page, beside running kernels | 1.3-2.3 s per page |
| Read through a second mapping | same bandwidth as the first |

The cost is per page handle, not per byte, and threads do not help (two to
eight threads mapped no faster than one).

## Remapped addresses

`hipMemUnmap` followed by `hipMemMap` with another page returns at once, but
the device keeps the translation it cached for an address it has used.
Kernels, runtime copies and copy-engine copies through that address go on
reading and writing the **old** page, with no error, until the process's
device work has been idle for a moment. Measured on gfx1151 in November 2026
(ROCm 10.1, Linux 7.3.3, `amdgpu` with the MES scheduler; probes and logs in
[state-memory-shared-pages](state-memory-shared-pages.md)):

| After the remap | Addresses that reach the new page |
| --- | --- |
| Used at once | 15% (44 of 300 batches) |
| Device idle 100 us first | 300 of 300; 3,000 of 3,000 at 150 us |
| Copy engine, device idle 200 us first | 400 of 400 (28 of 30 at 100 us) |
| Another thread runs kernels back to back | still the old page after 200 ms |

A second unmap, an access change, `hipDeviceSynchronize`, a busy kernel, or
freeing and reserving the address change nothing. An idle moment between the
last use and the unmap works as well as one after the remap. An address the
device never used since its last idle moment can be mapped and used at once.

So an address that loses a mapping is **dirty** until the device has been
idle, and memory mapped at a dirty address needs the **remap barrier**
(`mem::RemapBarrier::Settle`, `src/mem/remap_barrier.hpp`) before the device
uses it: wait for the device's work, keep the device idle for 400 us. Every
address unmapped before the barrier began is clean afterwards.

- Regions, the park and the cache's lendable range stamp each address when
  it loses a mapping (`RemapBarrier::Stamp`) and ask `RemapBarrier::Clean`
  before mapping there again. A clean address costs nothing.
- `Region::Ensure` runs the barrier itself when it maps a dirty address (a
  session that was trimmed and grows again). The page pool's thread never
  maps a dirty address: it leaves it to the writer.
- `PrepareWrite` and `Restore` remap many pages in many regions. They leave
  the barrier, and the copies and clears that must follow it, to one
  `Region::SettleRemaps` call per write or restore.
- The park uses an address again only once it is clean, so a parked page is
  readable as soon as it is parked.
- Pass scratch (`ScratchArena::MapTransient`) and a region created where
  another gave its address range back run the barrier themselves. Releasing
  the old page does not help: without the idle moment, 50 of 50 writes
  through the address reached the page the driver had already freed.
- Threads that submit device work on their own (transfer streams, the copy
  engine, the page pool's clears) bracket it with `RemapBarrier::Enter` and
  `Leave`. A barrier waits for open brackets and holds new ones back, for at
  most 25 ms. Device work it cannot see (another model's thread in the same
  process) is detected only while it runs; `event=remap_barrier` at shutdown
  counts the barriers that gave up as `unquiet`.
- The hold starts before the unmaps (`RemapBarrier::Hold`): an unmap waits
  for device work in flight, and a disk write's copies beside a request that
  replaced one page per buffer stalled that request for 290 ms.

A barrier costs about 0.4 ms with every decoder paused. One runs per restore
that maps or drops pages, per write that replaces shared pages (the first
pass of a request that takes over a session, then once per page the session
crosses), per wide prefill pass and per `Ensure` over dirty addresses. The
rotation run of the evidence page ran 29 in 12 requests, 11.7 ms in all.

> **Warning:** check a remapped address with a kernel. A `hipMemcpy` read
> can disagree with what kernels see through the same address.

## Shared session state

When the configured sessions do not all fit at full context, they may share
the memory of those that do (#715). A runner reports the demand-mapped part
of a session (`TextRunnerResourceClaim::demand_mapped_state_bytes`), and
`mem::FitSessions` decides at load:

- every session fits at full context: nothing changes;
- the fixed state of every session fits, with room for one session at full
  context: the sessions **share** `full_sessions × session_bytes −
  sessions × fixed_bytes` of demand-mapped state (the **shared state**);
- otherwise the load is refused, as before.

The ledger's **promise cap** is set to the shared state, so every other
budget (the RAM cache grant, snapshots, scratch, copy widths) sees what it
would see with only the sessions that fit at full context.

Each session has a **state account** (`mem::StateAccount`) that groups its
regions and charges their pages to the shared state: the larger of its
**need** (tokens it may write) and its mapped pages. Sessions together never
charge more than the shared state; a region maps a page only if its account
can charge it, and map-ahead stops at the need.

The scheduler drives it:

1. **Admission.** A request reserves its prompt plus up to 8,192 generated
   tokens (`mem::StateReservation`) before it takes a session; the session it
   lands on claims them from the reservation, before a restore writes rows.
2. **Reclaim.** When short, `ComponentTextCache::ReclaimState` first unmaps
   pages past each session's need (map-ahead, an older and longer
   conversation), then drops idle sessions' live frontiers, cheapest to lose
   first (cold admission's order) and never the one the request continues.
   Dropping takes the slot through cold reassignment, which copies rows that
   RAM checkpoints still borrow before they are unmapped; pages a checkpoint
   shares stay with it and are not copied. A session without a
   live frontier may still lend rows past its need, so only busy sessions and
   those with a frontier lose tails without reassignment. Unmapping waits for
   the device, so this runs on the scheduler thread between work units.
   Disk writes come first: a disk capture the RAM cache could not stage reads
   the session's rows in place until it is written. Reclaim never trims a
   session below such a write (a later piece would read pages mapped again,
   zeroed), and drops a frontier that a write still reads last: dropping it
   fails the write, and with it the checkpoint that brings the conversation
   back. Admission keeps that frontier for up to 30 s from its capture
   (`ComponentTextCache::kStateWriteWait`) and retries once the write has
   published; a decoding request does not wait for one. `event=state_reclaimed`
   reports `deferred_writes` and `dropped_writes`.
3. **Queue.** A request that still does not fit waits, in arrival order (a
   later one never passes the oldest), until its deadline; state frees as
   requests end. The log shows `event=state_wait` and `state_wait_end`.
4. **Decode.** A decoding request claims 8,192 more tokens when fewer than
   1,024 of its claim remain, up to its token limit. One that cannot pauses
   (`state_paused`) and resumes first when state frees (`state_resumed`).
5. **Preempt.** When every running request is paused, none can free state:
   the most recently admitted stops with 503 `memory_pressure` (retryable).
6. **Refuse.** A prompt the shared state never holds is refused at submit
   with 507 `insufficient_memory_permanent`; `/props` reports
   `max_prompt_tokens`, and a request's token limit ends where the shared
   state ends, as at the context.

### Pressure from other processes

The steps above share memory between sessions. Another process can still take
the host's memory while Gufo serves (#932). Gufo reacts to MemAvailable itself,
not to the state it promised sessions. Releasing a page the cache borrowed from
that promise raises the promise as much as MemAvailable, so only real memory
shows what a release achieved.

- **Give back at idle.** When the text scheduler is idle, on going idle and
  then about once a second, it compares MemAvailable with the host reserve,
  max(CmaFree, 4 GiB) (`ext::HostReserveShortfallBytes`). Below it, the RAM
  cache evicts checkpoints, least valuable first, and gives whole state pages
  and pinned segments back until MemAvailable is 2 GiB above the reserve.
  Disk copies stay. `event=host_pressure_release` logs the shortfall, the
  bytes released, the evictions and the time. The released backing comes
  back as captures need it, under the cache's usual growth rules (free 2 MiB
  blocks above the free floor), so it stays away while the other process
  holds the memory.
- **Only at idle.** Unmapping a device page waits until no GPU work is in
  flight. Beside a decode loop it waited 0.05–15 s per page, without slowing
  the loop; at idle it takes 1.5 ms per 128 MiB. Measured on Flash-Next: a
  6 GiB shortfall released 8 GiB in 219 ms (35 evictions), and MemAvailable
  rose by 8.2 GB. Greedy outputs were unchanged afterwards, and the cache
  regrew once the pressure ended.
- **Refuse new work.** While MemAvailable is more than 2 GiB below the
  reserve, new requests get 503 `insufficient_memory` (retryable). Running
  requests finish, and the scheduler goes idle and gives memory back.
- **No preemption for it.** A running request is not stopped for another
  process's memory. Its pages could only be given back at idle anyway.
- **No cgroup limit.** Demand-mapped state pages and `hipMalloc` memory are
  not charged to the process's memory cgroup (4 GiB each: `memory.current`
  +0.01 and +0.00 GiB on kernel 7.3.3; pinned host memory +4.02 GiB), so the
  ledger is not bounded by it.

When a session's request ends, its need falls to the live frontier it keeps.
`/metrics` publishes `gufo_shared_state_bytes`, `_charged_bytes`,
`_reclaimable_bytes`, `_max_prompt_tokens`, and the `_waits_total`,
`_pauses_total` and `_preemptions_total` counters. Where regions are not
demand-mapped (Windows today), runners report no demand-mapped state and
the full-context check stands.

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

| Model | Declared state | State in regions | Choke points |
| --- | --- | --- | --- |
| Qwen3.8 Flash-Next | `Executor::DeclareSessionState`: every buffer of a session | Trunk and MTP K/V and pooled indexer keys, one region per buffer; checkpoints share their pages | `Executor::EnsureKv` and `UnshareKv`: forward, batch, MTP, snapshot and cache restore |
| Qwen 27B | `QwenGpuArena::DeclareState`: the attention K/V | Attention K/V planes, one region per session | `QwenGpuArena::EnsureKvRows`: decode, batch, prefill chunk, verification commit, snapshot and cache restore |
| DeepSeek V4 Flash | `rocm_graph_declare_state`: the demand-mapped rows | Compressed attention, mirror and indexer rows, and the indexer score scratch, charged to the session's account | `rocm_graph_reserve_compressed_rows` and `rocm_graph_reserve_indexer_scores` grow them in place, before every write and restore |

DeepSeek V4 Flash no longer uses its managed-memory KV policy where state is
demand-mapped: at the default 1M context, regions decode as fast as managed
memory (1 session, depths 0/33K/99K: 20.91/19.37/18.80 vs 20.87/19.25/18.63
tokens/s, identical outputs). At load the runner creates and drops two
sessions to measure what each allocates whole (0.42 GB; 0.52 GB with DSpark);
the first also creates the shared batch workspace, which stays. The claim is
those bytes plus the regions at full context. The indexer score scratch is
a region too: it grows in place as prefill goes deeper instead of
reallocating, and a session is charged only for the depth it reaches.
Chunks whose score matrix would pass 256 MiB (past ~64K tokens of depth)
score and select their tokens in slices of a multiple of 16 tokens, with
identical results, so the scratch's worst case is ~0.27 GB per session at 1M
instead of 4.3 GB. At `--sessions 8 --context 262144` that leaves 32.0 GB of
shared state instead of 28.2 GB. A configuration that does not fit even sharing its
rows (two sessions at 1M on a 128 GB host) loads as before: managed policy,
nothing claimed, `event=state_unclaimed` warning. The F16 copies of Q8
weights (6 GB) are built once the last session binds at load, not inside the
first request, where allocating them took up to 2.2 s.

## Host buffers

GPU-visible host buffers (`hip::HostAllocations`) are anonymous pages, pinned
long-term and registered with HIP, not `hipHostMalloc` memory. Compaction keeps
trying to migrate pinned `hipHostMalloc` pages; each failed attempt invalidates
their GPU mapping and evicts every GPU queue of the process (a probe froze the
GPU for 45-109 s). Pinned anonymous pages are never isolated for migration.
The pin is an io_uring buffer registration and counts against
`RLIMIT_MEMLOCK`: where it fails (systemd's default 8 MiB fails it at every
size, and Windows has no long-term pin), a buffer stays `hipHostMalloc` memory,
because unpinned registered pages evict for longer under compaction. Kernels
take `hipHostGetDevicePointer`: on Windows the device address differs from the
host pointer. Never register, unregister or free pinned host memory while serving under
memory pressure: each costs 60-547 ms of queue evictions per 2 GiB.

## Restore streams

The cache restores into sessions on a pool of two high-priority streams and
writes to disk on a stream of its own. On the shared transfer streams, a
restore's copies used to wait behind other sessions' captures and persistence.

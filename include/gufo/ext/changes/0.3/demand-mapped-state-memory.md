# Demand-mapped state memory

`gufo/ext/experimental/state_memory.hpp` publishes the core state-memory
package (`src/mem`) and its HIP backend (`src/core/hip/state_*`):
- `gufo::mem::Region` and `RegionShape`: one buffer at a fixed device address
  for its full size, in equal segments that grow from their first row.
  `Region::Create` reserves the address range; `Ensure(rows)` backs rows
  `[0, rows)` of every segment before they are written or read, and
  `MapAhead(rows)` asks the process's page pool to back them in the
  background. Without virtual memory management a region allocates its full
  size and `Ensure` only checks bounds. Every writer of a region, cache
  restores included, calls `Ensure` first.
- `gufo::mem::MemGetInfo`: `hipMemGetInfo` less the state promised to regions
  but not mapped yet. Capacity decisions read free device memory through it.
- `gufo::mem::StateLedger`, `LessOutstanding`, `HostAvailableBytes`,
  `BackgroundFloorBytes` and `kPageBytes`: the process-wide account of
  promised, mapped and pooled state bytes and the host floor below which no
  page is created in the background.

`gufo::hip::HostAllocations` keeps its signature. Its buffers are anonymous
pages, pinned long-term and registered with HIP (coherent and mapped) instead
of `hipHostMalloc` memory; `flags` applies only where the long-term pin fails
(a low `RLIMIT_MEMLOCK`, and always on Windows) and the buffer stays
`hipHostMalloc` memory. Kernels take the device pointer from
`hipHostGetDevicePointer`.

The host snapshot budget (`HostSnapshotBudgetBytes`,
`HostSnapshotCeilingBytes`) and `QueryAdmissionMemory` subtract the
outstanding state bytes, so they report what they reported when state was
allocated whole at session creation.

`cache::ComponentCacheResources` gains optional `restore_stream` and
`background_stream` leases; empty ones fall back to `stream`. The HIP provider
fills them with a high-priority restore pool and a persistence pool.
`cache::StreamedStore::SetStreamClasses(write, restore)` routes disk writes and
disk restores to them. `hip::TransferPool` takes an optional HIP stream
`priority` for its native streams.

The RAM cache may borrow state that sessions were promised but have not
written:
- `StateLedger` counts RAM-cache pages that are state pool pages
  (`AddLendable`/`RemoveLendable`), the cache's own grant
  (`AddLenderGrant`/`RemoveLenderGrant`), and whether the pool can lend
  (`EnableLending`, `LendingEnabled`, `BorrowableBytes`). Cache pages beyond
  the grant cover the promise, so `StateMemoryTotals` gains `lendable_bytes`
  and `borrowed_bytes`, and `outstanding_bytes` excludes the borrowed bytes:
  every other budget reads what it read before the cache borrowed.
  `HostFloorBytes()` is `BackgroundFloorBytes` of this host's memory.
- `HostFreeBytes`, `ParseMeminfoFree`, `ParseBuddyinfoPageBlocks`,
  `HostFreeFloorBytes` and `CanCreatePagesWithoutReclaim` measure free pages
  and free 2 MiB blocks: while serving, state pages are made only from them,
  above the free floor (the background floor plus 4 GiB).
- `cache::ComponentCacheResources` gains an optional `set_shed`, through which
  the serving layer installs its eviction of one retained checkpoint. A
  provider whose backing sessions can borrow calls it when a session needs
  pages back.
- `platform::CanMapDeviceFiles`, `IsAddressRangeReserved`, `MapDeviceFileAt`,
  `UnmapDeviceFileAt` and `CloseDeviceFile` give exported device memory (a
  Linux dma-buf) a CPU view at the address the device runtime reserved for
  it. Windows reports no support.
- `platform::DevicePagesUseHostMemory` is true where device pages are host
  RAM, so outstanding state belongs in host budgets. Windows reports false
  (they come from the dedicated carve-out), and regions there allocate whole.
- `gufo::mem::HeapCoversHost(heap_total, host_total)`: whether the GPU heap
  reaches host memory less its background floor. The pool lends only then:
  borrowed pages are charged to the heap and pinned host pages are not, so
  with a GTT limit below host memory the cache keeps pinned host pages.

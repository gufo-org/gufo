# Shared state pages and the remap barrier

Additive: a checkpoint can keep the session pages that hold its rows instead
of a copy of them, a restore can take those pages by mapping them, and every
remap of a device address goes through a barrier before the device uses it.

- `mem::RemapBarrier` and `mem::DeviceWorkScope`
  (`src/mem/remap_barrier.hpp`, forwarded by
  `gufo/ext/experimental/state_memory.hpp`): the device keeps reaching the
  old page through an address that was unmapped and mapped again until its
  work has been idle. `Stamp`, `Clean`, `NoteRemap`, `Pending`, `Settle`,
  `Hold`, `Release`, `Enter`, `Leave`, `SetDeviceDrain` and `Stats`. A thread that submits
  device work on its own, outside `hip::TransferStream`, brackets it with
  `DeviceWorkScope`.
- `mem::SharedPage`, `mem::ParkedPage`, `mem::PagePark`, `mem::RestorePlan`
  and `mem::CreateParkedRows` (`src/core/hip/state_shared_page.hpp`, also
  forwarded): one physical page with several users, the address range where
  the RAM cache reads the pages it retains, and the pages a restore maps.
- `mem::SharedRows`, `SetSharedRowsFactory` and `CreateSharedRows`
  (`src/mem/shared_rows.hpp`): what the cache needs from shared pages,
  without HIP.
- `mem::Region::Find`, `Share`, `PrepareWrite`, `Restore`, `SettleRemaps`
  and `shared_pages` (`src/core/hip/state_region.hpp`). A model whose regions
  hold append-only rows calls `PrepareWrite(first_row, end_row, kept_rows)`
  on each region before every write, at the choke point where it calls
  `Ensure`, and `Restore(plan, rows)` before a cache restore loads rows,
  then `SettleRemaps()` once before the device uses those rows. Without
  them a write lands in a page a checkpoint still reads. Regions nobody
  shares are unaffected: `Share` returns null unless each page belongs to
  one segment and the cache may hold device pages.
- `mem::PagePool::ReleaseFree`, `AliasForCopy`, `QueueCopy`, `QueueClear`,
  `SettleRemaps` and `kScratchPages` (`src/core/hip/state_page_pool.hpp`).
- `cache::RestoreSource`, `cache::RestoredRows` and
  `cache::Adapter::BeginSharedRestore` (`gufo/ext/adapter.hpp`): a new
  virtual whose default calls `BeginRestore` and takes no rows. An adapter
  that shares pages overrides it, sets the new
  `cache::Capabilities::shared_restore` (default false) and returns the rows
  it mapped; the cache loads the rest.
- `cache::ComponentCacheResources::share`, `device_only`,
  `copy_device_only` and `settle` (`gufo/ext/component_cache_resources.hpp`), all optional. The HIP
  provider sets them where state regions are demand-mapped; a provider that
  leaves them empty copies or borrows rows as before.
- `cache::ResourceLedger::ReserveShared` (`gufo/ext/ledger.hpp`).
- `hip::BackingPoolConfig::state_pool_free_pages` and the `held_bytes`
  argument of `hip::CommittedBackingPool::CanFit`
  (`src/core/hip/committed_backing.hpp`), both optional.

Behavior: with the HIP provider, an adapter that sets
`Capabilities::shared_restore` has the completed append rows that
`BorrowRowsView` returns from a demand-mapped region shared instead of
borrowed, when the region's pages each belong to one segment and a chunk
lies within one 2 MiB page. Such an adapter makes its writes go through
`Region::PrepareWrite` and its restores through `Region::Restore`, each
followed by `Region::SettleRemaps`. Every other adapter borrows or copies as
before. In-tree: Flash-Next shares its K/V and pooled keys; Qwen 27B and
DeepSeek V4 Flash are unchanged.

`Region::Ensure` now runs the remap barrier (about 0.5 ms) when it maps an
address that lost a page since the device was last idle, as after `TrimTo`,
and so do `Region::Create` after another region gave its address range back
and `ScratchArena::MapTransient` after `ReleaseTransient`.
`hip::TransferStream` brackets its work for the barrier by itself.

`ComponentTextCache::Lease::CaptureBytes(boundary)` is internal to serving;
`Adapter::PrefillCheckpointPositions` may still throw `std::invalid_argument`
for a boundary it cannot capture inside a pass, which now skips that
checkpoint instead of failing the request (#1240).

Migration: none for adapters that do not use regions or return no borrowed
views from them.

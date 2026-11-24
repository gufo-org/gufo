#include "src/core/hip/component_cache_resources.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/cache/streaming.hpp"
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/copy_engine.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/lendable.hpp"
#include "src/mem/remap_barrier.hpp"
#include "src/mem/shared_rows.hpp"

namespace gufo::hip {

std::unique_ptr<cache::ComponentCacheResources>
CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter> adapter,
                                 cache::ResourceLedger& ledger,
                                 ComponentCacheHooks hooks) {
  struct Backing {
    Backing(std::size_t streams, std::size_t leases)
        : streams(leases ? leases : streams, streams) {}
    hip::TransferPool streams;
    // Foreground restores: two native streams at the greatest priority. A
    // restore's copies submit in milliseconds but used to complete only
    // after other sessions' captures and persistence queued on the shared
    // FIFO streams (up to 2.9 s measured); here 2.47 GB restore in 31 ms.
    // Restores beside decode (cache::BesideDecodeScope) copy on the GPU's
    // copy engine instead: on these streams each copy waited behind a whole
    // decode graph, as HIP pools them onto the executor's hardware queue
    // (2.2 GB took 2.6 s beside decode, 0.25 s inline). Inline restores keep
    // the copy kernel, faster alone (117-124 vs 54-65 GB/s).
    hip::TransferPool restore{16, 2, -1, hip::DeviceCopyEngine()};
    // Disk persistence: its own stream, so writes never queue in front of a
    // restore or capture.
    hip::TransferPool background{8, 1};
    std::unique_ptr<hip::CommittedBackingPool> pool;
    // Keeps checkpoint rows in the session pages that hold them; null where
    // rows are copied to the pool instead.
    std::shared_ptr<mem::SharedRows> shared;
    std::function<void(std::string_view)> info;
    ~Backing() {
      // What the remap barrier cost this server; `unquiet` counts barriers
      // that gave up waiting for a quiet device.
      const auto barrier = mem::RemapBarrier::Stats();
      if (info && barrier.settles != 0)
        info("schema=component-cache-v1 event=remap_barrier settles=" +
             std::to_string(barrier.settles) +
             " unquiet=" + std::to_string(barrier.unquiet) +
             " total_us=" + std::to_string(barrier.micros));
    }
    // Bytes of payloads that share session pages, and the payload bytes the
    // RAM budget covers. Shared payloads and the pool's committed blocks
    // together stay within it: what the budget holds beyond is the ledger's
    // room for metadata, and a payload that took it would fail every later
    // capture and disk write.
    std::atomic<std::size_t> shared_bytes{0};
    std::size_t payload_capacity{0};
    std::map<std::uint32_t, std::size_t> component_classes;
    // The serving layer's eviction for borrowed pages. The state pool's
    // thread calls it under this mutex, so clearing waits for a call.
    std::mutex shed_mutex;
    std::function<bool()> shed;
  };
  auto result = std::make_unique<cache::ComponentCacheResources>();
  result->adapter = std::move(adapter);
  auto backing =
      std::make_shared<Backing>(hooks.transfer_streams, hooks.transfer_leases);
  if (hooks.info)
    hooks.info(
        "schema=component-cache-v1 event=transfer_resources "
        "native_streams=" +
        std::to_string(hooks.transfer_streams) + " event_leases=" +
        std::to_string(hooks.transfer_leases ? hooks.transfer_leases
                                             : hooks.transfer_streams) +
        " allocation=eager restore_engine=" +
        (backing->restore.HasEngine() ? "sdma" : "none"));
  result->transfer_batch_size = 16;
  const auto rounded = [](std::size_t bytes) {
    return (bytes + 4095) / 4096 * 4096;
  };
  for (const auto& d : result->adapter->Components()) {
    const auto bytes = rounded(d.kind == cache::ComponentKind::kPrivateState
                                   ? d.state_bytes
                                   : d.row_bytes * d.rows_per_chunk);
    backing->component_classes.emplace(d.id.value, bytes);
  }
  // The remainder belongs to index, histories, disk catalog and queue
  // metadata. The pools charge their own per-block allocator headroom.
  result->commit_backing = [backing, &ledger, hooks,
                            adapter =
                                result->adapter.get()](std::size_t budget) {
    const auto plan = adapter->BackingPlan(budget);
    if (plan.size() != 1)
      throw std::invalid_argument(
          "a RAM cache budget of " + std::to_string(budget) +
          " bytes is too small for its backing; use a larger "
          "--cache-ram-bytes, or --cache-ram-bytes off");
    const auto& allocation = plan.front();
    // Pages commit as captures need them. A payload cannot span segments,
    // so segments of about 128 MiB bound registration pauses and stranded
    // edge space, fitting at least two of the largest payloads. Prefill
    // chunks prepare backing before GPU work, after any fallback capture.
    // Growth stops when the host keeps less than its reserve.
    std::size_t largest = 0;
    for (const auto& [id, bytes] : backing->component_classes)
      largest = std::max(largest, bytes);
    const auto block = allocation.block_bytes;
    const auto target =
        std::max<std::size_t>(std::size_t{128} << 20, 2 * largest) / block;
    const auto parts =
        std::max<std::size_t>((allocation.blocks + target - 1) / target, 1);
    const auto segment = (allocation.blocks + parts - 1) / parts * block;
    struct Limited {
      std::mutex mutex;
      std::chrono::steady_clock::time_point refused;
      bool active{};
      // A registration was charged to the device heap; later segments are
      // checked against the reserve before they are committed.
      bool charged{};
    };
    const auto limited = std::make_shared<Limited>();
    // Commit segments from state pages that sessions can borrow back, when
    // the state backend can lend them. The pages are device heap pages, so
    // the heap reserve applies from the first segment.
    auto lendable = mem::CreateLendableRange(allocation.blocks * block);
    if (lendable)
      limited->charged = true;
    const bool lends = lendable != nullptr;
    // The cache grows only from free pages, above mem::HostFreeFloorBytes().
    // Below it the cache does not grow; sessions borrow its pages back
    // instead of making new ones.
    const auto floor = mem::HostFreeFloorBytes();
    // Once serving, a segment also needs free 2 MiB blocks: making pages by
    // compaction beside running work evicts the GPU queues.
    const auto serving = std::make_shared<std::atomic<bool>>(false);
    // Pressure retries ask once per capture; recheck at most once a second
    // while growth is refused. Call with the mutex held.
    const auto refuse = [limited, hooks](const std::string& reason) {
      if (!limited->active)
        hooks.warn("schema=component-cache-v1 event=backing_limited reason=" +
                   reason);
      limited->active = true;
      limited->refused = std::chrono::steady_clock::now();
      return false;
    };
    backing->payload_capacity = allocation.blocks * allocation.block_bytes;
    backing->pool = std::make_unique<hip::CommittedBackingPool>(
        ledger,
        hip::BackingPoolConfig{
            allocation.blocks * allocation.block_bytes +
                allocation.metadata_bytes,
            allocation.block_bytes, allocation.metadata_bytes, segment,
            [limited, refuse, hooks, lends,
             weak = std::weak_ptr<Backing>(backing)](std::size_t bytes) {
              // Payloads that share session pages hold this part of the
              // budget. The capture evicts and asks again, so no latch: the
              // answer changes with the next eviction.
              if (const auto owner = weak.lock();
                  owner && owner->pool &&
                  owner->pool->CommittedBytes() + bytes +
                          owner->shared_bytes.load(std::memory_order_relaxed) >
                      owner->payload_capacity)
                return false;
              const std::lock_guard lock(limited->mutex);
              if (limited->active &&
                  std::chrono::steady_clock::now() - limited->refused <
                      std::chrono::seconds(1))
                return false;
              const auto ceiling = hooks.host_ceiling();
              if (ceiling < bytes)
                return refuse(
                    "host_reserve segment_bytes=" + std::to_string(bytes) +
                    " host_headroom_bytes=" + std::to_string(ceiling));
              // Pinned host pages beside state pages: within the cache's own
              // share, as without state pages. Only state pages borrow.
              if (const auto owner = weak.lock();
                  lends && owner && owner->pool &&
                  owner->pool->CommittedBytes() + bytes >
                      mem::StateLedger::Process().LenderGrantBytes())
                return refuse("own_share segment_bytes=" +
                              std::to_string(bytes) + " committed_bytes=" +
                              std::to_string(owner->pool->CommittedBytes()));
              if (!limited->charged)
                return true;
              // On a charging host, refuse before committing and registering
              // a segment that would only be released again.
              std::size_t heap_free = 0, heap_total = 0;
              if (mem::MemGetInfo(&heap_free, &heap_total) == hipSuccess &&
                  hip::HeapFitsAboveReserve(heap_free, bytes))
                return true;
              return refuse(
                  "device_heap_reserve segment_bytes=" + std::to_string(bytes) +
                  " device_heap_free_bytes=" + std::to_string(heap_free));
            },
            // Registered pages are charged to the device heap on some hosts
            // only. Where they are, keep its dispatch reserve so session and
            // scratch allocations do not fall back to managed memory.
            [limited, refuse](std::size_t bytes, std::size_t heap_before,
                              std::size_t heap_after) {
              const std::lock_guard lock(limited->mutex);
              if (hip::RegistrationCharged(bytes, heap_before, heap_after))
                limited->charged = true;
              if (hip::RegistrationKeepsReserve(heap_before, heap_after)) {
                limited->active = false;
                return true;
              }
              return refuse(
                  "device_heap_reserve segment_bytes=" + std::to_string(bytes) +
                  " device_heap_free_bytes=" + std::to_string(heap_before) +
                  " device_heap_after_bytes=" + std::to_string(heap_after));
            },
            std::move(lendable),
            // State pages: the cache's own share while loading, within the
            // host reserve and reclaiming page cache if it must (a
            // compaction stall costs nothing yet). Otherwise its own share or
            // state sessions were promised but have not written, leaving the
            // free floor. While loading, page cache counts as free: the
            // loader is still dropping the model files' pages (free pages read
            // 3.8 GB with 28 GB available). While serving, only free 2 MiB
            // blocks, so no page is made by compaction. A refusal is silent:
            // the segment takes pinned host pages instead.
            [hooks, floor, serving](std::size_t bytes) {
              const bool loaded = serving->load(std::memory_order_acquire);
              const auto& state = mem::StateLedger::Process();
              const auto ceiling = hooks.host_ceiling();
              if (!loaded &&
                  state.Totals().lendable_bytes + bytes <=
                      state.LenderGrantBytes() &&
                  ceiling >= bytes)
                return true;
              const auto free_bytes =
                  loaded ? mem::HostFreeBytes() : mem::HostAvailableBytes();
              return free_bytes >= floor + bytes &&
                     (!loaded || mem::CanCreatePagesWithoutReclaim(bytes)) &&
                     (ceiling >= bytes || state.OutstandingBytes() >= bytes);
            },
            [weak = std::weak_ptr<Backing>(backing)] {
              const auto owner = weak.lock();
              if (!owner)
                return false;
              const std::lock_guard lock(owner->shed_mutex);
              return owner->shed && owner->shed();
            },
            // Lend only to a cache that retains checkpoints, and so installs
            // a shed. With the RAM cache off the backing stages disk
            // captures only; idle sessions keep the pages they borrow, so
            // lending it would stop every later capture (#1260).
            [weak = std::weak_ptr<Backing>(backing)] {
              const auto owner = weak.lock();
              if (!owner)
                return false;
              const std::lock_guard lock(owner->shed_mutex);
              return bool(owner->shed);
            },
            [weak = std::weak_ptr<Backing>(backing)]() -> std::size_t {
              const auto owner = weak.lock();
              return owner && owner->shared ? owner->shared->FreePages() : 0;
            }});
    // Prepare a bounded reserve before readiness. The first shared-prefix
    // capture needs more than one segment; preparing 2 GiB avoids moving
    // registration onto that request's capture or decode path.
    const auto prepared = std::chrono::steady_clock::now();
    backing->pool->GrowAhead(cache::kPreparedBackingBytes);
    backing->pool->WaitForGrowth();
    const auto prepare_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - prepared)
            .count();
    backing->pool->StartServing();
    serving->store(true, std::memory_order_release);
    {
      // A refusal while loading is not a refusal while serving.
      const std::lock_guard lock(limited->mutex);
      limited->active = false;
    }
    // Afterwards the cache grows as captures need it, as without state
    // pages: state pages from free 2 MiB blocks above the free floor, else
    // pinned host pages within its own share. Filling the whole capacity
    // ahead competed with the first requests' decode (+10-15% in a
    // functional run) and took memory nothing needed yet.
    hooks.info(
        "schema=component-cache-v1 event=backing_prepared committed_bytes=" +
        std::to_string(backing->pool->CommittedBytes()) +
        " long_term_pinned_bytes=" +
        std::to_string(backing->pool->LongTermPinnedBytes()) +
        " capacity_bytes=" +
        std::to_string(allocation.blocks * allocation.block_bytes) +
        " backing=" + (backing->pool->Lends() ? "state_pages" : "host_pages") +
        " prepare_ms=" + std::to_string(prepare_ms));
  };
  backing->shared = mem::CreateSharedRows();
  backing->info = hooks.info;
  if (hooks.info)
    hooks.info(
        std::string("schema=component-cache-v1 event=row_storage mode=") +
        (backing->shared ? "shared_pages" : "backing"));
  if (backing->shared) {
    // Completed rows in a session's own pages: the checkpoint keeps the
    // pages, mapped a second time in the park, instead of a copy. The
    // session goes on appending to them and copies a page only when it
    // rewrites rows a checkpoint holds.
    result->share =
        [backing, &ledger](
            const cache::PayloadRequest& request,
            std::span<const std::byte> view) -> std::optional<cache::Payload> {
      if (view.size() != request.bytes)
        return std::nullopt;
      // Within the part of the budget the pool has not committed. Past it
      // the budget is in pool blocks, free or not: the rows go there.
      const auto committed =
          backing->pool ? backing->pool->CommittedBytes() : 0;
      if (committed + backing->shared_bytes.load(std::memory_order_relaxed) +
              request.bytes >
          backing->payload_capacity)
        return std::nullopt;
      auto rows = backing->shared->Share(view);
      if (!rows)
        return std::nullopt;
      cache::ResourceReservation reservation;
      try {
        reservation = ledger.ReserveShared(request.bytes);
      } catch (const std::bad_alloc&) {
        return std::nullopt;
      }
      // The payload's owner: the shared page, counted until it is dropped.
      struct Counted {
        std::shared_ptr<const void> rows;
        std::shared_ptr<Backing> backing;
        std::size_t bytes;
        ~Counted() {
          backing->shared_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        }
      };
      const auto* const data = rows.get();
      backing->shared_bytes.fetch_add(request.bytes, std::memory_order_relaxed);
      auto counted =
          std::make_shared<Counted>(std::move(rows), backing, request.bytes);
      return cache::Payload::Committed(
          reservation.Convert(),
          std::shared_ptr<const void>(std::move(counted), data));
    };
    result->device_only = [shared = backing->shared](const void* address) {
      return shared->Contains(address);
    };
    // A disk write reads shared pages on the device's copy engine: a copy
    // on the write's stream completes only behind the decode rounds that
    // share its queue (2.2 GB took 6-13 s beside a decoder), and mapping
    // the pages for the CPU from the writer's thread stalled the model's
    // own work for 100-250 ms at a time.
    if (auto* const engine = hip::DeviceCopyEngine())
      result->copy_device_only = [engine](std::span<const std::byte> source,
                                          std::span<std::byte> destination) {
        struct Fence {
          hip::CopyEngine* engine;
          hip::CopyEngine::Fence fence;
          explicit Fence(hip::CopyEngine* engine)
              : engine(engine), fence(engine->CreateFence()) {}
          ~Fence() { engine->DestroyFence(fence); }
        };
        thread_local std::unique_ptr<Fence> fence;
        try {
          if (!fence)
            fence = std::make_unique<Fence>(engine);
        } catch (const std::exception&) {
          return false;
        }
        // Device work of this thread's own, for the remap barrier.
        const mem::DeviceWorkScope work;
        if (!engine->Copy(destination.data(), source.data(), destination.size(),
                          fence->fence))
          return false;
        engine->Wait(fence->fence);
        return true;
      };
    result->settle = [shared = backing->shared] { shared->Sweep(); };
  }
  result->release_host = [backing](std::size_t bytes) -> std::size_t {
    // Evicted checkpoints' shared pages are back in the state pool; their
    // park addresses still map them until swept.
    std::size_t released = 0;
    if (backing->shared) {
      backing->shared->Sweep();
      released = backing->shared->ReleaseFree(bytes);
    }
    if (released < bytes && backing->pool)
      released += backing->pool->Surrender(bytes - released);
    return released;
  };
  result->committed_backing_bytes = [backing]() -> std::size_t {
    return backing->pool ? backing->pool->CommittedBytes() : 0;
  };
  result->set_shed = [backing](std::function<bool()> shed) {
    const std::lock_guard lock(backing->shed_mutex);
    backing->shed = std::move(shed);
  };
  result->prepare_prefill = [backing](std::size_t tokens) {
    // Prepare after any fallback capture, alongside actual prefill work.
    // Short turns need a smaller reserve for private states and row tails.
    backing->pool->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
  };
  result->prepare_capture = [backing](std::size_t bytes) {
    backing->pool->GrowAhead(bytes);
  };
  result->await_capture = [backing] { backing->pool->WaitWhileGrowingAhead(); };
  result->can_capture = [backing](auto payloads) {
    return backing->pool->CanFit(
        payloads, backing->shared_bytes.load(std::memory_order_relaxed));
  };
  result->owner = backing;
  result->stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->streams.TryAcquire();
  };
  result->restore_stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->restore.TryAcquire(cache::CopyingBesideDecode());
  };
  result->restore_beside_decode = backing->restore.HasEngine();
  result->background_stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->background.TryAcquire();
  };
  result->allocate = [backing, hooks](const cache::PayloadRequest& request) {
    // Bound the logical capture, then assign whole shared backing pages.
    const auto capacity =
        backing->component_classes.at(request.component.value);
    if (request.bytes > capacity)
      throw std::logic_error("component capture exceeds its backing class");
    const auto category =
        request.category == cache::ResourceCategory::kBackingAssigned
            ? cache::ResourceCategory::kBackingAssigned
            : request.category;
    const auto segments = backing->pool->PageCommitAllocations();
    const auto started = std::chrono::steady_clock::now();
    auto block = backing->pool->TryAcquire(category, request.bytes);
    // Backing committed where a page was given back to the host reaches the
    // device only after the remap barrier; two atomic reads otherwise.
    if (backing->shared)
      (void)backing->shared->SettleRemaps();
    if (backing->pool->PageCommitAllocations() != segments)
      hooks.info(
          "schema=component-cache-v1 event=backing_grown committed_bytes=" +
          std::to_string(backing->pool->CommittedBytes()) + " segments=" +
          std::to_string(backing->pool->PageCommitAllocations()) +
          " long_term_pinned_bytes=" +
          std::to_string(backing->pool->LongTermPinnedBytes()) +
          " acquire_ms=" +
          std::to_string(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count()));
    if (!block)
      throw cache::ResourceExhausted();
    // Do not start registration between a capture's allocations and copies.
    // Each prefill chunk prepares its own reserve before GPU work.
    auto owner = std::make_shared<hip::BackingBlock>(std::move(*block));
    return cache::ComponentCaptureBuffer{
        owner->Bytes(),
        [owner] {
          owner->Convert();
          return std::move(*owner).TakePayload();
        },
        backing->pool.get(),
        [owner, request](cache::SlotLease& lease, auto source) {
          return cache::Payload::Borrowed(
              std::move(*owner).Borrow(lease, request.component, request.first,
                                       request.end, request.bytes, source));
        }};
  };
  result->staging = [](std::size_t bytes) {
    auto allocation = std::make_shared<hip::HostAllocations>();
    void* data = nullptr;
    const auto status =
        allocation->Allocate(&data, bytes, hipHostMallocCoherent);
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
    auto owner = std::shared_ptr<void>(std::move(allocation), data);
    std::memset(data, 0, bytes);
    return cache::StagingAllocation{std::move(owner),
                                    {static_cast<std::uint8_t*>(data), bytes}};
  };
  result->copy_observed_rows = [](auto source, auto destination,
                                  cache::Stream& stream) {
    return dynamic_cast<hip::TransferStream&>(stream).Copy(
        destination.data(), source.data(), destination.size(),
        hip::CopyDirection::kDeviceToHost, destination.size());
  };
  return result;
}
}  // namespace gufo::hip

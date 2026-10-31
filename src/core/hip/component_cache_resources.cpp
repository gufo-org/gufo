#include "src/core/hip/component_cache_resources.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/transfer_pool.hpp"

namespace gufo::hip {

std::unique_ptr<cache::ComponentCacheResources>
CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter> adapter,
                                 cache::ResourceLedger& ledger,
                                 ComponentCacheHooks hooks) {
  struct Backing {
    hip::TransferPool streams{64};
    std::unique_ptr<hip::CommittedBackingPool> pool;
    std::map<std::uint32_t, std::size_t> component_classes;
  };
  auto result = std::make_unique<cache::ComponentCacheResources>();
  result->adapter = std::move(adapter);
  auto backing = std::make_shared<Backing>();
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
      throw std::invalid_argument("insufficient component backing budget");
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
    };
    backing->pool = std::make_unique<hip::CommittedBackingPool>(
        ledger,
        hip::BackingPoolConfig{
            allocation.blocks * allocation.block_bytes +
                allocation.metadata_bytes,
            allocation.block_bytes, allocation.metadata_bytes, segment,
            [limited = std::make_shared<Limited>(), hooks](std::size_t bytes) {
              const std::lock_guard lock(limited->mutex);
              const auto now = std::chrono::steady_clock::now();
              // Pressure retries ask once per capture; recheck at most
              // once a second while the host is short.
              if (limited->active &&
                  now - limited->refused < std::chrono::seconds(1))
                return false;
              const auto ceiling = hooks.host_ceiling();
              // Pinned host pages come out of the same device heap that
              // session and scratch allocations draw on; keep its dispatch
              // reserve so they do not fall back to managed memory.
              std::size_t heap_free = 0, heap_total = 0;
              const bool heap_fits =
                  hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
                  hip::HeapFitsAboveReserve(heap_free, bytes);
              if (ceiling >= bytes && heap_fits) {
                limited->active = false;
                return true;
              }
              if (!limited->active)
                hooks.warn(
                    "schema=component-cache-v1 "
                    "event=backing_limited reason=" +
                    std::string(heap_fits ? "host_reserve "
                                          : "device_heap_reserve ") +
                    "segment_bytes=" + std::to_string(bytes) +
                    " host_headroom_bytes=" + std::to_string(ceiling) +
                    " device_heap_free_bytes=" + std::to_string(heap_free));
              limited->active = true;
              limited->refused = now;
              return false;
            }});
    // Prepare a bounded reserve before readiness. The first shared-prefix
    // capture needs more than one segment; preparing 2 GiB avoids moving
    // registration onto that request's capture or decode path.
    backing->pool->GrowAhead(2ULL << 30);
    backing->pool->WaitForGrowth();
    hooks.info(
        "schema=component-cache-v1 event=backing_prepared committed_bytes=" +
        std::to_string(backing->pool->CommittedBytes()) +
        " long_term_pinned_bytes=" +
        std::to_string(backing->pool->LongTermPinnedBytes()));
  };
  result->prepare_prefill = [backing](std::size_t tokens) {
    // Prepare after any fallback capture, alongside actual prefill work.
    // Short turns need a smaller reserve for private states and row tails.
    backing->pool->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
  };
  result->can_capture = [backing](auto payloads) {
    return backing->pool->CanFit(payloads);
  };
  result->owner = backing;
  result->stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->streams.TryAcquire();
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

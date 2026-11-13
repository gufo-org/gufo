#include "src/core/hip/component_cache_resources.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/device_cache_budget.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/transfer_pool.hpp"

namespace gufo::hip {

std::unique_ptr<cache::ComponentCacheResources>
CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter> adapter,
                                 cache::ResourceLedger& ledger,
                                 ComponentCacheHooks hooks) {
  struct Backing {
    explicit Backing(std::size_t streams) : streams(streams) {}
    hip::TransferPool streams;
    std::unique_ptr<hip::CommittedBackingPool> pool;
    std::unique_ptr<hip::CommittedBackingPool> device_pool;
    std::function<std::unique_ptr<hip::CommittedBackingPool>()>
        make_device_pool;
    std::once_flag device_once;
    hip::CommittedBackingPool* DevicePool() {
      try {
        std::call_once(device_once, [&] { device_pool = make_device_pool(); });
      } catch (const cache::ResourceExhausted&) {
        // A device tier also needs host bookkeeping. Keep host fallback
        // available if its metadata cannot fit; a later call can retry.
        return nullptr;
      }
      return device_pool.get();
    }
    std::map<std::uint32_t, std::size_t> component_classes;
    cache::Rows device_after_rows{};
    bool UsesDevice(const cache::PayloadRequest& request) const {
      return make_device_pool &&
             request.category == cache::ResourceCategory::kBackingAssigned &&
             request.end > device_after_rows;
    }
  };
  auto result = std::make_unique<cache::ComponentCacheResources>();
  result->adapter = std::move(adapter);
  auto backing = std::make_shared<Backing>(hooks.transfer_streams);
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
    // Keep short histories on the existing host path. Device backing adds
    // room after host rows would compete with a prompt/continuation pair's
    // private state and tails; it need not change small working sets.
    const auto host_capacity = allocation.blocks * allocation.block_bytes;
    std::size_t capture_reserve = 0, row_bytes = 0;
    for (const auto& descriptor : adapter->Components()) {
      const auto component_bytes =
          backing->component_classes.at(descriptor.id.value);
      const auto bytes = (component_bytes + allocation.block_bytes - 1) /
                         allocation.block_bytes * allocation.block_bytes;
      capture_reserve += std::min(host_capacity - capture_reserve, bytes);
      capture_reserve += std::min(host_capacity - capture_reserve, bytes);
      if (cache::HasRows(descriptor.kind))
        row_bytes += descriptor.row_bytes;
    }
    // Captures can overlap retained checkpoints and persistence readers.
    // Keep at least half the host pool outside the row allowance so a large
    // capture reaches the device tier before host admission rejects it.
    const auto host_rows =
        std::min(host_capacity / 2, host_capacity - capture_reserve);
    backing->device_after_rows = row_bytes ? host_rows / row_bytes : 0;
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
    backing->pool = std::make_unique<hip::CommittedBackingPool>(
        ledger,
        hip::BackingPoolConfig{
            allocation.blocks * allocation.block_bytes +
                allocation.metadata_bytes,
            allocation.block_bytes, allocation.metadata_bytes, segment,
            [limited, refuse, hooks](std::size_t bytes) {
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
              if (!limited->charged)
                return true;
              // On a charging host, refuse before committing and registering
              // a segment that would only be released again.
              std::size_t heap_free = 0, heap_total = 0;
              if (hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
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
            }});
    // Prepare a bounded reserve before readiness. The first shared-prefix
    // capture needs more than one segment; preparing 2 GiB avoids moving
    // registration onto that request's capture or decode path.
    backing->pool->GrowAhead(2ULL << 30);
    backing->pool->WaitForGrowth();
    if (ledger.Limits().device_bytes && adapter->SupportsDeviceRows()) {
      const auto device = ledger.Limits().device_bytes;
      // Do not spend host headroom on an unused device tier. In particular,
      // short histories must retain the same complete checkpoints as host-only
      // serving. Initialize bookkeeping only when a capture reaches this tier.
      backing->make_device_pool = [&ledger, hooks, device] {
        constexpr std::size_t device_block = 512ULL << 10;
        const auto metadata = (device / device_block + 1) * 1024 + 4096;
        return std::make_unique<hip::CommittedBackingPool>(
            ledger,
            hip::BackingPoolConfig{
                device + metadata, device_block, metadata, 128ULL << 20,
                [hooks](std::size_t bytes) {
                  return hooks.host_ceiling() >= bytes &&
                         DeviceCacheBudgetBytes() >= bytes;
                },
                [hooks](std::size_t bytes, std::size_t, std::size_t after) {
                  return after >= hip::kDispatchReserve &&
                         hooks.host_ceiling() >= bytes &&
                         DeviceCacheBudgetBytes() >= bytes;
                },
                cache::MemoryDomain::kDevice});
      };
      hooks.info(
          "schema=component-cache-v1 event=device_backing_enabled "
          "capacity_bytes=" +
          std::to_string(device));
    }
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
    if (backing->make_device_pool &&
        std::ranges::any_of(payloads, [&](const auto& payload) {
          return backing->UsesDevice(payload);
        })) {
      std::vector<cache::PayloadRequest> host, device;
      for (const auto& payload : payloads)
        (backing->UsesDevice(payload) ? device : host).push_back(payload);
      auto* pool = backing->DevicePool();
      return (pool && backing->pool->CanFit(host) && pool->CanFit(device)) ||
             backing->pool->CanFit(payloads);
    }
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
    auto* pool = backing->pool.get();
    if (backing->UsesDevice(request))
      if (auto* device = backing->DevicePool())
        pool = device;
    auto segments = pool->PageCommitAllocations();
    const auto started = std::chrono::steady_clock::now();
    auto block = pool->TryAcquire(category, request.bytes);
    if (!block && pool != backing->pool.get()) {
      // Device pressure must not reject a capture the host pool can retain.
      pool = backing->pool.get();
      segments = pool->PageCommitAllocations();
      block = pool->TryAcquire(category, request.bytes);
    }
    if (pool->PageCommitAllocations() != segments)
      hooks.info(
          "schema=component-cache-v1 event=backing_grown committed_bytes=" +
          std::to_string(pool->CommittedBytes()) +
          " segments=" + std::to_string(pool->PageCommitAllocations()) +
          " long_term_pinned_bytes=" +
          std::to_string(pool->LongTermPinnedBytes()) + " acquire_ms=" +
          std::to_string(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count()));
    if (!block)
      throw cache::ResourceExhausted();
    // Do not start registration between a capture's allocations and copies.
    // Each prefill chunk prepares its own reserve before GPU work.
    auto owner = std::make_shared<hip::BackingBlock>(std::move(*block));
    if (pool != backing->pool.get())
      return cache::ComponentCaptureBuffer{
          owner->Bytes(),
          [owner] {
            owner->Convert();
            return std::move(*owner).TakePayload();
          },
          owner->Allocation(),
          [owner, request](cache::SlotLease& lease, auto source) {
            return cache::Payload::Borrowed(std::move(*owner).Borrow(
                lease, request.component, request.first, request.end,
                request.bytes, source));
          },
          cache::MemoryDomain::kDevice};
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
  result->copy_device_rows = result->copy_observed_rows;
  return result;
}
}  // namespace gufo::hip

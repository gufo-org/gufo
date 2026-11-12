#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "src/cache/component_resources.hpp"
#include "src/cache/ledger.hpp"

namespace gufo::hip {

struct ComponentTransferPlan {
  std::size_t native_streams;
  std::size_t leases;
};

// Each exclusively leased slot runs one capture/restore/reassignment batch
// (at most 16), and keeps one preservation lease. Also allow one already
// submitted idle piece per slot and one serialized disk-staging transfer.
// Native FIFO streams can carry several independently fenced transfers. The
// lane count matches the slot producers plus disk; transfers may queue on a
// shared lane and need no exclusive native lease. Logical leases, not native
// lanes, bound outstanding completion ownership.
[[nodiscard]] inline ComponentTransferPlan PlanComponentTransfers(
    std::size_t sessions) {
  if (!sessions)
    throw std::invalid_argument("cache transfer plan requires sessions");
  if (sessions > (std::numeric_limits<std::size_t>::max() - 1) / 18)
    throw std::overflow_error("cache transfer plan overflows");
  return {sessions + 1, sessions * 18 + 1};
}

// Policy inputs the backing provider must not own: the host RAM that stays
// free, and where diagnostics go. The serving layer supplies both, so this
// library reaches neither the snapshot budget nor the process logger itself.
struct ComponentCacheHooks {
  // Always-available host headroom, normally the runtime's
  // `ext::HostSnapshotCeilingBytes()`.
  std::function<std::size_t()> host_ceiling;
  std::function<void(std::string_view)> warn;
  std::function<void(std::string_view)> info;
  // Native streams created up front. With transfer_leases=0 each lease owns
  // an exclusive stream/event pair, preserving the unsized provider contract.
  // With more leases than streams, independent event fences share FIFO native
  // streams. PlanComponentTransfers sizes both from the actual cache slots.
  // Must be nonzero. This counts streams, not hardware queues.
  std::size_t transfer_streams{64};
  // Preallocated logical stream/event leases; zero means transfer_streams.
  // Must cover permanent preservation, foreground batches, idle and disk
  // work. Sharing native streams never waits for another lease or creates
  // native resources on acquisition. Unsupported/unsized providers retain
  // their existing exclusive pool by leaving this zero.
  std::size_t transfer_leases{0};
  // Only arriving backing segments reserve capacity. Retain the converted
  // charge until registered pages and asynchronous users are released.
  std::function<cache::ResourceReservation(std::size_t)> reserve_segment{};
};

// HIP component-cache backing for any adapter: pinned committed pages, a
// transfer-pool stream lease, coherent staging and the observed-row copy.
// The adapter's `BackingPlan()` (the cache's neutral contract) sizes the pool;
// an adapter without a plan is rejected rather than downcast to a model type.
[[nodiscard]] std::unique_ptr<cache::ComponentCacheResources>
CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter> adapter,
                                 cache::ResourceLedger& ledger,
                                 ComponentCacheHooks hooks);

}  // namespace gufo::hip

#pragma once

#include <functional>
#include <memory>
#include <string_view>

#include "src/cache/component_resources.hpp"
#include "src/cache/ledger.hpp"

namespace gufo::hip {

// Policy inputs the backing provider must not own: the host RAM that stays
// free, and where diagnostics go. The serving layer supplies both, so this
// library reaches neither the snapshot budget nor the process logger itself.
struct ComponentCacheHooks {
  // Always-available host headroom, normally the runtime's
  // `ext::HostSnapshotCeilingBytes()`.
  std::function<std::size_t()> host_ceiling;
  std::function<void(std::string_view)> warn;
  std::function<void(std::string_view)> info;
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

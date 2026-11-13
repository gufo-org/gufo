#pragma once

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <cstddef>

#include "src/core/hip/admission_query.hpp"
#include "src/core/hip/heap_placement.hpp"

namespace gufo::hip {
// Require independent heap and matching OS process-budget readings. Never
// add host RAM to either figure; platforms without a budget stay host-backed.
inline std::size_t DeviceCacheBudgetBytes() {
  const auto memory = QueryAdmissionMemory();
  if (!memory || !memory->local)
    return 0;
  // Host pressure is checked before and after each allocation using the
  // cache provider's host ceiling. The larger model-state admission margin
  // must not permanently freeze this pool after host backing has grown.
  const auto& local = *memory->local;
  const auto available = std::min<std::uint64_t>(
      memory->hip_free_bytes, local.budget_bytes > local.usage_bytes
                                  ? local.budget_bytes - local.usage_bytes
                                  : 0);
  const auto above_reserve =
      available > kDispatchReserve ? available - kDispatchReserve : 0;
  return std::min<std::uint64_t>(8ULL << 30, above_reserve / 4);
}
}  // namespace gufo::hip
#endif

#include "src/core/platform/gpu_budget.hpp"

namespace gufo::platform {

// Linux reports no per-process video-memory budget; callers keep the
// runtime's own figures.
std::optional<GpuLocalBudget> QueryGpuLocalBudget(
    std::span<const std::uint8_t, 8>, unsigned) noexcept {
  return std::nullopt;
}

}  // namespace gufo::platform

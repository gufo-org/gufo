#ifndef GUFO_CORE_PLATFORM_GPU_BUDGET_HPP_
#define GUFO_CORE_PLATFORM_GPU_BUDGET_HPP_

#include <cstdint>
#include <optional>
#include <span>

namespace gufo::platform {

/// The OS's accounting of one GPU's local video memory for this process.
struct GpuLocalBudget {
  std::uint64_t budget_bytes{0};
  std::uint64_t usage_bytes{0};
};

/// Budget and current usage of the local memory segment group of the GPU
/// adapter whose LUID is `luid`, on engine node `node`, for the calling
/// process. Windows asks DXGI once per call and keeps no adapter between
/// calls. The budget is a ceiling for this process: it does not see other
/// processes or pressure on host memory. Returns nullopt when the adapter or
/// the query is unavailable, and always on hosts without such a budget
/// (Linux).
std::optional<GpuLocalBudget> QueryGpuLocalBudget(
    std::span<const std::uint8_t, 8> luid, unsigned node) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_GPU_BUDGET_HPP_

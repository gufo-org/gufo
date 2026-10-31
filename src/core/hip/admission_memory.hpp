#ifndef GUFO_CORE_HIP_ADMISSION_MEMORY_HPP_
#define GUFO_CORE_HIP_ADMISSION_MEMORY_HPP_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string_view>

#include "src/core/platform/gpu_budget.hpp"

namespace gufo::hip {

/// The DXGI engine node of the adapter behind a HIP device, from the device's
/// luidDeviceNodeMask. One set bit names the node. TheRock 10.0.0 leaves the
/// mask zero on Strix Halo, a single-node APU, so a zero mask means node 0
/// there and on no other architecture. The architecture is the part of
/// `gcn_arch_name` before any feature flags, as in "gfx1151:xnack-", and must
/// be gfx1151 exactly: pass the name alone, without the NUL padding that
/// follows it in hipDeviceProp_t::gcnArchName. Several bits, or a zero mask on
/// another architecture, leave no node to query.
[[nodiscard]] inline std::optional<unsigned> DxgiNodeFromMask(
    unsigned node_mask, std::string_view gcn_arch_name) {
  if (node_mask == 0) {
    if (gcn_arch_name.substr(0, gcn_arch_name.find(':')) == "gfx1151")
      return 0u;
    return std::nullopt;
  }
  if (!std::has_single_bit(node_mask))
    return std::nullopt;
  return static_cast<unsigned>(std::countr_zero(node_mask));
}

/// Free bytes a state-capacity check may count on. On a Windows iGPU,
/// hipMemGetInfo reports a pool smaller than the video-memory budget the OS
/// grants the process, so weights can leave it empty while the budget still
/// has room. The result is the larger of HIP's figure and the budget left
/// (budget minus usage), the latter capped by the host RAM available above
/// `host_margin`, since that memory is system RAM. Without a budget or a RAM
/// reading, HIP's figure stands. The result is never below `hip_free`.
[[nodiscard]] inline std::uint64_t AdmissionFreeBytes(
    std::uint64_t hip_free, std::optional<platform::GpuLocalBudget> local,
    std::optional<std::uint64_t> host_available, std::uint64_t host_margin) {
  if (!local.has_value() || !host_available.has_value())
    return hip_free;
  const auto saturating_subtract = [](std::uint64_t value,
                                      std::uint64_t amount) {
    return value > amount ? value - amount : std::uint64_t{0};
  };
  const std::uint64_t budget_left =
      saturating_subtract(local->budget_bytes, local->usage_bytes);
  const std::uint64_t host_left =
      saturating_subtract(*host_available, host_margin);
  return std::max(hip_free, std::min(budget_left, host_left));
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_ADMISSION_MEMORY_HPP_

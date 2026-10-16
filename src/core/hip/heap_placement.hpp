#ifndef GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
#define GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

#include <cstdint>

namespace gufo::hip {

/// Keep this much free in the device heap for driver command submission.
inline constexpr std::uint64_t kDispatchReserve = 4ull << 30;

/// True when `bytes` can be taken from `free_bytes` and still leave
/// kDispatchReserve.
[[nodiscard]] inline bool HeapFitsAboveReserve(std::uint64_t free_bytes,
                                               std::uint64_t bytes) {
  return bytes <= free_bytes && free_bytes - bytes >= kDispatchReserve;
}

enum class Placement { kMalloc, kManaged };

[[nodiscard]] inline Placement ChooseWeightPlacement(
    std::uint64_t free_bytes, std::uint64_t total_weight_bytes) {
  return HeapFitsAboveReserve(free_bytes, total_weight_bytes)
             ? Placement::kMalloc
             : Placement::kManaged;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

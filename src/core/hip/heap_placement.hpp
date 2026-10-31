#ifndef GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
#define GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

#include <algorithm>
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

/// True when a host-page registration that moved device heap free memory
/// from `free_before` to `free_after` still leaves kDispatchReserve, or took
/// nothing from the heap. Whether registered pages are charged to the heap
/// differs between hosts, so the caller measures it rather than assuming.
[[nodiscard]] inline bool RegistrationKeepsReserve(std::uint64_t free_before,
                                                   std::uint64_t free_after) {
  return free_after >= std::min(free_before, kDispatchReserve);
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

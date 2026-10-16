#ifndef GUFO_CORE_HIP_MANAGED_ALLOC_HPP_
#define GUFO_CORE_HIP_MANAGED_ALLOC_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/hip/heap_placement.hpp"

namespace gufo::hip {

/// Managed memory is fine-grain by default, so the GPU keeps it coherent
/// with the CPU during a kernel. Weights and state are filled with hipMemcpy
/// or by kernels, so coarse-grain is enough and lets the GPU cache the pages
/// the way hipMalloc memory is cached.
[[nodiscard]] inline hipError_t AllocateManaged(void** pointer,
                                                std::size_t bytes) {
  const hipError_t status =
      hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
  if (status != hipSuccess || pointer == nullptr || *pointer == nullptr ||
      bytes == 0) {
    return status;
  }
  if (hipMemAdvise(*pointer, bytes, hipMemAdviseSetCoarseGrain, 0) !=
      hipSuccess) {
    // The error is sticky; clear it so later calls do not see it.
    (void)hipGetLastError();
  }
  return hipSuccess;
}

[[nodiscard]] inline hipError_t AllocateDeviceOrManaged(void** pointer,
                                                        std::size_t bytes) {
  hipError_t status = hipMalloc(pointer, bytes);
  if (status == hipErrorOutOfMemory) {
    // Clear the error as it remains sticky
    (void)hipGetLastError();
    status = AllocateManaged(pointer, bytes);
  }
  return status;
}

/// Session state and scratch. On the gfx1151 APU the device heap is a cap
/// below physical unified memory (ttm.pages_limit, half of RAM by default)
/// while managed allocations draw on the whole system pool. Both are the
/// same DRAM, so kernels see identical memory either way.
[[nodiscard]] inline hipError_t AllocateDevice(void** pointer,
                                               std::size_t bytes) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
      !HeapFitsAboveReserve(free_bytes, bytes)) {
    return AllocateManaged(pointer, bytes);
  }
  return AllocateDeviceOrManaged(pointer, bytes);
}

/// Typed-pointer overloads; hipMalloc itself accepts T**, so match it.
template<typename T>
[[nodiscard]] inline hipError_t AllocateDevice(T** pointer, std::size_t bytes) {
  return AllocateDevice(reinterpret_cast<void**>(pointer), bytes);
}

/// Allocates the weights of one model load with a placement decided once,
/// from the bytes about to be uploaded and the heap free at construction.
class WeightAllocator {
public:
  explicit WeightAllocator(std::uint64_t total_weight_bytes)
      : placement_(Decide(total_weight_bytes)) {}

  [[nodiscard]] Placement placement() const noexcept { return placement_; }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t bytes) const {
    if (placement_ == Placement::kMalloc)
      return AllocateDeviceOrManaged(pointer, bytes);
    return AllocateManaged(pointer, bytes);
  }

  template<typename T>
  [[nodiscard]] hipError_t Allocate(T** pointer, std::size_t bytes) const {
    return Allocate(reinterpret_cast<void**>(pointer), bytes);
  }

private:
  static Placement Decide(std::uint64_t total_weight_bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
      return Placement::kManaged;
    return ChooseWeightPlacement(free_bytes, total_weight_bytes);
  }

  Placement placement_;
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_MANAGED_ALLOC_HPP_

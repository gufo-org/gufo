#ifndef GUFO_CORE_HIP_MANAGED_ALLOC_HPP_
#define GUFO_CORE_HIP_MANAGED_ALLOC_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::hip {

/// Allocate device memory, preferring the device pool (hipMalloc) and
/// falling back to hipMallocManaged when the pool is exhausted. On the
/// gfx1151 APU the runtime-reported device pool is capped below physical
/// unified memory (a RAM/2 + VRAM heuristic, unaffected by amdgpu.gttsize),
/// while managed allocations draw on the full system pool. Both pools are
/// the same physical DRAM, so kernels see identical memory either way.
[[nodiscard]] inline hipError_t AllocateDevice(void** pointer,
                                               std::size_t bytes) {
  hipError_t status = hipMalloc(pointer, bytes);
  if (status == hipErrorOutOfMemory) {
    // Clear the error as it remains sticky
    (void)hipGetLastError();
    status = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
  }
  return status;
}

/// Weight allocation: managed memory directly, leaving the device pool free
/// for hot session state and scratch. Streamed read-only weights lose
/// nothing on an APU, and the runtime's device-pool accounting stays
/// available for state-capacity admission checks.
[[nodiscard]] inline hipError_t AllocateWeights(void** pointer,
                                                std::size_t bytes) {
  return hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
}

/// Typed-pointer overloads; hipMalloc itself accepts T**, so match it.
template<typename T>
[[nodiscard]] inline hipError_t AllocateDevice(T** pointer, std::size_t bytes) {
  return AllocateDevice(reinterpret_cast<void**>(pointer), bytes);
}

template<typename T>
[[nodiscard]] inline hipError_t AllocateWeights(T** pointer,
                                                std::size_t bytes) {
  return AllocateWeights(reinterpret_cast<void**>(pointer), bytes);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_MANAGED_ALLOC_HPP_

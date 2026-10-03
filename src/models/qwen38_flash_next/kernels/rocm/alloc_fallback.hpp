#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_ALLOC_FALLBACK_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_ALLOC_FALLBACK_HPP_

#include <cstddef>
#include <hip/hip_runtime.h>

namespace gufo::models::qwen38_flash_next::rocm {

/// Allocate device memory, preferring the device pool (hipMalloc) and
/// falling back to hipMallocManaged when the pool is exhausted. On the
/// gfx1151 APU the runtime-reported device pool is capped below physical
/// unified memory, while managed allocations draw on the full system
/// pool. Weights are read-only after upload and scratch is written only
/// through device pointers, so both paths service the same kernels.
[[nodiscard]] inline hipError_t AllocDevice(void** pointer, std::size_t bytes) {
  hipError_t status = hipMalloc(pointer, bytes);
  if (status != hipSuccess) {
    status = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
  }
  return status;
}



/// Weight allocation: managed memory directly, leaving the device pool free
/// for hot session state and scratch. On the gfx1151 APU both pools are the
/// same physical DRAM, so streamed read-only weights lose nothing, and the
/// runtime's device-pool accounting stays available for state claims.
[[nodiscard]] inline hipError_t AllocWeights(void** pointer, std::size_t bytes) {
  return hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
}

/// Typed-pointer overload; hipMalloc itself accepts T**, so match it.
template <typename T>
[[nodiscard]] inline hipError_t AllocDevice(T** pointer, std::size_t bytes) {
  return AllocDevice(reinterpret_cast<void**>(pointer), bytes);
}

template <typename T>
[[nodiscard]] inline hipError_t AllocWeights(T** pointer, std::size_t bytes) {
  return AllocWeights(reinterpret_cast<void**>(pointer), bytes);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_ALLOC_FALLBACK_HPP_

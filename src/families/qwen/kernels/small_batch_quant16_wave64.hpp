#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_QUANT16_WAVE64_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_QUANT16_WAVE64_HPP_

#include <cstddef>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::detail {

// Q4_K/Q5_K groups of sixteen rows, compiled with their qualified scheduler.
// The caller validates the projection shape and group count (one to four).
// A nonzero valid_rows masks one group to its first valid_rows tokens.
void LaunchKQuantSixteenWave64(core::GgmlType type, const void* w,
                               const float* x, float* y, std::size_t m,
                               std::size_t k, hipStream_t stream,
                               std::size_t groups, std::size_t valid_rows = 0);

// Adjacent Qwen27B gate/up matrices with the activation fused into their
// stores.
void LaunchPackedSwiGLUSixteenWave64(core::GgmlType type, const void* w,
                                     const float* x, float* y,
                                     std::size_t groups, hipStream_t stream);

}  // namespace gufo::hip::detail
#endif

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_QUANT16_WAVE64_HPP_

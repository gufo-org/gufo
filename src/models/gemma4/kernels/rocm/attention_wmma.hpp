#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_

#include <hip/hip_runtime.h>

#include "src/models/gemma4/kernels/rocm/kernels.hpp"

namespace gufo::models::gemma4::rocm {

/// Prefill attention on the WMMA matrix cores (binary16 operands, FP32
/// accumulation) for head_dim 256 and 512 with an even number of query heads
/// per KV head. Returns false without launching for other geometries.
[[nodiscard]] bool LaunchWmmaPrefillAttention(const AttentionArgs& args,
                                              hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_

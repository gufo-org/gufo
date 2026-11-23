#ifndef GUFO_MODELS_QWEN_HIP_LORA_SIDE_PATH_HPP_
#define GUFO_MODELS_QWEN_HIP_LORA_SIDE_PATH_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/families/qwen/kernels/ops/low_rank.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::hip {

/// Adds a layer's LoRA side-path update to `rows` contiguous output rows of
/// the projection it sits beside; `x` holds that projection's input rows.
/// Does nothing when the projection has no update.
inline void AddLoraSidePath(const models::QwenLowRankUpdate& update,
                            const float* x, float* y, std::size_t rows,
                            hipStream_t stream) {
  if (!update.empty()) {
    LaunchLowRankAdd(x, update.a, update.b, update.multiplier, y,
                     static_cast<std::uint32_t>(rows), update.in, update.out,
                     update.rank, stream);
  }
}

/// `AddLoraSidePath` reading the projection's IEEE half input rows.
inline void AddLoraSidePathFp16(const models::QwenLowRankUpdate& update,
                                const void* x, float* y, std::size_t rows,
                                hipStream_t stream) {
  if (!update.empty()) {
    LaunchLowRankAddFp16(x, update.a, update.b, update.multiplier, y,
                         static_cast<std::uint32_t>(rows), update.in,
                         update.out, update.rank, stream);
  }
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_LORA_SIDE_PATH_HPP_

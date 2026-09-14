#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_RECURRENT_STATE_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_RECURRENT_STATE_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <concepts>
#include <cstddef>

namespace gufo::hip::detail {

// Call only after the convolution launch has finished reading the old history.
// Each thread owns one channel; load the short-batch tail before overwriting it.
__device__ __forceinline__ void StoreSSMConvHistory(
    const float* input, float* state, std::size_t channel,
    std::size_t batch_size, std::size_t qkv_size) {
  float tail[4];
#pragma unroll
  for (std::size_t j = 0; j < 4; ++j) {
    tail[j] = (batch_size + j >= 4)
                  ? input[((batch_size + j - 4) * qkv_size) + channel]
                  : state[(channel * 4) + batch_size + j];
  }
#pragma unroll
  for (std::size_t j = 0; j < 4; ++j) {
    state[(channel * 4) + j] = tail[j];
  }
}

template<typename T>
concept QwenRecurrentStateElement =
    std::same_as<T, float> || std::same_as<T, hip_bfloat16>;

template<QwenRecurrentStateElement StateT>
__device__ __forceinline__ float4
LoadRecurrentState4(const StateT* row, std::size_t vector_index) {
  if constexpr (std::same_as<StateT, float>) {
    return reinterpret_cast<const float4*>(row)[vector_index];
  } else {
    const StateT* values = row + (vector_index * 4);
    return {static_cast<float>(values[0]), static_cast<float>(values[1]),
            static_cast<float>(values[2]), static_cast<float>(values[3])};
  }
}

template<QwenRecurrentStateElement StateT>
__device__ __forceinline__ void StoreRecurrentState4(StateT* row,
                                                     std::size_t vector_index,
                                                     float4 value) {
  if constexpr (std::same_as<StateT, float>) {
    reinterpret_cast<float4*>(row)[vector_index] = value;
  } else {
    StateT* values = row + (vector_index * 4);
    values[0] = hip_bfloat16(value.x);
    values[1] = hip_bfloat16(value.y);
    values[2] = hip_bfloat16(value.z);
    values[3] = hip_bfloat16(value.w);
  }
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_RECURRENT_STATE_HPP_

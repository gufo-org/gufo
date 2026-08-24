#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_query_attention_forward_wmma.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {

constexpr ck::index_t kCkNumKvHeads = detail::kCkAttentionKvHeads;

template<ck::index_t... Values>
using CkSequence = ck::Sequence<Values...>;

using CkPassThrough = ck::tensor_operation::element_wise::PassThrough;
using CkScale = ck::tensor_operation::element_wise::Scale;
using CkGqaOperation =
    ck::tensor_operation::device::DeviceGroupedQueryAttentionForward_Wmma<
        2, 1, 1, 1, 1, ck::half_t, ck::half_t, ck::half_t, float, ck::Tuple<>,
        float, ck::Tuple<>, float, float, CkPassThrough, CkPassThrough, CkScale,
        CkPassThrough, CkPassThrough,
        ck::tensor_operation::device::GemmSpecialization::MNKOPadding,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default, 1,
        kCkNumKvHeads, 256,
        // QK: 128x128 output tile, K tile 64.
        128, 128, 64, 8, 8,
        // PV: output width 64, reduction tile 64.
        64, 64, 8, 16, 16, 16, 1, 8, 4,
        // Q and K transfers.
        CkSequence<2, 128, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>, 2, 8,
        8, true, CkSequence<8, 32, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>,
        2, 8, 8, true,
        // V and output transfers.
        CkSequence<2, 16, 8>, CkSequence<0, 2, 1>, CkSequence<0, 2, 1>, 1, 1, 1,
        false, 1, 1, CkSequence<1, 128, 1, 2>, 8,
        ck::tensor_operation::device::MaskingSpecialization::
            MaskOutUpperTriangle>;


__global__ void PackCkAttentionInputsKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, __half* __restrict__ q_f16,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (token >= batch_size) {
    return;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  if (index < attention_width) {
    const std::size_t offset = (token * attention_width) + index;
    q_f16[offset] = __float2half_rn(q[offset]);
  }

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (index < kv_width) {
    const std::size_t source_offset = (token * kv_width) + index;
    const std::size_t position = start_pos + token;
    const std::size_t f16_cache_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + position) *
         kv_width) +
        index;
    const std::size_t kv_head = index / head_dim;
    const std::size_t dimension = index % head_dim;
    const std::size_t f32_cache_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
              max_context +
          position) *
         head_dim) +
        dimension;
    const float key = k[source_offset];
    const float value = v[source_offset];
    k_cache_f16[f16_cache_offset] = __float2half_rn(key);
    v_cache_f16[f16_cache_offset] = __float2half_rn(value);
    k_cache[f32_cache_offset] = key;
    v_cache[f32_cache_offset] = value;
  }
}

// This Qwen-specific tile follows the causal online-softmax structure used by
// llama.cpp's MIT-licensed ggml-cuda/fattn-tile.cuh (commit e9fa078). The data
// movement is native HIP and specialized for gfx1151, D=256 and GQA=6.

__global__ void ApplyCkAttentionGateKernel(const float* __restrict__ gate,
                                           float* __restrict__ output,
                                           std::size_t num_elements) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float gate_value = gate[index];
  output[index] *= 1.0F / (1.0F + expf(-gate_value));
}


bool LaunchBatchedAttentionCk(const float* q, const float* k, const float* v,
                              const float* gate, float* k_cache, float* v_cache,
                              void* k_cache_f16, void* v_cache_f16,
                              void* scratch_f16, float* out_context,
                              std::uint32_t layer_idx, std::uint32_t start_pos,
                              std::size_t batch_size, std::uint32_t max_context,
                              std::uint32_t num_heads,
                              std::uint32_t num_kv_heads,
                              std::uint32_t head_dim, hipStream_t stream) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = k_cache_f16 != nullptr,
      .has_v_cache_f16 = v_cache_f16 != nullptr,
      .has_scratch_f16 = scratch_f16 != nullptr,
  };
  if (!detail::IsCkAttentionSupported(support_params)) {
    return false;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  auto* q_f16 = static_cast<__half*>(scratch_f16);
  constexpr std::size_t block_size = 256;
  const std::size_t pack_width = std::max(attention_width, kv_width);
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  const auto* layer_k =
      static_cast<const ck::half_t*>(k_cache_f16) + layer_cache_offset;
  const auto* layer_v =
      static_cast<const ck::half_t*>(v_cache_f16) + layer_cache_offset;
  CkGqaOperation operation;
  const auto argument = operation.MakeArgument(
      reinterpret_cast<const ck::half_t*>(q_f16), layer_k, layer_v, out_context,
      static_cast<ck::index_t>(batch_size),
      static_cast<ck::index_t>(batch_size), static_cast<ck::index_t>(head_dim),
      static_cast<ck::index_t>(head_dim), 1,
      static_cast<ck::index_t>(num_heads),
      1.0F / sqrtf(static_cast<float>(head_dim)), true, true);
  if (!operation.IsSupportedArgument(argument)) {
    return false;
  }

  const dim3 pack_grid(
      static_cast<unsigned int>((pack_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  PackCkAttentionInputsKernel<<<pack_grid, block_size, 0, stream>>>(
      q, k, v, q_f16, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim);

  auto invoker = operation.MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});

  const std::size_t num_elements = batch_size * attention_width;
  if (gate != nullptr) {
    const std::size_t output_grid =
        (num_elements + block_size - 1) / block_size;
    ApplyCkAttentionGateKernel<<<output_grid, block_size, 0, stream>>>(
        gate, out_context, num_elements);
  }
  return true;
}


}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

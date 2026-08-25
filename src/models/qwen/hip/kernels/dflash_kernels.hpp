#ifndef STRIX_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
#define STRIX_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip::kernels {

void LaunchDFlashRMSNorm(
    const float* input,
    const float* weight,
    float* output,
    std::uint32_t dim,
    float eps,
    hipStream_t stream);

void LaunchDFlashDynamicConv2Tap(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint32_t num_tokens,
    std::uint32_t hidden_size,
    hipStream_t stream);

void LaunchDFlashSiLUMul(
    float* gate,
    const float* up,
    std::uint32_t total_elements,
    hipStream_t stream);

void LaunchDFlashNonCausalAttention(
    const float* q,
    const float* injected_k,
    const float* injected_v,
    const float* block_k,
    const float* block_v,
    float* out,
    std::uint32_t current_pos,
    std::uint32_t draft_count,
    std::uint32_t num_q_heads,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    float scale,
    hipStream_t stream);

}  // namespace strix::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

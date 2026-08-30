#ifndef GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::qwen3_asr::hip {

/// Reproduces the Qwen3 BF16 q/k RMSNorm, RoPE, value rounding, and optional
/// float32 KV-cache write in one dispatch.
[[nodiscard]] bool LaunchTextQkNormRoPE(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    float* key_cache, float* value_cache, std::uint32_t layer_index,
    std::uint32_t max_context, hipStream_t stream);

/// Exact Qwen3 RMSNorm ordering: normalize in float32, round to BF16, multiply
/// by the BF16 scale, and round again.
void LaunchTextRMSNorm(const float* input, const float* weight,
                       void* output_bfloat16, std::size_t batch_size,
                       std::size_t dimension, float epsilon,
                       hipStream_t stream);

/// Packs token-major normalized Q/K/V into head-major BF16 matrices and
/// appends K/V to head-major BF16 caches for the hipBLAS eager-attention path.
void LaunchTextPackQkv(const float* query, const float* key, const float* value,
                       void* query_bfloat16, void* key_cache_bfloat16,
                       void* value_cache_bfloat16, std::uint32_t layer_index,
                       std::uint32_t start_position, std::size_t batch_size,
                       std::uint32_t max_context, std::uint32_t num_heads,
                       std::uint32_t num_key_value_heads,
                       std::uint32_t head_dim, hipStream_t stream);

/// Scales BF16 QK products, applies the causal mask, performs float32 softmax,
/// and stores BF16 probabilities in place.
void LaunchTextCausalSoftmax(void* scores_bfloat16, float* probabilities,
                             std::uint32_t start_position,
                             std::size_t batch_size,
                             std::uint32_t context_length,
                             std::uint32_t num_heads, float scale,
                             hipStream_t stream);

/// Converts head-major BF16 attention output into the token-major BF16 layout
/// consumed by the output projection.
void LaunchTextUnpackAttention(const void* input_bfloat16,
                               void* output_bfloat16, std::size_t batch_size,
                               std::uint32_t num_heads, std::uint32_t head_dim,
                               hipStream_t stream);

/// Adds a BF16-rounded residual and emits the following BF16 RMSNorm in one
/// pass. `hidden` retains the unnormalized layer output as float32 values that
/// are exactly representable as BF16.
void LaunchTextResidualAddRMSNorm(float* hidden, const float* update,
                                  const float* weight, void* output_bfloat16,
                                  std::size_t batch_size, std::size_t dimension,
                                  float epsilon, hipStream_t stream);

void LaunchTextSwiGLU(const float* gate, const float* up, void* output_bfloat16,
                      std::size_t count, hipStream_t stream);

void LaunchTextRoundBfloat16(float* values, std::size_t count,
                             hipStream_t stream);

void LaunchTextBfloat16ToFloat(const void* input_bfloat16, float* output,
                               std::size_t count, hipStream_t stream);

/// Returns the lowest token index whose BF16 logit is within tie_tolerance of
/// the maximum. A zero tolerance matches torch.argmax.
void LaunchTextArgmax(const float* logits, std::uint32_t* output_token,
                      std::size_t count, float tie_tolerance,
                      hipStream_t stream);

}  // namespace gufo::models::qwen3_asr::hip
#endif

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_

#ifndef GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_OPS_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_OPS_HPP_

#include <cstddef>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::qwen3_asr::hip {

void LaunchChunkLogMel(const float* input, void* output_bfloat16,
                       std::size_t frames, std::size_t chunks,
                       hipStream_t stream);

void LaunchBiasGelu(void* values_bfloat16, const void* bias_bfloat16,
                    std::size_t elements, std::size_t channels,
                    std::size_t spatial_size, hipStream_t stream);

void LaunchPackConvolutionWeights(const void* input, void* output,
                                  hipStream_t stream);

void LaunchConvolution3x3(const void* input, const void* weight, void* output,
                          std::size_t chunks, std::size_t input_channels,
                          std::size_t height, std::size_t width,
                          hipStream_t stream);

void LaunchConvOutputLayout(const void* input_bfloat16, void* output_bfloat16,
                            std::size_t chunks, hipStream_t stream);

void LaunchAddPositionAndCompact(const float* projected, float* compact_output,
                                 std::size_t feature_frames, std::size_t chunks,
                                 hipStream_t stream);

void LaunchLayerNormToBfloat16(const float* input, const void* weight_bfloat16,
                               const void* bias_bfloat16, void* output_bfloat16,
                               std::size_t rows, std::size_t columns,
                               float epsilon, hipStream_t stream);

void LaunchBiasRoundBfloat16(float* values, const void* bias_bfloat16,
                             std::size_t rows, std::size_t columns,
                             hipStream_t stream);

void LaunchResidualAddBfloat16(float* residual, const float* update,
                               std::size_t elements, hipStream_t stream);

void LaunchBiasGeluToBfloat16(const float* input, const void* bias_bfloat16,
                              void* output_bfloat16, std::size_t rows,
                              std::size_t columns, hipStream_t stream);

void LaunchSegmentedSelfAttention(const float* query, const float* key,
                                  const float* value, float* output,
                                  void* output_bfloat16, std::size_t tokens,
                                  std::size_t segment_tokens, std::size_t heads,
                                  std::size_t head_dim, hipStream_t stream);

void LaunchAllFinite(const float* values, std::size_t count,
                     std::uint32_t* finite, hipStream_t stream);

}  // namespace gufo::models::qwen3_asr::hip
#endif

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_AUDIO_OPS_HPP_

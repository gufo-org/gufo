#ifndef GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_OPS_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::qwen3_tts::hip {

void LaunchDecodeCodebook(const std::uint32_t* codes, std::uint32_t code_groups,
                          std::uint32_t group_index, const float* embedding_sum,
                          const float* cluster_usage, float* output,
                          std::size_t frames, std::size_t dimension,
                          bool initialize, hipStream_t stream);

void LaunchAdd(const float* left, const float* right, float* output,
               std::size_t elements, hipStream_t stream);

void LaunchAddBias(float* values, const float* bias, std::size_t rows,
                   std::size_t columns, hipStream_t stream);

void LaunchAddScaledChannels(float* residual, const float* update,
                             const float* scale, std::size_t rows,
                             std::size_t columns, hipStream_t stream);

void LaunchRmsNorm(const float* input, const float* weight, float* output,
                   std::size_t rows, std::size_t columns, float epsilon,
                   hipStream_t stream);

void LaunchLayerNorm(const float* input, const float* weight, const float* bias,
                     float* output, std::size_t rows, std::size_t columns,
                     float epsilon, hipStream_t stream);

void LaunchRope(float* query, float* key, std::size_t frames, std::size_t heads,
                std::size_t head_dimension, float rope_theta,
                hipStream_t stream, std::size_t start_position = 0);

void LaunchSlidingCausalAttention(
    const float* query, const float* key, const float* value, float* output,
    std::size_t frames, std::size_t heads, std::size_t head_dimension,
    std::size_t sliding_window, hipStream_t stream,
    std::size_t query_offset = 0, std::size_t key_frames = 0);

void LaunchSwiGlu(const float* gate, const float* up, float* output,
                  std::size_t elements, hipStream_t stream);

void LaunchGelu(float* values, std::size_t elements, hipStream_t stream);

void LaunchSnakeBeta(const float* input, const float* alpha, const float* beta,
                     float* output, std::size_t rows, std::size_t columns,
                     hipStream_t stream);

void LaunchCausalConv1dIm2Col(const float* input, float* columns,
                              std::size_t input_length,
                              std::size_t input_channels,
                              std::size_t output_length, std::size_t kernel,
                              std::size_t dilation, hipStream_t stream);

void LaunchCausalDepthwiseConv1d(const float* input, const float* weight,
                                 const float* bias, float* output,
                                 std::size_t length, std::size_t channels,
                                 std::size_t kernel, std::size_t dilation,
                                 hipStream_t stream);

void LaunchTransposeConvWeight(const float* source, float* target,
                               std::size_t input_channels,
                               std::size_t output_channels, std::size_t kernel,
                               hipStream_t stream);

void LaunchAssembleCausalConvTranspose1d(const float* expanded,
                                         const float* bias, float* output,
                                         std::size_t input_length,
                                         std::size_t output_length,
                                         std::size_t output_channels,
                                         std::size_t kernel, std::size_t stride,
                                         hipStream_t stream);

void LaunchClamp(float* values, std::size_t elements, float minimum,
                 float maximum, hipStream_t stream);

}  // namespace gufo::models::qwen3_tts::hip
#endif

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_OPS_HPP_

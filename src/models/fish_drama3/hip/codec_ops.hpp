#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_OPS_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::fish_drama3::hip {

void LaunchCodecAdd(const float* left, const float* right, float* output,
                    std::size_t elements, hipStream_t stream);

void LaunchCodecSubInPlace(float* target, const float* subtrahend,
                           std::size_t elements, hipStream_t stream);

void LaunchCodecAddBias(float* values, const float* bias, std::size_t rows,
                        std::size_t columns, hipStream_t stream);

void LaunchCodecAddScaledChannels(float* residual, const float* update,
                                  const float* scale, std::size_t rows,
                                  std::size_t columns, hipStream_t stream);

void LaunchCodecRmsNorm(const float* input, const float* weight, float* output,
                        std::size_t rows, std::size_t columns, float epsilon,
                        hipStream_t stream);

void LaunchCodecLayerNorm(const float* input, const float* weight,
                          const float* bias, float* output, std::size_t rows,
                          std::size_t columns, float epsilon,
                          hipStream_t stream);

// Interleaved RoPE (GGML_ROPE_TYPE_NORMAL): rotates adjacent pairs (2*d,
// 2*d+1).
void LaunchCodecInterleavedRope(float* query, float* key, std::size_t frames,
                                std::size_t heads, std::size_t head_dimension,
                                float rope_theta, hipStream_t stream,
                                std::size_t start_position = 0);

void LaunchCodecSlidingCausalAttention(
    const float* query, const float* key, const float* value, float* output,
    std::size_t frames, std::size_t heads, std::size_t head_dimension,
    std::size_t sliding_window, hipStream_t stream,
    std::size_t query_offset = 0, std::size_t key_frames = 0);

void LaunchCodecSwiGlu(const float* gate, const float* up, float* output,
                       std::size_t elements, hipStream_t stream);

void LaunchCodecGelu(float* values, std::size_t elements, hipStream_t stream);

// Fish DAC Snake1d: x + sin^2(alpha * x) / (alpha + 1e-9f)
void LaunchCodecSnake1d(const float* input, const float* alpha, float* output,
                        std::size_t rows, std::size_t columns,
                        hipStream_t stream);

void LaunchCodecTanh(float* values, std::size_t elements, hipStream_t stream);

// Supports both stride == 1 and stride > 1 causal Conv1d:
// left_pad = (kernel - 1) * dilation + 1 - stride
void LaunchCodecCausalConv1dIm2Col(const float* input, float* columns,
                                   std::size_t input_length,
                                   std::size_t input_channels,
                                   std::size_t output_length,
                                   std::size_t kernel, std::size_t stride,
                                   std::size_t dilation, hipStream_t stream);

void LaunchCodecCausalDepthwiseConv1d(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      std::size_t length, std::size_t channels,
                                      std::size_t kernel, std::size_t dilation,
                                      hipStream_t stream);

void LaunchCodecTransposeConvWeight(const float* source, float* target,
                                    std::size_t input_channels,
                                    std::size_t output_channels,
                                    std::size_t kernel, hipStream_t stream);

void LaunchCodecAssembleCausalConvTranspose1d(
    const float* expanded, const float* bias, float* output,
    std::size_t input_length, std::size_t output_length,
    std::size_t output_channels, std::size_t kernel, std::size_t stride,
    hipStream_t stream);

// Quantizer codebook lookup + 1x1 out_proj accumulation for a single codebook:
// codes_row holds `frames` int32 indices for this codebook.
// codebook_weight is [codebook_size, 8], out_proj_weight is [1024, 8],
// out_proj_bias is [1024].
void LaunchCodecDequantizeCodebook(const std::int32_t* codes_row,
                                   std::size_t codebook_size,
                                   const float* codebook_weight,
                                   const float* out_proj_weight,
                                   const float* out_proj_bias, float* output,
                                   std::size_t frames, std::size_t latent_dim,
                                   bool initialize, hipStream_t stream);

// Encoder VQ quantization for one codebook:
// 1. Projects residual [frames, 1024] -> z_e [frames, 8] via in_proj_weight [8,
// 1024] + in_proj_bias [8]
// 2. L2-normalizes z_e and finds argmax cosine similarity against
// normalized_codebook [codebook_size, 8]
// 3. Writes chosen code to codes_row [frames]
// 4. Looks up codebook_weight[code] [8], projects via out_proj_weight [1024, 8]
// + out_proj_bias [1024], and subtracts from residual [frames, 1024].
void LaunchCodecQuantizeCodebookStep(
    float* residual, const float* in_proj_weight, const float* in_proj_bias,
    const float* codebook_weight, const float* normalized_codebook,
    const float* out_proj_weight, const float* out_proj_bias,
    std::int32_t* codes_row, std::size_t frames, std::size_t latent_dim,
    std::size_t codebook_size, hipStream_t stream);

}  // namespace gufo::models::fish_drama3::hip
#endif

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_CODEC_OPS_HPP_

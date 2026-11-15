#ifndef GUFO_MODELS_YUE2_HIP_KERNELS_HPP_
#define GUFO_MODELS_YUE2_HIP_KERNELS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::yue2::hip {

/// The acoustic input rows: `vae2llm(x) + time + position` with each upstream
/// BF16 boundary. Row 0 and row `frames + 1` are the zero latent boundaries;
/// row r takes `latent_pos_embed[r]`. `state` is FP32 `[frames][64]`;
/// `hidden` is FP32 `[frames + 2][2048]`.
void LaunchLatentInput(const float* state, std::size_t frames,
                       const void* weight, const void* bias,
                       const void* time_row, const void* positions,
                       float* hidden, hipStream_t stream);

/// `velocity[t] = bf16(projected[t + 1] + bias)` for the content rows of the
/// `llm2vae` projection, `projected` being `[frames + 2][64]`.
void LaunchVelocityOutput(const float* projected, const void* bias,
                          std::size_t frames, float* velocity,
                          hipStream_t stream);

/// `output = state - scale * velocity`, element-wise in FP32.
void LaunchOdeStep(const float* state, const float* velocity, float scale,
                   std::size_t count, float* output, hipStream_t stream);

/// Rounds FP32 logits to BF16 and copies them densely.
void LaunchBfloat16Logits(const float* input, std::size_t count, float* output,
                          hipStream_t stream);

/// One key segment of the acoustic attention: `count` key rows at
/// `keys + (kv_head * key_stride + r) * 128` and their values transposed at
/// `values + (kv_head * 128 + channel) * value_pitch + r`, `value_pitch`
/// being `count` rounded up to 32 with zero (finite) padding.
struct NarSegment {
  const std::uint16_t* keys{nullptr};
  const std::uint16_t* values{nullptr};
  std::uint32_t count{0};
  std::uint32_t key_stride{0};
  std::uint32_t value_pitch{0};
};

/// Rounds `count` up to the 32-key tile the acoustic attention consumes.
[[nodiscard]] constexpr std::uint32_t NarValuePitch(std::uint32_t count) {
  return (count + 31U) & ~31U;
}

/// Transposes head-major BF16 values `[kv_heads][stride][128]` (first
/// `count` rows) to `[kv_heads][128][NarValuePitch(count)]`, zero-padded.
void LaunchTransposeValues(const std::uint16_t* values, std::uint32_t stride,
                           std::uint32_t count, std::uint32_t kv_heads,
                           std::uint16_t* output, hipStream_t stream);

/// Unmasked acoustic attention of head-major BF16 queries `[heads][rows][128]`
/// over the prefix and the latent rows: BF16 QK on matrix instructions, FP32
/// online softmax over 32-key tiles, probabilities rounded to BF16 before a
/// matrix value product (PyTorch's fused BF16 attention). Writes BF16
/// `[rows][heads][128]`. Query head h reads key head `h / group`.
void LaunchNarAttention(const std::uint16_t* queries, NarSegment prefix,
                        NarSegment latent, std::uint16_t* output,
                        std::uint32_t heads, std::uint32_t rows,
                        std::uint32_t group, hipStream_t stream);

// Oobleck decoder operators. Activations are channel-major FP32 rows of
// `pitch = pad + length + pad` elements whose margins are zero, so every
// convolution tap is a plain shifted view.

/// `output = x + sin(alpha * x)^2 / (beta + 1e-9)` per channel, writing zero
/// margins. `alpha` and `inverse_beta` are already exponentiated.
void LaunchSnake(const float* input, float* output, std::size_t channels,
                 std::size_t length, std::size_t pad, const float* alpha,
                 const float* inverse_beta, hipStream_t stream);

/// Sets every interior element to its channel's bias (zero when null) and
/// the margins to zero.
void LaunchFillBias(float* output, std::size_t channels, std::size_t length,
                    std::size_t pad, const float* bias, hipStream_t stream);

/// Adds the channel bias to the interior.
void LaunchAddBias(float* values, std::size_t channels, std::size_t length,
                   std::size_t pad, const float* bias, hipStream_t stream);

/// Interleaves the `stride` polyphase outputs `[stride][channels][phase]` of
/// a transposed convolution into `output` with its bias:
/// `output[c][t] = phases[(t + padding) % stride][c][(t + padding) / stride]`.
void LaunchInterleavePhases(const float* phases, std::size_t phase_length,
                            std::size_t stride, std::size_t padding,
                            std::size_t channels, std::size_t length,
                            std::size_t pad, const float* bias, float* output,
                            hipStream_t stream);

/// Copies stereo samples `[first, first + count)` of a two-channel padded
/// activation to frame-interleaved `output`.
void LaunchInterleaveStereo(const float* input, std::size_t length,
                            std::size_t pad, std::size_t first,
                            std::size_t count, float* output,
                            hipStream_t stream);

}  // namespace gufo::models::yue2::hip

#endif  // GUFO_MODELS_YUE2_HIP_KERNELS_HPP_

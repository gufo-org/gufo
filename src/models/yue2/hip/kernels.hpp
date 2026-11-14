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

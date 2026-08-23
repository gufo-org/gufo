#ifndef STRIX_MODELS_QWEN_ORACLES_HPP_
#define STRIX_MODELS_QWEN_ORACLES_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace strix::models::qwen {

/// BF16 bit-exact bitcast and representation utilities.
[[nodiscard]] inline std::uint16_t FloatToBf16(float f) noexcept {
  std::uint32_t bits = 0;
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  // Round to nearest even / truncate upper 16 bits
  std::memcpy(&bits, &f, sizeof(float));
  const std::uint32_t lsb = (bits >> 16) & 1U;
  const std::uint32_t rounding_bias = 0x7FFFU + lsb;
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16);
}

[[nodiscard]] inline float Bf16ToFloat(std::uint16_t b) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

/// Computes pure FP64 reference RMSNorm across a 1D vector:
/// out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
void ReferenceRMSNorm(std::span<const float> x, std::span<const float> weight,
                      float eps, std::span<float> out) noexcept;

/// Computes pure FP64 reference Rotary Position Embedding (RoPE) for a single
/// head. head: span of size head_dim (must be even, typically 128) pos: token
/// sequence index (0, 1, 2, ...) rope_theta: base frequency (default 1000000.0)
void ReferenceRoPE(std::span<const float> head, std::size_t pos,
                   float rope_theta, std::span<float> out) noexcept;

/// Computes SiLU activation: x / (1.0 + exp(-x))
[[nodiscard]] inline float ReferenceSiLU(float x) noexcept {
  return x / (1.0F + std::exp(-x));
}

/// Computes SwiGLU gated activation: out[i] = SiLU(gate[i]) * up[i]
void ReferenceSwiGLU(std::span<const float> gate, std::span<const float> up,
                     std::span<float> out) noexcept;

/// Computes numerically stable FP64 reference Softmax across a 1D vector.
void ReferenceSoftmax(std::span<const float> x, std::span<float> out) noexcept;

/// Computes reference General Matrix-Vector Multiplication: y = W * x
/// W: row-major matrix of shape [rows, cols]
/// x: vector of length cols
/// y: vector of length rows
void ReferenceGEMV(std::span<const float> matrix, std::span<const float> x,
                   std::size_t rows, std::size_t cols,
                   std::span<float> y) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_ORACLES_HPP_

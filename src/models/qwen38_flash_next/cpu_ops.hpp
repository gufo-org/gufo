#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CPU_OPS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen38_flash_next/rope_scaling.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

/// Scalar CPU operators for the reference model: one row at a time, every
/// format decoded on the fly. Correctness oracle only; nothing here is tuned.
namespace gufo::models::qwen38_flash_next::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// Dot product of one weight row with x.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm over `x` in place with gamma `w` (nullptr keeps the raw scale).
void RmsNorm(std::span<float> x, const float* w, float eps);
/// x / sqrt(sum x^2 + eps), the DeltaNet query/key normalization.
void L2Norm(std::span<float> x, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
[[nodiscard]] float Softplus(float x) noexcept;

/// NEOX-style partial rotary embedding on the first `rotary_dim` elements of
/// each `head_dim` head: pair (i, i + rotary_dim/2) rotates by pos * theta_i.
/// Text-only IMRoPE has equal positions on every section, so it is exactly
/// this rotation. With `yarn` enabled each pair's frequency is multiplied by
/// YarnPairMultiplier and cos/sin by yarn.mscale.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta,
          const YarnRope& yarn = {});

}  // namespace gufo::models::qwen38_flash_next::cpu

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_CPU_OPS_HPP_

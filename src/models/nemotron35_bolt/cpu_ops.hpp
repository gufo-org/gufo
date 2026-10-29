#ifndef GUFO_MODELS_NEMOTRON35_BOLT_CPU_OPS_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_CPU_OPS_HPP_

#include <cstdint>
#include <span>

#include "src/models/nemotron35_bolt/weights.hpp"

/// Scalar CPU operators for the reference model: one row at a time, every
/// format decoded on the fly. Correctness oracle only; nothing here is tuned.
namespace gufo::models::nemotron35_bolt::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (t.cols floats). The
/// per-tensor scale is not applied.
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = scale * sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// RMSNorm over `x` in place with gamma `w`.
void RmsNorm(std::span<float> x, const float* w, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
/// ggml's softplus: identity above 20.
[[nodiscard]] float Softplus(float x) noexcept;
/// max(x, 0)^2, the ReLU^2 activation of every Nemotron-H MLP.
[[nodiscard]] float Relu2(float x) noexcept;

}  // namespace gufo::models::nemotron35_bolt::cpu

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_CPU_OPS_HPP_

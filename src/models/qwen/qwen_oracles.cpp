#include "src/models/qwen/qwen_oracles.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace strix::models::qwen {

void ReferenceRMSNorm(std::span<const float> x, std::span<const float> weight,
                      float eps, std::span<float> out) noexcept {
  if (x.empty() || x.size() != weight.size() || x.size() != out.size()) {
    return;
  }
  double sum_sq = 0.0;
  for (const float val : x) {
    sum_sq += static_cast<double>(val) * static_cast<double>(val);
  }
  const double mean_sq = sum_sq / static_cast<double>(x.size());
  const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = static_cast<float>(static_cast<double>(x[i]) * inv_rms *
                                static_cast<double>(weight[i]));
  }
}

void ReferenceRoPE(std::span<const float> head, std::size_t pos,
                   float rope_theta, std::span<float> out) noexcept {
  const std::size_t dim = head.size();
  if (dim == 0 || (dim % 2 != 0) || head.size() != out.size()) {
    return;
  }
  const std::size_t half_dim = dim / 2;
  for (std::size_t i = 0; i < half_dim; ++i) {
    const double freq_exp =
        2.0 * static_cast<double>(i) / static_cast<double>(dim);
    const double freq =
        1.0 / std::pow(static_cast<double>(rope_theta), freq_exp);
    const double angle = static_cast<double>(pos) * freq;
    const double cos_val = std::cos(angle);
    const double sin_val = std::sin(angle);

    const auto v0 = static_cast<double>(head[i]);
    const auto v1 = static_cast<double>(head[i + half_dim]);

    out[i] = static_cast<float>((v0 * cos_val) - (v1 * sin_val));
    out[i + half_dim] = static_cast<float>((v0 * sin_val) + (v1 * cos_val));
  }
}

void ReferenceSwiGLU(std::span<const float> gate, std::span<const float> up,
                     std::span<float> out) noexcept {
  if (gate.size() != up.size() || gate.size() != out.size()) {
    return;
  }
  for (std::size_t i = 0; i < gate.size(); ++i) {
    out[i] = ReferenceSiLU(gate[i]) * up[i];
  }
}

void ReferenceSoftmax(std::span<const float> x, std::span<float> out) noexcept {
  if (x.empty() || x.size() != out.size()) {
    return;
  }
  float max_val = x[0];
  for (const float v : x) {
    max_val = std::max(v, max_val);
  }
  double sum_exp = 0.0;
  for (const float val : x) {
    sum_exp += std::exp(static_cast<double>(val - max_val));
  }
  const double inv_sum = 1.0 / sum_exp;
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = static_cast<float>(std::exp(static_cast<double>(x[i] - max_val)) *
                                inv_sum);
  }
}

void ReferenceGEMV(std::span<const float> matrix, std::span<const float> x,
                   std::size_t rows, std::size_t cols,
                   std::span<float> y) noexcept {
  if (matrix.size() != (rows * cols) || x.size() != cols || y.size() != rows) {
    return;
  }
  for (std::size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    const std::size_t row_offset = r * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      acc += static_cast<double>(matrix[row_offset + c]) *
             static_cast<double>(x[c]);
    }
    y[r] = static_cast<float>(acc);
  }
}

}  // namespace strix::models::qwen

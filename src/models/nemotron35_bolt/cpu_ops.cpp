#include "src/models/nemotron35_bolt/cpu_ops.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::nemotron35_bolt::cpu {

using core::GgmlType;

void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  const std::size_t k = t.cols;
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        out[i] = gufo::quant::Fp16ToFloat(h);
      }
      break;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kIQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(src, out, k);
      break;
    case GgmlType::kNVFP4:
      gufo::quant::DequantizeNVFP4(src, out, k);
      break;
    default:
      std::memset(out, 0, k * sizeof(float));
      break;
  }
}

void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y) {
  const auto rows = static_cast<std::int64_t>(t.rows);
#pragma omp parallel
  {
    std::vector<float> w(t.cols);
#pragma omp for schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      DequantizeRow(t, e, static_cast<std::uint64_t>(r), w.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < t.cols; ++i) {
        acc += static_cast<double>(w[i]) * x[i];
      }
      y[static_cast<std::size_t>(r)] = static_cast<float>(acc) * t.scale;
    }
  }
}

void RmsNorm(std::span<float> x, const float* w, float eps) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(ss / static_cast<double>(x.size())) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * scale * w[i];
  }
}

float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}
float Silu(float x) noexcept {
  return x * Sigmoid(x);
}
float Softplus(float x) noexcept {
  return x > 20.0F ? x : std::log1p(std::exp(x));
}
float Relu2(float x) noexcept {
  const float r = x > 0.0F ? x : 0.0F;
  return r * r;
}

}  // namespace gufo::models::nemotron35_bolt::cpu

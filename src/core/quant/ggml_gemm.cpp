#include "src/core/quant/ggml_gemm.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"

namespace strix::quant {
namespace {

float Bf16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t u32 = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &u32, sizeof(float));
  return f;
}

bool IsQuantized(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
    case core::GgmlType::kQ8_0:
      return true;
    default:
      return false;
  }
}

}  // namespace

bool IsSupported(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kF16 ||
         type == core::GgmlType::kBF16 || IsQuantized(type);
}

void Dequantize(core::GgmlType type, const void* src, float* dst,
                std::size_t k) {
  switch (type) {
    case core::GgmlType::kF32:
      std::memcpy(dst, src, k * sizeof(float));
      return;
    case core::GgmlType::kF16: {
      const auto* p = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < k; ++i) {
        dst[i] = Fp16ToFloat(p[i]);
      }
      return;
    }
    case core::GgmlType::kBF16: {
      const auto* p = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < k; ++i) {
        dst[i] = Bf16ToFloat(p[i]);
      }
      return;
    }
    case core::GgmlType::kQ3_K:
      DequantizeQ3_K(src, dst, k);
      return;
    case core::GgmlType::kQ4_K:
      DequantizeQ4_K(src, dst, k);
      return;
    case core::GgmlType::kQ5_K:
      DequantizeQ5_K(src, dst, k);
      return;
    case core::GgmlType::kQ6_K:
      DequantizeQ6_K(src, dst, k);
      return;
    case core::GgmlType::kQ8_K:
      DequantizeQ8_K(src, dst, k);
      return;
    case core::GgmlType::kQ8_0:
      DequantizeQ8_0(src, dst, k);
      return;
    default:
      assert(false && "quant::Dequantize: unsupported GgmlType");
      std::abort();
  }
}

float Dot(core::GgmlType type, const void* row, std::span<const float> x,
          std::size_t k) {
  switch (type) {
    case core::GgmlType::kF32: {
      const auto* p = static_cast<const float*>(row);
      float sum = 0.0F;
      for (std::size_t i = 0; i < k; ++i) {
        sum += p[i] * x[i];
      }
      return sum;
    }
    case core::GgmlType::kF16: {
      const auto* p = static_cast<const std::uint16_t*>(row);
      float sum = 0.0F;
      for (std::size_t i = 0; i < k; ++i) {
        sum += Fp16ToFloat(p[i]) * x[i];
      }
      return sum;
    }
    case core::GgmlType::kBF16: {
      const auto* p = static_cast<const std::uint16_t*>(row);
      float sum = 0.0F;
      for (std::size_t i = 0; i < k; ++i) {
        sum += Bf16ToFloat(p[i]) * x[i];
      }
      return sum;
    }
    case core::GgmlType::kQ3_K:
      return DotProductQ3_K(row, x, k);
    case core::GgmlType::kQ4_K:
      return DotProductQ4_K(row, x, k);
    case core::GgmlType::kQ5_K:
      return DotProductQ5_K(row, x, k);
    case core::GgmlType::kQ6_K:
      return DotProductQ6_K(row, x, k);
    case core::GgmlType::kQ8_K:
      return DotProductQ8_K(row, x, k);
    case core::GgmlType::kQ8_0:
      return DotProductQ8_0(row, x, k);
    default:
      assert(false && "quant::Dot: unsupported GgmlType");
      std::abort();
  }
}

}  // namespace strix::quant

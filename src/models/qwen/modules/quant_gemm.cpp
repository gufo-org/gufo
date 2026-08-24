#include "src/models/qwen/modules/quant_gemm.hpp"

#include <cstdlib>
#include <limits>

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops/gemm.hpp"
#endif

namespace strix::models::qwen {

void QuantGemm(const CpuModuleContext&, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  TensorGEMV(A, x, M, K, y);
}

#if defined(ENGINE_ENABLE_HIP)
namespace {

[[nodiscard]] constexpr bool SupportsHipGemm(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K;
}

}  // namespace

void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  const bool shape_overflows =
      M != 0 && K > (std::numeric_limits<std::size_t>::max() / M);
  const bool quantized = A.type != core::GgmlType::kF32 &&
                         A.type != core::GgmlType::kBF16;
  if (A.empty() || M == 0 || K == 0 || !SupportsHipGemm(A.type) ||
      shape_overflows || x.size() < K || y.size() < M ||
      (!shape_overflows && A.num_elements < M * K) ||
      (quantized && strix::quant::QuantizedRowBytes(A.type, K) == 0)) {
    std::abort();
  }
  ::strix::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                           static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace strix::models::qwen

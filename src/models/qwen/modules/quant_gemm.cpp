#include "src/models/qwen/modules/quant_gemm.hpp"

#include <cstdlib>

#include "src/families/qwen/hip_common/gemm_route.hpp"
#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/families/qwen/kernels/ops/gemm.hpp"
#endif

namespace gufo::models::qwen {

void QuantGemm(const CpuModuleContext&, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  TensorGEMV(A, x, M, K, y);
}

#if defined(ENGINE_ENABLE_HIP)
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  const auto resolution =
      ResolveQwenGemmRoute({.type = A.type,
                            .batch_size = 1,
                            .m = M,
                            .k = K,
                            .mode = QwenGemmMode::kHipDecode});
  if (A.empty() || !resolution.accepted() || x.size() < K || y.size() < M ||
      A.num_elements < M * K) {
    std::abort();
  }
  ::gufo::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                          static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace gufo::models::qwen

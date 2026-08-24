#include "src/models/qwen/modules/quant_gemm.hpp"

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops.hpp"
#endif

namespace strix::models::qwen {

void QuantGemm(const CpuModuleContext&, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  TensorGEMV(A, x, M, K, y);
}

#if defined(ENGINE_ENABLE_HIP)
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  ::strix::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                           static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace strix::models::qwen

#include "src/models/qwen/modules/quant_gemm.hpp"

#include "src/models/qwen/qwen_forward.hpp"  // TensorGEMV

namespace strix::models::qwen {

void QuantGemm(ModuleCtx&, const QwenTensorRef& A, std::span<const float> x,
               std::size_t M, std::size_t K, std::span<float> y) noexcept {
  // CPU backend: delegates to the shared TensorGEMV dispatch (the canonical
  // quant_gemm seam). No body is moved — TensorGEMV has many callers.
  TensorGEMV(A, x, M, K, y);
}

}  // namespace strix::models::qwen

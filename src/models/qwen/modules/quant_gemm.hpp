#ifndef STRIX_MODELS_QWEN_MODULES_QUANT_GEMM_HPP_
#define STRIX_MODELS_QWEN_MODULES_QUANT_GEMM_HPP_

#include <cstddef>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/state.hpp"  // QwenTensorRef

namespace strix::models::qwen {

/// GEMV dispatch: y = A @ x.
///
/// The CPU backend supports F32, BF16, F16, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0,
/// and Q8_K through the canonical host dispatch. The HIP backend supports only
/// the formats implemented by `LaunchGEMV`: F32, BF16, Q5_K, Q6_K, Q8_0, and
/// Q8_K. Unsupported formats or invalid row geometry fail before launch.
///
/// CPU backend proxies the shared `TensorGEMV` dispatch (src/models/
/// forward.cpp), which routes through the canonical `quant::` helpers.
/// The module wraps the existing dispatch (the shared quant_gemm seam) rather
/// than moving `TensorGEMV`'s body in — `TensorGEMV` has many callers (SSM,
/// attention, MTP) that must not change. Behavior-identical.
void QuantGemm(const CpuModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept;
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_QUANT_GEMM_HPP_

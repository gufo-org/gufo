#ifndef STRIX_MODELS_QWEN_MODULES_QUANT_GEMM_HPP_
#define STRIX_MODELS_QWEN_MODULES_QUANT_GEMM_HPP_

#include <cstddef>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/state.hpp"  // QwenTensorRef

namespace strix::models::qwen {

/// Quantized GEMV dispatch: y = A @ x for quantized `A` (F32/BF16/F16/Q3_K/
/// Q4_K/Q5_K/Q6_K/Q8_0/Q8_K.
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

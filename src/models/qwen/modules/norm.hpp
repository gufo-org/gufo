#ifndef STRIX_MODELS_QWEN_MODULES_NORM_HPP_
#define STRIX_MODELS_QWEN_MODULES_NORM_HPP_

#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"

namespace strix::models::qwen {

/// Layer norm (attn pre-norm, ffn pre-norm, final output norm):
/// out = (x / rms(x) + eps) * weight.
///
/// CPU backend reproduces the existing `ForwardRMSNorm` computation exactly
/// (the pure FP64 `ReferenceRMSNorm` oracle, with non-F32 weight dequantized
/// to F32 first). The HIP backend lands in a later phase — `ctx.backend` is
/// not consulted yet. In-place is safe (`x == out`): the oracle reads all of
/// `x` before it writes any of `out`.
void NormForward(ModuleCtx& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_NORM_HPP_

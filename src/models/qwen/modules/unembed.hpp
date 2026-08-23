#ifndef STRIX_MODELS_QWEN_MODULES_UNEMBED_HPP_
#define STRIX_MODELS_QWEN_MODULES_UNEMBED_HPP_

#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen_state.hpp"

namespace strix::models::qwen {

/// Final output: RMSNorm the hidden state then project through the LM head
/// onto logits. `output_weight` may share storage with the token embeddings
/// (tied LM head).
void UnembedForward(ModuleCtx& ctx, const QwenTensorRef& output_norm,
                    const QwenTensorRef& output_weight,
                    std::span<const float> hidden,
                    std::span<float> logits_out) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_UNEMBED_HPP_

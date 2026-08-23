#include "src/models/qwen/modules/unembed.hpp"

#include "src/models/qwen_forward.hpp"  // ForwardRMSNorm, TensorGEMV

namespace strix::models::qwen {

void UnembedForward(ModuleCtx& ctx, const QwenTensorRef& output_norm,
                    const QwenTensorRef& output_weight,
                    std::span<const float> hidden,
                    std::span<float> logits_out) noexcept {
  // CPU backend: the final-norm + lm_head block lifted verbatim from
  // ForwardModel.
  ForwardRMSNorm(hidden, output_norm, 1e-6F, ctx.arena->normed);
  if (!output_weight.empty()) {
    TensorGEMV(output_weight, ctx.arena->normed, ctx.config->vocab_size,
               ctx.config->hidden_size, logits_out);
  }
}

}  // namespace strix::models::qwen

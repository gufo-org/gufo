#include "src/models/qwen/modules/unembed.hpp"

#include "src/models/qwen/forward.hpp"  // ForwardRMSNorm, TensorGEMV

namespace strix::models::qwen {

void UnembedForward(const CpuLayerContext& ctx,
                    const QwenTensorRef& output_norm,
                    const QwenTensorRef& output_weight,
                    std::span<const float> hidden,
                    std::span<float> logits_out) noexcept {
  // CPU backend: the final-norm + lm_head block lifted verbatim from
  // ForwardModel.
  auto& scratch = ctx.Scratch();
  const auto& config = ctx.Config();
  ForwardRMSNorm(hidden, output_norm, 1e-6F, scratch.normed);
  if (!output_weight.empty()) {
    TensorGEMV(output_weight, scratch.normed, config.vocab_size,
               config.hidden_size, logits_out);
  }
}

}  // namespace strix::models::qwen

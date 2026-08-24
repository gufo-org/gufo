#include "src/models/qwen/modules/modules.hpp"

#include <algorithm>

namespace strix::models::qwen {

void SsmForward(const CpuLayerContext& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept {
  // Thin forwarding shim (Phase 1 canary): the module signature is a slice over
  // the layer, but the existing production `ForwardSSM` still needs the full
  // QwenLayerWeights + config + the arena scratch buffers. The SsmLayerView
  // carries the backing layer (`source`); everything else comes from the typed
  // CPU layer context. The view's source is established by MakeSsmView.
  if (view.source == nullptr) {
    std::ranges::fill(out, 0.0F);
    return;
  }

  auto& scratch = ctx.Scratch();
  ::strix::models::ForwardSSM(x, *view.source, ctx.Config(), state,
                              ctx.LayerIndex(), scratch.ssm_qkv,
                              scratch.ssm_gate, scratch.ssm_out_buf, out);
}

}  // namespace strix::models::qwen

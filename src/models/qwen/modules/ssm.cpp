#include "src/models/qwen/modules/modules.hpp"

#include <algorithm>

namespace strix::models::qwen {

void SsmForward(ModuleCtx& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept {
  // Thin forwarding shim (Phase 1 canary): the module signature is a slice over
  // the layer, but the existing production `ForwardSSM` still needs the full
  // QwenLayerWeights + config + the arena scratch buffers. The SsmLayerView
  // carries the backing layer (`source`); everything else comes from ModuleCtx.
  // This is a straight forward — no duplicated math, no behavior change. Once
  // the module body is extracted in Phase 2 this shim is replaced by the
  // module's own implementation and `source` is dropped.
  if (ctx.config == nullptr || ctx.arena == nullptr || view.source == nullptr) {
    // Degenerate/unsanitized call site: mirror ForwardSSM's silent-fill guard.
    std::ranges::fill(out, 0.0F);
    return;
  }

  ::strix::models::ForwardSSM(x, *view.source, *ctx.config, state,
                              ctx.layer_idx, ctx.arena->ssm_qkv,
                              ctx.arena->ssm_gate, ctx.arena->ssm_out_buf, out);
}

}  // namespace strix::models::qwen

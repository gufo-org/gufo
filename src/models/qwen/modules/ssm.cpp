#include "src/models/qwen/modules/modules.hpp"

namespace strix::models::qwen {

void SsmForward(const CpuLayerContext& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept {
  auto& scratch = ctx.Scratch();
  ::strix::models::ForwardSSM(x, view, state, ctx.LayerIndex(), scratch.ssm_qkv,
                              scratch.ssm_gate, scratch.ssm_out_buf, out);
}

}  // namespace strix::models::qwen

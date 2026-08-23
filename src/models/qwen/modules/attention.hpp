#ifndef STRIX_MODELS_QWEN_MODULES_ATTENTION_HPP_
#define STRIX_MODELS_QWEN_MODULES_ATTENTION_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen_state.hpp"

namespace strix::models::qwen {

/// Full self-attention for one sequence position (gated GQA):
/// QKV proj -> QK-norm -> RoPE -> KV-cache write -> score/softmax/context ->
/// gate -> out proj.
///
/// CPU backend reproduces the exact attention branch formerly inlined in
/// `ForwardLayer` (both the fused `2*q_size` Q+gate projection and the plain
/// Q path), calling the kept `ForwardAttention` helper for the score/context
/// core. Stateless in signature; KV state lives in `kv` + `ctx.layer_idx`.
void AttnForward(ModuleCtx& ctx, const AttnLayerView& view,
                 std::span<const float> x, QwenKvCache& kv,
                 std::uint32_t pos, std::span<float> out) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_ATTENTION_HPP_

#ifndef STRIX_MODELS_QWEN_MODULES_ROPE_HPP_
#define STRIX_MODELS_QWEN_MODULES_ROPE_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"

namespace strix::models::qwen {

/// RoPE rotation on Q and K heads for the given position.
///
/// CPU backend reproduces the existing `ForwardRoPE` computation exactly
/// (`ReferenceRoPE` applied per head). In-place safe: `q`/`k` may be mutated
/// in place.
void RopeForward(ModuleCtx& ctx, const RopeLayerView& view,
                 std::span<float> q, std::span<float> k,
                 std::uint32_t pos) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_ROPE_HPP_

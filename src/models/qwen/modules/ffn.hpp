#ifndef STRIX_MODELS_QWEN_MODULES_FFN_HPP_
#define STRIX_MODELS_QWEN_MODULES_FFN_HPP_

#include <cstddef>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"

namespace strix::models::qwen {

/// SwiGLU FFN: out = (SiLU(gate) * up) @ down.
///
/// CPU backend reproduces the existing `ForwardFFN` computation exactly.
/// The three scratch buffers (`gate_scratch`, `up_scratch`, `act_scratch`)
/// hold the intermediate GEMV results; the composition layer supplies them
/// (typically from the arena's `mlp_gate`/`mlp_up`/`mlp_act`/`mlp_out`).
void FfnForward(ModuleCtx& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_FFN_HPP_

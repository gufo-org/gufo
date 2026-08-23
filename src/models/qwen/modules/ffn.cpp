#include "src/models/qwen/modules/ffn.hpp"

#include <cmath>

#include "src/models/qwen_forward.hpp"  // TensorGEMV

namespace strix::models::qwen {

void FfnForward(ModuleCtx&, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept {
  // CPU backend: the SwiGLU FFN body lifted verbatim from ForwardFFN.
  if (!view.gate.empty()) {
    TensorGEMV(view.gate, x, view.intermediate_size, view.hidden_size,
               gate_scratch);
  }
  if (!view.up.empty()) {
    TensorGEMV(view.up, x, view.intermediate_size, view.hidden_size,
               up_scratch);
  }

  // SwiGLU activation
  for (std::size_t i = 0; i < view.intermediate_size; ++i) {
    const float g = gate_scratch[i];
    const float silu_g = g / (1.0F + std::exp(-g));
    act_scratch[i] = silu_g * up_scratch[i];
  }

  if (!view.down.empty()) {
    TensorGEMV(view.down, act_scratch, view.hidden_size,
               view.intermediate_size, out);
  }
}

}  // namespace strix::models::qwen

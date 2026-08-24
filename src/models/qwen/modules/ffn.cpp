#include "src/models/qwen/modules/ffn.hpp"

#include <cmath>

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#endif

namespace strix::models::qwen {

void FfnForward(const CpuModuleContext&, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept {
  if (!view.gate.empty()) {
    TensorGEMV(view.gate, x, view.intermediate_size, view.hidden_size,
               gate_scratch);
  }
  if (!view.up.empty()) {
    TensorGEMV(view.up, x, view.intermediate_size, view.hidden_size,
               up_scratch);
  }

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

#if defined(ENGINE_ENABLE_HIP)
void FfnForward(const HipModuleContext& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept {
  (void)gate_scratch;
  (void)up_scratch;
  const auto stream = static_cast<hipStream_t>(ctx.Stream());
  ::strix::hip::LaunchFusedSwiGLUGEMV(
      view.gate.data, view.gate.type, view.up.data, view.up.type, x.data(),
      act_scratch.data(), view.intermediate_size, view.hidden_size, stream);
  ::strix::hip::LaunchGEMV(view.down.data, view.down.type, act_scratch.data(),
                           out.data(), view.hidden_size,
                           view.intermediate_size, stream);
}
#endif

}  // namespace strix::models::qwen

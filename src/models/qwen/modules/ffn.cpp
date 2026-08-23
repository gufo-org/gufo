#include "src/models/qwen/modules/ffn.hpp"

#include <cmath>

#include "src/models/qwen/qwen_forward.hpp"  // TensorGEMV

// GOP ops (LaunchFusedSwiGLUGEMV etc.) only exist under a HIP build; the
// CPU-only strix_core build must not reference them.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#endif

namespace strix::models::qwen {

void FfnForward(ModuleCtx& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept {
#if defined(ENGINE_ENABLE_HIP)
  // HIP backend: reproduce the decode path's fused SwiGLU FFN. The decode runs
  // the fused LaunchFusedSwiGLUGEMV (computes SiLU(gate) * up in one kernel —
  // no separate gate/up scratch) then LaunchGEMV for the down projection.
  // gate_scratch/up_scratch are unused here because the fused kernel computes
  // both GEMVs internally; x is the (pre-)normed input, act_scratch is the
  // arena's d_ffn_act slab, out is d_ffn_out. Behavior-identical to the former
  // inline calls (same kernels, same device pointers, same args).
  if (ctx.backend == Backend::Hip) {
    (void)gate_scratch;
    (void)up_scratch;
    ::strix::hip::LaunchFusedSwiGLUGEMV(
        view.gate.data, view.gate.type, view.up.data, view.up.type, x.data(),
        act_scratch.data(), view.intermediate_size, view.hidden_size,
        static_cast<hipStream_t>(ctx.stream));
    ::strix::hip::LaunchGEMV(view.down.data, view.down.type,
                             act_scratch.data(), out.data(),
                             view.hidden_size, view.intermediate_size,
                             static_cast<hipStream_t>(ctx.stream));
    return;
  }
#endif
  // CPU backend: the SwiGLU FFN body lifted verbatim from ForwardFFN. `ctx` is
  // not consulted here (only the HIP branch does); keep -Wunused-parameter
  // quiet when ENGINE_ENABLE_HIP is undefined (CPU-only build).
  (void)ctx;
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

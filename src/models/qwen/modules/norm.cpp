#include "src/models/qwen/modules/norm.hpp"

#include <cstddef>
#include <vector>

#include "src/models/qwen/qwen_oracles.hpp"

// GOP ops (LaunchRMSNorm etc.) only exist under a HIP build; the CPU-only
// strix_core build must not reference them.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#endif

namespace strix::models::qwen {

void NormForward(ModuleCtx& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept {
#if defined(ENGINE_ENABLE_HIP)
  // HIP backend: reproduce the decode path's unfused pre-RMSNorm launch. The
  // device input/output are handed in as `x`/`out` (the arena's d_hidden /
  // d_normed slices) by the composition layer; the forward weight is the view
  // handle (attn_norm / ffn_norm). Behavior-identical to the prior inline
  // `LaunchRMSNorm(arena_.d_hidden, (const float*)layer.attn_norm.data,
  //                 arena_.d_normed, hidden_size, 1e-6F, arena_.stream)` call.
  if (ctx.backend == Backend::Hip) {
    ::strix::hip::LaunchRMSNorm(x.data(),
                                static_cast<const float*>(view.weight.data),
                                out.data(), x.size(), view.eps,
                                static_cast<hipStream_t>(ctx.stream));
    return;
  }
#endif
  // CPU backend: the RMSNorm body lifted verbatim from ForwardRMSNorm. Weight
  // is dequantized to F32 when the view holds a non-F32 handle; the pure FP64
  // oracle does the actual RMSNorm. `ctx` is not consulted here (only the HIP
  // branch does); keep -Wunused-parameter quiet when ENGINE_ENABLE_HIP is
  // undefined (CPU-only build).
  (void)ctx;
  if (view.weight.type == core::GgmlType::kF32) {
    ReferenceRMSNorm(x, view.weight.AsFloatSpan(), view.eps, out);
  } else {
    std::vector<float> w_f32(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      w_f32[i] = view.weight.Get(i);
    }
    ReferenceRMSNorm(x, w_f32, view.eps, out);
  }
}

}  // namespace strix::models::qwen

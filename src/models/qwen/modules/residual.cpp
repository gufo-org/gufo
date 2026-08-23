#include "src/models/qwen/modules/residual.hpp"

#include <cstddef>

// GOP ops (LaunchResidualAdd etc.) only exist under a HIP build; the CPU-only
// strix_core build must not reference them.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#endif

namespace strix::models::qwen {

void ResidualAdd(ModuleCtx& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept {
#if defined(ENGINE_ENABLE_HIP)
  // HIP backend: reproduce the decode path's unfused residual-add launch. The
  // composition layer hands in the arena's device spans (d_hidden as `dst` and
  // d_attn_out / d_ffn_out as `src`). Behavior-identical to the former inline
  // `LaunchResidualAdd(arena_.d_hidden, arena_.d_attn_out, arena_.d_hidden, ...)`
  // call: out = dst + src, written in-place on dst.
  if (ctx.backend == Backend::Hip) {
    ::strix::hip::LaunchResidualAdd(dst.data(), src.data(), dst.data(),
                                    dst.size(),
                                    static_cast<hipStream_t>(ctx.stream));
    return;
  }
#endif
  // CPU backend: the residual add is the element-wise sum; the body was
  // previously inlined in ForwardLayer (the two `hidden += attn_out / mlp_out`
  // loops). Moved here so the composition drives the module instead. `ctx` is
  // not consulted here (only the HIP branch does); keep -Wunused-parameter
  // quiet when ENGINE_ENABLE_HIP is undefined (CPU-only build).
  (void)ctx;
  for (std::size_t i = 0; i < dst.size(); ++i) {
    dst[i] += src[i];
  }
}

}  // namespace strix::models::qwen

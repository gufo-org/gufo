#include "src/models/qwen/modules/quant_gemm.hpp"

#include "src/models/qwen/qwen_forward.hpp"  // TensorGEMV

// GOP ops (LaunchGEMV etc.) only exist under a HIP build; the CPU-only
// strix_core build must not reference them.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#endif

namespace strix::models::qwen {

void QuantGemm(ModuleCtx& ctx, const QwenTensorRef& A, std::span<const float> x,
               std::size_t M, std::size_t K, std::span<float> y) noexcept {
#if defined(ENGINE_ENABLE_HIP)
  // HIP backend: single-weight quantized GEMV (y = A @ x). Arg-maps to the
  // decode path's `LaunchGEMV`. Behavior-identical to the former inline call:
  // same device weight pointer/type, same x/y pointers, same M/K, same stream.
  //
  // NOTE: the decode's QKV projection is NOT routed here. It is a fused
  // multi-head kernel (`LaunchFusedQKVProjections`, three weight handles
  // q/k/v), which a single-weight `QuantGemm` cannot represent; it stays in the
  // composition layer. The attention-output projection (single weight) IS a
  // faithful `QuantGemm` and is wired through this module.
  if (ctx.backend == Backend::Hip) {
    ::strix::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                             static_cast<hipStream_t>(ctx.stream));
    return;
  }
#endif
  // CPU backend: delegates to the shared TensorGEMV dispatch (the canonical
  // quant_gemm seam). No body is moved — TensorGEMV has many callers. `ctx` is
  // only consulted by the HIP branch; keep -Wunused-parameter quiet on CPU-only
  // builds.
  (void)ctx;
  TensorGEMV(A, x, M, K, y);
}

}  // namespace strix::models::qwen

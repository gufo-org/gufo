#ifndef STRIX_MODELS_QWEN_MODULES_FWD_HPP_
#define STRIX_MODELS_QWEN_MODULES_FWD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen_ssm.hpp"
#include "src/models/qwen_state.hpp"

namespace strix::models::qwen {

// Module function declarations.
//
// One signature per architectural component, parameterized on the module's OWN
// LayerView (fixes the Data Clump) and a ModuleCtx. Activation tensors are
// `std::span<float>` (matching the existing CPU forward code); weights are
// QwenTensorRef inside the views. The plan's generic `const Tensor& x` maps to
// `std::span<const float>` / `std::span<float>` in this codebase.
//
// Each function has two backends (CPU reference / HIP kernels) behind the same
// signature; the CPU/HIP split is a Phase 2 implementation concern driven by
// `ctx.backend`. These are DECLARATIONS ONLY — bodies land in Phase 2 extraction.

/// Layer norm: out = (x / rms(x)) * weight. (attn pre-norm, ffn pre-norm, final.)
void NormForward(ModuleCtx& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;

/// RoPE rotation on Q and K heads for the given position.
void RopeForward(ModuleCtx& ctx, const RopeLayerView& view,
                 std::span<float> q, std::span<float> k,
                 std::uint32_t pos) noexcept;

/// Full attention: QKV proj -> RoPE -> KV write -> score -> out proj.
void AttnForward(ModuleCtx& ctx, const AttnLayerView& view,
                 std::span<const float> x, QwenKvCache& kv,
                 std::uint32_t pos, std::span<float> out) noexcept;

/// Gated DeltaNet linear attention: in-proj -> conv -> recurrent update -> gate
/// -> out.
void SsmForward(ModuleCtx& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept;

/// SwiGLU FFN.
//
// CPU backend operates on three intermediate scratch buffers (the GEMV
// results for gate/up and the activation); the composition layer supplies them
// from the arena's `mlp_gate`/`mlp_up`/`mlp_act`. This mirrors the existing
// `ForwardFFN` reference signature (which has no arena in its call — the old
// callers pass raw scratch spans), so the module can be a thin wrapper over it.
void FfnForward(ModuleCtx& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept;

/// Residual add: dst += src (standalone; the fused residual+norm route is a
/// composition-layer concern, not a module-local one).
void ResidualAdd(ModuleCtx& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;

/// Quantized GEMV dispatch: y = A @ x for quantized `A` (F32/BF16/Q3_K/Q4_K/
/// Q5_K/Q6_K/Q8_0/Q8_K), proxying src/core/quant/ggml_gemm.hpp.
void QuantGemm(ModuleCtx& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_FWD_HPP_

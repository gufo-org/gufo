#ifndef STRIX_MODELS_QWEN_MODULES_LAYER_VIEW_HPP_
#define STRIX_MODELS_QWEN_MODULES_LAYER_VIEW_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"

namespace strix::models::qwen {

// Lightweight per-module views over QwenLayerWeights.
//
// These are cheap value structs (QwenTensorRef is a small trivially-copyable
// handle), built from a const QwenLayerWeights& plus the config dims. They fix
// the current Data Clump: call sites drag the whole layer struct, config, both
// caches, and the arena into every forward. A module gets exactly the weight
// slice + dims it needs.

inline constexpr float kNormEps = 1e-6F;  ///< matches forward.cpp RMSNorm calls

/// Layer-norm slice (attn pre-norm, ffn pre-norm, or final output norm).
struct NormLayerView {
  QwenTensorRef weight;
  float eps = kNormEps;
};

/// RoPE rotation parameters + Q/K head dims.
struct RopeLayerView {
  std::uint32_t num_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t rotary_dim = 0;
  float rope_theta = 0.0F;
};

/// Full-attention layer slice (QKV projection weights + head dims).
struct AttnLayerView {
  QwenTensorRef q;
  QwenTensorRef k;
  QwenTensorRef v;
  QwenTensorRef output;
  QwenTensorRef q_norm;
  QwenTensorRef k_norm;
  std::uint32_t num_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;
};

/// Linear-attention / SSM layer slice (DeltaNet + conv + gating). The module
/// and compatibility wrapper share this exact tensor-and-shape contract.
using SsmLayerView = QwenSsmParameters;

/// SwiGLU FFN layer slice.
struct FfnLayerView {
  QwenTensorRef gate;
  QwenTensorRef up;
  QwenTensorRef down;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
};

/// Quantized-GEMM call parameters. Proxies the unified quant:: dispatch
/// (src/core/quant/ggml_gemm.hpp); carries no weights view of its own — the
/// caller passes the weight QwenTensorRef to QuantGemm directly.
struct QuantGemmLayerView {
  std::size_t M = 0;
  std::size_t K = 0;
};

// ---------------------------------------------------------------------------
// Factories. Bind a QwenLayerWeights slice so callers (the composition layer)
// don't hand-trace member names. `config` is unused where dims come from the
// weight handle itself.
// ---------------------------------------------------------------------------

inline NormLayerView MakeAttnNormView(const QwenLayerWeights& w,
                                      const core::ModelConfig&) {
  return NormLayerView{/*weight=*/w.attn_norm, kNormEps};
}
inline NormLayerView MakeFfnNormView(const QwenLayerWeights& w,
                                     const core::ModelConfig&) {
  return NormLayerView{/*weight=*/w.ffn_norm, kNormEps};
}
inline RopeLayerView MakeRopeView(const QwenLayerWeights&,
                                  const core::ModelConfig& c) {
  return RopeLayerView{c.num_attention_heads, c.num_key_value_heads,
                       c.head_dim, c.rotary_dim, c.rope_theta};
}
inline AttnLayerView MakeAttnView(const QwenLayerWeights& w,
                                  const core::ModelConfig& c) {
  return AttnLayerView{w.attn_q, w.attn_k, w.attn_v, w.attn_output,
                       w.attn_q_norm, w.attn_k_norm, c.num_attention_heads,
                       c.num_key_value_heads, c.head_dim};
}
inline SsmLayerView MakeSsmView(const QwenLayerWeights& w,
                                const core::ModelConfig& c) {
  return MakeQwenSsmParameters(w, c);
}
inline FfnLayerView MakeFfnView(const QwenLayerWeights& w,
                                const core::ModelConfig& c) {
  return FfnLayerView{w.ffn_gate, w.ffn_up, w.ffn_down, c.hidden_size,
                      c.intermediate_size};
}

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_LAYER_VIEW_HPP_

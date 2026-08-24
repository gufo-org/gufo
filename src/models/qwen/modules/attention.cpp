#include "src/models/qwen/modules/attention.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>

#include "src/models/qwen/forward.hpp"  // TensorGEMV, ForwardRMSNorm, ForwardRoPE, ForwardAttention

namespace strix::models::qwen {

void AttnForward(const CpuLayerContext& ctx, const AttnLayerView& view,
                 std::span<const float> x, QwenKvCache& kv,
                 std::uint32_t pos, std::span<float> out) noexcept {
  // CPU backend: the full-attention branch lifted verbatim from ForwardLayer
  // (see ForwardLayer's `if (layer.is_full_attention)` block). Weights come
  // from the view; scratch and dimensions come from the typed context. `pos`
  // is the caller's token index (matches the original ForwardLayer argument).
  const core::ModelConfig& config = ctx.Config();
  QwenScratchArena& arena = ctx.Scratch();
  const std::size_t hidden_size = config.hidden_size;
  const std::uint32_t head_dim = config.head_dim;
  const std::size_t q_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * head_dim;

  if (!view.q.empty()) {
    const std::size_t q_projection_size = view.q.num_elements / hidden_size;
    if (q_projection_size == 2 * q_size) {
      TensorGEMV(view.q, x, q_projection_size, hidden_size, arena.ssm_qkv);
      for (std::uint32_t h = 0; h < config.num_attention_heads; ++h) {
        const auto q_src = arena.ssm_qkv.subspan(
            (static_cast<std::size_t>(h) * head_dim * 2), head_dim);
        const auto g_src = arena.ssm_qkv.subspan(
            (static_cast<std::size_t>(h) * head_dim * 2) + head_dim,
            head_dim);
        std::ranges::copy(
            q_src,
            arena.q.begin() + static_cast<std::ptrdiff_t>(
                                  static_cast<std::size_t>(h) * head_dim));
        std::ranges::copy(g_src,
                          arena.ssm_gate.begin() +
                              static_cast<std::ptrdiff_t>(
                                  static_cast<std::size_t>(h) * head_dim));
      }
    } else {
      TensorGEMV(view.q, x, q_size, hidden_size, arena.q);
      std::ranges::fill(arena.ssm_gate, 0.0F);
    }
  }
  if (!view.k.empty()) {
    TensorGEMV(view.k, x, kv_size, hidden_size, arena.k);
  }
  if (!view.v.empty()) {
    TensorGEMV(view.v, x, kv_size, hidden_size, arena.v);
  }

  // Apply QK-Norm
  if (!view.q_norm.empty()) {
    for (std::uint32_t h = 0; h < config.num_attention_heads; ++h) {
      auto q_head =
          arena.q.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
      ForwardRMSNorm(q_head, view.q_norm, 1e-6F, q_head);
    }
  }
  if (!view.k_norm.empty()) {
    for (std::uint32_t h = 0; h < config.num_key_value_heads; ++h) {
      auto k_head =
          arena.k.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
      ForwardRMSNorm(k_head, view.k_norm, 1e-6F, k_head);
    }
  }

  ForwardRoPE(arena.q, arena.k, config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, config.rotary_dim,
              pos, config.rope_theta);

  ForwardAttention(arena.q, arena.k, arena.v, arena.ssm_gate.subspan(0, q_size),
                   view.output, kv,
                   ctx.LayerIndex() / config.full_attention_interval,
                   pos, config.num_attention_heads, config.num_key_value_heads,
                   config.head_dim, hidden_size, arena.attn_scores, out);
}

}  // namespace strix::models::qwen

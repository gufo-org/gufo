#include "src/models/qwen/modules/rope.hpp"

#include "src/models/qwen_oracles.hpp"

namespace strix::models::qwen {

void RopeForward(ModuleCtx&, const RopeLayerView& view,
                 std::span<float> q, std::span<float> k,
                 std::uint32_t pos) noexcept {
  // CPU backend: the RoPE body lifted verbatim from ForwardRoPE. Per-head
  // rotation via the pure FP64 ReferenceRoPE oracle.
  const std::uint32_t r_dim =
      (view.rotary_dim > 0 && view.rotary_dim <= view.head_dim)
          ? view.rotary_dim
          : view.head_dim;
  for (std::uint32_t h = 0; h < view.num_heads; ++h) {
    const auto q_slice =
        q.subspan(static_cast<std::size_t>(h) * view.head_dim, r_dim);
    ReferenceRoPE(q_slice, pos, view.rope_theta, q_slice);
  }
  for (std::uint32_t h = 0; h < view.num_kv_heads; ++h) {
    const auto k_slice =
        k.subspan(static_cast<std::size_t>(h) * view.head_dim, r_dim);
    ReferenceRoPE(k_slice, pos, view.rope_theta, k_slice);
  }
}

}  // namespace strix::models::qwen

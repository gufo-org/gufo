#include "src/models/qwen/modules/residual.hpp"

namespace strix::models::qwen {

void ResidualAdd(ModuleCtx&, std::span<float> dst,
                 std::span<const float> src) noexcept {
  // CPU backend: the residual add is the element-wise sum; the body was
  // previously inlined in ForwardLayer (the two `hidden += attn_out / mlp_out`
  // loops). Moved here so the composition drives the module instead.
  for (std::size_t i = 0; i < dst.size(); ++i) {
    dst[i] += src[i];
  }
}

}  // namespace strix::models::qwen

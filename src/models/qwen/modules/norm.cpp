#include "src/models/qwen/modules/norm.hpp"

#include <cstddef>
#include <vector>

#include "src/models/qwen_oracles.hpp"

namespace strix::models::qwen {

void NormForward(ModuleCtx&, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept {
  // CPU backend: the RMSNorm body lifted verbatim from ForwardRMSNorm. Weight
  // is dequantized to F32 when the view holds a non-F32 handle; the pure FP64
  // oracle does the actual RMSNorm.
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

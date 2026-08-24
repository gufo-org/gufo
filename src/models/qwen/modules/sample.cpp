#include "src/models/qwen/modules/sample.hpp"

#include "src/models/qwen/qwen_forward.hpp"  // GreedyArgmax

namespace strix::models::qwen {

std::uint32_t SampleForward(const CpuModuleContext&,
                            std::span<const float> logits) noexcept {
  // CPU backend: delegates to the existing greedy-argmax (already a small
  // standalone function). A sampling-policy module seam for the L1 tier.
  return GreedyArgmax(logits);
}

}  // namespace strix::models::qwen

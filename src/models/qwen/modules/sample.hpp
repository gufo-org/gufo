#ifndef STRIX_MODELS_QWEN_MODULES_SAMPLE_HPP_
#define STRIX_MODELS_QWEN_MODULES_SAMPLE_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"

namespace strix::models::qwen {

/// Greedy argmax over a logit distribution (sampling-policy module seam).
std::uint32_t SampleForward(ModuleCtx& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_SAMPLE_HPP_

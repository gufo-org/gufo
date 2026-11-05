#pragma once

#include <memory>
#include <string>

#include "gufo/ext/text.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::hip {
class QwenGpuModel;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::speculative {
struct SpeculativeOptions;
}  // namespace gufo::speculative

namespace gufo::models::qwen {

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy dflash_policy,
    std::string artifact_fingerprint, std::string draft_artifact_fingerprint);
#endif

}  // namespace gufo::models::qwen

#pragma once

#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {
class Model;

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {},
    std::uint32_t decode_concurrency = 1);
#endif

}  // namespace gufo::models::qwen38_flash_next

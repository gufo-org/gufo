#pragma once

#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::deepseek_v4_flash {
class Model;

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string support_fingerprint = {});
#endif

}  // namespace gufo::models::deepseek_v4_flash

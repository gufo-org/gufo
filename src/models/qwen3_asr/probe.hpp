#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "src/ext/asr.hpp"

namespace gufo::models::qwen3_asr {

[[nodiscard]] bool ProbeQwen3Asr(std::string_view model_root);
[[nodiscard]] bool ResolveQwen3AsrArtifacts(std::string_view model_root,
                                            std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
[[nodiscard]] std::unique_ptr<ext::AsrRunner> CreateQwen3AsrRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error = nullptr);
#endif

}  // namespace gufo::models::qwen3_asr

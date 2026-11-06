#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "src/ext/tts.hpp"

namespace gufo::models::qwen3_tts {

[[nodiscard]] bool ProbeQwen3Tts(std::string_view model_root);
[[nodiscard]] bool ResolveQwen3TtsArtifacts(std::string_view model_root,
                                            std::string* error = nullptr);

#if defined(ENGINE_ENABLE_HIP)
[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateQwen3TtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error = nullptr);
#endif

}  // namespace gufo::models::qwen3_tts

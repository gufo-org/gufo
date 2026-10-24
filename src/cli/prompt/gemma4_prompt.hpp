#ifndef GUFO_CLI_PROMPT_GEMMA4_PROMPT_HPP_
#define GUFO_CLI_PROMPT_GEMMA4_PROMPT_HPP_

#include <chrono>

#include "src/cli/prompt/prompt.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::cli {

[[nodiscard]] bool IsGemma4(const core::GgufReader& reader);

/// `gufo prompt` and `gufo chat` for Gemma 4 targets (ROCm only).
[[nodiscard]] int RunGemma4Prompt(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);
[[nodiscard]] int RunGemma4Chat(
    const PromptOptions& options,
    std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::cli

#endif  // GUFO_CLI_PROMPT_GEMMA4_PROMPT_HPP_

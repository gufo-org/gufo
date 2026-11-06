#ifndef GUFO_CLI_PROMPT_HPP_
#define GUFO_CLI_PROMPT_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "gufo/ext/text.hpp"

namespace gufo::cli {

/// Options for `gufo prompt` and `gufo chat` execution.
using PromptOptions = ext::TextPromptOptions;

/// Prints help for `gufo prompt`.
void PrintPromptHelp(std::string_view program_name);

/// Prints help for `gufo chat`.
void PrintChatHelp(std::string_view program_name);

/// Parses command line options for `gufo prompt`.
[[nodiscard]] std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg = nullptr);

/// Executes the prompt CLI workflow.
[[nodiscard]] int RunPrompt(std::span<const char* const> args);

/// Executes the interactive multi-turn terminal chat workflow.
[[nodiscard]] int RunChat(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_PROMPT_HPP_

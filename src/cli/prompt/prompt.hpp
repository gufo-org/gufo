#ifndef GUFO_CLI_PROMPT_HPP_
#define GUFO_CLI_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/core/sampling.hpp"

namespace gufo::cli {

/// Options for `gufo prompt` and `gufo chat` execution.
struct PromptOptions {
  std::string model_path;
  std::string prompt_text;
  std::string prompt_file;
  std::string system_prompt =
      "You are a helpful, respectful, and honest assistant.";
  std::size_t max_tokens = 128;
  sampling::SamplingConfig sampling;
  std::string chat_template;
  std::string reasoning_mode = "auto";
  std::int64_t reasoning_budget = -1;
  bool display_prompt = true;
  bool use_chat_template = true;
  bool verbose = false;
  bool force_cpu = false;
  std::string speculative_backend;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::size_t draft_tokens = 7;
  std::string draft_policy = "auto";
  std::size_t min_draft_tokens = 1;
  float draft_p_min = 0.0F;
};

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

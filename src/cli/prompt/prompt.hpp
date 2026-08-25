#ifndef STRIX_CLI_PROMPT_HPP_
#define STRIX_CLI_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace strix::cli {

/// Options for `strix prompt` and `strix chat` execution.
struct PromptOptions {
  std::string model_path;
  std::string prompt_text;
  std::string prompt_file;
  std::string system_prompt =
      "You are a helpful, respectful, and honest assistant.";
  std::size_t max_tokens = 128;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::int32_t top_k = 0;
  float min_p = 0.0F;
  std::int64_t seed = -1;
  float repeat_penalty = 1.0F;
  std::size_t repeat_last_n = 64;
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
  std::string draft_policy = "rolling";
  std::size_t min_draft_tokens = 1;
};

/// Prints help for `strix prompt`.
void PrintPromptHelp(std::string_view program_name);

/// Prints help for `strix chat`.
void PrintChatHelp(std::string_view program_name);

/// Parses command line options for `strix prompt`.
[[nodiscard]] std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg = nullptr);

/// Executes the prompt CLI workflow.
[[nodiscard]] int RunPrompt(std::span<const char* const> args);

/// Executes the interactive multi-turn terminal chat workflow.
[[nodiscard]] int RunChat(std::span<const char* const> args);

}  // namespace strix::cli

#endif  // STRIX_CLI_PROMPT_HPP_

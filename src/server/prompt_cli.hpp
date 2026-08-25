#ifndef STRIX_SERVER_PROMPT_CLI_HPP_
#define STRIX_SERVER_PROMPT_CLI_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace strix::server {

/// Options for `strix-server prompt` and `chat` execution.
struct PromptOptions {
  std::string model_path;
  std::string prompt_text;
  std::string system_prompt =
      "You are a helpful, respectful, and honest assistant.";
  std::size_t max_tokens = 128;
  float temperature = 0.0F;
  bool use_chat_template = true;
  bool verbose = false;
  bool force_cpu = false;
  std::string speculative_backend;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::size_t draft_tokens = 3;
};

/// Parses command line options for `strix-server prompt`.
[[nodiscard]] std::optional<PromptOptions> ParsePromptOptions(
    std::span<const char* const> args, std::string* error_msg = nullptr);

/// Executes the prompt CLI workflow.
[[nodiscard]] int RunPrompt(std::span<const char* const> args);

/// Executes the interactive multi-turn terminal chat workflow.
[[nodiscard]] int RunChat(std::span<const char* const> args);

}  // namespace strix::server

#endif  // STRIX_SERVER_PROMPT_CLI_HPP_

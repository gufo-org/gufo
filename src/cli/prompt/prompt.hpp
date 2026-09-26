#ifndef GUFO_CLI_PROMPT_HPP_
#define GUFO_CLI_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/text_sampling_defaults.hpp"

namespace gufo::cli {

/// Options for `gufo prompt` and `gufo chat` execution.
struct PromptOptions {
  std::string model_path;
  std::string vision_model_path;
  std::vector<std::string> image_paths;
  bool add_vision_id{false};
  std::string prompt_text;
  std::string prompt_file;
  std::string system_prompt;
  std::size_t max_tokens = 128;
  sampling::SamplingConfig sampling;
  sampling::SamplingOverrides sampling_supplied;
  std::string reasoning_mode = "auto";
  std::string reasoning_effort = "auto";
  std::string preserve_thinking = "auto";
  bool display_prompt = true;
  bool use_chat_template = true;
  bool verbose = false;
  bool force_cpu = false;
  std::string speculative_backend;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  // DeepSeek V4 Flash DSpark support model. DSpark is DS4's own drafter and
  // is unrelated to the Qwen DFlash paths above.
  std::string dspark_model_path;
  std::uint32_t draft_tokens = 7;
  std::uint32_t min_draft_tokens = 1;
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

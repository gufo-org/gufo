#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"

namespace gufo::models::deepseek_v4_flash {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete function definition, including optional fields such as strict.
  std::string definition_json;
};

struct ChatTemplateOptions {
  // Literal encoder options; user-facing defaults are resolved below.
  bool enable_thinking{false};
  ReasoningEffort reasoning_effort{ReasoningEffort::kLow};
  bool preserve_thinking{false};
  bool tools_present{false};
  bool require_tool_call{false};
};

/// User-facing defaults follow DeepSeek's API: thinking enabled, high effort.
/// Explicit options override them; the checkpoint encoder itself has no
/// default thinking mode.
[[nodiscard]] ChatTemplateOptions ResolveDeepSeekChatOptions(
    const ReasoningOptions& reasoning);

[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Renders the pinned DeepSeek V4 Flash 0731 conversation format.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Rejects DeepSeek artifacts whose embedded template is not the pinned
/// converter version implemented by the compiled formatter.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "deepseek-v4-flash-0731-compiled-v3";
}

[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27";
}

[[nodiscard]] constexpr std::string_view EncoderReferenceSha256() noexcept {
  return "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6";
}

/// Native DeepSeek effort name after mapping provider-neutral effort levels.
[[nodiscard]] std::string_view DeepSeekReasoningEffortName(
    ReasoningEffort effort) noexcept;

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_CHAT_TEMPLATE_HPP_

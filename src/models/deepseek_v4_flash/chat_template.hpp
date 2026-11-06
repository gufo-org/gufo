#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/reasoning.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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
};

/// User-facing defaults follow DeepSeek's API: thinking enabled, high effort.
/// Explicit options override them; the checkpoint encoder itself has no
/// default thinking mode.
[[nodiscard]] ChatTemplateOptions ResolveDeepSeekChatOptions(
    const ReasoningOptions& reasoning);

[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// A byte span of the rendered prompt that holds caller-supplied text: a
/// message body, reasoning, a tool definition, or a historical tool call's
/// name or argument. Such text is tokenized as text, never as framing: a
/// `</think>` a client sends is the string the client wrote, as in DeepSeek's
/// reference encoder (deepseek-ai/deepseek-recipe) and the Qwen renderer.
struct ContentSpan {
  std::size_t offset{0};
  std::size_t size{0};
};

/// Renders the pinned DeepSeek V4 Flash 0731 conversation format.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});
/// As above, and appends the sorted, disjoint spans of caller-supplied text.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options,
                                     std::vector<ContentSpan>* content_spans);

/// The control tokens the renderer writes as framing. The runtime tokenizer's
/// rendered-chat table (runtime/tokenizer.cpp) holds the same spellings.
enum class ControlToken : std::uint8_t {
  kBeginOfSentence,
  kEndOfSentence,
  kUser,
  kAssistant,
  kThinkStart,
  kThinkEnd,
  kDsml,
};
inline constexpr std::size_t kControlTokenCount = 7;

[[nodiscard]] std::string_view ControlTokenText(ControlToken token) noexcept;

/// A run of ordinary text, or one control token.
struct PromptPiece {
  std::string_view text;
  std::optional<ControlToken> control;
};

/// Splits a rendered prompt at control-token spellings, in the runtime
/// tokenizer's order, except that a spelling overlapping a content span stays
/// ordinary text. Text runs are delimited only by control tokens, so a prompt
/// without such spellings in its content splits exactly as before.
[[nodiscard]] std::vector<PromptPiece> SplitControlTokens(
    std::string_view rendered, std::span<const ContentSpan> content_spans);

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

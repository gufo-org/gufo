#ifndef GUFO_MODELS_NEMOTRON35_BOLT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/reasoning.hpp"

namespace gufo::models::nemotron35_bolt {

class Tokenizer;

enum class ChatRole : std::uint8_t { kSystem, kUser, kAssistant, kTool };

struct ChatToolArgument {
  std::string name;
  /// The string itself, or the argument's JSON text (`tojson` spacing).
  std::string value;
  bool is_string{true};
};

struct ChatToolCall {
  std::string name;
  std::vector<ChatToolArgument> arguments;
};

struct ChatMessage {
  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string reasoning;  ///< Assistant reasoning_content.
  std::vector<ChatToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  /// JSON schema object; its properties, required list and other keys render
  /// as the template's XML-like tool block.
  std::string parameters_json{"{}"};
  /// Complete tool object when supplied, for keys beyond the four standard
  /// ones; may be empty.
  std::string definition_json;
};

struct ChatOptions {
  bool enable_thinking{true};
  /// Appends the template's "{reasoning effort: efficient}" hint.
  bool medium_effort{false};
  bool truncate_history_thinking{true};
  bool add_generation_prompt{true};
};

/// Thinking on unless disabled; minimal, low and medium effort select the
/// template's efficient mode.
[[nodiscard]] ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning);

/// A byte range of a rendered prompt that holds client text (message
/// content, reasoning, tool-call arguments) rather than template framing.
struct ContentSpan {
  std::size_t offset{0};
  std::size_t size{0};
};

/// Renders the artifact's chat template (n5h_5p5_nanov2) for text messages,
/// with the client text's ranges in ascending order when `content_spans` is
/// given. Throws std::invalid_argument for malformed tool schemas.
[[nodiscard]] std::string RenderChat(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatOptions& options,
    std::vector<ContentSpan>* content_spans = nullptr);

/// Tokenizes a rendered prompt with its framing's control tokens. Client text
/// that spells a control token (a literal "<|im_end|>") is encoded as text;
/// only such ranges are encoded apart, so every other prompt keeps the token
/// IDs of encoding the whole text, as llama.cpp does.
[[nodiscard]] std::vector<std::uint32_t> TokenizeChat(
    const Tokenizer& tokenizer, std::string_view rendered,
    std::span<const ContentSpan> content_spans);

/// Python str.strip(), as Jinja's `trim` filter applies it.
[[nodiscard]] std::string_view PythonStrip(std::string_view text);

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_CHAT_TEMPLATE_HPP_

// Renderer reproducing official chat_template.jinja (Inkling 2 Mini) byte for
// byte, without a Jinja engine. Verified against jinja2/transformers goldens
// (tests/text/goldens/render_cases.json).
//
// The template has no generation prompt: the prompt ends after the last message
// and the model writes "<|message_model|>" itself.
// RenderOptions::generation_prefix can append the header (and optionally the
// text-content token) to the prompt.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "src/models/inkling2/text/json.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"

namespace inkling2::text {

struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;  // a JSON object (empty means {})
};

struct ContentPart {
  /// kThinking: an assistant thought in its place among the text parts, as
  /// the reference renderer reads {"type": "thinking", "thinking": ...}.
  enum class Kind { kText, kImage, kAudio, kThinking };
  Kind kind{Kind::kText};
  std::string text;
};

struct Message {
  std::string role;  // system | user | assistant | tool
  enum class Content { kNone, kString, kParts };
  Content content_kind{Content::kString};
  std::string content;             // when kString
  std::vector<ContentPart> parts;  // when kParts
  std::string name;
  std::string reasoning_content;
  std::string tool_call_id;
  std::vector<ToolCall> tool_calls;

  static Message Text(std::string role, std::string text) {
    Message m;
    m.role = std::move(role);
    m.content = std::move(text);
    return m;
  }
};

/// One OpenAI-style tool object ({"type":"function","function":{...}}) or a
/// bare function dictionary, as JSON text.
struct Tool {
  std::string definition_json;
};

/// Provider-neutral effort levels (Gufo's ReasoningEffort) and the template's
/// names.
enum class ReasoningLevel { kMinimal, kLow, kMedium, kHigh, kXHigh, kMax };

struct RenderOptions {
  /// Template name: none, minimal, low, medium, high, xhigh, max. Unset and
  /// no numeric effort: the template default (0.9, same as "high").
  std::optional<std::string> effort_name;
  /// Numeric effort in [0, 0.99]; used when effort_name is unset.
  std::optional<double> effort_number;
  /// Unset or true: reasoning_content is rendered for every assistant message,
  /// as the template does. False: dropped for assistant messages before the
  /// last user message (a Gufo-level option; the template has no such rule).
  std::optional<bool> preserve_thinking;
  /// Placeholder tokens written per image (the template writes one).
  std::size_t image_placeholders{1};
  /// Per-image placeholder counts, in the order the images appear; image k uses
  /// image_counts[k] when present, else image_placeholders. The runner sets one
  /// per image (its token count).
  std::vector<std::size_t> image_counts;
  /// Not the template: an assistant message without a (shown) thought is
  /// rendered with a finished empty thinking block before its content and
  /// calls. It is the framing a request with generation_prefix == kEmptyThought
  /// puts in the prompt, so what the model then writes and the next turn's
  /// re-render of that message are the same tokens (see
  /// docs/internal/models/inkling-2-mini/README.md, Reasoning and tool calls).
  bool empty_thought_for_assistants{false};
  enum class Prefix {
    kNone,
    kHeader,
    kHeaderText,
    /// A finished empty thinking block (header, thinking token, end): the
    /// answer or a call follows.
    kEmptyThought,
  };
  /// What to append after the template output so generation continues inside an
  /// assistant block. kNone is exactly the template.
  Prefix generation_prefix{Prefix::kNone};
};

/// Maps Gufo's reasoning controls onto the template: enabled=false is "none",
/// an effort maps to its namesake, enabled=true or unset with no effort leaves
/// the template default.
[[nodiscard]] std::optional<std::string> EffortNameFor(
    std::optional<bool> enabled, std::optional<ReasoningLevel> level);

/// The prompt as a sequence of control-token literals and plain-text spans.
/// Text spans are user data or framing between control tokens, so they never
/// parse special tokens (a client cannot inject `<|end_message|>`).
struct Rendered {
  struct Item {
    bool control{false};  // true: a control-token spelling
    std::string text;
  };
  std::vector<Item> items;
  /// Concatenation, equal to the jinja2 output.
  [[nodiscard]] std::string Text() const;
  /// Item indices that are the image placeholder token.
  std::vector<std::size_t> image_items;
};

[[nodiscard]] bool RenderChat(const std::vector<Message>& messages,
                              const std::vector<Tool>& tools,
                              const RenderOptions& options, Rendered* out,
                              std::string* error);

struct PromptTokens {
  std::vector<TokenId> ids;
  std::vector<std::size_t>
      image_positions;  // token indices of image placeholders
};
[[nodiscard]] bool RenderAndTokenize(const Tokenizer& tokenizer,
                                     const std::vector<Message>& messages,
                                     const std::vector<Tool>& tools,
                                     const RenderOptions& options,
                                     PromptTokens* out, std::string* error);

}  // namespace inkling2::text

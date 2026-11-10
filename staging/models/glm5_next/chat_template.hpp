#ifndef GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::models::glm5_next {

// Delimiters and control-token spellings are hex-escaped everywhere so no
// source, object file or log line ever contains a literal special-token
// sequence (flint convention, extended to every GLM-5 tag).
inline constexpr std::string_view kCtrlOpen = "\x3C" "|";
inline constexpr std::string_view kCtrlClose = "|" "\x3E";

inline constexpr std::string_view kThinkOpenTag = "\x3C" "think" "\x3E";
inline constexpr std::string_view kThinkCloseTag = "\x3C" "/think" "\x3E";
inline constexpr std::string_view kToolCallOpenTag = "\x3C" "tool_call" "\x3E";
inline constexpr std::string_view kToolCallCloseTag = "\x3C" "/tool_call" "\x3E";
inline constexpr std::string_view kArgKeyOpenTag = "\x3C" "arg_key" "\x3E";
inline constexpr std::string_view kArgKeyCloseTag = "\x3C" "/arg_key" "\x3E";
inline constexpr std::string_view kArgValueOpenTag = "\x3C" "arg_value" "\x3E";
inline constexpr std::string_view kArgValueCloseTag = "\x3C" "/arg_value" "\x3E";
inline constexpr std::string_view kToolResponseOpenTag =
    "\x3C" "tool_response" "\x3E";
inline constexpr std::string_view kToolResponseCloseTag =
    "\x3C" "/tool_response" "\x3E";

/// GGUF tokenizer.ggml.eos_token_id (154820) spelling, declared once here.
/// It is the only stop token the runtime honors numerically (through the
/// tokenizer GetEosTokenId/IsStopToken accessors); this module keeps its
/// literal spelling out of every rendered prompt so no stage that scans
/// prompt strings or prompt token ids for stops can halt generation before
/// the model produced anything.
inline constexpr std::string_view kGlmEosSpelling = "\x3C" "|endoftext|" "\x3E";

/// Substitute installed for every kGlmEosSpelling occurrence in
/// request-supplied text. The placeholder is inert for the tokenizer (no
/// special entry registers it), so it reaches the model as ordinary
/// content without arming the stop detector. The substitution rule lives
/// here and nowhere else.
inline constexpr std::string_view kEosPlaceholder = "{eos}";

/// Replaces every (left-to-right, non-overlapping) occurrence of the EOS
/// spelling in text with kEosPlaceholder. Idempotent: the placeholder byte
/// string never contains the EOS spelling.
void HideEosLiterals(std::string* text);

/// One rendering knob matching a reference-template variable.
struct ChatRenderOptions {
  /// reasoning_effort: only low/high are honored; anything else (or unset)
  /// renders the default effort label Max.
  std::optional<std::string> reasoning_effort;
  /// clear_thinking: drops the think block from assistant turns up to and
  /// including the last user turn; later turns keep it (matching the
  /// loop-index guard of the reference template).
  bool clear_thinking = false;
};

/// Renders the GLM-5 GGUF tokenizer.chat_template framing for
/// string-content messages: the two header tokens ([gMASK] then the
/// short-opener tag) followed by the Reasoning Effort system line, then one
/// tagged turn per message with no inter-role newlines (every reference
/// template boundary tag carries the dash-strip), and, with
/// add_generation_prompt, the trailing assistant role token plus the
/// think-open tag. Assistant history carries reasoning inside think
/// delimiters (explicit thought first, else split from a literal
/// close-think boundary inside content), then stripped content, then calls
/// as newline + call-open + name + (arg-key/arg-value pairs) + call-close
/// per call. Runs of tool turns open one observation role token and wrap
/// every result in tool-response delimiters, ordered by the preceding
/// assistant call list when ids match. Requests that carry tool schemas
/// render nullopt: the schema-preamble branch is not implemented yet and
/// must fail closed rather than silently drop the schema.
[[nodiscard]] std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt = true,
    std::span<const tokenization::ChatTool> tools = {},
    const ChatRenderOptions& options = {});

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_
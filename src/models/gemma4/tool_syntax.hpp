#ifndef GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_
#define GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_

#include <optional>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::models::gemma4 {

struct ToolCall {
  std::string name;
  json::Value arguments;  ///< Always a JSON object.
};

/// Parses the body of one `<|tool_call>…<tool_call|>` block:
/// `call:NAME{key:value,…}`. Values use the Gemma 4 argument syntax the chat
/// template renders: `<|"|>`-delimited strings, numbers, true/false/null,
/// `{…}` objects with bare or `<|"|>`-quoted keys, and `[…]` arrays. A bare
/// value that is not a JSON literal is kept as a string.
[[nodiscard]] std::optional<ToolCall> ParseToolCall(
    std::string_view body, std::string* error_msg = nullptr);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_

// Inkling 2 tool-call dialect and output-block parser (the extension-API glue
// lives in gufo_adapter.hpp).
//
// The model writes a turn as blocks, each closed by <|end_message|>:
//   <|message_model|><|content_thinking|>REASONING<|end_message|>
//   <|message_model|><|content_text|>ANSWER<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
// and ends with <|content_model_end_sampling|> (eos). Calls come last; several
// calls are several blocks.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inkling2::text::dialect {

inline constexpr std::string_view kId = "inkling2-json";

inline constexpr std::string_view kHeader = "<|message_model|>";
inline constexpr std::string_view kThinking = "<|content_thinking|>";
inline constexpr std::string_view kContentText = "<|content_text|>";
inline constexpr std::string_view kInvokeJson = "<|content_invoke_tool_json|>";
inline constexpr std::string_view kEndMessage = "<|end_message|>";
inline constexpr std::string_view kEndSampling =
    "<|content_model_end_sampling|>";

// Reasoning budget tags: control tokens (each one token).
inline constexpr std::string_view kBudgetStart = "<|content_thinking|>";
inline constexpr std::string_view kBudgetForcedEnd = "<|end_message|>";
// Literal reasoning delimiters in the generated text.
inline constexpr std::string_view kReasoningOpen =
    "<|message_model|><|content_thinking|>";
inline constexpr std::string_view kReasoningClose =
    "<|end_message|><|message_model|><|content_text|>";

inline constexpr std::string_view kMarkers[] = {kHeader};
inline constexpr std::string_view kClosers[] = {kEndMessage};

struct Argument {
  std::string name;
  std::string value;  // string: the decoded text; otherwise the raw JSON text
  bool is_string{true};
};

struct Call {
  std::string name;
  std::vector<Argument> arguments;
  std::optional<Argument>
      unfinished;  // a string argument still open (partial scans)
  bool valid{false};
  bool complete{false};
};

/// `text` starts at the first marker. Appends every call found and, to `spans`,
/// the [begin, end) range of each complete call block (header through
/// <|end_message|>). With `partial`, the text may end inside a call: the one
/// unfinished call that later text could still complete is appended last with
/// complete=false.
void ScanToolCalls(std::string_view text, bool partial,
                   std::vector<Call>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans);

struct ParsedOutput {
  std::string reasoning;
  std::string content;
  struct ToolCall {
    std::string name;
    std::string arguments_json;  // the args object exactly as written
  };
  std::vector<ToolCall> tool_calls;
  bool truncated{false};  // the text ended inside a block
};

/// Splits a complete model output (control tokens as text) into reasoning,
/// content and tool calls. The trailing <|content_model_end_sampling|> is
/// optional.
[[nodiscard]] ParsedOutput ParseAssistantOutput(std::string_view text);

}  // namespace inkling2::text::dialect

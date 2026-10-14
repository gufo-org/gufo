#include "src/models/deepseek_v4_flash/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::deepseek_v4_flash {
namespace {

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";

constexpr std::string_view kHighReasoningPrompt =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively "
    "decompose the problem to resolve the root cause, rigorously "
    "stress-testing your logic against all potential paths, edge cases, and "
    "adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting "
    "every intermediate step, considered alternative, and rejected "
    "hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

constexpr std::string_view kMaxReasoningPrompt =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
    "uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely "
    "nothing to chance: exhaustively decompose the problem into its most "
    "fundamental components, trace every causal chain to its root, and "
    "resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the "
    "solution from multiple angles and are certain that no assumption "
    "remains unchecked and no error remains undiscovered.\n\n";

enum class NativeEffort : std::uint8_t {
  kLow,
  kHigh,
  kMax,
};

NativeEffort MapEffort(ReasoningEffort effort) noexcept {
  switch (effort) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      return NativeEffort::kLow;
    case ReasoningEffort::kMedium:
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
      return NativeEffort::kHigh;
    case ReasoningEffort::kMax:
      return NativeEffort::kMax;
  }
  return NativeEffort::kLow;
}

void AppendEffortPrompt(std::string& output, ReasoningEffort effort) {
  switch (MapEffort(effort)) {
    case NativeEffort::kLow:
      return;
    case NativeEffort::kHigh:
      output.append(kHighReasoningPrompt);
      return;
    case NativeEffort::kMax:
      output.append(kMaxReasoningPrompt);
      return;
  }
}

std::string Sha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

void AssignError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

void AppendJsonString(std::string& output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        output.push_back(static_cast<char>(character));
        break;
    }
  }
  output.push_back('"');
}

std::string PythonJsonSpacing(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    output.push_back(character);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools,
                       bool require_tool_call) {
  if (tools.empty()) {
    return;
  }
  output.append(
      "## Tools\n\n"
      "You have access to a set of tools to help answer the user's "
      "question. You can invoke tools by writing a \"");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.append("\" block like the following:\n\n");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$TOOL_NAME\">\n");
  output.append(DeepSeekToolSyntax::kParameterOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$PARAMETER_NAME\" ");
  output.append(DeepSeekToolSyntax::kStringAttribute);
  output.append("true|false\">$PARAMETER_VALUE");
  output.append(DeepSeekToolSyntax::kParameterCloses[0]);
  output.append("\n...\n");
  output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$TOOL_NAME2\">\n...\n");
  output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kCallsClose);
  output.append(
      "\n\n"
      "String parameters should be specified as is and set `string=\"true\"`. "
      "For all other types (numbers, booleans, arrays, objects), pass the "
      "value in JSON format and set `string=\"false\"`.\n\n"
      "If thinking_mode is enabled (triggered by <think>), you MUST output "
      "your complete reasoning inside <think>...</think> BEFORE any tool "
      "calls or final response.\n\n"
      "Otherwise, output directly after </think> with tool calls or final "
      "response.\n\n"
      "### Available Tool Schemas\n\n");
  for (std::size_t index = 0; index < tools.size(); ++index) {
    const auto& tool = tools[index];
    if (!tool.definition_json.empty()) {
      output.append(PythonJsonSpacing(tool.definition_json));
      if (index + 1 < tools.size())
        output.push_back('\n');
      continue;
    }
    output.append("{\"name\": ");
    AppendJsonString(output, tool.name);
    output.append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output.append(", \"parameters\": ");
    output.append(tool.parameters_json.empty()
                      ? "{}"
                      : PythonJsonSpacing(tool.parameters_json));
    output.push_back('}');
    if (index + 1 < tools.size()) {
      output.push_back('\n');
    }
  }
  output.append(
      "\n\nYou MUST strictly follow the above defined tool name and parameter "
      "schemas to invoke tool calls.\n");
  if (require_tool_call) {
    output.append(
        "\n\nYou MUST invoke at least one available tool before answering.");
  }
}

void AppendToolCalls(std::string& output,
                     std::span<const ChatMessage::ToolCall> calls) {
  if (calls.empty()) {
    return;
  }
  output.append("\n\n");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.push_back('\n');
  for (std::size_t call_index = 0; call_index < calls.size(); ++call_index) {
    const auto& call = calls[call_index];
    output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
    output.push_back(' ');
    output.append(DeepSeekToolSyntax::kNameAttribute);
    output.append(call.name);
    output.append("\">\n");
    for (std::size_t argument_index = 0; argument_index < call.arguments.size();
         ++argument_index) {
      const auto& argument = call.arguments[argument_index];
      output.append(DeepSeekToolSyntax::kParameterOpens[0]);
      output.push_back(' ');
      output.append(DeepSeekToolSyntax::kNameAttribute);
      output.append(argument.name);
      output.append("\" ");
      output.append(DeepSeekToolSyntax::kStringAttribute);
      output.append(argument.is_string ? "true" : "false");
      output.append("\">");
      output.append(argument.value);
      output.append(DeepSeekToolSyntax::kParameterCloses[0]);
      if (argument_index + 1 < call.arguments.size()) {
        output.push_back('\n');
      }
    }
    output.push_back('\n');
    output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
    if (call_index + 1 < calls.size()) {
      output.push_back('\n');
    }
  }
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kCallsClose);
}

}  // namespace

namespace {

// The API surface trims framing with ASCII whitespace; the scanner keeps the
// same reading so parser decisions do not depend on Unicode table updates.
std::string_view TrimAscii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0)
    value.remove_prefix(1);
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0)
    value.remove_suffix(1);
  return value;
}

std::optional<std::string> Attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const std::size_t start = tag.find(prefix);
  if (start == std::string_view::npos)
    return std::nullopt;
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = tag.find('"', value_start);
  if (end == std::string_view::npos)
    return std::nullopt;
  return std::string(tag.substr(value_start, end - value_start));
}

// Hold a possible closing delimiter until its next byte disambiguates it.
std::string_view BeforePartialDelimiter(std::string_view value,
                                        std::string_view delimiter) {
  for (auto size = std::min(value.size(), delimiter.size()); size > 0; --size)
    if (delimiter.starts_with(value.substr(value.size() - size)))
      return value.substr(0, value.size() - size);
  return value;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace

void ScanDeepSeekToolCalls(
    std::string_view text, std::span<const std::string_view> markers,
    std::vector<ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  std::size_t cursor = 0;
  std::size_t envelope_begin = std::string_view::npos;
  std::string envelope_end;
  while (cursor < text.size()) {
    std::size_t invoke_start = std::string_view::npos;
    std::size_t syntax = 0;
    for (std::size_t index = 0; index < DeepSeekToolSyntax::kInvokeOpens.size();
         ++index) {
      const std::size_t position =
          text.find(DeepSeekToolSyntax::kInvokeOpens[index], cursor);
      if (position < invoke_start) {
        invoke_start = position;
        syntax = index;
      }
    }
    if (spans) {
      // Examine envelope boundaries only outside the parameter/invocation
      // ranges consumed below: an outer closing tag can be argument data.
      if (envelope_begin == std::string_view::npos) {
        std::string_view opening;
        std::size_t begin = std::string_view::npos;
        for (const auto marker : markers) {
          const auto position = text.find(marker, cursor);
          if (position < begin) {
            begin = position;
            opening = marker;
          }
        }
        // Bare invocation examples outside an outer envelope are prose.
        if (begin == std::string_view::npos)
          break;
        envelope_begin = begin;
        envelope_end =
            "</" + std::string(opening.substr(opening.find('<') + 1));
        cursor = begin + opening.size();
        continue;
      }
      if (const auto end = text.find(envelope_end, cursor);
          end < invoke_start) {
        cursor = end + envelope_end.size();
        spans->emplace_back(envelope_begin, cursor);
        envelope_begin = std::string_view::npos;
        continue;
      }
    }
    if (invoke_start == std::string_view::npos) {
      break;
    }
    const std::size_t tag_end = text.find('>', invoke_start);
    if (tag_end == std::string_view::npos) {
      break;
    }
    ScannedToolCall call;
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    bool valid = !call.name.empty();
    std::size_t protected_end = tag_end + 1;
    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < text.size()) {
      auto remainder = TrimAscii(text.substr(parameter_cursor));
      parameter_cursor =
          static_cast<std::size_t>(remainder.data() - text.data());
      if (!remainder.starts_with(DeepSeekToolSyntax::kParameterOpens[syntax])) {
        break;
      }
      const std::size_t parameter_start = parameter_cursor;
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      std::size_t parameter_end = text.find(
          DeepSeekToolSyntax::kParameterCloses[syntax], parameter_tag_end);
      if (parameter_tag_end == std::string_view::npos ||
          parameter_end == std::string_view::npos) {
        if (partial && parameter_tag_end != std::string_view::npos) {
          const auto tag = text.substr(parameter_start,
                                       parameter_tag_end - parameter_start + 1);
          const auto name = Attribute(tag, "name");
          if (name && !name->empty() &&
              Attribute(tag, "string").value_or("true") == "true")
            call.unfinished = ChatMessage::ToolArgument{
                .name = *name,
                .value = std::string(BeforePartialDelimiter(
                    text.substr(parameter_tag_end + 1),
                    DeepSeekToolSyntax::kParameterCloses[syntax])),
                .is_string = true};
        }
        protected_end = text.size();
        valid = false;
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      if (parameter_name && !parameter_name->empty() &&
          (!string_value || *string_value == "true" ||
           *string_value == "false")) {
        const bool is_string = string_value.value_or("true") == "true";
        // JSON strings may contain DSML delimiters. Find the end of the JSON
        // value before interpreting the enclosing markup.
        while (
            !is_string && parameter_end != std::string_view::npos &&
            !TryParseJson(TrimAscii(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))))
          parameter_end = text.find(
              DeepSeekToolSyntax::kParameterCloses[syntax],
              parameter_end +
                  DeepSeekToolSyntax::kParameterCloses[syntax].size());
        if (parameter_end == std::string_view::npos) {
          protected_end = text.size();
          valid = false;
          break;
        }
        const auto raw = text.substr(parameter_tag_end + 1,
                                     parameter_end - parameter_tag_end - 1);
        ChatMessage::ToolArgument argument{
            .name = *parameter_name,
            .value = std::string(is_string ? raw : TrimAscii(raw)),
            .is_string = is_string,
        };
        // As for Qwen calls: drop an identical repeat and reject a
        // conflicting one instead of letting the last value win.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(std::move(argument));
        } else if (previous->value != argument.value ||
                   previous->is_string != argument.is_string) {
          valid = false;
        }
      } else {
        valid = false;
      }
      parameter_cursor =
          parameter_end + DeepSeekToolSyntax::kParameterCloses[syntax].size();
      protected_end = parameter_cursor;
    }
    const bool complete =
        text.substr(parameter_cursor)
            .starts_with(DeepSeekToolSyntax::kInvokeCloses[syntax]);
    call.valid = valid;
    call.complete = complete;
    if ((valid && complete) || (partial && !call.name.empty())) {
      calls->push_back(std::move(call));
      if (!(valid && complete)) {
        break;
      }
    }
    cursor = complete ? parameter_cursor +
                            DeepSeekToolSyntax::kInvokeCloses[syntax].size()
                      : protected_end;
  }
}

ChatTemplateOptions ResolveDeepSeekChatOptions(
    const ReasoningOptions& reasoning) {
  return {
      .enable_thinking = reasoning.enabled.value_or(true),
      .reasoning_effort = reasoning.effort.value_or(ReasoningEffort::kHigh),
      .preserve_thinking = reasoning.preserve_thinking.value_or(false),
  };
}

std::string_view DeepSeekReasoningEffortName(ReasoningEffort effort) noexcept {
  switch (MapEffort(effort)) {
    case NativeEffort::kLow:
      return "low";
    case NativeEffort::kHigh:
      return "high";
    case NativeEffort::kMax:
      return "max";
  }
  return "low";
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       const ChatTemplateOptions& options) {
  return RenderChat(messages, {}, options);
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  std::string output(kBos);
  if (options.enable_thinking) {
    AppendEffortPrompt(output, options.reasoning_effort);
  }

  std::size_t last_user_index = messages.size();
  for (std::size_t index = messages.size(); index > 0; --index) {
    const std::string_view role = messages[index - 1].role;
    if (role == "user" || role == "developer") {
      last_user_index = index - 1;
      break;
    }
  }
  const bool preserve_thinking =
      options.preserve_thinking || options.tools_present || !tools.empty();

  std::size_t tool_host_index = messages.size();
  if (!tools.empty()) {
    for (std::size_t index = 0; index < messages.size(); ++index) {
      if (messages[index].role == "system" ||
          messages[index].role == "developer") {
        tool_host_index = index;
        break;
      }
    }
    if (tool_host_index == messages.size()) {
      AppendToolsPrompt(output, tools, options.require_tool_call);
    }
  }

  std::unordered_map<std::string_view, std::size_t> tool_call_order;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    const std::string_view role = message.role;
    if (role == "system") {
      output.append(message.content);
      if (index == tool_host_index) {
        if (!message.content.empty()) {
          output.append("\n\n");
        }
        AppendToolsPrompt(output, tools, options.require_tool_call);
      }
      continue;
    }
    if (role == "developer") {
      if (options.enable_thinking && !preserve_thinking &&
          index < last_user_index) {
        continue;
      }
      output.append(kUser);
      output.append(message.content);
      if (index == tool_host_index) {
        if (!message.content.empty()) {
          output.append("\n\n");
        }
        AppendToolsPrompt(output, tools, options.require_tool_call);
      }
      continue;
    }
    if (role == "user" || role == "tool" || role == "function") {
      std::vector<const ChatMessage*> tool_results;
      for (std::size_t end = index; end < messages.size(); ++end) {
        const auto& next = messages[end];
        if (next.role != "user" && next.role != "tool" &&
            next.role != "function")
          break;
        if (next.role != "user")
          tool_results.push_back(&next);
      }
      // Match sort_tool_results_by_call_order in the official encoder.
      // Reorder only tool slots, retaining interleaved user text. Unknown
      // IDs have rank zero and ties preserve their arrival order.
      const auto rank = [&](const ChatMessage* result) {
        const auto found = tool_call_order.find(result->tool_call_id);
        return found == tool_call_order.end() ? std::size_t{0} : found->second;
      };
      std::stable_sort(tool_results.begin(), tool_results.end(),
                       [&](const auto* left, const auto* right) {
                         return rank(left) < rank(right);
                       });
      std::size_t result_index = 0;
      output.append(kUser);
      bool first_block = true;
      while (index < messages.size() && (messages[index].role == "user" ||
                                         messages[index].role == "tool" ||
                                         messages[index].role == "function")) {
        if (!first_block) {
          output.append("\n\n");
        }
        if (messages[index].role == "user") {
          output.append(messages[index].content);
        } else {
          output.append("<tool_result>");
          output.append(tool_results[result_index++]->content);
          output.append("</tool_result>");
        }
        first_block = false;
        ++index;
      }
      --index;
      continue;
    }
    if (role == "assistant") {
      if (!message.tool_calls.empty()) {
        tool_call_order.clear();
        for (std::size_t call = 0; call < message.tool_calls.size(); ++call) {
          if (!message.tool_calls[call].id.empty())
            tool_call_order[message.tool_calls[call].id] = call;
        }
      }
      output.append(kAssistant);
      if (options.enable_thinking &&
          (preserve_thinking || index > last_user_index)) {
        output.append(kThinkStart);
        output.append(message.reasoning_content);
        output.append(kThinkEnd);
      } else {
        output.append(kThinkEnd);
      }
      output.append(message.content);
      AppendToolCalls(output, message.tool_calls);
      output.append(kEos);
      continue;
    }
  }

  if (!messages.empty()) {
    const std::string_view last_role = messages.back().role;
    if (last_role == "user" || last_role == "developer" ||
        last_role == "tool" || last_role == "function") {
      output.append(GenerationPrompt(options.enable_thinking));
    }
  }
  return output;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>";
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  if (reader.GetMetadataString("general.architecture") != "deepseek4") {
    AssignError(error_msg,
                "DeepSeek V4 Flash artifact has unexpected architecture");
    return false;
  }
  const auto name = reader.GetMetadataString("general.name");
  if (!name.has_value() ||
      name->find("DeepSeek V4 Flash") == std::string_view::npos) {
    AssignError(error_msg,
                "DeepSeek artifact is not the supported V4 Flash revision");
    return false;
  }
  const auto raw_template = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw_template.has_value() || raw_template->empty()) {
    AssignError(error_msg,
                "DeepSeek GGUF is missing tokenizer.chat_template metadata");
    return false;
  }
  const std::string actual_hash = Sha256(*raw_template);
  if (actual_hash != ArtifactTemplateSha256()) {
    AssignError(error_msg,
                "DeepSeek GGUF chat template SHA-256 is not the pinned 0731 "
                "artifact version: " +
                    actual_hash);
    return false;
  }
  return true;
}

}  // namespace gufo::models::deepseek_v4_flash

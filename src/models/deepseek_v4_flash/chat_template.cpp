#include "src/models/deepseek_v4_flash/chat_template.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"

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
      "You have access to a set of tools to help answer the user's question. "
      "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like "
      "the following:\n\n"
      "<｜DSML｜tool_calls>\n"
      "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
      "<｜DSML｜parameter name=\"$PARAMETER_NAME\" "
      "string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
      "...\n"
      "</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
      "...\n"
      "</｜DSML｜invoke>\n"
      "</｜DSML｜tool_calls>\n\n"
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
  output.append("\n\n<｜DSML｜tool_calls>\n");
  for (std::size_t call_index = 0; call_index < calls.size(); ++call_index) {
    const auto& call = calls[call_index];
    output.append("<｜DSML｜invoke name=\"");
    output.append(call.name);
    output.append("\">\n");
    for (std::size_t argument_index = 0; argument_index < call.arguments.size();
         ++argument_index) {
      const auto& argument = call.arguments[argument_index];
      output.append("<｜DSML｜parameter name=\"");
      output.append(argument.name);
      output.append("\" string=\"");
      output.append(argument.is_string ? "true" : "false");
      output.append("\">");
      output.append(argument.value);
      output.append("</｜DSML｜parameter>");
      if (argument_index + 1 < call.arguments.size()) {
        output.push_back('\n');
      }
    }
    output.append("\n</｜DSML｜invoke>");
    if (call_index + 1 < calls.size()) {
      output.push_back('\n');
    }
  }
  output.append("\n</｜DSML｜tool_calls>");
}

}  // namespace

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

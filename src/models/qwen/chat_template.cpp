#include "src/models/qwen/chat_template.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

namespace {

constexpr std::string_view kDefaultChatmlTemplate =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

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

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools,
                       bool require_tool_call) {
  if (tools.empty()) {
    return;
  }

  output.append(
      "\n\n# Tools\n\nYou have access to the following "
      "functions:\n\n<tools>\n");
  for (const auto& tool : tools) {
    output.append("{\"type\":\"function\",\"function\":{\"name\":");
    AppendJsonString(output, tool.name);
    output.append(",\"description\":");
    AppendJsonString(output, tool.description);
    output.append(",\"parameters\":");
    output.append(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
    output.append("}}\n");
  }
  output.append(
      "</tools>\n\nIf you choose to call a function, reply using this exact "
      "format with no suffix:\n\n<tool_call>\n<function=FUNCTION_NAME>\n"
      "<parameter=PARAMETER_NAME>\nPARAMETER_VALUE\n</parameter>\n"
      "</function>\n</tool_call>\n\nRequired parameters must be present. "
      "Multiple tool calls may be emitted as consecutive <tool_call> "
      "blocks.");
  if (require_tool_call) {
    output.append(
        "\n\nYou must call at least one available function. Do not answer the "
        "user directly.");
  }
}

void AppendToolCalls(std::string& output,
                     std::span<const ChatMessage::ToolCall> calls) {
  bool first_call = true;
  for (const auto& call : calls) {
    if (!first_call) {
      output.push_back('\n');
    }
    first_call = false;
    output.append("<tool_call>\n<function=");
    output.append(call.name);
    output.append(">\n");
    for (const auto& argument : call.arguments) {
      output.append("<parameter=");
      output.append(argument.name);
      output.append(">\n");
      output.append(argument.value);
      output.append("\n</parameter>\n");
    }
    output.append("</function>\n</tool_call>");
  }
}

}  // namespace

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto template_str = reader.GetMetadataString("tokenizer.chat_template");
  if (!template_str.has_value() || template_str->empty()) {
    if (error_msg != nullptr) {
      *error_msg =
          "GGUF metadata missing 'tokenizer.chat_template', using default "
          "ChatML template";
    }
    return CreateDefault();
  }
  return std::unique_ptr<QwenChatTemplate>(
      new QwenChatTemplate(std::string(*template_str)));
}

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateDefault(
    std::string_view raw_template) {
  if (raw_template.empty()) {
    return std::unique_ptr<QwenChatTemplate>(
        new QwenChatTemplate(std::string(kDefaultChatmlTemplate)));
  }
  return std::unique_ptr<QwenChatTemplate>(
      new QwenChatTemplate(std::string(raw_template)));
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, const ChatTemplateOptions& options,
    std::string* error_msg) {
  return Render(messages, {}, options, error_msg);
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg) {
  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len += msg.content.size() + msg.thought.size() + 32;
    for (const auto& call : msg.tool_calls) {
      estimated_len += call.name.size() + 64;
      for (const auto& argument : call.arguments) {
        estimated_len += argument.name.size() + argument.value.size() + 40;
      }
    }
  }
  for (const auto& tool : tools) {
    estimated_len += tool.name.size() + tool.description.size() +
                     tool.parameters_json.size() + 96;
  }
  if (options.add_generation_prompt) {
    estimated_len += 32;
  }

  if (estimated_len > options.max_output_bytes) {
    if (error_msg != nullptr) {
      *error_msg = "Rendered prompt estimated length (" +
                   std::to_string(estimated_len) +
                   " bytes) exceeds maximum bound (" +
                   std::to_string(options.max_output_bytes) + " bytes)";
    }
    return std::nullopt;
  }

  output.reserve(estimated_len);

  bool tools_rendered = tools.empty();
  if (!tools_rendered &&
      (messages.empty() || (messages.front().role != ChatRole::kSystem &&
                            messages.front().role != ChatRole::kDeveloper))) {
    output.append("<|im_start|>system\n");
    AppendToolsPrompt(output, tools, options.require_tool_call);
    output.append("<|im_end|>\n");
    tools_rendered = true;
  }

  for (const auto& msg : messages) {
    const bool tool_result = msg.role == ChatRole::kTool;
    const auto role_name =
        tool_result ? std::string_view{"user"} : ToString(msg.role);
    output.append("<|im_start|>");
    output.append(role_name);
    output.push_back('\n');

    if (!tools_rendered &&
        (msg.role == ChatRole::kSystem || msg.role == ChatRole::kDeveloper)) {
      output.append(msg.content);
      AppendToolsPrompt(output, tools, options.require_tool_call);
      tools_rendered = true;
      output.append("<|im_end|>\n");
      continue;
    }

    if (msg.role == ChatRole::kAssistant && !msg.thought.empty()) {
      output.append("<think>\n");
      output.append(msg.thought);
      output.append("\n</think>\n");
    }

    if (tool_result) {
      output.append("<tool_response>\n");
      output.append(msg.content);
      output.append("\n</tool_response>");
    } else {
      output.append(msg.content);
      if (msg.role == ChatRole::kAssistant && !msg.tool_calls.empty()) {
        AppendToolCalls(output, msg.tool_calls);
      }
    }
    output.append("<|im_end|>\n");

    if (output.size() > options.max_output_bytes) {
      if (error_msg != nullptr) {
        *error_msg =
            "Prompt exceeded max output bytes during message rendering";
      }
      return std::nullopt;
    }
  }

  if (options.add_generation_prompt) {
    output.append("<|im_start|>assistant\n");
    if (options.enable_thinking) {
      output.append("<think>\n");
    }
  }

  if (output.size() > options.max_output_bytes) {
    if (error_msg != nullptr) {
      *error_msg = "Prompt exceeded max output bytes after generation prompt";
    }
    return std::nullopt;
  }

  return output;
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    const ChatTemplateOptions& options, std::string* error_msg) {
  return RenderAndTokenize(tokenizer, messages, {}, options, error_msg);
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  const auto rendered = Render(messages, tools, options, error_msg);
  if (!rendered.has_value()) {
    return std::nullopt;
  }

  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;

  return tokenizer.Encode(*rendered, tok_opts);
}

}  // namespace gufo::tokenization

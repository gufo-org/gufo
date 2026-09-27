#include "src/models/qwen/chat_template.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

std::size_t RenderedPromptBoundBytes(std::uint32_t context_tokens) noexcept {
  constexpr std::size_t kFloor = 1024ULL * 1024ULL;
  return std::max(kFloor,
                  std::size_t{context_tokens} * kMaxRenderedBytesPerToken);
}

ChatTemplateOptions ResolveQwenChatOptions(const ReasoningOptions& reasoning,
                                           bool add_vision_id) {
  ChatTemplateOptions options;
  options.add_vision_id = add_vision_id;
  options.enable_thinking = reasoning.enabled.value_or(options.enable_thinking);
  options.preserve_thinking =
      reasoning.preserve_thinking.value_or(options.preserve_thinking);
  switch (reasoning.effort.value_or(ReasoningEffort::kXHigh)) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      options.reasoning_effort = QwenReasoningEffort::kLow;
      break;
    case ReasoningEffort::kMedium:
      options.reasoning_effort = QwenReasoningEffort::kMedium;
      break;
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      options.reasoning_effort = QwenReasoningEffort::kXHigh;
      break;
  }
  return options;
}

namespace {

constexpr std::string_view kDefaultChatmlTemplate =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

constexpr std::string_view kLowReasoningInstruction =
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstruction =
    "Reasoning effort is set to xhigh. Please think carefully through the "
    "task, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness, consistency, and clarity in the final answer.";

std::string TemplateSha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

bool IsQwen38ReasoningTemplate(std::string_view value) {
  const std::string hash = TemplateSha256(value);
  return hash == QwenChatTemplate::OfficialTemplateSha256() ||
         hash == QwenChatTemplate::UnslothArtifactTemplateSha256();
}

bool IsQwen38Artifact(const core::GgufReader& reader) {
  const auto name = reader.GetMetadataString("general.name");
  if (name.has_value() && name->find("Qwen3.8") != std::string_view::npos) {
    return true;
  }
  const auto config = reader.ExtractModelConfig();
  return config.has_value() && config->num_layers == 64 &&
         config->hidden_size == 5120 && config->vocab_size == 248320;
}

std::string_view Trim(std::string_view value) {
  const auto whitespace = [](UChar32 cp) {
    // Python str.strip also includes the four ASCII information separators.
    return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
  };
  std::int32_t first = 0;
  auto last = static_cast<std::int32_t>(value.size());
  while (first < last) {
    auto next = first;
    UChar32 cp;
    U8_NEXT(value.data(), next, last, cp);
    if (!whitespace(cp))
      break;
    first = next;
  }
  while (first < last) {
    auto previous = last;
    UChar32 cp;
    U8_PREV(value.data(), first, previous, cp);
    if (!whitespace(cp))
      break;
    last = previous;
  }
  return value.substr(first, last - first);
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

void AppendReasoningInstruction(std::string& output,
                                const ChatTemplateOptions& options) {
  if (!options.enable_thinking) {
    return;
  }
  switch (options.reasoning_effort) {
    case QwenReasoningEffort::kLow:
      output.append(kLowReasoningInstruction);
      return;
    case QwenReasoningEffort::kMedium:
      return;
    case QwenReasoningEffort::kXHigh:
      output.append(kXHighReasoningInstruction);
      return;
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

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools,
                       bool require_tool_call) {
  if (tools.empty()) {
    return;
  }

  if (!output.empty()) {
    output.append("\n\n");
  }
  output.append(
      "# Tools\n\nYou have access to the following functions:\n\n<tools>");
  for (const auto& tool : tools) {
    if (!tool.definition_json.empty()) {
      output.push_back('\n');
      output.append(PythonJsonSpacing(tool.definition_json));
      continue;
    }
    output.append("\n{\"type\": \"function\", \"function\": {\"name\": ");
    AppendJsonString(output, tool.name);
    output.append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output.append(", \"parameters\": ");
    output.append(tool.parameters_json.empty()
                      ? "{}"
                      : PythonJsonSpacing(tool.parameters_json));
    output.append("}}");
  }
  output.append(
      "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
      "following format with NO suffix:\n\n<tool_call>\n"
      "<function=example_function_name>\n<parameter=example_parameter_1>\n"
      "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
      "This is the value for the second parameter\nthat can span\nmultiple "
      "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
      "Reminder:\n- Function calls MUST follow the specified format: an inner "
      "<function=...></function> block must be nested within "
      "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
      "specified\n- You may provide optional reasoning for your function call "
      "in natural language BEFORE the function call, but NOT after\n- If "
      "there is no function call available, answer the question like normal "
      "with your current knowledge and do not tell the user about function "
      "calls\n</IMPORTANT>");
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
    if (IsQwen38Artifact(reader)) {
      if (error_msg != nullptr) {
        *error_msg = "Qwen3.8 GGUF is missing tokenizer.chat_template metadata";
      }
      return nullptr;
    }
    if (error_msg != nullptr) {
      *error_msg =
          "GGUF metadata missing 'tokenizer.chat_template', using default "
          "ChatML template";
    }
    return CreateDefault();
  }
  const Profile profile = IsQwen38ReasoningTemplate(*template_str)
                              ? Profile::kQwen38Reasoning
                              : Profile::kLegacyChatMl;
  if (IsQwen38Artifact(reader) && profile != Profile::kQwen38Reasoning) {
    if (error_msg != nullptr) {
      *error_msg =
          "Qwen3.8 GGUF chat template SHA-256 is not a recognized pinned "
          "version: " +
          TemplateSha256(*template_str);
    }
    return nullptr;
  }
  return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
      std::string(*template_str), TemplateSha256(*template_str), profile));
}

bool QwenChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                            std::string* error_msg) {
  if (!IsQwen38Artifact(reader)) {
    return true;
  }
  return CreateFromGguf(reader, error_msg) != nullptr;
}

std::unique_ptr<QwenChatTemplate> QwenChatTemplate::CreateDefault(
    std::string_view raw_template) {
  if (raw_template.empty()) {
    return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
        std::string(kDefaultChatmlTemplate),
        TemplateSha256(kDefaultChatmlTemplate), Profile::kLegacyChatMl));
  }
  return std::unique_ptr<QwenChatTemplate>(new QwenChatTemplate(
      std::string(raw_template), TemplateSha256(raw_template),
      IsQwen38ReasoningTemplate(raw_template) ? Profile::kQwen38Reasoning
                                              : Profile::kLegacyChatMl));
}

std::string_view QwenChatTemplate::GetTemplateId() const noexcept {
  switch (profile_) {
    case Profile::kLegacyChatMl:
      return "qwen-chatml-compiled-v1";
    case Profile::kQwen38Reasoning:
      return "qwen38-reasoning-compiled-v3";
  }
  return "qwen-chatml-compiled-v1";
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, const ChatTemplateOptions& options,
    std::string* error_msg) {
  return Render(messages, {}, options, error_msg);
}

std::optional<std::string> QwenChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets) {
  if (image_offsets != nullptr)
    image_offsets->clear();
  if (messages.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "No messages provided";
    }
    return std::nullopt;
  }

  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len +=
        msg.content.size() + msg.thought.size() + 32 + msg.images.size() * 64;
    std::size_t previous = 0;
    for (const auto& image : msg.images) {
      if (msg.role != ChatRole::kUser || image.bytes == nullptr ||
          image.offset < previous || image.offset > msg.content.size()) {
        if (error_msg != nullptr)
          *error_msg = "invalid user image content";
        return std::nullopt;
      }
      previous = image.offset;
    }
    for (const auto& call : msg.tool_calls) {
      estimated_len += call.name.size() + 64;
      for (const auto& argument : call.arguments) {
        estimated_len += argument.name.size() + argument.value.size() + 40;
      }
    }
  }
  for (const auto& tool : tools) {
    estimated_len += tool.name.size() + tool.description.size() +
                     tool.parameters_json.size() +
                     2 * tool.definition_json.size() + 96;
  }
  if (options.add_generation_prompt) {
    estimated_len += 64;
  }
  estimated_len += kXHighReasoningInstruction.size() + 64;

  if (estimated_len > options.max_output_bytes ||
      estimated_len >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    if (error_msg != nullptr) {
      *error_msg = "Rendered prompt estimated length (" +
                   std::to_string(estimated_len) +
                   " bytes) exceeds maximum bound (" +
                   std::to_string(options.max_output_bytes) + " bytes)";
    }
    return std::nullopt;
  }

  output.reserve(estimated_len);

  std::size_t message_index = 0;
  std::string system_content;
  while (message_index < messages.size() &&
         (messages[message_index].role == ChatRole::kSystem ||
          messages[message_index].role == ChatRole::kDeveloper)) {
    const auto content = Trim(messages[message_index].content);
    if (!content.empty()) {
      if (!system_content.empty())
        system_content.push_back('\n');
      system_content.append(content);
    }
    ++message_index;
  }

  std::string system_prefix;
  AppendReasoningInstruction(system_prefix, options);
  AppendToolsPrompt(system_prefix, tools, options.require_tool_call);
  if (!system_prefix.empty() && !system_content.empty()) {
    system_prefix.append("\n\n");
  }
  system_prefix.append(system_content);
  if (!system_prefix.empty()) {
    output.append("<|im_start|>system\n");
    output.append(system_prefix);
    output.append("<|im_end|>\n");
  }

  std::size_t last_user_index = messages.size();
  for (std::size_t index = messages.size(); index > 0; --index) {
    if (messages[index - 1].role == ChatRole::kUser) {
      const auto content = Trim(messages[index - 1].content);
      if (messages[index - 1].images.empty() &&
          content.starts_with("<tool_response>") &&
          content.ends_with("</tool_response>"))
        continue;
      last_user_index = index - 1;
      break;
    }
  }
  if (last_user_index == messages.size()) {
    if (error_msg != nullptr)
      *error_msg = "No user query found in messages.";
    return std::nullopt;
  }

  std::size_t image_count = 0;
  for (; message_index < messages.size(); ++message_index) {
    const auto& msg = messages[message_index];
    if (msg.role == ChatRole::kSystem || msg.role == ChatRole::kDeveloper) {
      if (error_msg != nullptr)
        *error_msg = "System message must be at the beginning.";
      return std::nullopt;
    }
    const bool tool_result = msg.role == ChatRole::kTool;
    if (tool_result) {
      output.append("<|im_start|>user\n");
      while (message_index < messages.size() &&
             messages[message_index].role == ChatRole::kTool) {
        const auto& tool_message = messages[message_index];
        output.append("<tool_response>\n");
        output.append(Trim(tool_message.content));
        output.append("\n</tool_response>");
        ++message_index;
        if (message_index < messages.size() &&
            messages[message_index].role == ChatRole::kTool) {
          output.push_back('\n');
        }
      }
      --message_index;
      output.append("<|im_end|>\n");
      continue;
    }
    const auto role_name = ToString(msg.role);
    output.append("<|im_start|>");
    output.append(role_name);
    output.push_back('\n');

    std::string image_content;
    std::vector<std::size_t> local_image_offsets;
    if (!msg.images.empty()) {
      std::size_t cursor = 0;
      for (const auto& image : msg.images) {
        image_content.append(std::string_view(msg.content)
                                 .substr(cursor, image.offset - cursor));
        ++image_count;
        if (options.add_vision_id)
          image_content.append("Picture " + std::to_string(image_count) + ": ");
        image_content.append("<|vision_start|>");
        local_image_offsets.push_back(image_content.size());
        image_content.append("<|image_pad|><|vision_end|>");
        cursor = image.offset;
      }
      image_content.append(std::string_view(msg.content).substr(cursor));
    }
    // Trim the fully rendered content, including image markers. Whitespace
    // between text and images remains significant; image offsets follow the
    // trim.
    const std::string_view untrimmed = msg.images.empty()
                                           ? std::string_view(msg.content)
                                           : std::string_view(image_content);
    const std::string_view content = Trim(untrimmed);
    const std::string_view thought = Trim(msg.thought);
    if (msg.role == ChatRole::kAssistant &&
        (options.preserve_thinking || message_index > last_user_index)) {
      output.append("<think>\n");
      output.append(thought);
      output.append("\n</think>\n\n");
    }

    if (image_offsets != nullptr && !local_image_offsets.empty()) {
      const auto removed =
          static_cast<std::size_t>(content.data() - untrimmed.data());
      for (const auto offset : local_image_offsets)
        image_offsets->push_back(output.size() + offset - removed);
    }
    output.append(content);
    if (msg.role == ChatRole::kAssistant && !msg.tool_calls.empty()) {
      if (!content.empty()) {
        output.append("\n\n");
      }
      AppendToolCalls(output, msg.tool_calls);
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

  if (options.add_generation_prompt)
    output.append(GenerationPrompt(options.enable_thinking));

  if (output.size() > options.max_output_bytes) {
    if (error_msg != nullptr) {
      *error_msg = "Prompt exceeded max output bytes after generation prompt";
    }
    return std::nullopt;
  }

  return output;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<|im_start|>assistant\n<think>\n"
                         : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
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
  if (std::any_of(messages.begin(), messages.end(), [](const auto& message) {
        return !message.images.empty();
      })) {
    if (error_msg != nullptr) {
      *error_msg =
          "image messages require the model's vision prompt preparation";
    }
    return std::nullopt;
  }
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

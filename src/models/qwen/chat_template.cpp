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
#include "src/models/qwen/control_tokens.hpp"
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

// Message text is text. Content spans are tokenized without special matching,
// so a vocabulary token a client sends — or the model itself wrote — reaches
// the model as the characters it contains, never as a control token that ends
// a turn in the middle of a conversation (#383).
void AppendContent(std::string& output, std::vector<ContentSpan>* spans,
                   std::string_view text) {
  const auto begin = output.size();
  output.append(text);
  if (spans != nullptr && !text.empty()) {
    spans->push_back({begin, output.size() - begin});
  }
}

// Keep images inside their owning message, including grouped tool results.
// Only the inserted vision markers are framing; surrounding text stays literal.
bool AppendMessageContent(std::string& output, const ChatMessage& message,
                          const ChatTemplateOptions& options,
                          std::size_t& image_count,
                          std::vector<std::size_t>* image_offsets,
                          std::vector<ContentSpan>* content_spans) {
  if (message.images.empty()) {
    const auto content = Trim(message.content);
    AppendContent(output, content_spans, content);
    return !content.empty();
  }
  std::string rendered;
  std::vector<ContentSpan> spans;
  std::vector<std::size_t> offsets;
  std::size_t cursor = 0;
  for (const auto& image : message.images) {
    AppendContent(rendered, &spans,
                  std::string_view(message.content)
                      .substr(cursor, image.offset - cursor));
    ++image_count;
    if (options.add_vision_id)
      rendered.append("Picture " + std::to_string(image_count) + ": ");
    rendered.append(kVisionStart);
    offsets.push_back(rendered.size());
    rendered.append(kImagePad).append(kVisionEnd);
    cursor = image.offset;
  }
  AppendContent(rendered, &spans,
                std::string_view(message.content).substr(cursor));
  const auto content = Trim(rendered);
  const auto removed =
      static_cast<std::size_t>(content.data() - rendered.data());
  const auto start = output.size();
  if (image_offsets != nullptr)
    for (const auto offset : offsets)
      image_offsets->push_back(start + offset - removed);
  if (content_spans != nullptr) {
    for (const auto& span : spans) {
      const auto begin = std::max(span.offset, removed);
      const auto end =
          std::min(span.offset + span.size, removed + content.size());
      if (end > begin)
        content_spans->push_back({start + begin - removed, end - begin});
    }
  }
  output.append(content);
  return !content.empty();
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
                     std::span<const ChatMessage::ToolCall> calls,
                     std::vector<ContentSpan>* content_spans) {
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
      AppendContent(output, content_spans, argument.value);
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
    std::vector<std::size_t>* image_offsets, std::size_t* stable_prefix_bytes,
    std::vector<ContentSpan>* content_spans) {
  if (image_offsets != nullptr)
    image_offsets->clear();
  if (content_spans != nullptr)
    content_spans->clear();
  if (messages.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "No messages provided";
    }
    return std::nullopt;
  }

  // The reference template accepts one leading system turn, but OpenAI clients
  // such as Codex send system/developer messages mid-conversation. Hoist them,
  // in order, into that turn; templates that allow them keep them in place.
  const auto is_system = [](const ChatMessage& message) {
    return message.role == ChatRole::kSystem ||
           message.role == ChatRole::kDeveloper;
  };
  std::vector<ChatMessage> hoisted;
  if (std::any_of(std::find_if_not(messages.begin(), messages.end(), is_system),
                  messages.end(), is_system)) {
    hoisted.assign(messages.begin(), messages.end());
    std::stable_partition(hoisted.begin(), hoisted.end(), is_system);
    messages = hoisted;
  }

  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len += msg.content.size() + msg.framing_suffix.size() +
                     msg.thought.size() + 32 + msg.images.size() * 64;
    std::size_t previous = 0;
    for (const auto& image : msg.images) {
      if ((msg.role != ChatRole::kUser && msg.role != ChatRole::kTool) ||
          image.bytes == nullptr || image.offset < previous ||
          image.offset > msg.content.size()) {
        if (error_msg != nullptr)
          *error_msg = "invalid user or tool image content";
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
  std::vector<ContentSpan> system_content_spans;
  while (message_index < messages.size() &&
         (messages[message_index].role == ChatRole::kSystem ||
          messages[message_index].role == ChatRole::kDeveloper)) {
    const auto& message = messages[message_index];
    const std::string untrimmed = message.content + message.framing_suffix;
    const auto content = Trim(untrimmed);
    if (!content.empty()) {
      if (!system_content.empty())
        system_content.push_back('\n');
      const auto begin =
          static_cast<std::size_t>(content.data() - untrimmed.data());
      const auto client_end =
          std::min(message.content.size(), begin + content.size());
      const auto client_size = client_end > begin ? client_end - begin : 0;
      AppendContent(system_content, &system_content_spans,
                    content.substr(0, client_size));
      system_content.append(content.substr(client_size));
    }
    ++message_index;
  }

  std::string system_prefix;
  std::vector<ContentSpan> system_spans;
  AppendReasoningInstruction(system_prefix, options);
  AppendToolsPrompt(system_prefix, tools, options.require_tool_call);
  if (!system_prefix.empty() && !system_content.empty()) {
    system_prefix.append("\n\n");
  }
  const auto system_content_offset = system_prefix.size();
  system_prefix.append(system_content);
  for (const auto& span : system_content_spans)
    system_spans.push_back({system_content_offset + span.offset, span.size});
  if (!system_prefix.empty()) {
    output.append(kImStart).append("system\n");
    const auto prefix_offset = output.size();
    output.append(system_prefix);
    output.append(kImEnd).append("\n");
    if (content_spans != nullptr) {
      for (const auto& span : system_spans)
        content_spans->push_back({prefix_offset + span.offset, span.size});
    }
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
  std::optional<std::size_t> mutable_reasoning;
  // Agent clients may end every request with a user message that the next
  // request replaces (per-turn runtime context) rather than keeps. A
  // checkpoint at the end of the prompt then never prefixes the next request.
  // Such context follows a tool result or another user message, not an
  // assistant reply, so only then keep the checkpoint before the final
  // text-only user turn. A client that keeps the message prefills it again.
  const bool final_user_turn =
      last_user_index + 1 == messages.size() &&
      messages.back().images.empty() && messages.size() >= 2 &&
      (messages[messages.size() - 2].role == ChatRole::kTool ||
       messages[messages.size() - 2].role == ChatRole::kUser);
  bool seen_assistant = false;
  std::optional<std::size_t> final_user_start;
  for (; message_index < messages.size(); ++message_index) {
    const auto& msg = messages[message_index];
    const bool tool_result = msg.role == ChatRole::kTool;
    if (tool_result) {
      output.append(kImStart).append("user\n");
      while (message_index < messages.size() &&
             messages[message_index].role == ChatRole::kTool) {
        const auto& tool_message = messages[message_index];
        output.append("<tool_response>\n");
        AppendMessageContent(output, tool_message, options, image_count,
                             image_offsets, content_spans);
        output.append("\n</tool_response>");
        ++message_index;
        if (message_index < messages.size() &&
            messages[message_index].role == ChatRole::kTool) {
          output.push_back('\n');
        }
      }
      --message_index;
      output.append(kImEnd).append("\n");
      continue;
    }
    // Current tool-cycle assistants retain their reasoning even when older
    // reasoning is disabled. A new user turn removes it, including empty
    // <think> framing. Keep a checkpoint before the first affected assistant.
    if (!options.preserve_thinking && msg.role == ChatRole::kAssistant &&
        message_index > last_user_index && !mutable_reasoning)
      mutable_reasoning = output.size();
    if (msg.role == ChatRole::kAssistant)
      seen_assistant = true;
    else if (final_user_turn && seen_assistant &&
             message_index == last_user_index)
      final_user_start = output.size();
    const auto role_name = ToString(msg.role);
    output.append(kImStart);
    output.append(role_name);
    output.push_back('\n');

    const std::string_view thought = Trim(msg.thought);
    if (msg.role == ChatRole::kAssistant &&
        (options.preserve_thinking || message_index > last_user_index)) {
      output.append("<think>\n");
      AppendContent(output, content_spans, thought);
      output.append("\n</think>\n\n");
    }

    const bool has_content = AppendMessageContent(
        output, msg, options, image_count, image_offsets, content_spans);
    if (msg.role == ChatRole::kAssistant && !msg.tool_calls.empty()) {
      if (has_content) {
        output.append("\n\n");
      }
      AppendToolCalls(output, msg.tool_calls, content_spans);
    }
    output.append(kImEnd).append("\n");

    if (output.size() > options.max_output_bytes) {
      if (error_msg != nullptr) {
        *error_msg =
            "Prompt exceeded max output bytes during message rendering";
      }
      return std::nullopt;
    }
  }

  if (stable_prefix_bytes != nullptr)
    *stable_prefix_bytes = std::min(mutable_reasoning.value_or(output.size()),
                                    final_user_start.value_or(output.size()));
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
  static constexpr std::string_view kThinkingSuffix = "assistant\n<think>\n";
  static constexpr std::string_view kAnsweredSuffix =
      "assistant\n<think>\n\n</think>\n\n";
  static constexpr auto kThinking =
      ConcatControlText<kImStart, kThinkingSuffix>();
  static constexpr auto kAnswered =
      ConcatControlText<kImStart, kAnsweredSuffix>();
  return enable_thinking ? std::string_view(kThinking.data(), kThinking.size())
                         : std::string_view(kAnswered.data(), kAnswered.size());
}

std::optional<std::vector<TokenId>> QwenChatTemplate::RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    const ChatTemplateOptions& options, std::string* error_msg) {
  return RenderAndTokenize(tokenizer, messages, {}, options, error_msg);
}

std::vector<TokenId> QwenChatTemplate::EncodeRendered(
    const QwenTokenizer& tokenizer, std::string_view rendered,
    std::size_t begin, std::size_t end,
    std::span<const ContentSpan> content_spans,
    const TokenizerOptions& options) {
  const auto slice = rendered.substr(begin, end - begin);
  // Only split spans containing literal vocabulary-token spellings. Splitting
  // every content span when a later message contains one changes earlier BPE
  // merges at content/framing boundaries and invalidates unchanged history.
  TokenizerOptions as_framing = options;
  as_framing.parse_special_tokens = true;
  TokenizerOptions as_content = options;
  as_content.parse_special_tokens = false;
  const auto encode = [&](std::size_t from, std::size_t to,
                          const TokenizerOptions& opts) {
    return tokenizer.Encode(rendered.substr(from, to - from), opts);
  };

  std::vector<TokenId> tokens;
  std::size_t cursor = begin;
  for (const auto& span : content_spans) {
    const auto span_end = span.offset + span.size;
    if (span_end <= begin || span.offset >= end) {
      continue;
    }
    const auto from = std::max(span.offset, begin);
    const auto to = std::min(span_end, end);
    const auto content_text = rendered.substr(from, to - from);
    if (std::none_of(tokenizer.SpecialTokens().begin(),
                     tokenizer.SpecialTokens().end(), [&](const auto& token) {
                       return content_text.find(token.first) !=
                              std::string_view::npos;
                     }))
      continue;
    if (from > cursor) {
      const auto framing = encode(cursor, from, as_framing);
      tokens.insert(tokens.end(), framing.begin(), framing.end());
    }
    const auto content = encode(from, to, as_content);
    tokens.insert(tokens.end(), content.begin(), content.end());
    cursor = to;
  }
  if (cursor == begin)
    return tokenizer.Encode(slice, options);
  if (cursor < end) {
    const auto tail = encode(cursor, end, as_framing);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
  }
  return tokens;
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
  std::vector<ContentSpan> content_spans;
  const auto rendered = Render(messages, tools, options, error_msg, nullptr,
                               nullptr, &content_spans);
  if (!rendered.has_value()) {
    return std::nullopt;
  }

  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;
  return EncodeRendered(tokenizer, *rendered, 0, rendered->size(),
                        content_spans, tok_opts);
}

}  // namespace gufo::tokenization

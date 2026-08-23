#include "src/models/qwen/qwen_chat_template.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

namespace strix::tokenization {

namespace {

constexpr std::string_view kDefaultChatmlTemplate =
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

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
  std::string output;

  std::size_t estimated_len = 0;
  for (const auto& msg : messages) {
    estimated_len += msg.content.size() + msg.thought.size() + 32;
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

  for (const auto& msg : messages) {
    const auto role_name = ToString(msg.role);
    output.append("<|im_start|>");
    output.append(role_name);
    output.push_back('\n');

    if (msg.role == ChatRole::kAssistant && options.enable_thinking &&
        !msg.thought.empty()) {
      output.append("<think>\n");
      output.append(msg.thought);
      output.append("\n</think>\n");
    }

    output.append(msg.content);
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
  const auto rendered = Render(messages, options, error_msg);
  if (!rendered.has_value()) {
    return std::nullopt;
  }

  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;

  return tokenizer.Encode(*rendered, tok_opts);
}

}  // namespace strix::tokenization

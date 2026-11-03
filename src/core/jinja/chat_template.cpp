#include "src/core/jinja/chat_template.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/jinja/vendor/lexer.h"
#include "src/core/jinja/vendor/parser.h"

namespace gufo::core {

void ChatTemplateConfig::Validate(bool raw) const {
  if (!path.empty() && !enabled)
    throw std::invalid_argument(
        "--chat-template-path requires --chat-template");
  if (raw && (enabled || !path.empty()))
    throw std::invalid_argument(
        "--chat-template cannot be combined with --raw");
}

struct JinjaChatTemplate::Impl {
  std::string source;
  std::string tool_source;
  std::string bos;
  std::string eos;
  std::string identity;
  jinja::program program;
  std::optional<jinja::program> tool_program;
};

JinjaChatTemplate::JinjaChatTemplate(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
JinjaChatTemplate::~JinjaChatTemplate() = default;

std::shared_ptr<const JinjaChatTemplate> JinjaChatTemplate::Compile(
    std::string source, std::string bos, std::string eos,
    std::string tool_source) {
  if (source.find_first_not_of(" \t\r\n") == std::string::npos)
    throw std::invalid_argument("chat template is empty");
  auto impl = std::make_unique<Impl>();
  impl->source = std::move(source);
  impl->tool_source = std::move(tool_source);
  impl->bos = std::move(bos);
  impl->eos = std::move(eos);
  impl->program =
      jinja::parse_from_tokens(jinja::lexer{}.tokenize(impl->source));
  if (!impl->tool_source.empty() &&
      impl->tool_source.find_first_not_of(" \t\r\n") == std::string::npos)
    throw std::invalid_argument("tool_use chat template is empty");
  if (!impl->tool_source.empty())
    impl->tool_program =
        jinja::parse_from_tokens(jinja::lexer{}.tokenize(impl->tool_source));
  const auto hash = [](const std::string& text) {
    return crypto::Sha256Hex(
        {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
  };
  impl->identity = "llama-jinja-68d9053a-gufo-v1:" + hash(impl->source) + ":" +
                   hash(impl->tool_source) + ":" + hash(impl->bos) + ":" +
                   hash(impl->eos);
  return std::shared_ptr<const JinjaChatTemplate>(
      new JinjaChatTemplate(std::move(impl)));
}

std::shared_ptr<const JinjaChatTemplate> JinjaChatTemplate::Load(
    const GgufReader& reader, const ChatTemplateConfig& config) {
  config.Validate();
  if (!config.enabled)
    return {};
  std::string source;
  std::string tool_source;
  if (!config.path.empty()) {
    std::ifstream file(std::filesystem::path(config.path), std::ios::binary);
    if (!file)
      throw std::invalid_argument("cannot read chat template: " + config.path);
    source.assign(std::istreambuf_iterator<char>(file), {});
    if (file.bad())
      throw std::invalid_argument("error reading chat template: " +
                                  config.path);
  } else {
    const auto embedded = reader.GetMetadataString("tokenizer.chat_template");
    if (!embedded)
      throw std::invalid_argument("GGUF is missing tokenizer.chat_template");
    source = *embedded;
    if (const auto tools =
            reader.GetMetadataString("tokenizer.chat_template.tool_use")) {
      if (tools->empty())
        throw std::invalid_argument("tool_use chat template is empty");
      tool_source = *tools;
    }
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto special = [&](const char* key) {
    const auto id = reader.GetMetadataUint32(key);
    return id && *id < tokens.size() ? std::string(tokens[*id]) : std::string{};
  };
  return Compile(std::move(source), special("tokenizer.ggml.bos_token_id"),
                 special("tokenizer.ggml.eos_token_id"),
                 std::move(tool_source));
}

std::vector<TemplatePart> JinjaChatTemplate::Render(const json::Value& inputs,
                                                    const json::Value& kwargs,
                                                    std::size_t max_bytes) const
    try {
  if (!kwargs.is_object())
    throw std::invalid_argument("chat_template_kwargs must be an object");
  const auto* tools = inputs.find("tools");
  const bool use_tools = tools && !tools->empty() && impl_->tool_program;
  jinja::context context(use_tools ? impl_->tool_source : impl_->source);
  for (const auto& [key, value] : kwargs.members()) {
    if (key == "messages" || key == "tools" || key == "bos_token" ||
        key == "eos_token" || key == "add_generation_prompt")
      throw std::invalid_argument("reserved chat template variable: " + key);
  }
  jinja::global_from_json(context, kwargs, true);
  // Validated controls override kwargs; only message/tool strings are input.
  for (const auto& [key, value] : inputs.members()) {
    auto entry = json::Value::object();
    entry[key] = value;
    jinja::global_from_json(context, entry,
                            key == "messages" || key == "tools");
  }
  context.set_val("bos_token", jinja::mk_val<jinja::value_string>(impl_->bos));
  context.set_val("eos_token", jinja::mk_val<jinja::value_string>(impl_->eos));
  jinja::runtime runtime(context);
  const auto rendered = jinja::runtime::gather_string_parts(
      runtime.execute(use_tools ? *impl_->tool_program : impl_->program));
  std::vector<TemplatePart> result;
  std::size_t size = 0;
  for (const auto& part : rendered->val_str.parts) {
    if (part.val.size() > max_bytes - size)
      throw std::invalid_argument(
          "rendered chat template exceeds output limit");
    size += part.val.size();
    if (part.val.empty())
      continue;
    if (!result.empty() && result.back().input == part.is_input)
      result.back().text += part.val;
    else
      result.push_back({part.val, part.is_input});
  }
  return result;
} catch (const jinja::raised_exception& error) {
  throw std::invalid_argument(error.what());
} catch (const jinja::rethrown_exception& error) {
  throw std::invalid_argument(error.what());
} catch (const std::runtime_error& error) {
  throw std::invalid_argument(error.what());
}

const std::string& JinjaChatTemplate::Identity() const {
  return impl_->identity;
}

bool PromptOpensReasoning(std::string_view prompt) {
  const auto open = prompt.rfind("<think>");
  const auto close = prompt.rfind("</think>");
  return open != std::string_view::npos &&
         (close == std::string_view::npos || open > close);
}

}  // namespace gufo::core

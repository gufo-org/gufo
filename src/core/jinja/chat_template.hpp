#ifndef GUFO_CORE_JINJA_CHAT_TEMPLATE_HPP_
#define GUFO_CORE_JINJA_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace gufo::core {

struct ChatTemplateConfig {
  bool enabled{false};
  std::string path;
  /// Throws on incompatible CLI selections, before loading a model.
  void Validate(bool raw = false) const;
};

struct TemplatePart {
  std::string text;
  bool input{false};
};

/// Immutable compiled Jinja programs. Each render owns its evaluation context.
class JinjaChatTemplate {
public:
  static std::shared_ptr<const JinjaChatTemplate> Load(
      const GgufReader& reader, const ChatTemplateConfig& config);
  static std::shared_ptr<const JinjaChatTemplate> Compile(
      std::string source, std::string bos = {}, std::string eos = {},
      std::string tool_source = {});
  ~JinjaChatTemplate();
  std::vector<TemplatePart> Render(const json::Value& inputs,
                                   const json::Value& kwargs,
                                   std::size_t max_bytes) const;
  const std::string& Identity() const;

private:
  struct Impl;
  explicit JinjaChatTemplate(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// State of Gufo's native reasoning tags at the end of a rendered prompt.
bool PromptOpensReasoning(std::string_view prompt);

}  // namespace gufo::core
#endif

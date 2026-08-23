#ifndef STRIX_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_
#define STRIX_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

namespace strix::tokenization {

/// Chat message roles supported by the Qwen chat template.
enum class ChatRole : std::uint8_t {
  kSystem = 0,
  kDeveloper = 1,
  kUser = 2,
  kAssistant = 3,
  kTool = 4,
};

[[nodiscard]] constexpr std::string_view ToString(ChatRole role) noexcept {
  switch (role) {
    case ChatRole::kSystem:
    case ChatRole::kDeveloper:
      return "system";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

/// A structured input message for chat formatting.
struct ChatMessage {
  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string name;     ///< Optional function/tool name
  std::string thought;  ///< Optional thinking/reasoning prefix
};

/// Formatting options for rendering a conversation into a text prompt.
struct ChatTemplateOptions {
  bool add_generation_prompt{true};
  bool enable_thinking{false};
  std::size_t max_output_bytes{1024ULL * 1024ULL};  ///< 1 MiB upper bound
};

/// Deterministic, bounded Qwen ChatML formatter.
class QwenChatTemplate {
public:
  ~QwenChatTemplate() = default;

  QwenChatTemplate(const QwenChatTemplate&) = delete;
  QwenChatTemplate& operator=(const QwenChatTemplate&) = delete;
  QwenChatTemplate(QwenChatTemplate&&) noexcept = default;
  QwenChatTemplate& operator=(QwenChatTemplate&&) noexcept = default;

  /// Creates a chat template by extracting the template string from GGUF
  /// metadata.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Creates a default Qwen ChatML template formatter.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateDefault(
      std::string_view raw_template = "");

  [[nodiscard]] std::string_view GetTemplateString() const noexcept {
    return template_string_;
  }

  /// Formats a list of messages into a deterministic UTF-8 prompt string.
  [[nodiscard]] static std::optional<std::string> Render(
      std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

  /// Formats messages and tokenizes the rendered prompt with the given
  /// tokenizer.
  [[nodiscard]] static std::optional<std::vector<TokenId>> RenderAndTokenize(
      const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

private:
  explicit QwenChatTemplate(std::string template_str)
      : template_string_(std::move(template_str)) {}

  std::string template_string_;
};

}  // namespace strix::tokenization

#endif  // STRIX_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

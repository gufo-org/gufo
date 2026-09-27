#ifndef GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_
#define GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::tokenization {

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
  ChatMessage() = default;
  ChatMessage(ChatRole message_role, std::string message_content,
              std::string message_name = {}, std::string message_thought = {})
      : role(message_role),
        content(std::move(message_content)),
        name(std::move(message_name)),
        thought(std::move(message_thought)) {}

  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string name;     ///< Optional function/tool name
  std::string thought;  ///< Optional thinking/reasoning prefix
  std::string tool_call_id;
  struct ImagePart {
    /// Insert an image before this byte of content. Equal offsets preserve
    /// input order; image-only messages use offset zero.
    std::size_t offset{0};
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
  };
  std::vector<ImagePart> images;

  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string id;
    std::string name;
    std::vector<ToolArgument> arguments;
  };

  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete validated HTTP tool object, preserving field order/extensions.
  std::string definition_json;
};

enum class QwenReasoningEffort : std::uint8_t {
  kLow,
  kMedium,
  kXHigh,
};

/// Formatting options for rendering a conversation into a text prompt.
struct ChatTemplateOptions {
  bool add_generation_prompt{true};
  bool enable_thinking{true};
  QwenReasoningEffort reasoning_effort{QwenReasoningEffort::kXHigh};
  bool preserve_thinking{true};
  bool add_vision_id{false};
  bool require_tool_call{false};
  std::size_t max_output_bytes{1024ULL * 1024ULL};  ///< 1 MiB upper bound
};

/// Longest Qwen3.8 vocabulary entry in bytes (Flash-Next GGUF, 248,320
/// tokens). A prompt of N tokens cannot render to more than N times this.
inline constexpr std::size_t kMaxRenderedBytesPerToken = 128;

/// Rendered-prompt bound for a session of `context_tokens`: any prompt that
/// fits the context passes, never below the 1 MiB default.
[[nodiscard]] std::size_t RenderedPromptBoundBytes(
    std::uint32_t context_tokens) noexcept;

/// Suffix opened for a new assistant turn, outside the stable conversation.
[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Resolve CLI/API controls against the official Qwen3.8 template defaults.
/// Provider-neutral minimal/high/max map to native low/xhigh/xhigh.
[[nodiscard]] ChatTemplateOptions ResolveQwenChatOptions(
    const ReasoningOptions& reasoning, bool add_vision_id = false);

/// Deterministic, bounded Qwen ChatML formatter.
class QwenChatTemplate {
public:
  enum class Profile : std::uint8_t {
    kLegacyChatMl,
    kQwen38Reasoning,
  };

  ~QwenChatTemplate() = default;

  QwenChatTemplate(const QwenChatTemplate&) = delete;
  QwenChatTemplate& operator=(const QwenChatTemplate&) = delete;
  QwenChatTemplate(QwenChatTemplate&&) noexcept = default;
  QwenChatTemplate& operator=(QwenChatTemplate&&) noexcept = default;

  /// Creates a chat template by extracting the template string from GGUF
  /// metadata.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Validates that a Qwen3.8 artifact carries a recognized, pinned chat
  /// template. Other Qwen model revisions retain their legacy formatter.
  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Creates a default Qwen ChatML template formatter.
  [[nodiscard]] static std::unique_ptr<QwenChatTemplate> CreateDefault(
      std::string_view raw_template = "");

  [[nodiscard]] std::string_view GetTemplateString() const noexcept {
    return template_string_;
  }
  [[nodiscard]] std::string_view GetTemplateSha256() const noexcept {
    return template_sha256_;
  }
  [[nodiscard]] Profile GetProfile() const noexcept { return profile_; }
  [[nodiscard]] std::string_view GetTemplateId() const noexcept;

  [[nodiscard]] static constexpr std::string_view
  OfficialTemplateSha256() noexcept {
    return "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
  }

  [[nodiscard]] static constexpr std::string_view
  UnslothArtifactTemplateSha256() noexcept {
    return "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";
  }

  /// Formats a list of messages into a deterministic UTF-8 prompt string.
  [[nodiscard]] static std::optional<std::string> Render(
      std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<std::string> Render(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options = {}, std::string* error_msg = nullptr,
      std::vector<std::size_t>* image_offsets = nullptr);

  /// Formats messages and tokenizes the rendered prompt with the given
  /// tokenizer.
  [[nodiscard]] static std::optional<std::vector<TokenId>> RenderAndTokenize(
      const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
      const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<std::vector<TokenId>> RenderAndTokenize(
      const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
      std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
      std::string* error_msg = nullptr);

private:
  QwenChatTemplate(std::string template_str, std::string template_sha256,
                   Profile profile)
      : template_string_(std::move(template_str)),
        template_sha256_(std::move(template_sha256)),
        profile_(profile) {}

  std::string template_string_;
  std::string template_sha256_;
  Profile profile_{Profile::kLegacyChatMl};
};

}  // namespace gufo::tokenization

#endif  // GUFO_TOKENIZATION_QWEN_CHAT_TEMPLATE_HPP_

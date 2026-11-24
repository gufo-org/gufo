#ifndef GUFO_SERVER_CHAT_REQUEST_HPP_
#define GUFO_SERVER_CHAT_REQUEST_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

/// The request bodies the Rust decoder (src/core/chat_request) reads.
enum class ChatApi : std::uint8_t { kChat, kResponses, kMessages };

/// What decoding needs from the server. ChatEnvironment fills it from a
/// backend; tests replace the hooks.
struct ChatDecodeEnvironment {
  /// Chat: the name a request without "model" selects when one is optional.
  std::string model_id;
  bool accepts_images{false};
  /// Chat merges these after parsing; Responses and Messages start from them.
  ReasoningOptions reasoning_defaults;
  TextGenerationBackend::SamplingDefaults sampling_defaults;
  std::function<bool(std::string_view)> serves_model;
  /// Throws std::exception with the message a client gets.
  std::function<std::vector<std::uint8_t>(std::string_view,
                                          core::ImageReadBudget&)>
      read_image;
  /// Compiles a strict tool's parameters; throws std::exception.
  std::function<void(const json::Value&)> compile_schema;
  /// Responses and Messages: ReadCompatibilityOptions (http_server.cpp) on
  /// the body, with the thinking mode that resolves sampling defaults.
  std::function<std::optional<HttpResponse>(
      ChatApi, const json::Value& body, std::optional<bool> thinking,
      std::size_t* max_tokens, sampling::SamplingConfig* sampling)>
      compatibility;
};

/// The backend's names, defaults, image reader and schema compiler, without
/// a compatibility validator.
ChatDecodeEnvironment ChatEnvironment(const TextGenerationBackend& backend);

/// A decoded body: the request and the fields the endpoint handlers read.
struct DecodedChatRequest {
  ChatRequest chat;
  /// Chat: the selected name; Responses and Messages: "model" when it is a
  /// string.
  std::optional<std::string> model;
  /// Chat: the requested limit; Responses and Messages: the validated limit.
  std::optional<std::size_t> max_tokens;
  sampling::SamplingConfig sampling;
  bool stream{false};
  bool include_usage{false};
};

/// Decodes `body` as `api` reads it (ParseRequest, ReadOpenAiResponseChat or
/// ReadAnthropicChat). Returns the rejection, or fills `output`. Running out
/// of memory is a 400 rejection whose message is std::bad_alloc's, as the C++
/// decoders answered; a hook's exception that the API does not report is
/// rethrown.
std::optional<HttpResponse> DecodeChatRequest(
    ChatApi api, std::string_view body,
    const ChatDecodeEnvironment& environment, bool require_model,
    DecodedChatRequest* output);

/// The Responses controls of `body` applied to `chat`, which keeps its value
/// on a rejection (ParseOpenAiResponseControls).
std::optional<HttpResponse> DecodeResponseControls(const json::Value& body,
                                                   ChatRequest* chat);
/// The Messages tools of `body` applied to `chat`
/// (ParseAnthropicToolControls).
std::optional<HttpResponse> DecodeAnthropicToolControls(const json::Value& body,
                                                        ChatRequest* chat);
/// One Responses item as a new message (ParseOpenAiResponseMessage).
bool DecodeResponseItem(const json::Value& item,
                        tokenization::ChatMessage* message,
                        core::ImageReadBudget& budget, std::string* error);

/// The rejection body Error (openai_chat.cpp) and Err (http_server.cpp) share.
HttpResponse ChatRejection(int status, std::string message,
                           std::string_view code);

}  // namespace gufo::server

#endif  // GUFO_SERVER_CHAT_REQUEST_HPP_

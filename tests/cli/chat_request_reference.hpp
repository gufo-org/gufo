// Frozen copies of the C++ chat request decoders that src/core/chat_request
// replaced (src/cli/serve/openai_chat.cpp and http_server.cpp before #1351),
// for tests/cli/chat_request_test.cpp. Their backend calls, image reads and
// strict-schema compiles go through a ChatDecodeEnvironment instead.
#ifndef GUFO_TESTS_CLI_CHAT_REQUEST_REFERENCE_HPP_
#define GUFO_TESTS_CLI_CHAT_REQUEST_REFERENCE_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/serve/chat_request.hpp"

namespace gufo::server::chat_request_reference {

/// ParseRequest, ReadOpenAiResponseChat or ReadAnthropicChat, with what the
/// endpoint handlers read of the body (stream, model and max tokens).
std::optional<HttpResponse> Decode(ChatApi api, std::string_view body,
                                   const ChatDecodeEnvironment& environment,
                                   bool require_model,
                                   DecodedChatRequest* output);

/// ParseOpenAiResponseControls, ParseAnthropicToolControls and
/// ParseOpenAiResponseMessage.
std::optional<HttpResponse> ResponseControls(
    const json::Value& body, ChatRequest* chat,
    const ChatDecodeEnvironment& environment);
std::optional<HttpResponse> AnthropicToolControls(
    const json::Value& body, ChatRequest* chat,
    const ChatDecodeEnvironment& environment);
bool ResponseItem(const json::Value& item, tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error,
                  const ChatDecodeEnvironment& environment);

/// ReadCompatibilityOptions with each API's allowances, which both decoders
/// get as their environment's compatibility check.
std::optional<HttpResponse> Compatibility(
    ChatApi api, const json::Value& body, std::optional<bool> thinking,
    std::size_t* max_tokens, sampling::SamplingConfig* sampling,
    const ChatDecodeEnvironment& environment);

}  // namespace gufo::server::chat_request_reference

#endif  // GUFO_TESTS_CLI_CHAT_REQUEST_REFERENCE_HPP_

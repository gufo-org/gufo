#ifndef GUFO_SERVER_OPENAI_CHAT_HPP_
#define GUFO_SERVER_OPENAI_CHAT_HPP_

#include "src/cli/serve/http_server.hpp"
#include "src/core/image.hpp"

namespace gufo::server {

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// consume scheduler-published token pieces from HttpResponse::streaming_body;
/// socket writes never own or execute model state.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

/// Responses nests output schema under text.format and effort under reasoning.
/// Shared controls stay validated as in Chat Completions, but the Responses
/// adapter tolerates standard Responses fields with no native effect (hosted
/// tool types, include, reasoning.summary, text.verbosity) rather than reject
/// the whole request, because conforming Responses clients routinely send them.
std::optional<HttpResponse> ParseOpenAiResponseControls(const json::Value& body,
                                                        ChatRequest* chat);
bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error);
/// Shared effort names; each API applies its own thinking and alias rules.
std::optional<ReasoningEffort> ParseReasoningEffortName(std::string_view value);

/// Messages tools and tool_choice, mapped onto the Chat tool declarations so
/// framing, schema constraints and replayed tool turns match Chat.
std::optional<HttpResponse> ParseAnthropicToolControls(const json::Value& body,
                                                       ChatRequest* chat);
/// One Messages turn with tool_use or tool_result blocks. Each tool_result
/// becomes a tool message; text in the same user turn follows as user text.
bool ParseAnthropicToolMessage(const json::Value& item,
                               std::vector<tokenization::ChatMessage>* messages,
                               std::string* error);
/// Responses text output uses the same reasoning/UTF-8 filter and scheduler
/// as Chat Completions, including streaming cancellation and cache retention.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream);
/// Messages output through the same path: content blocks when buffered,
/// Anthropic SSE events when streamed.
HttpResponse CreateAnthropicMessage(const HttpRequest& request,
                                    TextGenerationBackend& backend,
                                    const ChatRequest& chat,
                                    std::size_t max_tokens,
                                    const sampling::SamplingConfig& sampling,
                                    bool stream);

}  // namespace gufo::server

#endif  // GUFO_SERVER_OPENAI_CHAT_HPP_

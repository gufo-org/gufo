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

/// Reasoning and visible text of a generation without tools or schemas, split
/// exactly as Chat Completions reports them.
struct GeneratedText {
  std::string reasoning;
  std::string text;
};
GeneratedText SplitGeneratedText(
    std::string_view text, TextGenerationBackend::InitialOutputState initial);
/// Responses text output uses the same reasoning/UTF-8 filter and scheduler
/// as Chat Completions, including streaming cancellation and cache retention.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream);

}  // namespace gufo::server

#endif  // GUFO_SERVER_OPENAI_CHAT_HPP_

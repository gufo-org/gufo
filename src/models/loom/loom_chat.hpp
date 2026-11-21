// OpenAI request bodies for a package's chat program, and the canonical
// tool-call markup Gufo's parsers read.
#ifndef GUFO_LOOM_CHAT_HPP_
#define GUFO_LOOM_CHAT_HPP_

#include <cstdint>
#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>
#include <span>
#include <string>
#include <string_view>

namespace gufo::loom {

const char* RoleName(ext::ChatRole role);
json::Value ParseOr(const std::string& text, json::Value fallback);
std::string Base64(std::span<const std::uint8_t> bytes);
std::string_view MediaType(std::span<const std::uint8_t> bytes);
json::Value ContentParts(const ext::ChatMessage& message);
std::string ChatBody(const std::string& model, const ext::ChatRequest& request,
                     bool with_effort = true);
// `logits` adds the package's documented request key ("logits": true), which
// makes the row's selecting spans leave their full logit row.
std::string SamplingBody(const std::string& model,
                         const sampling::SamplingConfig& config,
                         bool logits = false);
std::string ToolMarkup(std::string_view calls_json);

}  // namespace gufo::loom

#endif  // GUFO_LOOM_CHAT_HPP_

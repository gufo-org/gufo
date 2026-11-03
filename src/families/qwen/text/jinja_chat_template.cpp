#include <algorithm>
#include <stdexcept>

#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::tokenization {
bool RenderedOpensReasoning(std::string_view rendered,
                            std::span<const ContentSpan> content_spans) {
  const auto start = content_spans.empty() ? 0
                                           : content_spans.back().offset +
                                                 content_spans.back().size;
  return core::PromptOpensReasoning(rendered.substr(start));
}
namespace {

json::Value MessageJson(const ChatMessage& message) {
  auto out = json::Value::object();
  out["role"] = message.role == ChatRole::kDeveloper
                    ? "developer"
                    : std::string(ToString(message.role));
  out["content"] = message.content + message.framing_suffix;
  if (message.template_content.is_array()) {
    out["content"] = message.template_content;
    if (!message.framing_suffix.empty()) {
      auto suffix = json::Value::object();
      suffix["type"] = "text";
      suffix["text"] = message.framing_suffix;
      out["content"].push_back(std::move(suffix));
    }
  } else if (!message.images.empty()) {
    auto content = json::Value::array();
    std::size_t offset = 0;
    for (const auto& image : message.images) {
      if ((message.role != ChatRole::kUser && message.role != ChatRole::kTool &&
           message.role != ChatRole::kDeveloper) ||
          !image.bytes || image.offset < offset ||
          image.offset > message.content.size())
        throw std::invalid_argument("invalid image offset in chat message");
      auto text = json::Value::object();
      text["type"] = "text";
      text["text"] = message.content.substr(offset, image.offset - offset);
      content.push_back(std::move(text));
      auto part = json::Value::object();
      part["type"] = "image";
      content.push_back(std::move(part));
      offset = image.offset;
    }
    auto text = json::Value::object();
    text["type"] = "text";
    text["text"] = message.content.substr(offset) + message.framing_suffix;
    content.push_back(std::move(text));
    out["content"] = std::move(content);
  }
  out["reasoning_content"] = message.thought;
  if (!message.name.empty())
    out["name"] = message.name;
  if (!message.tool_call_id.empty())
    out["tool_call_id"] = message.tool_call_id;
  auto calls = json::Value::array();
  for (const auto& call : message.tool_calls) {
    auto entry = json::Value::object();
    entry["id"] = call.id;
    entry["type"] = "function";
    entry["function"]["name"] = call.name;
    auto args = json::Value::object();
    for (const auto& arg : call.arguments)
      args[arg.name] =
          arg.is_string ? json::Value(arg.value) : json::parse(arg.value);
    entry["function"]["arguments"] = std::move(args);
    calls.push_back(std::move(entry));
  }
  out["tool_calls"] = std::move(calls);
  return out;
}

}  // namespace

std::optional<std::string> QwenChatTemplate::RenderJinja(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets, std::size_t* stable_prefix_bytes,
    std::vector<ContentSpan>* content_spans) {
  try {
    if (image_offsets)
      image_offsets->clear();
    if (content_spans)
      content_spans->clear();
    if (messages.empty())
      throw std::invalid_argument("No messages provided");
    auto inputs = json::Value::object();
    inputs["messages"] = json::Value::array();
    std::size_t image_count = 0;
    for (const auto& message : messages) {
      inputs["messages"].push_back(MessageJson(message));
      image_count += message.images.size();
    }
    inputs["tools"] = json::Value::array();
    for (const auto& tool : tools) {
      auto entry = json::Value::object();
      if (!tool.definition_json.empty()) {
        entry = json::parse(tool.definition_json);
      } else {
        entry["type"] = "function";
        entry["function"]["name"] = tool.name;
        entry["function"]["description"] = tool.description;
        entry["function"]["parameters"] = json::parse(tool.parameters_json);
      }
      inputs["tools"].push_back(std::move(entry));
    }
    inputs["add_generation_prompt"] = options.add_generation_prompt;
    inputs["enable_thinking"] = options.enable_thinking;
    inputs["preserve_thinking"] = options.preserve_thinking;
    inputs["add_vision_id"] = options.add_vision_id;
    inputs["reasoning_effort"] =
        options.reasoning_effort == QwenReasoningEffort::kLow      ? "low"
        : options.reasoning_effort == QwenReasoningEffort::kMedium ? "medium"
                                                                   : "xhigh";
    const auto parts =
        options.jinja->Render(inputs, options.kwargs, options.max_output_bytes);
    std::string output;
    std::vector<std::size_t> offsets;
    for (const auto& part : parts) {
      if (part.input && content_spans && !part.text.empty())
        content_spans->push_back({output.size(), part.text.size()});
      // Only template-authored image placeholders can consume image inputs.
      if (!part.input) {
        std::size_t pos = 0;
        while ((pos = part.text.find(kImagePad, pos)) != std::string::npos) {
          offsets.push_back(output.size() + pos);
          pos += kImagePad.size();
        }
      }
      output += part.text;
    }
    if (offsets.size() != image_count)
      throw std::invalid_argument(
          "chat template must render one native image placeholder per image");
    for (const auto offset : offsets) {
      if (offset < kVisionStart.size() ||
          std::string_view(output).substr(offset - kVisionStart.size(),
                                          kVisionStart.size()) !=
              kVisionStart ||
          std::string_view(output).substr(offset + kImagePad.size(),
                                          kVisionEnd.size()) != kVisionEnd)
        throw std::invalid_argument(
            "chat template image placeholder must use native vision framing");
    }
    if (image_offsets)
      *image_offsets = std::move(offsets);
    if (stable_prefix_bytes) {
      *stable_prefix_bytes = 0;
      // Some templates require strict turn alternation. A speculative probe
      // must never reject the real conversation when its next turn is unknown.
      try {
        // Compare actual renders: templates may rewrite prior reasoning on a
        // new turn.
        inputs["messages"].push_back(
            MessageJson(ChatMessage{ChatRole::kUser, ""}));
        std::string continued;
        for (const auto& part : options.jinja->Render(inputs, options.kwargs,
                                                      options.max_output_bytes))
          continued += part.text;
        *stable_prefix_bytes = static_cast<std::size_t>(
            std::mismatch(output.begin(), output.end(), continued.begin(),
                          continued.end())
                .first -
            output.begin());
      } catch (const std::exception&) {
        *stable_prefix_bytes = 0;
      }
    }
    return output;
  } catch (const std::exception& error) {
    if (error_msg)
      *error_msg = error.what();
    return std::nullopt;
  }
}

}  // namespace gufo::tokenization

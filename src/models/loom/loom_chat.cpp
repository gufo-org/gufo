#include "loom_chat.hpp"

#include <cstdio>
#include <exception>

namespace gufo::loom {

// ------------------------------------------------------- chat requests ---

const char* RoleName(ext::ChatRole role) {
  switch (role) {
    case ext::ChatRole::kSystem:
    case ext::ChatRole::kDeveloper:
      return "system";
    case ext::ChatRole::kUser:
      return "user";
    case ext::ChatRole::kAssistant:
      return "assistant";
    case ext::ChatRole::kTool:
      return "tool";
  }
  return "user";
}

json::Value ParseOr(const std::string& text, json::Value fallback) {
  try {
    return json::parse(text);
  } catch (const std::exception&) {
    return fallback;
  }
}

std::string Base64(std::span<const std::uint8_t> bytes) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t n = (bytes[i] << 16) |
                            (i + 1 < bytes.size() ? bytes[i + 1] << 8 : 0) |
                            (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out += kAlphabet[(n >> 18) & 63];
    out += kAlphabet[(n >> 12) & 63];
    out += i + 1 < bytes.size() ? kAlphabet[(n >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? kAlphabet[n & 63] : '=';
  }
  return out;
}

std::string_view MediaType(std::span<const std::uint8_t> bytes) {
  if (bytes.size() >= 8 && bytes[0] == 0x89 && bytes[1] == 'P')
    return "image/png";
  if (bytes.size() >= 12 && bytes[0] == 'R' && bytes[8] == 'W')
    return "image/webp";
  return "image/jpeg";
}

// Text and images in order, as OpenAI content parts with data URLs.
json::Value ContentParts(const ext::ChatMessage& message) {
  json::Value parts = json::Value::array();
  std::size_t at = 0;
  const auto text = [&](std::size_t end) {
    if (end > at) {
      json::Value part = json::Value::object();
      part["type"] = "text";
      part["text"] = message.content.substr(at, end - at);
      parts.push_back(std::move(part));
    }
    at = end;
  };
  for (const auto& image : message.images) {
    text(std::min(image.offset, message.content.size()));
    const std::span<const std::uint8_t> bytes(*image.bytes);
    json::Value url = json::Value::object();
    url["url"] =
        "data:" + std::string(MediaType(bytes)) + ";base64," + Base64(bytes);
    if (image.detail == ext::ChatMessage::ImageDetail::kLow)
      url["detail"] = "low";
    json::Value part = json::Value::object();
    part["type"] = "image_url";
    part["image_url"] = std::move(url);
    parts.push_back(std::move(part));
  }
  text(message.content.size());
  return parts;
}

// The OpenAI chat body the package's chat program renders.
std::string ChatBody(const std::string& model, const ext::ChatRequest& request,
                     bool with_effort) {
  json::Value body = json::Value::object();
  body["model"] = model;
  json::Value messages = json::Value::array();
  for (const auto& message : request.messages) {
    json::Value out = json::Value::object();
    out["role"] = RoleName(message.role);
    out["content"] = message.images.empty() ? json::Value(message.content)
                                            : ContentParts(message);
    if (message.role == ext::ChatRole::kAssistant) {
      if (!message.thought.empty())
        out["reasoning_content"] = message.thought;
      if (!message.tool_calls.empty()) {
        json::Value calls = json::Value::array();
        for (const auto& call : message.tool_calls) {
          json::Value arguments = json::Value::object();
          for (const auto& argument : call.arguments)
            arguments[argument.name] =
                argument.is_string
                    ? json::Value(argument.value)
                    : ParseOr(argument.value, json::Value(argument.value));
          json::Value function = json::Value::object();
          function["name"] = call.name;
          function["arguments"] = arguments.dump();
          json::Value entry = json::Value::object();
          entry["id"] = call.id;
          entry["type"] = "function";
          entry["function"] = std::move(function);
          calls.push_back(std::move(entry));
        }
        out["tool_calls"] = std::move(calls);
      }
    }
    if (message.role == ext::ChatRole::kTool && !message.tool_call_id.empty())
      out["tool_call_id"] = message.tool_call_id;
    messages.push_back(std::move(out));
  }
  body["messages"] = std::move(messages);
  if (!request.tools.empty()) {
    json::Value tools = json::Value::array();
    for (const auto& tool : request.tools) {
      auto definition = ParseOr(tool.definition_json, json::Value());
      if (!definition.is_object()) {
        json::Value function = json::Value::object();
        function["name"] = tool.name;
        if (!tool.description.empty())
          function["description"] = tool.description;
        function["parameters"] =
            ParseOr(tool.parameters_json, json::Value::object());
        definition = json::Value::object();
        definition["type"] = "function";
        definition["function"] = std::move(function);
      }
      // "strict" is Gufo's to enforce (the host grammar over the returned
      // logits); Loom's chat parser (hrx-system v0.1.0) refuses strict=true
      // outright ("strict constrained tool sampling is not implemented"), so
      // the rendered prompt carries it off.
      if (definition.is_object() && definition.contains("function") &&
          definition["function"].is_object() &&
          definition["function"].contains("strict"))
        definition["function"]["strict"] = false;
      tools.push_back(std::move(definition));
    }
    body["tools"] = std::move(tools);
    if (request.tool_choice == ext::ChatRequest::ToolChoice::kNone) {
      body["tool_choice"] = "none";
    }
    // Required and named tool calls are enforced on the host (the grammar over
    // the returned logits); Loom's chat parser (hrx-system v0.1.0) accepts only
    // "auto" and "none" for tool_choice, so the prompt is rendered with the
    // default (auto).
  }
  if (request.reasoning.enabled.has_value()) {
    json::Value kwargs = json::Value::object();
    kwargs["enable_thinking"] = *request.reasoning.enabled;
    body["chat_template_kwargs"] = std::move(kwargs);
  }
  if (with_effort && request.reasoning.effort.has_value() &&
      request.reasoning.enabled.value_or(true)) {
    static constexpr const char* kEfforts[] = {"minimal", "low",   "medium",
                                               "high",    "xhigh", "max"};
    body["reasoning_effort"] =
        kEfforts[static_cast<std::size_t>(*request.reasoning.effort)];
  }
  return body.dump();
}

// A request body that only carries sampling, for row binding.
std::string SamplingBody(const std::string& model,
                         const sampling::SamplingConfig& config, bool logits) {
  json::Value body = json::Value::object();
  body["model"] = model;
  json::Value message = json::Value::object();
  message["role"] = "user";
  message["content"] = ".";
  json::Value messages = json::Value::array();
  messages.push_back(std::move(message));
  body["messages"] = std::move(messages);
  body["temperature"] = static_cast<double>(config.temperature);
  body["top_p"] = static_cast<double>(config.top_p);
  if (config.top_k > 0)
    body["top_k"] = config.top_k;
  if (config.seed >= 0)
    body["seed"] = config.seed;
  if (logits)
    body["logits"] = true;
  return body.dump();
}

// Tool calls as Gufo's JSON dialect reads them.
std::string ToolMarkup(std::string_view calls_json) {
  if (calls_json.empty())
    return {};
  std::string markup;
  try {
    const auto calls = json::parse(calls_json);
    for (const auto& call : calls.items()) {
      const auto* function = call.find("function");
      if (function == nullptr)
        continue;
      json::Value block = json::Value::object();
      block["name"] = function->member_str("name");
      const auto arguments = function->member_str("arguments");
      block["arguments"] = ParseOr(arguments, json::Value(arguments));
      markup += "<tool_call>\n" + block.dump() + "\n</tool_call>";
    }
  } catch (const std::exception& error) {
    // Loom writes this JSON itself; a parse failure is a bug, not input.
    std::fprintf(stderr, "loom: dropped unreadable tool calls: %s\n",
                 error.what());
  }
  return markup;
}

}  // namespace gufo::loom

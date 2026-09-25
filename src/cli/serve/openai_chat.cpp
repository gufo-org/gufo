#include "src/cli/serve/openai_chat.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

struct ParsedChatRequest {
  ChatRequest chat;
  std::string model;
  std::size_t max_tokens{0};
  sampling::SamplingConfig sampling;
  bool stream{false};
  bool include_usage{false};
};

struct ParsedToolCall {
  std::string id;
  std::string name;
  std::vector<tokenization::ChatMessage::ToolArgument> arguments;
};

struct ParsedGeneration {
  std::string text;
  std::string reasoning_content;
  std::vector<ParsedToolCall> tool_calls;
  bool hide_tool_markup{false};
};

constexpr std::array<std::string_view, 7> kToolMarkers{
    "<tool_call>",          "<｜DSML｜tool_calls｜>", "<｜DSML｜tool_calls>",
    "<DSML｜tool_calls｜>", "<DSML｜tool_calls>",     "<tool_calls｜>",
    "<tool_calls>",
};

long long Now() {
  return static_cast<long long>(std::time(nullptr));
}

std::string RandomId(std::string_view prefix) {
  static constexpr std::string_view kCharacters =
      "abcdefghijklmnopqrstuvwxyz0123456789";
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<std::size_t> distribution{
      0, kCharacters.size() - 1};

  std::string result(prefix);
  result.reserve(prefix.size() + 20);
  for (int index = 0; index < 20; ++index) {
    result.push_back(kCharacters[distribution(generator)]);
  }
  return result;
}

HttpResponse Error(int status, const char* reason, std::string message,
                   const char* code) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = code;
  response["error"] = std::move(error);
  return {.status = status, .reason = reason, .body = response.dump()};
}

const char* StatusReason(int status) noexcept {
  switch (status) {
    case 408:
      return "Request Timeout";
    case 429:
      return "Too Many Requests";
    case 503:
      return "Service Unavailable";
    case 502:
      return "Bad Gateway";
    default:
      return "Internal Server Error";
  }
}

HttpResponse GenerationError(const TextGenerationError& exception) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = exception.what();
  error["type"] = "server_error";
  error["code"] = exception.stable_code();
  response["error"] = std::move(error);
  HttpResponse output{
      .status = exception.http_status(),
      .reason = StatusReason(exception.http_status()),
      .body = response.dump(),
      .headers = {},
      .streaming_body = {},
  };
  if (exception.retryable()) {
    output.headers.emplace_back("Retry-After", "1");
  }
  return output;
}

bool IsKnownRole(std::string_view role) {
  return role == "system" || role == "developer" || role == "user" ||
         role == "assistant" || role == "tool";
}

tokenization::ChatRole ParseRole(std::string_view role) {
  if (role == "system") {
    return tokenization::ChatRole::kSystem;
  }
  if (role == "developer") {
    return tokenization::ChatRole::kDeveloper;
  }
  if (role == "assistant") {
    return tokenization::ChatRole::kAssistant;
  }
  if (role == "tool") {
    return tokenization::ChatRole::kTool;
  }
  return tokenization::ChatRole::kUser;
}

bool ParseContent(const json::Value* content,
                  tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error) {
  auto* output = &message->content;
  if (content == nullptr || content->is_null()) {
    return true;
  }
  if (content->is_string()) {
    *output = content->get_str();
    return true;
  }
  if (!content->is_array()) {
    *error = "message content must be a string, null, or content-part array";
    return false;
  }

  for (const auto& part : content->items()) {
    if (!part.is_object()) {
      *error = "message content parts must be objects";
      return false;
    }
    const std::string type = part.member_str("type", "text");
    if (type == "image_url") {
      const auto* image = part.find("image_url");
      const auto* url =
          image != nullptr && image->is_object() ? image->find("url") : nullptr;
      if (message->role != tokenization::ChatRole::kUser || url == nullptr ||
          !url->is_string() || message->images.size() >= 16) {
        *error =
            "image_url requires a user message and a string URL (at most 16 "
            "images)";
        return false;
      }
      // Resolution is model-owned; accept only the automatic policy rather
      // than silently ignoring a requested low/high preprocessing policy.
      const auto* detail = image->find("detail");
      if (detail != nullptr &&
          (!detail->is_string() || detail->get_str() != "auto")) {
        *error = "image_url.detail supports only 'auto'";
        return false;
      }
      try {
        message->images.push_back(
            {output->size(), std::make_shared<const std::vector<std::uint8_t>>(
                                 core::ReadImageUrl(url->get_str(), budget))});
      } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
      }
      continue;
    }
    if (type != "text" && type != "input_text") {
      *error = "message content parts must use text or image_url";
      return false;
    }
    const json::Value* text = part.find("text");
    if (text == nullptr || !text->is_string()) {
      *error = "text message content parts require a string 'text'";
      return false;
    }
    output->append(text->get_str());
  }
  return true;
}

bool ParseArguments(std::string_view arguments,
                    std::vector<tokenization::ChatMessage::ToolArgument>* out,
                    std::string* error) {
  json::Value parsed;
  try {
    parsed = json::parse(arguments);
  } catch (const std::exception& exception) {
    *error =
        std::string("tool arguments are not valid JSON: ") + exception.what();
    return false;
  }
  if (!parsed.is_object()) {
    *error = "tool arguments must encode a JSON object";
    return false;
  }
  for (const auto& [name, value] : parsed.members()) {
    out->push_back({
        .name = name,
        .value = value.is_string() ? value.get_str() : value.dump(),
        .is_string = value.is_string(),
    });
  }
  return true;
}

bool ParseMessage(const json::Value& value, tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error) {
  if (!value.is_object()) {
    *error = "each message must be an object";
    return false;
  }
  const std::string role = value.member_str("role");
  if (!IsKnownRole(role)) {
    *error = "message role must be system, developer, user, assistant, or tool";
    return false;
  }
  message->role = ParseRole(role);
  message->name = value.member_str("name");
  message->tool_call_id = value.member_str("tool_call_id");
  if (!ParseContent(value.find("content"), message, budget, error)) {
    return false;
  }
  if (const json::Value* reasoning = value.find("reasoning_content");
      reasoning != nullptr && !reasoning->is_null()) {
    if (message->role != tokenization::ChatRole::kAssistant ||
        !reasoning->is_string()) {
      *error =
          "'reasoning_content' is only valid as a string on assistant "
          "messages";
      return false;
    }
    message->thought = reasoning->get_str();
  }

  const json::Value* tool_calls = value.find("tool_calls");
  if (tool_calls == nullptr) {
    return true;
  }
  if (message->role != tokenization::ChatRole::kAssistant ||
      !tool_calls->is_array()) {
    *error = "'tool_calls' is only valid as an array on assistant messages";
    return false;
  }
  for (const auto& item : tool_calls->items()) {
    if (!item.is_object() ||
        item.member_str("type", "function") != "function") {
      *error = "only function tool calls are supported";
      return false;
    }
    const json::Value* function = item.find("function");
    if (function == nullptr || !function->is_object()) {
      *error = "assistant tool calls require a function object";
      return false;
    }
    tokenization::ChatMessage::ToolCall call;
    call.id = item.member_str("id");
    call.name = function->member_str("name");
    const std::string arguments = function->member_str("arguments");
    if (call.name.empty() || arguments.empty() ||
        !ParseArguments(arguments, &call.arguments, error)) {
      if (error->empty()) {
        *error = "assistant tool calls require a name and JSON arguments";
      }
      return false;
    }
    message->tool_calls.push_back(std::move(call));
  }
  return true;
}

bool ParseTools(const json::Value* tools,
                std::vector<tokenization::ChatTool>* output,
                std::string* error) {
  if (tools == nullptr || tools->is_null()) {
    return true;
  }
  if (!tools->is_array()) {
    *error = "'tools' must be an array";
    return false;
  }
  for (const auto& item : tools->items()) {
    if (!item.is_object()) {
      *error = "'tools' entries must be objects";
      return false;
    }
    if (item.member_str("type") != "function") {
      *error = "only function tools are supported";
      return false;
    }

    const json::Value* function = item.find("function");
    if (function != nullptr && !function->is_object()) {
      *error = "'function' must be an object";
      return false;
    }
    const json::Value* src = function != nullptr ? function : &item;
    // OpenAI uses "parameters"; some agent clients send parametersJsonSchema.
    const json::Value* params_src = src->find("parameters");
    if (params_src == nullptr || params_src->is_null()) {
      params_src = src->find("parametersJsonSchema");
    }

    tokenization::ChatTool tool;
    tool.name = src->member_str("name");
    tool.description = src->member_str("description");
    if (tool.name.empty()) {
      *error = "function tools require a nonempty name";
      return false;
    }
    if (params_src != nullptr && !params_src->is_null() &&
        !params_src->is_object()) {
      *error = "function tools require an object parameters schema";
      return false;
    }
    // Preserve nested definitions and their field order. For flat tools, move
    // the complete function body (including strict) under "function".
    json::Value function_obj = json::Value::object();
    for (const auto& [key, value] : src->members()) {
      if (key == "parametersJsonSchema" ||
          (function == nullptr && key == "type")) {
        continue;
      }
      function_obj.append_member(key, value);
    }
    function_obj["parameters"] =
        params_src != nullptr && params_src->is_object()
            ? *params_src
            : json::Value::object();
    tool.parameters_json = function_obj.find("parameters")->dump();
    json::Value definition = function != nullptr ? item : json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function_obj);
    tool.definition_json = definition.dump();
    output->push_back(std::move(tool));
  }
  return true;
}

bool ParseToolChoice(const json::Value* value, ParsedChatRequest* request,
                     std::string* error) {
  if (value == nullptr || value->is_null()) {
    return true;
  }
  if (value->is_string()) {
    const std::string choice = value->get_str();
    if (choice == "auto") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kAuto;
      return true;
    }
    if (choice == "none") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kNone;
      return true;
    }
    if (choice == "required") {
      request->chat.tool_choice = ChatRequest::ToolChoice::kRequired;
      return true;
    }
  }
  *error = "'tool_choice' must be auto, none, or required";
  return false;
}

std::optional<ReasoningEffort> ParseReasoningEffortName(
    std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

bool AssignReasoningEnabled(ReasoningOptions* options, bool enabled,
                            std::string* error) {
  if (options->enabled.has_value() && *options->enabled != enabled) {
    *error = "reasoning controls disagree about whether thinking is enabled";
    return false;
  }
  options->enabled = enabled;
  return true;
}

bool AssignReasoningEffort(ReasoningOptions* options, std::string_view value,
                           std::string* error, bool enable_thinking = true) {
  if (value == "off" || value == "none") {
    return AssignReasoningEnabled(options, false, error);
  }
  const auto effort = ParseReasoningEffortName(value);
  if (!effort.has_value()) {
    *error =
        "reasoning_effort must be off, minimal, low, medium, high, xhigh, or "
        "max";
    return false;
  }
  if (options->effort.has_value() && options->effort != effort) {
    *error = "top-level and chat_template_kwargs reasoning_effort disagree";
    return false;
  }
  options->effort = effort;
  return !enable_thinking || AssignReasoningEnabled(options, true, error);
}

bool ParseReasoningOptions(const json::Value& body, ReasoningOptions* options,
                           std::string* error) {
  if (const json::Value* thinking = body.find("thinking")) {
    if (!thinking->is_object()) {
      *error = "'thinking' must be an object";
      return false;
    }
    const json::Value* type = thinking->find("type");
    if (type == nullptr || !type->is_string()) {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    const std::string value = type->get_str();
    if (value != "enabled" && value != "disabled") {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    if (!AssignReasoningEnabled(options, value == "enabled", error)) {
      return false;
    }
  }

  if (const json::Value* effort = body.find("reasoning_effort");
      effort != nullptr && !effort->is_null()) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error)) {
      if (error->empty()) {
        *error = "'reasoning_effort' must be a string";
      }
      return false;
    }
  }

  const json::Value* kwargs = body.find("chat_template_kwargs");
  if (kwargs == nullptr) {
    return true;
  }
  if (!kwargs->is_object()) {
    *error = "'chat_template_kwargs' must be an object";
    return false;
  }
  if (const json::Value* enabled = kwargs->find("enable_thinking")) {
    if (!enabled->is_bool() ||
        !AssignReasoningEnabled(options, enabled->as_bool(), error)) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.enable_thinking' must be a boolean";
      }
      return false;
    }
  }
  if (const json::Value* mode = kwargs->find("thinking_mode")) {
    if (!mode->is_string()) {
      *error = "'chat_template_kwargs.thinking_mode' must be a string";
      return false;
    }
    const std::string value = mode->get_str();
    if (value != "auto") {
      const bool enabled = value == "thinking" || value == "on";
      if ((!enabled && value != "chat" && value != "off" && value != "none") ||
          !AssignReasoningEnabled(options, enabled, error)) {
        if (error->empty()) {
          *error =
              "'chat_template_kwargs.thinking_mode' must be auto, thinking, "
              "or chat";
        }
        return false;
      }
    }
  }
  if (const json::Value* effort = kwargs->find("reasoning_effort")) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error,
                               options->enabled.value_or(true))) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.reasoning_effort' must be a string";
      }
      return false;
    }
  }
  if (const json::Value* preserve = kwargs->find("preserve_thinking")) {
    if (!preserve->is_bool()) {
      *error = "'chat_template_kwargs.preserve_thinking' must be a boolean";
      return false;
    }
    options->preserve_thinking = preserve->as_bool();
  }
  return true;
}

std::optional<HttpResponse> ParseRequest(const HttpRequest& request,
                                         TextGenerationBackend& backend,
                                         ParsedChatRequest* output) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& exception) {
    return Error(400, "Bad Request", exception.what(), "parse_error");
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }

  output->model = body.member_str("model");
  if (output->model.empty()) {
    return Error(400, "Bad Request", "'model' is required", "missing_model");
  }
  if (output->model != backend.model_id()) {
    return Error(404, "Not Found",
                 "model '" + output->model + "' is not served by this process",
                 "model_not_found");
  }

  output->chat.client_id = request.client_id;
  if (const auto* cache_prompt = body.find("cache_prompt")) {
    if (!cache_prompt->is_bool()) {
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    }
    output->chat.cache_prompt = cache_prompt->as_bool();
  }
  if (const auto error =
          ParseStopSequences(body.find("stop"), StopSequenceFormat::kOpenAi,
                             &output->chat.stop_sequences)) {
    return Error(400, "Bad Request", *error, "invalid_stop");
  }

  const json::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() || messages->empty()) {
    return Error(400, "Bad Request", "'messages' must be a non-empty array",
                 "missing_messages");
  }
  core::ImageReadBudget image_budget;
  for (const auto& item : messages->items()) {
    tokenization::ChatMessage message;
    std::string parse_error;
    if (!ParseMessage(item, &message, image_budget, &parse_error)) {
      return Error(400, "Bad Request", std::move(parse_error),
                   "invalid_messages");
    }
    output->chat.messages.push_back(std::move(message));
  }

  std::string parse_error;
  if (!ParseTools(body.find("tools"), &output->chat.tools, &parse_error) ||
      !ParseToolChoice(body.find("tool_choice"), output, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error), "invalid_tools");
  }
  if (output->chat.tool_choice == ChatRequest::ToolChoice::kRequired &&
      output->chat.tools.empty()) {
    return Error(400, "Bad Request",
                 "'tool_choice' cannot be required without tools",
                 "invalid_tool_choice");
  }

  if (!ParseReasoningOptions(body, &output->chat.reasoning, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error),
                 "invalid_reasoning");
  }
  if (const auto* kwargs = body.find("chat_template_kwargs")) {
    if (const auto* vision_id = kwargs->find("add_vision_id")) {
      if (!vision_id->is_bool())
        return Error(400, "Bad Request",
                     "'chat_template_kwargs.add_vision_id' must be a boolean",
                     "invalid_template_options");
      output->chat.add_vision_id = vision_id->as_bool();
    }
  }
  const ReasoningOptions defaults = backend.reasoning_defaults();
  if (!output->chat.reasoning.enabled.has_value()) {
    output->chat.reasoning.enabled = defaults.enabled;
  }
  if (!output->chat.reasoning.effort.has_value()) {
    output->chat.reasoning.effort = defaults.effort;
  }
  if (!output->chat.reasoning.preserve_thinking.has_value()) {
    output->chat.reasoning.preserve_thinking = defaults.preserve_thinking;
  }

  if (const json::Value* stream = body.find("stream");
      stream != nullptr && !stream->is_null()) {
    if (!stream->is_bool()) {
      return Error(400, "Bad Request", "'stream' must be a boolean",
                   "invalid_stream");
    }
    output->stream = stream->as_bool();
  }
  if (const json::Value* options = body.find("stream_options");
      options != nullptr && !options->is_null()) {
    if (!options->is_object()) {
      return Error(400, "Bad Request", "'stream_options' must be an object",
                   "invalid_stream_options");
    }
    if (const json::Value* include_usage = options->find("include_usage")) {
      if (!include_usage->is_bool()) {
        return Error(400, "Bad Request",
                     "'stream_options.include_usage' must be a boolean",
                     "invalid_stream_options");
      }
      output->include_usage = include_usage->as_bool();
    }
  }

  const json::Value* max_tokens = body.find("max_completion_tokens");
  if (max_tokens == nullptr || max_tokens->is_null()) {
    max_tokens = body.find("max_tokens");
  }
  if (max_tokens != nullptr && !max_tokens->is_null()) {
    const double value =
        max_tokens->is_number() ? max_tokens->as_double() : 0.0;
    if (!max_tokens->is_number() || !std::isfinite(value) ||
        std::floor(value) != value || value < 1.0 ||
        value >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
      return Error(400, "Bad Request",
                   "'max_tokens' must be a positive integer",
                   "invalid_max_tokens");
    }
    output->max_tokens = max_tokens->as_size();
  }

  sampling::SamplingConfig parsed_sampling;
  if (const auto sampling_error =
          ParseSamplingConfig(body, output->sampling, &parsed_sampling)) {
    return Error(400, "Bad Request", sampling_error->message,
                 sampling_error->code.c_str());
  }
  if (parsed_sampling.temperature > 2.0F) {
    return Error(400, "Bad Request", "'temperature' must be between 0 and 2",
                 "invalid_temperature");
  }
  output->sampling = parsed_sampling;

  if (const json::Value* choices = body.find("n");
      choices != nullptr && !choices->is_null() &&
      (!choices->is_number() || choices->as_double() != 1.0)) {
    return Error(400, "Bad Request", "only n=1 is supported", "unsupported_n");
  }
  for (const std::string_view unsupported :
       {"logprobs", "top_logprobs", "response_format", "modalities", "audio"}) {
    if (const auto* value = body.find(std::string(unsupported));
        value != nullptr && !value->is_null()) {
      if ((unsupported == "logprobs" && value->is_bool() &&
           !value->as_bool()) ||
          (unsupported == "response_format" && value->is_object() &&
           value->size() == 1 && value->member_str("type") == "text") ||
          (unsupported == "modalities" && value->is_array() &&
           value->size() == 1 && value->items().front().is_string() &&
           value->items().front().str() == "text"))
        continue;
      return Error(
          400, "Bad Request",
          "request field '" + std::string(unsupported) + "' is not implemented",
          "unsupported_field");
    }
  }
  return std::nullopt;
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

std::size_t EarliestMarker(std::string_view text,
                           std::string_view* marker = nullptr) {
  std::size_t earliest = std::string_view::npos;
  for (const auto candidate : kToolMarkers) {
    const std::size_t position = text.find(candidate);
    if (position < earliest) {
      earliest = position;
      if (marker != nullptr) {
        *marker = candidate;
      }
    }
  }
  return earliest;
}

std::size_t HeldMarkerPrefix(std::string_view text) {
  std::size_t maximum_marker = 0;
  for (const auto marker : kToolMarkers) {
    maximum_marker = std::max(maximum_marker, marker.size());
  }
  const std::size_t maximum =
      std::min(text.size(), maximum_marker > 0 ? maximum_marker - 1 : 0);
  for (std::size_t length = maximum; length > 0; --length) {
    const std::string_view suffix = text.substr(text.size() - length);
    if (std::ranges::any_of(kToolMarkers, [&](std::string_view marker) {
          return marker.starts_with(suffix);
        })) {
      return length;
    }
  }
  return 0;
}

std::string ArgumentsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> arguments) {
  json::Value object = json::Value::object();
  for (const auto& argument : arguments) {
    if (argument.is_string) {
      object[argument.name] = argument.value;
      continue;
    }
    try {
      object[argument.name] = json::parse(argument.value);
    } catch (...) {
      object[argument.name] = argument.value;
    }
  }
  return object.dump();
}

// Qwen's XML-like arguments carry no type marker. The advertised schema is
// needed to distinguish a string such as 42 from the JSON number 42.
bool SchemaAccepts(const json::Value& schema, const json::Value& value) {
  const auto matches = [&](std::string_view type) {
    return (type == "string" && value.is_string()) ||
           (type == "number" && value.is_number()) ||
           (type == "integer" && value.is_number() &&
            std::floor(value.as_double()) == value.as_double()) ||
           (type == "boolean" && value.is_bool()) ||
           (type == "null" && value.is_null()) ||
           (type == "array" && value.is_array()) ||
           (type == "object" && value.is_object());
  };
  if (const auto* type = schema.find("type")) {
    if (type->is_string())
      return matches(type->get_str());
    if (type->is_array())
      return std::ranges::any_of(type->items(), [&](const auto& item) {
        return item.is_string() && matches(item.get_str());
      });
    return false;
  }
  for (const auto* name : {"anyOf", "oneOf"}) {
    if (const auto* choices = schema.find(name); choices && choices->is_array())
      return std::ranges::any_of(choices->items(), [&](const auto& item) {
        return SchemaAccepts(item, value);
      });
  }
  return true;
}

void ParseQwenCalls(std::string_view text,
                    std::span<const tokenization::ChatTool> tools,
                    std::vector<ParsedToolCall>* calls) {
  constexpr std::string_view start = "<tool_call>";
  constexpr std::string_view end = "</tool_call>";
  std::size_t cursor = 0;
  while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
    const auto begin = cursor + start.size();
    cursor = begin;  // A malformed call may be followed by a valid call.
    auto body = text.substr(begin);
    const auto consume = [&](std::string_view tag) {
      body = Trim(body);
      if (!body.starts_with(tag))
        return false;
      body.remove_prefix(tag.size());
      return true;
    };
    body = Trim(body);
    ParsedToolCall call;
    bool complete = false;
    if (consume("<function=")) {
      const auto name_end = body.find('>');
      if (name_end == std::string_view::npos)
        continue;
      call.name = std::string(Trim(body.substr(0, name_end)));
      body.remove_prefix(name_end + 1);
      std::optional<json::Value> schema;
      for (const auto& tool : tools) {
        if (tool.name == call.name) {
          schema = TryParseJson(tool.parameters_json);
          break;
        }
      }
      if (call.name.empty() || (!tools.empty() && !schema))
        continue;
      bool valid = true;
      while (consume("<parameter=")) {
        const auto name_end = body.find('>');
        if (name_end == std::string_view::npos) {
          valid = false;
          break;
        }
        const std::string name(Trim(body.substr(0, name_end)));
        body.remove_prefix(name_end + 1);
        // Outer closing tags inside a parameter are data, not structure.
        const auto close = body.find("</parameter>");
        if (close == std::string_view::npos || body.find(start) < close ||
            name.empty() ||
            std::ranges::any_of(call.arguments, [&](const auto& arg) {
              return arg.name == name;
            })) {
          valid = false;
          break;
        }
        const auto raw = Trim(body.substr(0, close));
        const auto parsed = TryParseJson(raw);
        const auto* properties = schema ? schema->find("properties") : nullptr;
        const auto* property = properties ? properties->find(name) : nullptr;
        const bool string_allowed =
            !property ||
            SchemaAccepts(*property, json::Value(std::string(raw)));
        // Prefer text if the schema permits it; parsing ambiguous scalars
        // as JSON would silently change a caller's declared string type.
        const bool is_string = string_allowed;
        if (!is_string && (!parsed || !SchemaAccepts(*property, *parsed))) {
          valid = false;
          break;
        }
        call.arguments.push_back(
            {.name = name, .value = std::string(raw), .is_string = is_string});
        body.remove_prefix(close + std::string_view{"</parameter>"}.size());
      }
      complete = valid && consume("</function>") && consume(end);
    } else {
      // JSON calls already encode their argument types. Try closing markers
      // until the preceding payload is complete JSON (a marker in a quoted
      // string cannot terminate the call).
      std::size_t close = 0;
      while ((close = body.find(end, close)) != std::string_view::npos) {
        const auto parsed = TryParseJson(Trim(body.substr(0, close)));
        if (parsed && parsed->is_object()) {
          call.name = parsed->member_str("name");
          const auto* arguments = parsed->find("arguments");
          if (!call.name.empty() && arguments && arguments->is_object()) {
            for (const auto& [name, value] : arguments->members())
              call.arguments.push_back(
                  {.name = name,
                   .value = value.is_string() ? value.get_str() : value.dump(),
                   .is_string = value.is_string()});
            complete = true;
            body.remove_prefix(close + end.size());
          }
          break;
        }
        close += end.size();
      }
    }
    if (complete && !tools.empty() &&
        std::ranges::none_of(
            tools, [&](const auto& tool) { return tool.name == call.name; }))
      complete = false;
    if (complete) {
      call.id = RandomId("call_");
      calls->push_back(std::move(call));
      cursor = text.size() - body.size();
    }
  }
}

std::optional<std::string> Attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const std::size_t start = tag.find(prefix);
  if (start == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = tag.find('"', value_start);
  if (end == std::string_view::npos) {
    return std::nullopt;
  }
  return std::string(tag.substr(value_start, end - value_start));
}

void ParseDsmlCalls(std::string_view text, std::vector<ParsedToolCall>* calls) {
  constexpr std::array<std::string_view, 4> kInvokeStarts{
      "<｜DSML｜invoke",
      "<DSML｜invoke",
      "<｜DS｜invoke",
      "<DS｜invoke",
  };
  constexpr std::array<std::string_view, 4> kInvokeEnds{
      "</｜DSML｜invoke>",
      "</DSML｜invoke>",
      "</｜DS｜invoke>",
      "</DS｜invoke>",
  };
  constexpr std::array<std::string_view, 4> kParameterStarts{
      "<｜DSML｜parameter",
      "<DSML｜parameter",
      "<｜DS｜parameter",
      "<DS｜parameter",
  };
  constexpr std::array<std::string_view, 4> kParameterEnds{
      "</｜DSML｜parameter>",
      "</DSML｜parameter>",
      "</｜DS｜parameter>",
      "</DS｜parameter>",
  };

  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t invoke_start = std::string_view::npos;
    std::size_t syntax = 0;
    for (std::size_t index = 0; index < kInvokeStarts.size(); ++index) {
      const std::size_t position = text.find(kInvokeStarts[index], cursor);
      if (position < invoke_start) {
        invoke_start = position;
        syntax = index;
      }
    }
    if (invoke_start == std::string_view::npos) {
      break;
    }
    const std::size_t tag_end = text.find('>', invoke_start);
    const std::size_t invoke_end = text.find(kInvokeEnds[syntax], tag_end);
    if (tag_end == std::string_view::npos ||
        invoke_end == std::string_view::npos) {
      break;
    }
    ParsedToolCall call;
    call.id = RandomId("call_");
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < invoke_end) {
      const std::size_t parameter_start =
          text.find(kParameterStarts[syntax], parameter_cursor);
      if (parameter_start == std::string_view::npos ||
          parameter_start >= invoke_end) {
        break;
      }
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      const std::size_t parameter_end =
          text.find(kParameterEnds[syntax], parameter_tag_end);
      if (parameter_tag_end == std::string_view::npos ||
          parameter_end == std::string_view::npos ||
          parameter_end > invoke_end) {
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      if (parameter_name.has_value()) {
        call.arguments.push_back({
            .name = *parameter_name,
            .value = std::string(Trim(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))),
            .is_string = string_value.value_or("true") != "false",
        });
      }
      parameter_cursor = parameter_end + kParameterEnds[syntax].size();
    }
    if (!call.name.empty()) {
      calls->push_back(std::move(call));
    }
    cursor = invoke_end + kInvokeEnds[syntax].size();
  }
}

ParsedGeneration ParseGeneration(
    std::string_view raw,
    TextGenerationBackend::InitialOutputState initial_output_state,
    std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required) {
  ParsedGeneration parsed;
  std::string_view content = raw;

  constexpr std::string_view kThinkStart = "<think>";
  constexpr std::string_view kThinkEnd = "</think>";
  if (initial_output_state ==
      TextGenerationBackend::InitialOutputState::kReasoning) {
    if (content.starts_with(kThinkStart)) {
      content.remove_prefix(kThinkStart.size());
    }
    std::size_t think_end = content.find(kThinkEnd);
    if (EarliestMarker(content) < think_end)
      think_end = std::string_view::npos;
    if (think_end == std::string_view::npos) {
      const auto marker = EarliestMarker(content);
      parsed.reasoning_content = std::string(Trim(content.substr(0, marker)));
      if (marker == std::string_view::npos) {
        if (enforce_required && choice == ChatRequest::ToolChoice::kRequired)
          throw TextGenerationError(
              TextGenerationErrorCode::kToolChoiceUnsatisfied,
              "model did not produce a declared tool call");
        return parsed;
      }
      parsed.text = std::string(content.substr(marker));
    } else {
      parsed.reasoning_content =
          std::string(Trim(content.substr(0, think_end)));
      content.remove_prefix(think_end + kThinkEnd.size());
      while (!content.empty() &&
             (content.front() == '\n' || content.front() == '\r')) {
        content.remove_prefix(1);
      }
      parsed.text = std::string(content);
    }
  } else {
    const std::size_t think_start = content.find(kThinkStart);
    if (think_start != std::string_view::npos) {
      const std::size_t think_content_start = think_start + kThinkStart.size();
      std::size_t think_end = content.find(kThinkEnd, think_content_start);
      const auto marker = EarliestMarker(content.substr(think_content_start));
      if (marker != std::string_view::npos &&
          think_content_start + marker < think_end)
        think_end = std::string_view::npos;
      if (think_end != std::string_view::npos) {
        parsed.reasoning_content = std::string(Trim(content.substr(
            think_content_start, think_end - think_content_start)));
        std::string_view remaining =
            content.substr(think_end + kThinkEnd.size());
        if (remaining.starts_with("\n")) {
          remaining.remove_prefix(1);
        }
        if (think_start > 0) {
          parsed.text = std::string(content.substr(0, think_start)) +
                        std::string(remaining);
        } else {
          parsed.text = std::string(remaining);
        }
      } else {
        const auto remaining = content.substr(think_content_start);
        const auto marker = EarliestMarker(remaining);
        parsed.reasoning_content =
            std::string(Trim(remaining.substr(0, marker)));
        parsed.text = std::string(content.substr(0, think_start));
        if (marker != std::string_view::npos)
          parsed.text += remaining.substr(marker);
      }
    } else {
      parsed.text = std::string(content);
    }
  }

  const std::size_t marker = EarliestMarker(parsed.text);
  if (choice != ChatRequest::ToolChoice::kNone && !tools.empty() &&
      marker != std::string_view::npos) {
    const std::string text_before_tools = parsed.text.substr(0, marker);
    const std::string text_from_tools = parsed.text.substr(marker);
    ParseQwenCalls(text_from_tools, tools, &parsed.tool_calls);
    ParseDsmlCalls(text_from_tools, &parsed.tool_calls);
    std::erase_if(parsed.tool_calls, [&](const auto& call) {
      return std::ranges::none_of(
          tools, [&](const auto& tool) { return tool.name == call.name; });
    });
    // An explicit stop can interrupt a call before its closing tags. Keep
    // complete calls, but do not expose an unfinished call as ordinary text.
    if (!parsed.tool_calls.empty() || !enforce_required) {
      parsed.text = text_before_tools;
      parsed.hide_tool_markup = true;
    }
  }
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired &&
      parsed.tool_calls.empty())
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not produce a declared tool call");
  return parsed;
}

const char* FinishReason(const TextGenerationBackend::Result& result,
                         bool has_tool_calls) {
  if (result.finish_reason ==
      TextGenerationBackend::FinishReason::kStopSequence)
    return "stop";
  if (has_tool_calls) {
    return "tool_calls";
  }
  if (result.finish_reason == TextGenerationBackend::FinishReason::kLength) {
    return "length";
  }
  return "stop";
}

json::Value Usage(const TextGenerationBackend::Result& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  json::Value prompt_details = json::Value::object();
  prompt_details["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_details"] = std::move(prompt_details);

  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double predicted_per_second =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  usage["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_per_second"] = prompt_per_second;
  usage["completion_tokens_per_second"] = predicted_per_second;
  usage["draft_tokens"] = result.draft_tokens;
  usage["draft_tokens_accepted"] = result.draft_accepted_tokens;

  json::Value metrics = json::Value::object();
  metrics["cache_hit"] = result.cache_hit;
  if (!result.cache_miss_reason.empty()) {
    metrics["cache_miss_reason"] = result.cache_miss_reason;
    metrics["cache_common_prefix_tokens"] = result.cache_common_prefix_tokens;
    metrics["cache_checkpoint_tokens"] = result.cache_checkpoint_tokens;
  }
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_queued_bytes"] = result.cache_disk_queued_bytes;
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
  metrics["cache_shared_prefix_snapshots"] =
      result.cache_shared_prefix_snapshots;
  metrics["cache_shared_prefix_bytes"] = result.cache_shared_prefix_bytes;
  metrics["cache_shared_prefix_ms"] = result.cache_shared_prefix_ms;
  metrics["prefill_tokens"] = result.prefill_tokens;
  metrics["prefill_chunks"] = result.prefill_chunks;
  metrics["active_decode_prefill_chunks"] = result.active_decode_prefill_chunks;
  metrics["max_prefill_chunk_tokens"] = result.max_prefill_chunk_tokens;
  metrics["queue_depth_at_submit"] = result.queue_depth_at_submit;
  metrics["client_queue_depth_at_submit"] = result.client_queue_depth_at_submit;
  metrics["resident_requests_at_admission"] =
      result.resident_requests_at_admission;
  metrics["requested_logical_concurrency"] =
      result.requested_logical_concurrency;
  metrics["physical_execution_width"] = result.physical_execution_width;
  metrics["queue_ms"] = result.queue_ms;
  metrics["prefill_ms"] = result.prefill_ms;
  metrics["decode_ms"] = result.decode_ms;
  metrics["ttft_ms"] = result.ttft_ms;
  metrics["mean_inter_token_ms"] = result.mean_inter_token_ms;
  metrics["max_inter_token_ms"] = result.max_inter_token_ms;
  metrics["execution_plan"] = result.execution_plan;
  usage["gufo"] = std::move(metrics);
  return usage;
}

json::Value ToolCallsJson(std::span<const ParsedToolCall> calls) {
  json::Value output = json::Value::array();
  for (const auto& call : calls) {
    json::Value item = json::Value::object();
    item["id"] = call.id;
    item["type"] = "function";
    json::Value function = json::Value::object();
    function["name"] = call.name;
    function["arguments"] = ArgumentsJson(call.arguments);
    item["function"] = std::move(function);
    output.push_back(std::move(item));
  }
  return output;
}

std::string Sse(const json::Value& value) {
  return "data: " + value.dump() + "\n\n";
}

json::Value BaseChunk(std::string_view id, long long created,
                      std::string_view model) {
  json::Value chunk = json::Value::object();
  chunk["id"] = std::string(id);
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = std::string(model);
  return chunk;
}

json::Value ChoiceChunk(std::string_view id, long long created,
                        std::string_view model, json::Value delta,
                        const char* finish_reason = nullptr) {
  json::Value chunk = BaseChunk(id, created, model);
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = std::move(delta);
  if (finish_reason == nullptr) {
    choice["finish_reason"] = json::Value();
  } else {
    choice["finish_reason"] = finish_reason;
  }
  choices.push_back(std::move(choice));
  chunk["choices"] = std::move(choices);
  return chunk;
}

class StreamingTextFilter {
public:
  using EmitCallback =
      std::function<bool(std::string_view piece, bool is_reasoning)>;

  StreamingTextFilter(
      TextGenerationBackend::InitialOutputState initial_output_state,
      EmitCallback emit_piece)
      : emit_piece_(std::move(emit_piece)) {
    if (initial_output_state ==
        TextGenerationBackend::InitialOutputState::kReasoning) {
      state_ = State::kThinking;
    } else if (initial_output_state ==
               TextGenerationBackend::InitialOutputState::kContent) {
      state_ = State::kContent;
    }
  }

  bool Push(std::string_view bytes, bool final = false) {
    const auto piece = decoder_.Push(bytes, final);
    raw_.append(piece);
    if (tool_mode_) {
      hidden_.append(piece);
      return true;
    }
    pending_.append(piece);

    if (state_ == State::kInitial) {
      constexpr std::string_view kThinkStart = "<think>";
      std::string_view view = pending_;
      while (!view.empty() &&
             std::isspace(static_cast<unsigned char>(view.front())) != 0) {
        view.remove_prefix(1);
      }
      if (view.empty()) {
        return true;
      }
      if (kThinkStart.starts_with(view)) {
        if (view == kThinkStart) {
          state_ = State::kThinking;
          pending_.clear();
        }
        return true;
      }
      state_ = State::kContent;
    }

    if (state_ == State::kThinking) {
      constexpr std::string_view kThinkEnd = "</think>";
      const std::size_t end_pos = pending_.find(kThinkEnd);
      const auto marker = EarliestMarker(pending_);
      if (marker < end_pos) {
        if (marker > 0 && !emit_piece_(pending_.substr(0, marker), true))
          return false;
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }
      if (end_pos != std::string::npos) {
        if (end_pos > 0 && !emit_piece_(pending_.substr(0, end_pos), true)) {
          return false;
        }
        std::string_view remaining = pending_;
        remaining.remove_prefix(end_pos + kThinkEnd.size());
        pending_ = std::string(remaining);
        state_ = State::kContent;
        trim_reasoning_separator_ = true;
      } else {
        std::size_t held = HeldMarkerPrefix(pending_);
        for (std::size_t len = std::min(pending_.size(), kThinkEnd.size() - 1);
             len > 0; --len) {
          if (kThinkEnd.starts_with(pending_.substr(pending_.size() - len))) {
            held = std::max(held, len);
            break;
          }
        }
        const std::size_t ready = pending_.size() - held;
        if (ready > 0 && !emit_piece_(pending_.substr(0, ready), true)) {
          return false;
        }
        pending_.erase(0, ready);
        return true;
      }
    }

    if (state_ == State::kContent) {
      if (trim_reasoning_separator_) {
        const auto first = pending_.find_first_not_of("\r\n");
        if (first == std::string::npos) {
          pending_.clear();
          return true;
        }
        pending_.erase(0, first);
        trim_reasoning_separator_ = false;
      }
      const std::size_t marker = EarliestMarker(pending_);
      if (marker != std::string::npos) {
        if (marker > 0 && !emit_piece_(pending_.substr(0, marker), false)) {
          return false;
        }
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }

      const std::size_t held = HeldMarkerPrefix(pending_);
      const std::size_t ready = pending_.size() - held;
      if (ready > 0 && !emit_piece_(pending_.substr(0, ready), false)) {
        return false;
      }
      pending_.erase(0, ready);
      return true;
    }

    return true;
  }

  bool Finish(bool hide_tool_markup) {
    if (!tool_mode_) {
      if (!pending_.empty()) {
        const bool is_reasoning = (state_ == State::kThinking);
        const bool emitted = emit_piece_(pending_, is_reasoning);
        pending_.clear();
        return emitted;
      }
      return true;
    }
    if (!hide_tool_markup && !hidden_.empty()) {
      return emit_piece_(hidden_, false);
    }
    return true;
  }

  [[nodiscard]] std::string_view raw() const noexcept { return raw_; }

private:
  enum class State : std::uint8_t {
    kInitial,
    kThinking,
    kContent,
  };

  EmitCallback emit_piece_;
  core::Utf8Decoder decoder_;
  std::string raw_;
  std::string pending_;
  std::string hidden_;
  State state_{State::kInitial};
  bool tool_mode_{false};
  bool trim_reasoning_separator_{false};
};

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Build the same output items for streaming and buffered responses.
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, HttpResponse::BodyWriter writer)
      : writer_(std::move(writer)) {
    response_ = json::Value::object();
    response_["id"] = RandomId("resp_");
    response_["object"] = "response";
    response_["created_at"] = Now();
    response_["model"] = std::move(model);
    response_["status"] = "in_progress";
    response_["error"] = json::Value();
    response_["incomplete_details"] = json::Value();
    response_["usage"] = json::Value();
    response_["output"] = json::Value::array();
    response_["store"] = false;
    response_["parallel_tool_calls"] = false;
    response_["tool_choice"] = "none";
    response_["tools"] = json::Value::array();
  }

  bool Begin() {
    return Lifecycle("response.created") && Lifecycle("response.in_progress");
  }

  bool Append(std::string_view text, bool reasoning) {
    if (text.empty())
      return true;
    if (!active_ || reasoning_ != reasoning) {
      if (!CloseItem("completed"))
        return false;
      reasoning_ = reasoning;
      active_ = true;
      item_ = json::Value::object();
      item_["id"] = RandomId(reasoning ? "rs_" : "msg_");
      item_["type"] = reasoning ? "reasoning" : "message";
      item_["status"] = "in_progress";
      item_[reasoning ? "summary" : "content"] = json::Value::array();
      if (!reasoning)
        item_["role"] = "assistant";
      auto added = IndexedEvent("response.output_item.added");
      added["item"] = item_;
      if (!Emit(std::move(added)))
        return false;
      text_.clear();
      auto part = PartEvent(reasoning ? "response.reasoning_summary_part.added"
                                      : "response.content_part.added");
      part["part"] = Part();
      if (!Emit(std::move(part)))
        return false;
    }
    text_.append(text);
    auto delta = PartEvent(reasoning ? "response.reasoning_summary_text.delta"
                                     : "response.output_text.delta");
    delta["delta"] = std::string(text);
    if (!reasoning)
      delta["logprobs"] = json::Value::array();
    return Emit(std::move(delta));
  }

  json::Value Complete(const TextGenerationBackend::Result& result) {
    const bool limited =
        result.finish_reason == TextGenerationBackend::FinishReason::kLength;
    CloseItem(limited ? "incomplete" : "completed");
    response_["status"] = limited ? "incomplete" : "completed";
    if (limited)
      response_["incomplete_details"]["reason"] = "max_output_tokens";
    auto usage = json::Value::object();
    usage["input_tokens"] = result.prompt_tokens;
    usage["input_tokens_details"]["cached_tokens"] =
        result.cached_prompt_tokens;
    usage["input_tokens_details"]["cache_write_tokens"] = result.prefill_tokens;
    usage["output_tokens"] = result.completion_tokens;
    usage["output_tokens_details"]["reasoning_tokens"] =
        result.reasoning_tokens;
    usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
    response_["usage"] = std::move(usage);
    response_["timings"] = GenerationTimings(result);
    Lifecycle(limited ? "response.incomplete" : "response.completed");
    return response_;
  }

  bool Fail(std::string_view message) {
    response_["status"] = "failed";
    // Responses defines a closed error-code enum. Keep the specific runtime
    // code in the request log, rather than emitting an invalid wire value.
    response_["error"]["code"] = "server_error";
    response_["error"]["message"] = std::string(message);
    return Lifecycle("response.failed");
  }

private:
  json::Value IndexedEvent(std::string_view type) const {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["output_index"] = output_index_;
    return event;
  }

  json::Value PartEvent(std::string_view type) const {
    auto event = IndexedEvent(type);
    event["item_id"] = item_.member_str("id");
    event[reasoning_ ? "summary_index" : "content_index"] = 0;
    return event;
  }

  json::Value Part() const {
    auto part = json::Value::object();
    part["type"] = reasoning_ ? "summary_text" : "output_text";
    part["text"] = text_;
    if (!reasoning_) {
      part["annotations"] = json::Value::array();
      part["logprobs"] = json::Value::array();
    }
    return part;
  }

  bool CloseItem(const char* status) {
    if (!active_)
      return connected_;
    auto done = PartEvent(reasoning_ ? "response.reasoning_summary_text.done"
                                     : "response.output_text.done");
    done["text"] = text_;
    if (!reasoning_)
      done["logprobs"] = json::Value::array();
    if (!Emit(std::move(done)))
      return false;
    auto part = Part();
    auto part_done =
        PartEvent(reasoning_ ? "response.reasoning_summary_part.done"
                             : "response.content_part.done");
    part_done["part"] = part;
    if (!Emit(std::move(part_done)))
      return false;
    item_[reasoning_ ? "summary" : "content"].push_back(std::move(part));
    item_["status"] = status;
    auto item_done = IndexedEvent("response.output_item.done");
    item_done["item"] = item_;
    response_["output"].push_back(item_);
    active_ = false;
    ++output_index_;
    return Emit(std::move(item_done));
  }

  bool Lifecycle(std::string_view type) {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["response"] = response_;
    return Emit(std::move(event));
  }

  bool Emit(json::Value event) {
    if (!writer_)
      return true;
    if (!connected_)
      return false;
    event["sequence_number"] = sequence_++;
    connected_ =
        writer_("event: " + event.member_str("type") + "\n" + Sse(event));
    return connected_;
  }

  HttpResponse::BodyWriter writer_;
  json::Value response_;
  json::Value item_;
  std::string text_;
  std::size_t sequence_{0};
  std::size_t output_index_{0};
  bool active_{false};
  bool reasoning_{false};
  bool connected_{true};
};

HttpResponse NonStreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>& generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  const auto result = generation->Wait();
  core::Utf8Decoder decoder;
  const ParsedGeneration generated =
      ParseGeneration(decoder.Push(result.text, true), initial_output_state,
                      request.chat.tools, request.chat.tool_choice,
                      result.finish_reason !=
                          TextGenerationBackend::FinishReason::kStopSequence);

  json::Value response = json::Value::object();
  response["id"] = RandomId("chatcmpl-");
  response["object"] = "chat.completion";
  response["created"] = Now();
  response["model"] = backend.model_id();
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  if (!generated.reasoning_content.empty()) {
    message["reasoning_content"] = generated.reasoning_content;
  }
  if (generated.text.empty() && !generated.tool_calls.empty()) {
    message["content"] = json::Value();
  } else {
    message["content"] = generated.text;
  }
  if (!generated.tool_calls.empty()) {
    message["tool_calls"] = ToolCallsJson(generated.tool_calls);
  }
  choice["message"] = std::move(message);
  choice["finish_reason"] = FinishReason(result, !generated.tool_calls.empty());
  choices.push_back(std::move(choice));
  response["choices"] = std::move(choices);
  response["usage"] = Usage(result);
  response["timings"] = GenerationTimings(result);
  RecordServerMetrics(result);

  std::ostringstream timing;
  timing << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
         << ", inter_token;dur=" << result.mean_inter_token_ms
         << ", max_inter_token;dur=" << result.max_inter_token_ms;

  return {
      .status = 200,
      .reason = "OK",
      .body = response.dump(),
      .headers = {{"Server-Timing", timing.str()}},
      .streaming_body = {},
      .log_details = GenerationLogDetails(result),
  };
}

HttpResponse StreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  const std::string id = RandomId("chatcmpl-");
  const long long created = Now();
  const std::string model = backend.model_id();
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  return {
      .status = 200,
      .reason = "OK",
      .body = {},
      .headers =
          {
              {"Content-Type", "text/event-stream"},
              {"Cache-Control", "no-cache"},
              {"X-Accel-Buffering", "no"},
          },
      .streaming_body =
          [request, generation = std::move(generation), id, created, model,
           initial_output_state,
           stream_log](const HttpResponse::BodyWriter& writer) {
            json::Value role_delta = json::Value::object();
            role_delta["role"] = "assistant";
            if (!writer(Sse(
                    ChoiceChunk(id, created, model, std::move(role_delta))))) {
              generation->Cancel();
              return;
            }

            bool connected = true;
            StreamingTextFilter filter(
                initial_output_state,
                [&](std::string_view text, bool is_reasoning) {
                  if (text.empty()) {
                    return true;
                  }
                  json::Value delta = json::Value::object();
                  if (is_reasoning) {
                    delta["reasoning_content"] = std::string(text);
                  } else {
                    delta["content"] = std::string(text);
                  }
                  connected = writer(
                      Sse(ChoiceChunk(id, created, model, std::move(delta))));
                  return connected;
                });

            try {
              const auto result = generation->Wait([&](std::string_view piece) {
                return connected && filter.Push(piece);
              });
              stream_log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (!connected || result.cancelled) {
                return;
              }
              if (!filter.Push({}, true))
                return;

              const ParsedGeneration generated = ParseGeneration(
                  filter.raw(), initial_output_state, request.chat.tools,
                  request.chat.tool_choice,
                  result.finish_reason !=
                      TextGenerationBackend::FinishReason::kStopSequence);
              if (!filter.Finish(generated.hide_tool_markup)) {
                return;
              }
              for (std::size_t index = 0; index < generated.tool_calls.size();
                   ++index) {
                const auto& call = generated.tool_calls[index];
                json::Value delta = json::Value::object();
                json::Value tool_calls = json::Value::array();
                json::Value item = json::Value::object();
                item["index"] = index;
                item["id"] = call.id;
                item["type"] = "function";
                json::Value function = json::Value::object();
                function["name"] = call.name;
                function["arguments"] = ArgumentsJson(call.arguments);
                item["function"] = std::move(function);
                tool_calls.push_back(std::move(item));
                delta["tool_calls"] = std::move(tool_calls);
                if (!writer(Sse(
                        ChoiceChunk(id, created, model, std::move(delta))))) {
                  return;
                }
              }

              json::Value terminal_delta = json::Value::object();
              auto terminal_chunk = ChoiceChunk(
                  id, created, model, std::move(terminal_delta),
                  FinishReason(result, !generated.tool_calls.empty()));
              // llama.cpp reports timings on the terminal choice regardless
              // of the optional OpenAI usage chunk. Proxies need these even
              // when a client does not request stream_options.include_usage.
              terminal_chunk["timings"] = GenerationTimings(result);
              if (!writer(Sse(terminal_chunk))) {
                return;
              }
              if (request.include_usage) {
                json::Value usage_chunk = BaseChunk(id, created, model);
                usage_chunk["choices"] = json::Value::array();
                usage_chunk["usage"] = Usage(result);
                usage_chunk["timings"] = GenerationTimings(result);
                if (!writer(Sse(usage_chunk))) {
                  return;
                }
              }
              (void)writer("data: [DONE]\n\n");
            } catch (const TextGenerationError& exception) {
              stream_log->error_code = exception.stable_code();
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = exception.what();
              detail["type"] = "server_error";
              detail["code"] = exception.stable_code();
              error["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            } catch (const std::exception&) {
              stream_log->error_code = "generation_failed";
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = "generation failed";
              detail["type"] = "server_error";
              detail["code"] = "generation_failed";
              error["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            }
          },
      .stream_log = std::move(stream_log),
  };
}

}  // namespace

HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream) {
  const auto initial = backend.initial_output_state(chat);
  auto generation = backend.start_chat(chat, max_tokens, sampling,
                                       request.is_cancelled, stream);
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  auto timing = std::make_shared<std::string>();
  const auto run = [generation, initial, model = backend.model_id(), stream_log,
                    timing](const HttpResponse::BodyWriter& writer) {
    ResponsesOutput output(model, writer);
    if (!output.Begin()) {
      generation->Cancel();
      return json::Value();
    }
    StreamingTextFilter filter(initial,
                               [&](std::string_view piece, bool reasoning) {
                                 if (output.Append(piece, reasoning))
                                   return true;
                                 generation->Cancel();
                                 return false;
                               });
    try {
      const auto result = writer
                              ? generation->Wait([&](std::string_view piece) {
                                  return filter.Push(piece);
                                })
                              : generation->Wait();
      stream_log->details = GenerationLogDetails(result);
      RecordServerMetrics(result);
      if (!writer) {
        std::ostringstream value;
        value << std::fixed << std::setprecision(3)
              << "ttft;dur=" << result.ttft_ms
              << ", inter_token;dur=" << result.mean_inter_token_ms
              << ", max_inter_token;dur=" << result.max_inter_token_ms;
        *timing = value.str();
      }
      if (result.cancelled)
        return json::Value();
      if (!writer)
        filter.Push(result.text);
      if (!filter.Push({}, true) || !filter.Finish(false)) {
        generation->Cancel();
        return json::Value();
      }
      return output.Complete(result);
    } catch (const std::exception& error) {
      if (!writer)
        throw;
      const auto* generation_error =
          dynamic_cast<const TextGenerationError*>(&error);
      stream_log->error_code = generation_error
                                   ? generation_error->stable_code()
                                   : "generation_failed";
      stream_log->error_event_sent =
          output.Fail(generation_error ? error.what() : "generation failed");
      generation->Cancel();
      return json::Value();
    }
  };
  if (stream) {
    return {.status = 200,
            .reason = "OK",
            .body = {},
            .headers = {{"Content-Type", "text/event-stream"},
                        {"Cache-Control", "no-cache"},
                        {"X-Accel-Buffering", "no"}},
            .streaming_body =
                [run](const HttpResponse::BodyWriter& writer) {
                  (void)run(writer);
                },
            .stream_log = std::move(stream_log)};
  }
  auto response = run({});
  return {.status = 200,
          .reason = "OK",
          .body = response.dump(),
          .headers = {{"Server-Timing", *timing}},
          .log_details = stream_log->details};
}

HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend) {
  ParsedChatRequest parsed;
  const auto defaults = backend.sampling_defaults();
  parsed.max_tokens = defaults.max_tokens;
  parsed.sampling = defaults.sampling;
  if (auto error = ParseRequest(request, backend, &parsed); error.has_value()) {
    return std::move(*error);
  }
  try {
    const auto initial_output_state = backend.initial_output_state(parsed.chat);
    auto generation =
        backend.start_chat(parsed.chat, parsed.max_tokens, parsed.sampling,
                           request.is_cancelled, parsed.stream);
    if (parsed.stream) {
      return StreamingResponse(parsed, backend, std::move(generation),
                               initial_output_state);
    }
    return NonStreamingResponse(parsed, backend, generation,
                                initial_output_state);
  } catch (const TextGenerationError& exception) {
    return GenerationError(exception);
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

}  // namespace gufo::server

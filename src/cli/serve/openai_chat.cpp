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

#include "src/cli/serve/json.hpp"
#include "src/cli/serve/sampling_request.hpp"

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

bool ValidClientId(std::string_view client_id) {
  return !client_id.empty() && client_id.size() <= 64 &&
         std::ranges::all_of(client_id, [](unsigned char character) {
           return std::isalnum(character) != 0 || character == '-' ||
                  character == '_' || character == '.' || character == ':';
         });
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

bool ParseContent(const json::Value* content, std::string* output,
                  std::string* error) {
  if (content == nullptr || content->is_null()) {
    return true;
  }
  if (content->is_string()) {
    *output = content->get_str();
    return true;
  }
  if (!content->is_array()) {
    *error = "message content must be a string, null, or text-part array";
    return false;
  }

  for (const auto& part : content->items()) {
    if (!part.is_object()) {
      *error = "message content parts must be objects";
      return false;
    }
    const std::string type = part.member_str("type", "text");
    if (type != "text" && type != "input_text") {
      *error = "only text message content is supported";
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
                  std::string* error) {
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
  if (!ParseContent(value.find("content"), &message->content, error)) {
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
  if (tools == nullptr) {
    return true;
  }
  if (!tools->is_array()) {
    *error = "'tools' must be an array";
    return false;
  }
  for (const auto& item : tools->items()) {
    if (!item.is_object() || item.member_str("type") != "function") {
      *error = "only function tools are supported";
      return false;
    }
    const json::Value* function = item.find("function");
    if (function == nullptr || !function->is_object()) {
      *error = "function tools require a function object";
      return false;
    }
    tokenization::ChatTool tool;
    tool.name = function->member_str("name");
    tool.description = function->member_str("description");
    const json::Value* parameters = function->find("parameters");
    if (tool.name.empty() || parameters == nullptr ||
        !parameters->is_object()) {
      *error = "function tools require a name and object parameters schema";
      return false;
    }
    tool.parameters_json = parameters->dump();
    output->push_back(std::move(tool));
  }
  return true;
}

bool ParseToolChoice(const json::Value* value, ParsedChatRequest* request,
                     std::string* error) {
  if (value == nullptr) {
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

  const std::string client_id = request.header("X-Client-ID");
  if (!client_id.empty()) {
    if (!ValidClientId(client_id)) {
      return Error(400, "Bad Request",
                   "'X-Client-ID' must contain 1-64 letters, digits, '.', "
                   "'_', '-', or ':'",
                   "invalid_client_id");
    }
    output->chat.client_id = client_id;
  }

  const json::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() || messages->empty()) {
    return Error(400, "Bad Request", "'messages' must be a non-empty array",
                 "missing_messages");
  }
  for (const auto& item : messages->items()) {
    tokenization::ChatMessage message;
    std::string parse_error;
    if (!ParseMessage(item, &message, &parse_error)) {
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

  if (const json::Value* stream = body.find("stream")) {
    if (!stream->is_bool()) {
      return Error(400, "Bad Request", "'stream' must be a boolean",
                   "invalid_stream");
    }
    output->stream = stream->as_bool();
  }
  if (const json::Value* options = body.find("stream_options")) {
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
  if (max_tokens == nullptr) {
    max_tokens = body.find("max_tokens");
  }
  if (max_tokens != nullptr) {
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
      choices != nullptr && (!choices->is_number() || choices->as_int() != 1)) {
    return Error(400, "Bad Request", "only n=1 is supported", "unsupported_n");
  }
  for (const std::string_view unsupported :
       {"logprobs", "top_logprobs", "logit_bias", "stop", "response_format",
        "modalities", "audio"}) {
    if (body.contains(std::string(unsupported))) {
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

void ParseQwenCalls(std::string_view text, std::vector<ParsedToolCall>* calls) {
  std::size_t cursor = 0;
  while ((cursor = text.find("<tool_call>", cursor)) !=
         std::string_view::npos) {
    const std::size_t block_end = text.find("</tool_call>", cursor);
    if (block_end == std::string_view::npos) {
      return;
    }
    const std::string_view block = text.substr(
        cursor, block_end + std::string_view{"</tool_call>"}.size() - cursor);
    const std::size_t function = block.find("<function=");
    if (function == std::string_view::npos) {
      const std::size_t payload_start =
          block.find('>', std::string_view{"<tool_call"}.size());
      if (payload_start != std::string_view::npos) {
        const std::string_view payload =
            Trim(block.substr(payload_start + 1,
                              block.rfind("</tool_call>") - payload_start - 1));
        const auto parsed = TryParseJson(payload);
        if (parsed.has_value()) {
          ParsedToolCall call;
          call.id = RandomId("call_");
          call.name = parsed->member_str("name");
          const json::Value* arguments = parsed->find("arguments");
          if (!call.name.empty() && arguments != nullptr &&
              arguments->is_object()) {
            for (const auto& [name, value] : arguments->members()) {
              call.arguments.push_back({
                  .name = name,
                  .value = value.is_string() ? value.get_str() : value.dump(),
                  .is_string = value.is_string(),
              });
            }
            calls->push_back(std::move(call));
          }
        }
      }
      cursor = block_end + std::string_view{"</tool_call>"}.size();
      continue;
    }
    const std::size_t name_start =
        function + std::string_view{"<function="}.size();
    const std::size_t name_end = block.find('>', name_start);
    const std::size_t function_end = block.find("</function>", name_end);
    if (name_end == std::string_view::npos ||
        function_end == std::string_view::npos) {
      return;
    }

    ParsedToolCall call;
    call.id = RandomId("call_");
    call.name =
        std::string(Trim(block.substr(name_start, name_end - name_start)));
    std::size_t parameter_cursor = name_end + 1;
    while ((parameter_cursor = block.find("<parameter=", parameter_cursor)) !=
           std::string_view::npos) {
      if (parameter_cursor >= function_end) {
        break;
      }
      const std::size_t parameter_name_start =
          parameter_cursor + std::string_view{"<parameter="}.size();
      const std::size_t parameter_name_end =
          block.find('>', parameter_name_start);
      const std::size_t parameter_end =
          block.find("</parameter>", parameter_name_end);
      if (parameter_name_end == std::string_view::npos ||
          parameter_end == std::string_view::npos ||
          parameter_end > function_end) {
        break;
      }
      const std::string_view value = Trim(block.substr(
          parameter_name_end + 1, parameter_end - parameter_name_end - 1));
      const bool is_string = !TryParseJson(value).has_value();
      call.arguments.push_back({
          .name = std::string(
              Trim(block.substr(parameter_name_start,
                                parameter_name_end - parameter_name_start))),
          .value = std::string(value),
          .is_string = is_string,
      });
      parameter_cursor =
          parameter_end + std::string_view{"</parameter>"}.size();
    }
    if (!call.name.empty()) {
      calls->push_back(std::move(call));
    }
    cursor = block_end + std::string_view{"</tool_call>"}.size();
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

ParsedGeneration ParseGeneration(std::string_view raw) {
  ParsedGeneration parsed;
  const std::string_view content = raw;

  constexpr std::string_view kThinkStart = "<think>";
  constexpr std::string_view kThinkEnd = "</think>";
  const std::size_t think_start = content.find(kThinkStart);
  if (think_start != std::string_view::npos) {
    const std::size_t think_content_start = think_start + kThinkStart.size();
    const std::size_t think_end = content.find(kThinkEnd, think_content_start);
    if (think_end != std::string_view::npos) {
      parsed.reasoning_content = std::string(Trim(content.substr(
          think_content_start, think_end - think_content_start)));
      std::string_view remaining = content.substr(think_end + kThinkEnd.size());
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
      parsed.reasoning_content =
          std::string(Trim(content.substr(think_content_start)));
      parsed.text = "";
    }
  } else {
    parsed.text = std::string(content);
  }

  const std::size_t marker = EarliestMarker(parsed.text);
  if (marker != std::string_view::npos) {
    const std::string text_before_tools = parsed.text.substr(0, marker);
    const std::string text_from_tools = parsed.text.substr(marker);
    ParseQwenCalls(text_from_tools, &parsed.tool_calls);
    ParseDsmlCalls(text_from_tools, &parsed.tool_calls);
    if (!parsed.tool_calls.empty()) {
      parsed.text = text_before_tools;
    }
  }
  return parsed;
}

const char* FinishReason(const TextGenerationBackend::Result& result,
                         bool has_tool_calls) {
  if (has_tool_calls) {
    return "tool_calls";
  }
  if (result.finish_reason == TextGenerationBackend::FinishReason::kLength) {
    return "length";
  }
  return "stop";
}

json::Value Timings(const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  const double prompt_per_second =
      (result.prefill_ms > 0.0 && result.prompt_tokens > 0)
          ? (static_cast<double>(result.prompt_tokens) /
             (result.prefill_ms / 1000.0))
          : 0.0;
  const double predicted_per_second =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;
  const double prompt_per_token_ms =
      (result.prompt_tokens > 0)
          ? (result.prefill_ms / static_cast<double>(result.prompt_tokens))
          : 0.0;
  const double predicted_per_token_ms =
      (result.completion_tokens > 0)
          ? (result.decode_ms / static_cast<double>(result.completion_tokens))
          : 0.0;

  timings["prompt_n"] = result.prompt_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] = prompt_per_token_ms;
  timings["prompt_per_second"] = prompt_per_second;
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] = predicted_per_token_ms;
  timings["predicted_per_second"] = predicted_per_second;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_write_ms"] = result.cache_disk_write_ms;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  return timings;
}

json::Value Metrics(const TextGenerationBackend::Result& result) {
  json::Value metrics = json::Value::object();
  metrics["time_to_first_token_ms"] = result.ttft_ms;
  metrics["generation_time_ms"] = result.decode_ms;
  metrics["queue_time_ms"] = result.queue_ms;
  metrics["mean_itl_ms"] = result.mean_inter_token_ms;
  metrics["prompt_tokens"] = result.prompt_tokens;
  metrics["completion_tokens"] = result.completion_tokens;
  metrics["cached_tokens"] = result.cached_prompt_tokens;
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_write_bytes"] = result.cache_disk_write_bytes;
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_write_ms"] = result.cache_disk_write_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
  return metrics;
}

json::Value Usage(const TextGenerationBackend::Result& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  json::Value prompt_details = json::Value::object();
  prompt_details["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_details"] = std::move(prompt_details);

  const double prompt_per_second =
      (result.prefill_ms > 0.0 && result.prompt_tokens > 0)
          ? (static_cast<double>(result.prompt_tokens) /
             (result.prefill_ms / 1000.0))
          : 0.0;
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
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_write_bytes"] = result.cache_disk_write_bytes;
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_write_ms"] = result.cache_disk_write_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
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

  explicit StreamingTextFilter(EmitCallback emit_piece)
      : emit_piece_(std::move(emit_piece)) {}

  bool Push(std::string_view piece) {
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
      if (end_pos != std::string::npos) {
        if (end_pos > 0 && !emit_piece_(pending_.substr(0, end_pos), true)) {
          return false;
        }
        std::string_view remaining = pending_;
        remaining.remove_prefix(end_pos + kThinkEnd.size());
        if (!remaining.empty() && remaining.front() == '\n') {
          remaining.remove_prefix(1);
        }
        pending_ = std::string(remaining);
        state_ = State::kContent;
      } else {
        std::size_t held = 0;
        for (std::size_t len = std::min(pending_.size(), kThinkEnd.size() - 1);
             len > 0; --len) {
          if (kThinkEnd.starts_with(pending_.substr(pending_.size() - len))) {
            held = len;
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

  bool Finish(bool valid_tool_calls) {
    if (!tool_mode_) {
      if (!pending_.empty()) {
        const bool is_reasoning = (state_ == State::kThinking);
        const bool emitted = emit_piece_(pending_, is_reasoning);
        pending_.clear();
        return emitted;
      }
      return true;
    }
    if (!valid_tool_calls && !hidden_.empty()) {
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
  std::string raw_;
  std::string pending_;
  std::string hidden_;
  State state_{State::kInitial};
  bool tool_mode_{false};
};

HttpResponse NonStreamingResponse(
    TextGenerationBackend& backend,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>&
        generation) {
  const auto result = generation->Wait();
  const ParsedGeneration generated = ParseGeneration(result.text);

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
  response["timings"] = Timings(result);
  response["metrics"] = Metrics(result);
  RecordServerMetrics(result);

  std::ostringstream timing;
  timing << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
         << ", inter_token;dur=" << result.mean_inter_token_ms
         << ", max_inter_token;dur=" << result.max_inter_token_ms;

  std::ostringstream details;
  const double prompt_per_second =
      (result.prefill_ms > 0.0 && result.prompt_tokens > 0)
          ? (static_cast<double>(result.prompt_tokens) /
             (result.prefill_ms / 1000.0))
          : 0.0;
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  details << result.prompt_tokens << " prompt tok";
  if (prompt_per_second > 0.0) {
    details << " (" << std::fixed << std::setprecision(1) << prompt_per_second
            << " tok/s)";
  }
  details << " | " << result.completion_tokens << " gen tok";
  if (tok_per_sec > 0.0) {
    details << " (" << std::fixed << std::setprecision(1) << tok_per_sec
            << " tok/s)";
  }
  if (result.cached_prompt_tokens > 0) {
    details << " | cache: " << result.cached_prompt_tokens << " tok";
  }
  if (result.draft_tokens > 0) {
    const double accept_pct =
        (static_cast<double>(result.draft_accepted_tokens) * 100.0) /
        static_cast<double>(result.draft_tokens);
    details << " | draft: " << result.draft_accepted_tokens << "/"
            << result.draft_tokens << " (" << std::fixed << std::setprecision(1)
            << accept_pct << "%)";
  }
  if (result.ttft_ms > 0.0) {
    details << " | TTFT: " << std::fixed << std::setprecision(1)
            << result.ttft_ms << "ms";
  }

  return {
      .status = 200,
      .reason = "OK",
      .body = response.dump(),
      .headers = {{"Server-Timing", timing.str()}},
      .streaming_body = {},
      .log_details = details.str(),
  };
}

HttpResponse StreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation) {
  const std::string id = RandomId("chatcmpl-");
  const long long created = Now();
  const std::string model = backend.model_id();
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
          [request, generation = std::move(generation), id, created,
           model](const HttpResponse::BodyWriter& writer) {
            json::Value role_delta = json::Value::object();
            role_delta["role"] = "assistant";
            if (!writer(Sse(
                    ChoiceChunk(id, created, model, std::move(role_delta))))) {
              generation->Cancel();
              return;
            }

            bool connected = true;
            StreamingTextFilter filter(
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
              if (!connected || result.cancelled) {
                return;
              }

              const ParsedGeneration generated = ParseGeneration(filter.raw());
              if (!filter.Finish(!generated.tool_calls.empty())) {
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
              if (!writer(Sse(ChoiceChunk(
                      id, created, model, std::move(terminal_delta),
                      FinishReason(result, !generated.tool_calls.empty()))))) {
                return;
              }
              json::Value usage_chunk = BaseChunk(id, created, model);
              usage_chunk["choices"] = json::Value::array();
              usage_chunk["usage"] = Usage(result);
              usage_chunk["timings"] = Timings(result);
              usage_chunk["metrics"] = Metrics(result);
              RecordServerMetrics(result);
              if (!writer(Sse(usage_chunk))) {
                return;
              }
              (void)writer("data: [DONE]\n\n");
            } catch (const TextGenerationError& exception) {
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = exception.what();
              detail["type"] = "server_error";
              detail["code"] = exception.stable_code();
              error["error"] = std::move(detail);
              (void)writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            } catch (const std::exception&) {
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = "generation failed";
              detail["type"] = "server_error";
              detail["code"] = "generation_failed";
              error["error"] = std::move(detail);
              (void)writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            }
          },
  };
}

}  // namespace

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
    auto generation =
        backend.start_chat(parsed.chat, parsed.max_tokens, parsed.sampling,
                           request.is_cancelled, parsed.stream);
    if (parsed.stream) {
      return StreamingResponse(parsed, backend, std::move(generation));
    }
    return NonStreamingResponse(backend, generation);
  } catch (const TextGenerationError& exception) {
    return GenerationError(exception);
  }
}

}  // namespace gufo::server

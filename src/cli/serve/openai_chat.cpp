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

#include "src/cli/serve/quote_tracker.hpp"
#include "src/cli/serve/response_format.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

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
using ToolMarkerSet = std::span<const std::string_view>;

ToolMarkerSet ToolMarkers(
    std::optional<sampling::JsonConstraint::ToolFormat> format) {
  using Format = sampling::JsonConstraint::ToolFormat;
  static constexpr std::array<std::string_view, 1> qwen{"<tool_call>"};
  // The template writes "\n\n" before the call block. As in llama.cpp
  // common/parsers/deepseek.cpp (TC_SEPARATOR + FC_START), that separator is
  // framing: returning it as content would double it when history is replayed.
  static constexpr std::array<std::string_view, 2> deepseek{
      "\n\n<｜DSML｜tool_calls>", "<｜DSML｜tool_calls>"};
  // The JSON fallback has no native counterpart in llama.cpp. Its grammar
  // leaves text before <tool_call> unconstrained, where models still write
  // their native call syntax; keep recognizing every opener there.
  if (!format || *format == Format::kJson)
    return kToolMarkers;
  // Match the canonical opener selected by JsonConstraint::WithTools. Other
  // dialects are ordinary prose under that grammar, not additional calls.
  return *format == Format::kDeepSeek ? ToolMarkerSet(deepseek)
                                      : ToolMarkerSet(qwen);
}

// A native function header, rather than a bare marker mentioned in prose,
// permits the implicit reasoning boundary used by llama.cpp's native parsers.
ToolMarkerSet ReasoningToolMarkers(ToolMarkerSet markers) {
  static constexpr std::array<std::string_view, 1> qwen{
      "<tool_call>\n<function="};
  static constexpr std::array<std::string_view, 2> deepseek{
      "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"",
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\""};
  if (markers.size() == 1 && markers.front() == "<tool_call>")
    return qwen;
  if (markers.size() == 2 && markers.back() == "<｜DSML｜tool_calls>")
    return deepseek;
  return {};
}

// Closers eligible for removal immediately after an accepted call. Standalone
// closers and XML remain literal; the declared-tool envelope rule is separate.
constexpr std::array<std::string_view, 4> kQwenClosers{
    "</parameter>",
    "</function>",
    "</tool_call>",
    // The client's envelope, whose block rules already strip it whole.
    "</invoke>",
};
constexpr std::array<std::string_view, 9> kToolClosers{
    // Qwen family.
    "</parameter>",
    "</function>",
    "</tool_call>",
    // Envelope dialects the clients and harness prompts render; Qwen models
    // echo these back verbatim, so they are framing here too.
    "</invoke>",
    "</function_results>",
    "</tool_calls>",
    // Model-native dialects, which this fallback admits like any other opener.
    "</｜DSML｜parameter>",
    "</｜DSML｜invoke>",
    "</｜DSML｜tool_calls>",
};
using ToolCloserSet = std::span<const std::string_view>;

ToolCloserSet ToolClosers(
    std::optional<sampling::JsonConstraint::ToolFormat> format) {
  using Format = sampling::JsonConstraint::ToolFormat;
  if (!format)
    return kToolClosers;
  // As in llama.cpp, DeepSeek content is everything outside its native call
  // block, which ends the output: there is no echo to strip, and client
  // envelope syntax the model writes stays visible text.
  if (*format == Format::kDeepSeek)
    return {};
  // Mirror ToolMarkers: stripping follows the dialect the request admitted,
  // and another dialect's closing tags are ordinary prose like its openers —
  // the client's envelope excepted, which no dialect admits and every one of
  // them parses.
  return kQwenClosers;
}

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
          !url->is_string()) {
        *error = "image_url requires a user message and a string URL";
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

// Declared names reach Qwen and DeepSeek unescaped inside "<function=NAME>"
// and "name=\"NAME\"", so the characters that frame a call
// are excluded. The dots, colons and slashes that agent harnesses give bridged
// tool names are data and are kept. Non-ASCII bytes are excluded as well: a
// name is placed in a prompt the model reads and in operator logs, where
// confusable and invisible characters buy a client nothing.
// Historical names reach the same renderers unchanged; they describe past
// output and do not declare a tool the model is allowed to call now.
constexpr std::string_view kToolNameRule =
    "function names require 1-64 printable ASCII characters other than "
    "spaces, '<', '>', '\"' and '\\'";

bool RenderableToolName(std::string_view name) {
  return !name.empty() && name.size() <= 64 &&
         std::ranges::none_of(name, [](unsigned char c) {
           return c <= 0x20 || c >= 0x7F || c == '<' || c == '>' || c == '"' ||
                  c == '\\';
         });
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
  // Render typed values as the reference chat templates' tojson does (and as
  // llama.cpp's Jinja runtime does), so a replayed call matches the tokens
  // the model generated and its continuation checkpoint is reused.
  for (const auto& [name, value] : parsed.members()) {
    out->push_back({
        .name = name,
        .value = value.is_string() ? value.get_str() : value.tojson(),
        .is_string = value.is_string(),
    });
  }
  return true;
}

bool ParseHistoricalFunction(const json::Value& function,
                             tokenization::ChatMessage::ToolCall* call,
                             std::string* error) {
  const auto* name = function.find("name");
  const auto* arguments = function.find("arguments");
  if (!name || !name->is_string() || name->str().empty() || !arguments ||
      !arguments->is_string()) {
    *error =
        "historical function calls require a non-empty name and string "
        "arguments";
    return false;
  }
  // History is a record, not a declaration of a tool the model may call now.
  // Preserve names as llama.cpp common/chat.cpp does at
  // f1cee9941b0e843ea260bf8dd9a090fbd9711b6a. NUL cannot pass through the
  // DeepSeek tokenizer's C-string interface.
  if (name->str().find('\0') != std::string::npos) {
    *error = "historical function names cannot contain NUL";
    return false;
  }
  call->name = name->str();
  return ParseArguments(arguments->str(), &call->arguments, error);
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
    if (!ParseHistoricalFunction(*function, &call, error))
      return false;
    message->tool_calls.push_back(std::move(call));
  }
  return true;
}

bool ParseTools(const json::Value* tools,
                std::vector<tokenization::ChatTool>* output, std::string* error,
                bool allow_non_function = false) {
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
    if (allow_non_function && item.member_str("type") == "namespace") {
      // A Responses namespace only groups client-executed function tools for
      // organization; calls replay by the plain function name. Flatten the
      // nested functions and let the uniqueness check reject ambiguous
      // namespaces. Nested hosted tools skip like top-level hosted types.
      const auto* nested = item.find("tools");
      if (nested == nullptr || !nested->is_array()) {
        *error = "namespace tools require a tools array";
        return false;
      }
      if (!ParseTools(nested, output, error, allow_non_function))
        return false;
      continue;
    }
    if (item.member_str("type") != "function") {
      // The Responses API declares hosted tool types (web_search, file_search,
      // code_interpreter, mcp, ...) that only the provider can execute.
      // Skip them so the request still reaches the function tools the model can
      // call; Chat Completions declares only functions and keeps its contract.
      if (allow_non_function)
        continue;
      *error = "only function tools are supported";
      return false;
    }

    const json::Value* function = item.find("function");
    if (function != nullptr && !function->is_object()) {
      *error = "'function' must be an object";
      return false;
    }
    const json::Value* src = function != nullptr ? function : &item;
    if (const auto* strict = src->find("strict");
        strict && !strict->is_null() && !strict->is_bool()) {
      *error = "function strict must be a boolean or null";
      return false;
    }
    // OpenAI uses "parameters"; some agent clients send parametersJsonSchema.
    const json::Value* params_src = src->find("parameters");
    if (params_src == nullptr || params_src->is_null()) {
      params_src = src->find("parametersJsonSchema");
    }

    tokenization::ChatTool tool;
    tool.name = src->member_str("name");
    tool.description = src->member_str("description");
    if (!RenderableToolName(tool.name)) {
      *error = std::string(kToolNameRule);
      return false;
    }
    if (std::ranges::any_of(*output, [&](const auto& previous) {
          return previous.name == tool.name;
        })) {
      *error = "function names must be unique";
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
    if (const auto* strict = function_obj.find("strict");
        strict && strict->as_bool()) {
      // Omitting parameters defines a function with no arguments.
      if (!params_src || params_src->is_null())
        function_obj["parameters"] = json::parse(
            R"({"type":"object","properties":{},"required":[],"additionalProperties":false})");
      try {
        sampling::JsonConstraint::Compile(*function_obj.find("parameters"),
                                          true);
      } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
      }
    }
    tool.parameters_json = function_obj.find("parameters")->dump();
    json::Value definition = function != nullptr ? item : json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function_obj);
    tool.definition_json = definition.dump();
    // Counted per flattened function, so namespace recursion cannot exceed
    // the cap by ordering hosted-adjacent entries around a full namespace.
    if (output->size() >= 128) {
      *error = "'tools' supports at most 128 functions";
      return false;
    }
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
    request->chat.forced_tool_name.clear();
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
  if (value->is_object() && value->member_str("type") == "function") {
    const auto* function = value->find("function");
    const auto name =
        function ? function->member_str("name") : value->member_str("name");
    auto& tools = request->chat.tools;
    const auto found = std::ranges::find_if(
        tools, [&](const auto& tool) { return tool.name == name; });
    if (found != tools.end()) {
      auto tool = *found;
      tools = {std::move(tool)};
      request->chat.tool_choice = ChatRequest::ToolChoice::kRequired;
      request->chat.forced_tool_name = name;
      return true;
    }
    *error = "'tool_choice' must name a declared function";
    return false;
  }
  *error = "'tool_choice' must be auto, none, required, or a declared function";
  return false;
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

std::optional<HttpResponse> ParseToolControls(const json::Value& body,
                                              ParsedChatRequest* output,
                                              bool nullable_parallel = false,
                                              bool allow_non_function = false) {
  std::string parse_error;
  if (!ParseTools(body.find("tools"), &output->chat.tools, &parse_error,
                  allow_non_function) ||
      !ParseToolChoice(body.find("tool_choice"), output, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error), "invalid_tools");
  }
  if (output->chat.tool_choice == ChatRequest::ToolChoice::kRequired &&
      output->chat.tools.empty()) {
    return Error(400, "Bad Request",
                 "'tool_choice' cannot be required without tools",
                 "invalid_tool_choice");
  }
  if (const auto* parallel = body.find("parallel_tool_calls");
      parallel && !(nullable_parallel && parallel->is_null())) {
    if (!parallel->is_bool())
      return Error(400, "Bad Request",
                   "'parallel_tool_calls' must be a boolean", "invalid_tools");
    output->chat.parallel_tool_calls = parallel->as_bool();
  }
  if (const auto* choice = body.find("tool_choice");
      choice && choice->is_object())
    output->chat.parallel_tool_calls =
        false;  // Forced functions execute exactly once.
  output->chat.constrained_tools =
      !output->chat.tools.empty() &&
      output->chat.tool_choice != ChatRequest::ToolChoice::kNone;

  return {};
}

json::Value NormalizeToolSchema(const json::Value& schema,
                                std::size_t depth = 0) {
  if (!schema.is_object() || depth > 16 || (depth && schema.empty()))
    throw std::invalid_argument("tool schema cannot be made strict");
  auto out = schema;
  for (const auto* key : {"properties", "$defs", "definitions"}) {
    if (const auto* fields = schema.find(key); fields && fields->is_object()) {
      auto normalized = json::Value::object();
      for (const auto& [name, field] : fields->members())
        normalized.append_member(name, NormalizeToolSchema(field, depth + 1));
      out[key] = std::move(normalized);
    }
  }
  if (const auto* items = schema.find("items"))
    out["items"] = NormalizeToolSchema(*items, depth + 1);
  for (const auto* key : {"anyOf", "allOf", "oneOf"}) {
    if (const auto* choices = schema.find(key);
        choices && choices->is_array()) {
      auto normalized = json::Value::array();
      for (const auto& choice : choices->items())
        normalized.push_back(NormalizeToolSchema(choice, depth + 1));
      out[key] = std::move(normalized);
    }
  }
  if (schema.empty() || schema.member_str("type") == "object" ||
      schema.contains("properties")) {
    if (!out.contains("type"))
      out["type"] = "object";
    if (!out.contains("properties"))
      out["properties"] = json::Value::object();
    if (!out.contains("additionalProperties"))
      out["additionalProperties"] = false;
    out["required"] = json::Value::array();
    for (const auto& [name, value] : out.find("properties")->members()) {
      (void)value;
      out["required"].push_back(name);
    }
  }
  return out;
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

  try {
    output->chat.response_format =
        ParseResponseFormat(body.find("response_format"));
    if (output->chat.response_format) {
      if (const auto* specification =
              body.find("response_format")->find("json_schema"))
        output->chat.response_format_description =
            specification->member_str("description");
    }
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", error.what(), "invalid_response_format");
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

  if (auto error = ParseToolControls(body, output))
    return error;
  std::string parse_error;

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
  if (const json::Value* progress = body.find("return_progress");
      progress != nullptr && !progress->is_null()) {
    if (!progress->is_bool()) {
      return Error(400, "Bad Request", "'return_progress' must be a boolean",
                   "invalid_return_progress");
    }
    output->chat.return_progress = progress->as_bool();
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
  if (const auto sampling_error = ParseSamplingConfig(
          body,
          backend.sampling_defaults().Resolve(output->chat.reasoning.enabled),
          &parsed_sampling)) {
    return Error(400, "Bad Request", sampling_error->message,
                 sampling_error->code.c_str());
  }
  output->sampling = parsed_sampling;

  if (const json::Value* choices = body.find("n");
      choices != nullptr && !choices->is_null() &&
      (!choices->is_number() || choices->as_double() != 1.0)) {
    return Error(400, "Bad Request", "only n=1 is supported", "unsupported_n");
  }
  // Raw Completions owns the fixed-length benchmark contract. Rejecting the
  // field here keeps a harness from measuring silently shortened runs.
  if (const auto* value = body.find("ignore_eos");
      value != nullptr && !value->is_null()) {
    return Error(400, "Bad Request",
                 "request field 'ignore_eos' is not supported on this endpoint",
                 "unsupported_field");
  }
  for (const std::string_view unsupported :
       {"logprobs", "top_logprobs", "modalities", "audio"}) {
    if (const auto* value = body.find(std::string(unsupported));
        value != nullptr && !value->is_null()) {
      if ((unsupported == "logprobs" && value->is_bool() &&
           !value->as_bool()) ||
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

// `value` without its trailing whitespace.
std::string_view TrimTrailing(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  return TrimTrailing(value);
}

std::string_view AfterReasoningSeparator(std::string_view value,
                                         ToolMarkerSet markers) {
  // Qwen's llama.cpp PEG uses `reasoning << content`: `<<` consumes space().
  // Keep other dialects' existing line-break rule. Only the boundary after an
  // explicit </think> is framing; answer and argument interiors are untouched.
  const bool qwen = markers.size() == 1 && markers.front() == "<tool_call>";
  const auto first = value.find_first_not_of(qwen ? " \t\r\n\f\v" : "\r\n");
  return first == std::string_view::npos ? std::string_view{}
                                         : value.substr(first);
}

// Qwen writes a parameter as "<parameter=name>\nVALUE\n</parameter>": one
// newline on each side is framing, everything else (a file's final newline,
// indentation, blank lines) belongs to the value.
std::string_view StripFramingNewlines(std::string_view value) {
  // Match the opening frame: a literal final CR before an LF frame is data.
  const std::string_view frame = value.starts_with("\r\n") ? "\r\n" : "\n";
  if (value.starts_with(frame)) {
    value.remove_prefix(frame.size());
    if (value.ends_with(frame))
      value.remove_suffix(frame.size());
  }
  return value;
}

// Qwen's XML-like calls look like Python keyword arguments, and the models
// sometimes write Python's True, False and None where the schema asks for a
// JSON boolean or null. Rewrites those words outside JSON string literals.
std::string PythonLiteralsToJson(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  bool in_string = false;
  for (std::size_t i = 0; i < value.size();) {
    const char c = value[i];
    if (in_string) {
      const std::size_t length = c == '\\' && i + 1 < value.size() ? 2 : 1;
      out += value.substr(i, length);
      in_string = c != '"';
      i += length;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) == 0) {
      out += c;
      in_string = c == '"';
      ++i;
      continue;
    }
    std::size_t word_end = i;
    while (word_end < value.size() &&
           (std::isalnum(static_cast<unsigned char>(value[word_end])) != 0 ||
            value[word_end] == '_')) {
      ++word_end;
    }
    const auto word = value.substr(i, word_end - i);
    out += word == "True"    ? std::string_view{"true"}
           : word == "False" ? std::string_view{"false"}
           : word == "None"  ? std::string_view{"null"}
                             : word;
    i = word_end;
  }
  return out;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

// Earliest call marker at or after `from` that is not quoted prose. A marker
// the model writes inside a backtick span or a code fence is the model
// *explaining* the format; an answer that names its tags must not be parsed
// or cut at them (#383). `full` is the whole response so the surrounding
// lines are visible.
std::size_t EarliestMarker(std::string_view full, ToolMarkerSet markers,
                           QuoteTracker& quotes, std::size_t from = 0) {
  for (std::size_t position = from; position < full.size();) {
    const std::size_t next = full.find('<', position);
    if (next == std::string_view::npos) {
      break;
    }
    // The cheap question first: only a position that carries a marker is worth
    // asking about quoting, which is what keeps a line full of '<' linear.
    for (const auto marker : markers) {
      const auto tag_offset = marker.find('<');
      if (tag_offset != std::string_view::npos && next - from >= tag_offset &&
          full.substr(next - tag_offset).starts_with(marker) &&
          !quotes.QuotedAt(next)) {
        return next - tag_offset;
      }
    }
    position = next + 1;
  }
  return std::string_view::npos;
}

std::size_t ReasoningToolMarker(std::string_view full, ToolMarkerSet markers,
                                QuoteTracker& quotes, std::size_t from = 0) {
  auto found = EarliestMarker(full, markers, quotes, from);
  while (found != std::string_view::npos && quotes.UnclosedAt(found))
    found = EarliestMarker(full, markers, quotes, found + 1);
  return found;
}

// Bytes at the end of [0, end) that could still open a call marker. A marker
// that would open inside quoted prose is prose, so it is not held back.
std::size_t HeldMarkerPrefix(std::string_view full, std::size_t end,
                             ToolMarkerSet markers, QuoteTracker& quotes) {
  std::size_t maximum_marker = 0;
  for (const auto marker : markers) {
    maximum_marker = std::max(maximum_marker, marker.size());
  }
  const std::size_t maximum =
      std::min(end, maximum_marker > 0 ? maximum_marker - 1 : 0);
  for (std::size_t length = maximum; length > 0; --length) {
    const std::string_view suffix = full.substr(end - length, length);
    if (std::ranges::any_of(markers,
                            [&](std::string_view marker) {
                              return marker.starts_with(suffix);
                            }) &&
        !quotes.QuotedAt(end - length)) {
      return length;
    }
  }
  return 0;
}

bool ToolMarkerPrefix(std::string_view text, ToolMarkerSet markers) {
  return std::ranges::any_of(
      markers, [&](auto marker) { return marker.starts_with(text); });
}

// Client envelope syntax is removed only when it names a declared tool, or
// immediately echoes a call the parser accepted. Other XML stays content.
constexpr std::array<std::string_view, 1> kEnvelopeHeads{"<invoke name="};
constexpr std::array<std::string_view, 1> kEnvelopeParameters{
    "<parameter name="};
constexpr std::array<std::string_view, 2> kEnvelopeClosers{"</invoke>",
                                                           "</parameter>"};
// The last of `tags` at or before `stop`, or npos when there is none or the
// last one sits in quoted prose.
template<std::size_t N>
std::size_t LastUnquotedTag(std::string_view full, std::size_t stop,
                            std::size_t floor,
                            const std::array<std::string_view, N>& tags,
                            QuoteTracker& quotes) {
  if (stop <= floor) {
    return std::string_view::npos;
  }
  // Search only what the caller can use: a tag starting before `floor` is
  // discarded there, and these tags hold no `<`, so none can straddle a marker
  // or a closer and be missed.
  const std::string_view window = full.substr(floor, stop - floor);
  std::size_t last = std::string_view::npos;
  for (const auto tag : tags) {
    const auto found = window.rfind(tag);
    if (found == std::string_view::npos) {
      continue;
    }
    if (last == std::string_view::npos || found > last) {
      last = found;
    }
  }
  if (last == std::string_view::npos || quotes.QuotedAt(floor + last)) {
    return std::string_view::npos;
  }
  return floor + last;
}

bool DeclaredEnvelope(std::string_view tag,
                      std::span<const tokenization::ChatTool> tools) {
  const auto value = tag.substr(kEnvelopeHeads.front().size());
  if (value.empty() || (value.front() != '\"' && value.front() != '\''))
    return false;
  const auto end = value.find(value.front(), 1);
  return end != std::string_view::npos &&
         std::ranges::any_of(tools, [&](const auto& tool) {
           return tool.name == value.substr(1, end - 1);
         });
}

// Find a closed or truncated client envelope whose request context identifies
// a tool attempt. A bare parameter block is not enough evidence.
std::size_t EnvelopeBlockStart(std::string_view full, std::size_t stop,
                               std::size_t floor, ToolCloserSet closers,
                               std::span<const tokenization::ChatTool> tools,
                               bool after_call, QuoteTracker& quotes) {
  const auto opening =
      LastUnquotedTag(full, stop, floor, kEnvelopeHeads, quotes);
  if (opening == std::string_view::npos) {
    return std::string_view::npos;
  }
  const auto tag_end = full.find('>', opening);
  if (!after_call) {
    if (tag_end == std::string_view::npos || tag_end >= stop) {
      return std::string_view::npos;
    }
    if (!DeclaredEnvelope(full.substr(opening, tag_end - opening), tools)) {
      return std::string_view::npos;
    }
  }
  const auto region = TrimTrailing(full.substr(opening, stop - opening));
  for (const auto closer : kEnvelopeClosers) {
    if (region.ends_with(closer)) {
      return opening;
    }
  }
  if (region.find(kEnvelopeParameters.front()) == std::string_view::npos) {
    return std::string_view::npos;
  }
  for (const auto closer : closers) {
    if (region.find(closer) != std::string_view::npos) {
      return std::string_view::npos;
    }
  }
  return opening;
}

// Remove framing only with request context: a closer echo directly after a
// parsed call, or a client envelope naming a declared tool. Other XML is data.
// A dialect without closers removes nothing.
std::string_view ContentBefore(std::string_view full, std::size_t begin,
                               std::size_t end, ToolCloserSet closers,
                               QuoteTracker& quotes,
                               std::span<const tokenization::ChatTool> tools,
                               bool after_call = false) {
  const auto stop = std::min(end, full.size());
  if (stop <= begin)
    return {};
  if (closers.empty())
    return full.substr(begin, stop - begin);
  if (after_call) {
    auto cursor = begin;
    bool removed = false;
    while (cursor < stop) {
      while (cursor < stop &&
             std::isspace(static_cast<unsigned char>(full[cursor])))
        ++cursor;
      const auto tag = std::ranges::find_if(closers, [&](auto closer) {
        return full.substr(cursor, stop - cursor).starts_with(closer);
      });
      if (tag == closers.end() || quotes.QuotedAt(cursor))
        break;
      cursor += tag->size();
      removed = true;
    }
    if (removed)
      begin = cursor;
  }
  const auto slice = full.substr(begin, stop - begin);
  const bool adjacent_envelope =
      after_call && Trim(slice).starts_with(kEnvelopeHeads.front());
  if (const auto block = EnvelopeBlockStart(full, stop, begin, closers, tools,
                                            adjacent_envelope, quotes);
      block != std::string_view::npos) {
    return TrimTrailing(full.substr(begin, block - begin));
  }
  return slice;
}

// Hold only a possible client envelope. A declared-tool head must stay pending
// until final parsing decides whether it is quoted documentation or framing.
std::size_t FramingHold(std::string_view full, std::size_t end,
                        std::size_t floor, ToolCloserSet closers,
                        QuoteTracker& quotes,
                        std::span<const tokenization::ChatTool> tools) {
  if (closers.empty())
    return 0;
  std::size_t hold = 0;
  if (end > 0) {
    // Hold only a suffix that can still become admitted framing. A bare
    // comparison such as "3 < 5" is already prose once the space arrives.
    const auto open = full.find_last_of('<', end - 1);
    if (open != std::string_view::npos && open >= floor &&
        !quotes.QuotedAt(open)) {
      const auto suffix = full.substr(open, end - open);
      const auto prefix = [&](auto tags) {
        return std::ranges::any_of(
            tags, [&](auto tag) { return tag.starts_with(suffix); });
      };
      if (prefix(kEnvelopeHeads)) {
        hold = suffix.size();
      }
    }
  }
  if (const auto opening =
          LastUnquotedTag(full, end, floor, kEnvelopeHeads, quotes);
      opening != std::string_view::npos) {
    const auto tag_end = full.find('>', opening);
    if (tag_end == std::string_view::npos || tag_end >= end ||
        DeclaredEnvelope(full.substr(opening, tag_end - opening), tools)) {
      hold = std::max(hold, end - opening);
    }
  }
  // The separator before framing is removed with it. Keep trailing whitespace
  // undecided until the next piece establishes whether it introduces prose or
  // a held tag, rather than streaming a separator we cannot retract later.
  const auto ready = full.substr(floor, end - hold - floor);
  hold += ready.size() - TrimTrailing(ready).size();
  return hold;
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
const json::Value* ResolveToolSchema(const json::Value& root,
                                     const json::Value& schema) {
  const auto* target = &schema;
  std::vector<const json::Value*> seen;
  while (const auto* reference = target->find("$ref")) {
    if (std::ranges::find(seen, target) != seen.end())
      return nullptr;
    seen.push_back(target);
    try {
      target = sampling::JsonConstraint::ResolveReference(root, *reference);
      // A parameter pointing to its containing tool schema is recursive.
      // Native best-effort grammar cannot enforce that recursion; do not
      // infer a definite scalar type from the outer parameter container.
      if (target == &root)
        return nullptr;
    } catch (const std::invalid_argument&) {
      return nullptr;  // Non-strict schemas may not be compilable.
    }
  }
  return target;
}

bool SchemaAccepts(const json::Value& root, const json::Value& original,
                   const json::Value& value, std::size_t depth = 0) {
  const auto* resolved = ResolveToolSchema(root, original);
  if (!resolved || depth > 16)
    return true;  // Retain text for unknown non-strict argument types.
  const auto& schema = *resolved;
  // Match value kinds, as llama.cpp's value_types(), rather than guessing
  // from the lexeme. {"const":"42"} is text even without an explicit type.
  const auto same_kind = [&](const json::Value& candidate) {
    return (candidate.is_string() && value.is_string()) ||
           (candidate.is_number() && value.is_number()) ||
           (candidate.is_bool() && value.is_bool()) ||
           (candidate.is_null() && value.is_null()) ||
           (candidate.is_array() && value.is_array()) ||
           (candidate.is_object() && value.is_object());
  };
  if (const auto* constant = schema.find("const"))
    return same_kind(*constant);
  if (const auto* values = schema.find("enum"); values && values->is_array())
    return std::ranges::any_of(values->items(), same_kind);
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
        return SchemaAccepts(root, item, value, depth + 1);
      });
  }
  if (schema.contains("properties") ||
      (schema.contains("additionalProperties") &&
       !(schema.find("additionalProperties")->is_bool() &&
         schema.find("additionalProperties")->as_bool())))
    return value.is_object();
  if (const auto* parts = schema.find("allOf"); parts && parts->is_array())
    return std::ranges::all_of(parts->items(), [&](const auto& item) {
      return SchemaAccepts(root, item, value, depth + 1);
    });
  if (schema.contains("items") || schema.contains("prefixItems"))
    return value.is_array();
  if (schema.contains("pattern") || schema.contains("minLength") ||
      schema.contains("maxLength"))
    return value.is_string();
  return true;
}

void ParseQwenCalls(
    std::string_view text, std::span<const tokenization::ChatTool> tools,
    std::vector<ParsedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr,
    bool require_schema = false) {
  constexpr std::string_view start = "<tool_call>";
  constexpr std::string_view end = "</tool_call>";
  QuoteTracker call_quotes;
  call_quotes.Reset(text);
  std::size_t cursor = 0;
  while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
    if (call_quotes.QuotedAt(cursor)) {
      cursor += start.size();
      continue;
    }
    const auto marker_begin = cursor;
    const bool canonical_header =
        text.substr(cursor).starts_with("<tool_call>\n<function=");
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
      if (call.name.empty())
        continue;
      const auto* object =
          schema ? ResolveToolSchema(*schema, *schema) : nullptr;
      const auto* properties = object ? object->find("properties") : nullptr;
      bool valid = true;
      bool canonical_parameters = canonical_header;
      while (consume("<parameter=")) {
        const auto name_end = body.find('>');
        if (name_end == std::string_view::npos) {
          valid = false;
          break;
        }
        // Preserve the schema's exact key. llama.cpp's PEG matches this
        // spelling too, but its JSON mapper trims it; doing that here would
        // corrupt a declared key containing surrounding spaces.
        const std::string spelled(body.substr(0, name_end));
        const std::string name = properties && properties->is_object() &&
                                         properties->contains(spelled)
                                     ? spelled
                                     : std::string(Trim(spelled));
        body.remove_prefix(name_end + 1);
        // Decoded characters are argument data, including vocabulary token
        // spellings. Actual EOS IDs are handled by the backend before parsing.
        auto close = body.find("</parameter>");
        const bool framed_value =
            body.starts_with('\n') || body.starts_with("\r\n");
        canonical_parameters &= framed_value;
        if (framed_value) {
          // The canonical Qwen delimiter includes the following newline.
          // A line such as "</parameter> is literal" is argument data.
          auto framed = body.find("\n</parameter>");
          while (framed != std::string_view::npos) {
            const auto tail =
                body.substr(framed + std::string_view("\n</parameter>").size());
            // Compact legacy headers also admit compact closers. A canonical
            // header uses the complete delimiter: apparent tags on the same
            // line remain argument data, as in llama.cpp.
            if (tail.starts_with('\n') || tail.starts_with("\r\n") ||
                (!canonical_parameters &&
                 (Trim(tail).starts_with("</function>") ||
                  Trim(tail).starts_with("<parameter="))))
              break;
            framed = body.find("\n</parameter>", framed + 1);
          }
          close = framed == std::string_view::npos ? framed : framed + 1;
          // Never recover a nested example from inside an unfinished value.
          cursor = close == std::string_view::npos
                       ? text.size()
                       : static_cast<std::size_t>(body.data() - text.data()) +
                             close + std::string_view{"</parameter>"}.size();
        }
        if (close == std::string_view::npos || name.empty()) {
          valid = false;
          break;
        }
        const auto nested = body.find(start);
        if (!body.starts_with('\n') && !body.starts_with("\r\n") &&
            nested < close &&
            Trim(body.substr(nested + start.size()))
                .starts_with("<function=")) {
          // Recover an unframed, unterminated legacy call at the next complete
          // opener. A literal <tool_call> alone is always argument data.
          valid = false;
          break;
        }
        const auto value = StripFramingNewlines(body.substr(0, close));
        const auto* property = properties ? properties->find(name) : nullptr;
        const bool string_allowed =
            !property ||
            SchemaAccepts(*schema, *property, json::Value(std::string(value)));
        // Prefer text if the schema permits it; parsing ambiguous scalars
        // as JSON would silently change a caller's declared string type.
        // A union also admitting other types tries those first, as
        // llama.cpp's qwen3-coder parser does.
        std::optional<json::Value> typed;
        if (string_allowed && property) {
          typed = TryParseJson(Trim(value));
          if (!typed || typed->is_string() ||
              !SchemaAccepts(*schema, *property, *typed))
            typed.reset();
        }
        const bool is_string = string_allowed && !typed;
        std::string raw(is_string ? value : Trim(value));
        if (!is_string && !typed) {
          auto parsed = TryParseJson(raw);
          if (parsed && !SchemaAccepts(*schema, *property, *parsed))
            parsed.reset();
          if (!parsed) {
            auto converted = PythonLiteralsToJson(raw);
            auto alternative = TryParseJson(converted);
            if (alternative &&
                SchemaAccepts(*schema, *property, *alternative)) {
              raw = std::move(converted);
              parsed = std::move(alternative);
            }
          }
          // As in llama.cpp, a value the grammar admitted is kept when its
          // schema cannot be enforced natively (e.g. a recursive or empty
          // one). Recovery without a grammar checks the whole call below.
          if (!parsed) {
            valid = false;
            break;
          }
        }
        // Models sometimes repeat a parameter. An identical copy is harmless;
        // conflicting copies leave no safe choice.
        const auto previous = std::ranges::find(
            call.arguments, name, [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(
              {.name = name, .value = std::move(raw), .is_string = is_string});
        } else if (previous->value != raw || previous->is_string != is_string) {
          valid = false;
          break;
        }
        body.remove_prefix(close + std::string_view{"</parameter>"}.size());
      }
      complete = valid && consume("</function>") && consume(end);
    } else {
      // Recovery boundaries belong to the envelope, not quoted JSON data.
      // An unfinished string owns the remaining bytes, including tool tags.
      bool in_string = false;
      bool escaped = false;
      std::size_t close = 0;
      for (; close < body.size(); ++close) {
        const char byte = body[close];
        if (in_string) {
          if (escaped)
            escaped = false;
          else if (byte == '\\')
            escaped = true;
          else if (byte == '"')
            in_string = false;
          continue;
        }
        if (byte == '"') {
          in_string = true;
          continue;
        }
        if (body.substr(close).starts_with(end) ||
            body.substr(close).starts_with(start))
          break;
      }
      cursor = static_cast<std::size_t>(body.data() - text.data()) + close;
      if (body.substr(close).starts_with(end)) {
        cursor += end.size();
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
        }
      }
    }
    if (complete && !tools.empty() &&
        std::ranges::none_of(
            tools, [&](const auto& tool) { return tool.name == call.name; }))
      complete = false;
    if (complete && (require_schema || call_quotes.UnclosedAt(marker_begin))) {
      // An unfinished quotation is ambiguous. Restore the legacy fallback only
      // for a declared call whose complete arguments satisfy its schema.
      const auto tool =
          std::ranges::find(tools, call.name, &tokenization::ChatTool::name);
      if (tool == tools.end()) {
        complete = false;
      } else {
        try {
          const auto grammar = sampling::JsonConstraint::Compile(
              json::parse(tool->parameters_json), false);
          auto state = grammar->Start();
          for (const unsigned char byte : ArgumentsJson(call.arguments))
            state = grammar->Advance(state, byte);
          complete = grammar->Complete(state);
        } catch (const std::exception&) {
          complete = false;
        }
      }
    }
    if (complete) {
      call.id = RandomId("call_");
      calls->push_back(std::move(call));
      cursor = static_cast<std::size_t>(body.data() - text.data());
      if (spans)
        spans->emplace_back(marker_begin, cursor);
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

void ParseDsmlCalls(
    std::string_view text, ToolMarkerSet markers,
    std::vector<ParsedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr) {
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
  std::size_t envelope_begin = std::string_view::npos;
  std::string envelope_end;
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
    if (spans) {
      // Examine envelope boundaries only outside the parameter/invocation
      // ranges consumed below: an outer closing tag can be argument data.
      if (envelope_begin == std::string_view::npos) {
        std::string_view opening;
        std::size_t begin = std::string_view::npos;
        for (const auto marker : markers) {
          if (marker == "<tool_call>")
            continue;
          const auto position = text.find(marker, cursor);
          if (position < begin) {
            begin = position;
            opening = marker;
          }
        }
        // Bare invocation examples outside an outer envelope are prose.
        if (begin == std::string_view::npos)
          break;
        envelope_begin = begin;
        envelope_end =
            "</" + std::string(opening.substr(opening.find('<') + 1));
        cursor = begin + opening.size();
        continue;
      } else if (const auto end = text.find(envelope_end, cursor);
                 end < invoke_start) {
        cursor = end + envelope_end.size();
        spans->emplace_back(envelope_begin, cursor);
        envelope_begin = std::string_view::npos;
        continue;
      }
    }
    if (invoke_start == std::string_view::npos) {
      break;
    }
    const std::size_t tag_end = text.find('>', invoke_start);
    if (tag_end == std::string_view::npos) {
      break;
    }
    ParsedToolCall call;
    call.id = RandomId("call_");
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    bool valid = !call.name.empty();
    std::size_t protected_end = tag_end + 1;
    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < text.size()) {
      auto remainder = Trim(text.substr(parameter_cursor));
      parameter_cursor =
          static_cast<std::size_t>(remainder.data() - text.data());
      if (!remainder.starts_with(kParameterStarts[syntax])) {
        break;
      }
      const std::size_t parameter_start = parameter_cursor;
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      std::size_t parameter_end =
          text.find(kParameterEnds[syntax], parameter_tag_end);
      if (parameter_tag_end == std::string_view::npos ||
          parameter_end == std::string_view::npos) {
        protected_end = text.size();
        valid = false;
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      if (parameter_name && !parameter_name->empty() &&
          (!string_value || *string_value == "true" ||
           *string_value == "false")) {
        const bool is_string = string_value.value_or("true") == "true";
        // JSON strings may contain DSML delimiters. Find the end of the JSON
        // value before interpreting the enclosing markup.
        while (
            !is_string && parameter_end != std::string_view::npos &&
            !TryParseJson(Trim(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))))
          parameter_end =
              text.find(kParameterEnds[syntax],
                        parameter_end + kParameterEnds[syntax].size());
        if (parameter_end == std::string_view::npos) {
          protected_end = text.size();
          valid = false;
          break;
        }
        const auto raw = text.substr(parameter_tag_end + 1,
                                     parameter_end - parameter_tag_end - 1);
        tokenization::ChatMessage::ToolArgument argument{
            .name = *parameter_name,
            .value = std::string(is_string ? raw : Trim(raw)),
            .is_string = is_string,
        };
        // As for Qwen calls: drop an identical repeat and reject a
        // conflicting one instead of letting the last value win.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(std::move(argument));
        } else if (previous->value != argument.value ||
                   previous->is_string != argument.is_string) {
          valid = false;
        }
      } else {
        valid = false;
      }
      parameter_cursor = parameter_end + kParameterEnds[syntax].size();
      protected_end = parameter_cursor;
    }
    const bool complete =
        text.substr(parameter_cursor).starts_with(kInvokeEnds[syntax]);
    if (valid && complete) {
      calls->push_back(std::move(call));
    }
    cursor = complete ? parameter_cursor + kInvokeEnds[syntax].size()
                      : protected_end;
  }
}

ParsedGeneration ParseGeneration(
    std::string_view raw,
    TextGenerationBackend::InitialOutputState initial_output_state,
    std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required,
    ToolMarkerSet markers, ToolCloserSet closers, QuoteTracker& quotes) {
  quotes.Reset(raw);
  ParsedGeneration parsed;
  std::string_view content = raw;
  std::string_view reasoning_call_context;
  bool unfinished_reasoning_quote = false;
  const bool recognize_tools =
      !tools.empty() && choice != ChatRequest::ToolChoice::kNone;
  const auto tool_marker = [recognize_tools, markers,
                            &quotes](std::string_view text) {
    quotes.Reset(text);
    return recognize_tools ? EarliestMarker(text, markers, quotes)
                           : std::string_view::npos;
  };

  if (initial_output_state ==
      TextGenerationBackend::InitialOutputState::kReasoning) {
    if (content.starts_with(kThinkStart)) {
      content.remove_prefix(kThinkStart.size());
    }
    std::size_t think_end = content.find(kThinkEnd);
    if (tool_marker(content) < think_end)
      think_end = std::string_view::npos;
    if (think_end == std::string_view::npos) {
      const auto marker = tool_marker(content);
      parsed.reasoning_content = std::string(Trim(content.substr(0, marker)));
      if (marker == std::string_view::npos) {
        if (enforce_required && choice == ChatRequest::ToolChoice::kRequired)
          throw TextGenerationError(
              TextGenerationErrorCode::kToolChoiceUnsatisfied,
              "model did not produce a declared tool call");
        return parsed;
      }
      reasoning_call_context = content;
      unfinished_reasoning_quote = quotes.UnclosedAt(marker);
      parsed.text = std::string(content.substr(marker));
    } else {
      parsed.reasoning_content =
          std::string(Trim(content.substr(0, think_end)));
      content.remove_prefix(think_end + kThinkEnd.size());
      parsed.text = std::string(AfterReasoningSeparator(content, markers));
    }
  } else if (initial_output_state ==
             TextGenerationBackend::InitialOutputState::kAuto) {
    // Only the initial phase has reasoning markup semantics. A literal tag
    // inside an argument (or quoted ordinary text) must never be stripped.
    const auto leading = content.find_first_not_of(" \t\r\n");
    const std::size_t think_start =
        leading != std::string_view::npos &&
                content.substr(leading).starts_with(kThinkStart)
            ? leading
            : std::string_view::npos;
    if (think_start != std::string_view::npos) {
      const std::size_t think_content_start = think_start + kThinkStart.size();
      std::size_t think_end = content.find(kThinkEnd, think_content_start);
      const auto marker = tool_marker(content.substr(think_content_start));
      if (marker != std::string_view::npos &&
          think_content_start + marker < think_end)
        think_end = std::string_view::npos;
      if (think_end != std::string_view::npos) {
        parsed.reasoning_content = std::string(Trim(content.substr(
            think_content_start, think_end - think_content_start)));
        std::string_view remaining =
            content.substr(think_end + kThinkEnd.size());
        remaining = AfterReasoningSeparator(remaining, markers);
        if (think_start > 0) {
          parsed.text = std::string(content.substr(0, think_start)) +
                        std::string(remaining);
        } else {
          parsed.text = std::string(remaining);
        }
      } else {
        const auto remaining = content.substr(think_content_start);
        const auto marker = tool_marker(remaining);
        parsed.reasoning_content =
            std::string(Trim(remaining.substr(0, marker)));
        parsed.text = std::string(content.substr(0, think_start));
        if (marker != std::string_view::npos) {
          reasoning_call_context = remaining;
          unfinished_reasoning_quote = quotes.UnclosedAt(marker);
          parsed.text += remaining.substr(marker);
        }
      }
    } else {
      parsed.text = std::string(content);
    }
  } else {
    parsed.text = std::string(content);
  }

  quotes.Reset(parsed.text);
  const std::size_t marker = EarliestMarker(parsed.text, markers, quotes);
  if (choice != ChatRequest::ToolChoice::kNone && !tools.empty() &&
      marker != std::string_view::npos) {
    const auto text_from_tools = std::string_view(parsed.text).substr(marker);
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    // The outer envelope selects the format, as in llama.cpp's model parsers.
    // Scanning both dialects would turn a literal call inside an argument into
    // an additional API invocation.
    if (text_from_tools.starts_with("<tool_call>"))
      ParseQwenCalls(parsed.text, tools, &parsed.tool_calls, nullptr,
                     unfinished_reasoning_quote);
    else
      ParseDsmlCalls(text_from_tools, markers, &parsed.tool_calls, &spans);
    std::erase_if(parsed.tool_calls, [&](const auto& call) {
      return std::ranges::none_of(
          tools, [&](const auto& tool) { return tool.name == call.name; });
    });
    if (parsed.tool_calls.empty() && unfinished_reasoning_quote &&
        choice != ChatRequest::ToolChoice::kRequired) {
      // The tentative implicit end of reasoning was an invalid quoted example.
      // Preserve it in its original phase, including a later explicit boundary.
      const auto end = reasoning_call_context.find(kThinkEnd);
      if (end != std::string_view::npos) {
        parsed = ParseGeneration(
            reasoning_call_context.substr(end + kThinkEnd.size()),
            TextGenerationBackend::InitialOutputState::kContent, tools, choice,
            enforce_required, markers, closers, quotes);
      } else {
        parsed.text.clear();
      }
      parsed.reasoning_content =
          std::string(reasoning_call_context.substr(0, end));
      return parsed;
    }
    if (parsed.tool_calls.empty() && quotes.UnclosedAt(marker) &&
        choice != ChatRequest::ToolChoice::kRequired) {
      return parsed;
    }
    // An explicit stop can interrupt a call before its closing tags. Keep
    // complete calls, but do not expose an unfinished call as ordinary text.
    if (!parsed.tool_calls.empty() || !enforce_required) {
      // Text outside complete envelopes is content. Framing echoed directly
      // after a call is removed, as for Qwen calls (#383).
      const std::string full = std::move(parsed.text);
      std::string text(ContentBefore(full, 0, marker, closers, quotes, tools));
      std::size_t cursor = marker;
      for (const auto& [begin, end] : spans) {
        text += ContentBefore(full, cursor, marker + begin, closers, quotes,
                              tools, cursor > marker);
        cursor = marker + end;
      }
      if (!spans.empty()) {
        auto unfinished = EarliestMarker(full, markers, quotes, cursor);
        if (unfinished == std::string_view::npos && !enforce_required &&
            choice == ChatRequest::ToolChoice::kRequired)
          if (const auto held =
                  HeldMarkerPrefix(full, full.size(), markers, quotes))
            unfinished = full.size() - held;
        text += ContentBefore(full, cursor, unfinished, closers, quotes, tools,
                              true);
      }
      parsed.text = std::move(text);
      parsed.hide_tool_markup = true;
    }
  } else if (recognize_tools) {
    // No marker: the text is content, but the framing it carries still goes.
    // The streamed response drops it as the turn ends, and both transports
    // must report the same content.
    parsed.text = std::string(ContentBefore(parsed.text, 0, parsed.text.size(),
                                            closers, quotes, tools));
  }
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired &&
      parsed.tool_calls.empty())
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not produce a declared tool call");
  return parsed;
}

ParsedGeneration ParseStructuredGeneration(
    std::string_view raw, TextGenerationBackend::InitialOutputState initial,
    std::span<const tokenization::ChatTool> tools,
    ChatRequest::ToolChoice choice, bool enforce_required, bool tool_only,
    ToolMarkerSet markers, ToolCloserSet closers, QuoteTracker& quotes) {
  ParsedGeneration parsed;
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning) {
    if (raw.starts_with("<think>"))
      raw.remove_prefix(7);
    const auto end = raw.find("</think>");
    quotes.Reset(raw);
    const auto call =
        !tools.empty() && choice != ChatRequest::ToolChoice::kNone
            ? ReasoningToolMarker(raw, ReasoningToolMarkers(markers), quotes)
            : std::string_view::npos;
    if (call < end) {
      parsed.reasoning_content = std::string(raw.substr(0, call));
      raw.remove_prefix(call);
    } else {
      parsed.reasoning_content = std::string(raw.substr(0, end));
      if (end == std::string_view::npos)
        return parsed;
      raw.remove_prefix(end + 8);
      raw = AfterReasoningSeparator(raw, markers);
    }
  }
  const auto content = tool_only ? raw : Trim(raw);
  quotes.Reset(content);
  if (const auto marker = EarliestMarker(content, markers, quotes);
      !tools.empty() && choice != ChatRequest::ToolChoice::kNone &&
      (tool_only || marker == 0) && marker != std::string_view::npos &&
      !content.substr(marker).starts_with("<tool_call>")) {
    auto native = ParseGeneration(
        content, TextGenerationBackend::InitialOutputState::kContent, tools,
        choice, enforce_required, markers, closers, quotes);
    native.reasoning_content = std::move(parsed.reasoning_content);
    return native;
  }
  constexpr std::string_view tool_start = "<tool_call>";
  auto marker = tool_only ? EarliestMarker(content, markers, quotes) : 0;
  if (tool_only && choice == ChatRequest::ToolChoice::kRequired &&
      marker == std::string_view::npos) {
    if (const auto held =
            HeldMarkerPrefix(content, content.size(), markers, quotes)) {
      marker = content.size() - held;
    }
  }
  if (!tools.empty() && choice != ChatRequest::ToolChoice::kNone &&
      !content.empty() && marker != std::string_view::npos &&
      (tool_only || content.starts_with(tool_start) ||
       ToolMarkerPrefix(content, markers))) {
    ParsedGeneration call;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    ParseQwenCalls(content, tools, &call.tool_calls,
                   tool_only ? &spans : nullptr);
    if (tool_only) {
      std::size_t cursor = 0;
      for (const auto& [begin, end] : spans) {
        call.text += ContentBefore(content, cursor, begin, closers, quotes,
                                   tools, cursor > 0);
        cursor = end;
      }
      const auto tail = content.substr(cursor);
      auto unfinished = EarliestMarker(content, markers, quotes, cursor);
      if (unfinished != std::string_view::npos) {
        unfinished -= cursor;
      } else if (!enforce_required &&
                 choice == ChatRequest::ToolChoice::kRequired) {
        if (const auto held =
                HeldMarkerPrefix(content, content.size(), markers, quotes)) {
          unfinished = tail.size() - held;
        }
      }
      const std::size_t limit = spans.empty() ? marker : unfinished;
      const std::size_t stop = limit == std::string_view::npos
                                   ? content.size()
                                   : std::min(cursor + limit, content.size());
      // The model repeats the framing of the call it just made; that echo is
      // markup, not prose the client should replay on the next turn (#383).
      call.text += ContentBefore(content, cursor, stop, closers, quotes, tools,
                                 cursor > 0);
    }
    if (call.tool_calls.empty() && marker != std::string_view::npos &&
        quotes.UnclosedAt(marker) &&
        choice != ChatRequest::ToolChoice::kRequired) {
      parsed.text = std::string(raw);
      return parsed;
    }
    call.hide_tool_markup = true;
    if (call.tool_calls.empty() && enforce_required &&
        choice == ChatRequest::ToolChoice::kRequired)
      throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                                "model did not complete a declared tool call");
    call.reasoning_content = std::move(parsed.reasoning_content);
    return call;
  }
  // Markers in JSON strings are data. Only the initial reasoning phase has
  // markup semantics; ordinary tool/reasoning parsing must not run here.
  if (enforce_required && choice == ChatRequest::ToolChoice::kRequired)
    throw TextGenerationError(TextGenerationErrorCode::kToolChoiceUnsatisfied,
                              "model did not complete a declared tool call");
  // A constrained turn that produced no call is still reported without
  // framing: the streamed response drops it as the turn ends, and both
  // transports must report the same content.
  parsed.text = tool_only ? std::string(ContentBefore(raw, 0, raw.size(),
                                                      closers, quotes, tools))
                          : std::string(raw);
  return parsed;
}

const char* FinishReason(const TextGenerationBackend::Result& result,
                         bool has_tool_calls) {
  if (result.finish_reason ==
      TextGenerationBackend::FinishReason::kStopSequence)
    return "stop";
  if (result.finish_reason == TextGenerationBackend::FinishReason::kLength) {
    return "length";
  }
  if (has_tool_calls) {
    return "tool_calls";
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
  metrics["draft_rounds"] = result.draft_rounds;
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
  metrics["shared_prefix_wait_ms"] = result.shared_prefix_wait_ms;
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
      EmitCallback emit_piece, bool structured, bool tool_only,
      bool recognize_tools, ToolMarkerSet markers, ToolCloserSet closers,
      std::span<const tokenization::ChatTool> tools)
      : emit_piece_(std::move(emit_piece)),
        raw_content_(structured || tool_only),
        tool_only_(tool_only),
        recognize_tools_(recognize_tools),
        markers_(markers),
        closers_(closers),
        tools_(tools) {
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
    quotes_.Append(piece);
    if (tool_mode_) {
      hidden_.append(piece);
      return true;
    }
    pending_.append(piece);
    if (raw_content_ && state_ != State::kThinking &&
        !trim_reasoning_separator_)
      return StructuredContent();

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
      const auto offset = pending_offset();
      const auto think_found = raw_.find(kThinkEnd, offset);
      const std::size_t end_pos = think_found == std::string::npos
                                      ? std::string::npos
                                      : think_found - offset;
      const auto thinking_markers =
          raw_content_ ? ReasoningToolMarkers(markers_) : markers_;
      const auto found =
          !recognize_tools_ ? std::string::npos
          : raw_content_
              ? ReasoningToolMarker(raw_, thinking_markers, quotes_, offset)
              : EarliestMarker(raw_, thinking_markers, quotes_, offset);
      const std::size_t marker =
          found == std::string::npos ? std::string::npos : found - offset;
      if (marker < end_pos) {
        if (marker > 0 && !Emit(pending_.substr(0, marker), true))
          return false;
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }
      if (end_pos != std::string::npos) {
        if (end_pos > 0 && !Emit(pending_.substr(0, end_pos), true)) {
          return false;
        }
        std::string_view remaining = pending_;
        remaining.remove_prefix(end_pos + kThinkEnd.size());
        pending_ = std::string(remaining);
        state_ = State::kContent;
        quotes_.Reset(raw_, offset + end_pos + kThinkEnd.size());
        trim_reasoning_separator_ = true;
      } else {
        std::size_t held = !recognize_tools_
                               ? 0
                               : HeldMarkerPrefix(raw_, raw_.size(),
                                                  thinking_markers, quotes_);
        held = std::min(held, pending_.size());
        for (std::size_t len = std::min(pending_.size(), kThinkEnd.size() - 1);
             len > 0; --len) {
          if (kThinkEnd.starts_with(pending_.substr(pending_.size() - len))) {
            held = std::max(held, len);
            break;
          }
        }
        const std::size_t ready = pending_.size() - held;
        if (ready > 0 && !Emit(pending_.substr(0, ready), true)) {
          return false;
        }
        pending_.erase(0, ready);
        return true;
      }
    }

    if (state_ == State::kContent) {
      if (trim_reasoning_separator_) {
        const auto content = AfterReasoningSeparator(pending_, markers_);
        if (content.empty()) {
          pending_.clear();
          return true;
        }
        pending_.erase(0, pending_.size() - content.size());
        trim_reasoning_separator_ = false;
      }
      if (raw_content_) {
        return StructuredContent();
      }
      const auto offset = pending_offset();
      const auto found = recognize_tools_
                             ? EarliestMarker(raw_, markers_, quotes_, offset)
                             : std::string_view::npos;
      const std::size_t marker =
          found == std::string::npos ? std::string::npos : found - offset;
      if (marker != std::string::npos) {
        const auto prose =
            recognize_tools_
                ? ContentBefore(raw_, offset, found, closers_, quotes_, tools_)
                : std::string_view(pending_).substr(0, marker);
        if (!prose.empty() && !Emit(prose, false)) {
          return false;
        }
        hidden_ = pending_.substr(marker);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }

      const std::size_t held =
          recognize_tools_
              ? std::min(std::max(HeldMarkerPrefix(raw_, raw_.size(), markers_,
                                                   quotes_),
                                  FramingHold(raw_, raw_.size(), offset,
                                              closers_, quotes_, tools_)),
                         pending_.size())
              : 0;
      const std::size_t ready = pending_.size() - held;
      if (ready > 0 && !Emit(pending_.substr(0, ready), false)) {
        return false;
      }
      pending_.erase(0, ready);
      return true;
    }

    return true;
  }

  bool Finish(bool hide_tool_markup, std::string_view content = {},
              std::string_view reasoning = {}) {
    if (tool_mode_ && state_ == State::kThinking) {
      const auto tail = reasoning.substr(
          std::min(emitted_reasoning_bytes_, reasoning.size()));
      return (tail.empty() || Emit(tail, true)) &&
             (content.empty() || Emit(content, false));
    }
    if (tool_only_ && tool_mode_ && hide_tool_markup) {
      // The prefix was streamed before the first call. Preserve prose between
      // or after complete calls without exposing their markup or partial calls.
      const auto offset = std::min(emitted_content_bytes_, content.size());
      const auto tail = content.substr(offset);
      return tail.empty() || Emit(tail, false);
    }
    if (raw_content_ && !structured_started_ && hide_tool_markup) {
      pending_.clear();
      return true;
    }
    if (!tool_mode_) {
      if (!pending_.empty()) {
        // With tools declared, the closing tags the model repeats after a call
        // are markup; a client must never store them as assistant prose (#383).
        const auto offset = pending_offset();
        const auto text = recognize_tools_ && state_ == State::kContent
                              ? ContentBefore(raw_, offset, raw_.size(),
                                              closers_, quotes_, tools_)
                              : std::string_view(pending_);
        const bool is_reasoning = (state_ == State::kThinking);
        const bool emitted = Emit(text, is_reasoning);
        pending_.clear();
        if (!emitted) {
          return false;
        }
      }
      return true;
    }
    if (!hide_tool_markup && !hidden_.empty()) {
      return Emit(hidden_, false);
    }
    return true;
  }

  [[nodiscard]] std::string_view raw() const noexcept { return raw_; }

  // `pending_` always holds a suffix of what the model generated, so the
  // context of a marker inside it is decided against the whole response.
  [[nodiscard]] std::size_t pending_offset() const noexcept {
    return raw_.size() - pending_.size();
  }

private:
  bool Emit(std::string_view piece, bool reasoning) {
    if (!emit_piece_(piece, reasoning))
      return false;
    if (reasoning)
      emitted_reasoning_bytes_ += piece.size();
    return true;
  }

  bool StructuredContent() {
    if (tool_only_) {
      const auto offset = pending_offset();
      const auto found = EarliestMarker(raw_, markers_, quotes_, offset);
      const auto start =
          found == std::string::npos ? std::string::npos : found - offset;
      if (start != std::string::npos) {
        const auto prose =
            ContentBefore(raw_, offset, found, closers_, quotes_, tools_);
        if (!prose.empty() && !Emit(prose, false)) {
          return false;
        }
        emitted_content_bytes_ += prose.size();
        hidden_ = pending_.substr(start);
        pending_.clear();
        tool_mode_ = true;
        return true;
      }
      const auto held = std::min(
          std::max(HeldMarkerPrefix(raw_, raw_.size(), markers_, quotes_),
                   FramingHold(raw_, raw_.size(), offset, closers_, quotes_,
                               tools_)),
          pending_.size());
      const auto ready = pending_.size() - held;
      if (ready) {
        const auto text = ContentBefore(raw_, offset, offset + ready, closers_,
                                        quotes_, tools_);
        if (!text.empty() && !Emit(text, false)) {
          return false;
        }
        emitted_content_bytes_ += text.size();
      }
      pending_.erase(0, ready);
      return true;
    }
    if (!structured_started_) {
      const auto content = Trim(pending_);
      if (content.empty() || (ToolMarkerPrefix(content, markers_) &&
                              EarliestMarker(content, markers_, quotes_) != 0))
        return true;
      structured_started_ = true;
      if (EarliestMarker(content, markers_, quotes_) == 0) {
        tool_mode_ = true;
        hidden_ = std::exchange(pending_, {});
        return true;
      }
    }
    auto text = std::exchange(pending_, {});
    return text.empty() || Emit(text, false);
  }
  enum class State : std::uint8_t {
    kInitial,
    kThinking,
    kContent,
  };

  EmitCallback emit_piece_;
  bool structured_started_{false};
  core::Utf8Decoder decoder_;
  std::string raw_;
  std::string pending_;
  std::string hidden_;
  State state_{State::kInitial};
  bool tool_mode_{false};
  bool raw_content_{false};
  bool tool_only_{false};
  bool recognize_tools_{false};
  ToolMarkerSet markers_;
  QuoteTracker quotes_;
  ToolCloserSet closers_;
  std::span<const tokenization::ChatTool> tools_;
  std::size_t emitted_content_bytes_{0};
  std::size_t emitted_reasoning_bytes_{0};
  bool trim_reasoning_separator_{false};
};

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Build the same output items for streaming and buffered responses.
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, HttpResponse::BodyWriter writer,
                  const ChatRequest& chat)
      : writer_(std::move(writer)), namespaces_(chat.tool_namespaces) {
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
    response_["parallel_tool_calls"] =
        !chat.tools.empty() && chat.parallel_tool_calls;
    response_["tool_choice"] =
        chat.tools.empty() || chat.tool_choice == ChatRequest::ToolChoice::kNone
            ? "none"
        : chat.tool_choice == ChatRequest::ToolChoice::kRequired ? "required"
                                                                 : "auto";
    if (!chat.forced_tool_name.empty()) {
      response_["tool_choice"] = json::Value::object();
      response_["tool_choice"]["type"] = "function";
      response_["tool_choice"]["name"] = chat.forced_tool_name;
    }
    response_["tools"] = json::Value::array();
    for (const auto& tool : chat.tools) {
      const auto definition = json::parse(tool.definition_json);
      auto function = *definition.find("function");
      function["type"] = "function";
      response_["tools"].push_back(std::move(function));
    }
  }

  bool Begin() {
    return Lifecycle("response.created") && Lifecycle("response.in_progress");
  }

  bool Progress(const TextGenerationBackend::PromptProgress& progress) {
    auto event = json::Value::object();
    event["type"] = "response.in_progress";
    event["response"] = response_;
    event["prompt_progress"] = PromptProgressJson(progress);
    return Emit(std::move(event));
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

  bool Tool(const ParsedToolCall& call) {
    if (!CloseItem("completed"))
      return false;
    auto item = json::Value::object();
    item["id"] = RandomId("fc_");
    item["type"] = "function_call";
    item["call_id"] = call.id;
    item["name"] = call.name;
    if (const auto it = namespaces_.find(call.name); it != namespaces_.end())
      item["namespace"] = it->second;
    item["arguments"] = "";
    item["status"] = "in_progress";
    auto added = IndexedEvent("response.output_item.added");
    added["item"] = item;
    if (!Emit(std::move(added)))
      return false;
    const auto arguments = ArgumentsJson(call.arguments);
    auto delta = IndexedEvent("response.function_call_arguments.delta");
    delta["item_id"] = item.member_str("id");
    delta["delta"] = arguments;
    if (!Emit(std::move(delta)))
      return false;
    auto done = IndexedEvent("response.function_call_arguments.done");
    done["item_id"] = item.member_str("id");
    done["name"] = call.name;
    done["arguments"] = arguments;
    if (!Emit(std::move(done)))
      return false;
    item["arguments"] = arguments;
    item["status"] = "completed";
    auto closed = IndexedEvent("response.output_item.done");
    closed["item"] = item;
    response_["output"].push_back(std::move(item));
    ++output_index_;
    return Emit(std::move(closed));
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
  std::map<std::string, std::string> namespaces_;
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
  const auto format = generation->ToolFormat();
  const auto markers = ToolMarkers(format);
  const auto closers = ToolClosers(format);
  core::Utf8Decoder decoder;
  const auto decoded = decoder.Push(result.text, true);
  QuoteTracker quotes;
  const ParsedGeneration generated =
      request.chat.response_format || request.chat.constrained_tools
          ? ParseStructuredGeneration(
                decoded, initial_output_state, request.chat.tools,
                request.chat.tool_choice,
                result.finish_reason ==
                    TextGenerationBackend::FinishReason::kStop,
                request.chat.constrained_tools && !request.chat.response_format,
                markers, closers, quotes)
          : ParseGeneration(decoded, initial_output_state, request.chat.tools,
                            request.chat.tool_choice,
                            result.finish_reason ==
                                TextGenerationBackend::FinishReason::kStop,
                            markers, closers, quotes);

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
  const auto format = generation->ToolFormat();
  const auto markers = ToolMarkers(format);
  const auto closers = ToolClosers(format);
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
           initial_output_state, markers, closers,
           stream_log](const HttpResponse::BodyWriter& writer) {
            bool connected = true;
            bool started = false;
            const auto begin = [&] {
              if (!started) {
                started = true;
                json::Value role_delta = json::Value::object();
                role_delta["role"] = "assistant";
                connected = writer(Sse(
                    ChoiceChunk(id, created, model, std::move(role_delta))));
              }
              return connected;
            };
            if (request.chat.return_progress && !begin()) {
              generation->Cancel();
              return;
            }
            QuoteTracker quotes;
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
                },
                request.chat.response_format != nullptr,
                request.chat.constrained_tools && !request.chat.response_format,
                !request.chat.tools.empty() &&
                    request.chat.tool_choice != ChatRequest::ToolChoice::kNone,
                markers, closers, request.chat.tools);

            TextGenerationBackend::ProgressCallback on_progress;
            if (request.chat.return_progress) {
              on_progress =
                  [&](const TextGenerationBackend::PromptProgress& progress) {
                    auto chunk =
                        ChoiceChunk(id, created, model, json::Value::object());
                    chunk["prompt_progress"] = PromptProgressJson(progress);
                    connected = begin() && writer(Sse(chunk));
                    return connected;
                  };
            }

            try {
              const auto result = generation->Wait(
                  [&](std::string_view piece) {
                    return begin() && filter.Push(piece);
                  },
                  on_progress, begin);
              stream_log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (!connected || result.cancelled) {
                return;
              }
              if (!begin())
                return;
              if (!filter.Push({}, true))
                return;

              const ParsedGeneration generated =
                  request.chat.response_format || request.chat.constrained_tools
                      ? ParseStructuredGeneration(
                            filter.raw(), initial_output_state,
                            request.chat.tools, request.chat.tool_choice,
                            result.finish_reason ==
                                TextGenerationBackend::FinishReason::kStop,
                            request.chat.constrained_tools &&
                                !request.chat.response_format,
                            markers, closers, quotes)
                      : ParseGeneration(
                            filter.raw(), initial_output_state,
                            request.chat.tools, request.chat.tool_choice,
                            result.finish_reason ==
                                TextGenerationBackend::FinishReason::kStop,
                            markers, closers, quotes);
              if (!filter.Finish(generated.hide_tool_markup, generated.text,
                                 generated.reasoning_content)) {
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
              if (!started)
                throw;
              stream_log->error_code = exception.stable_code();
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = exception.what();
              detail["type"] = "server_error";
              detail["code"] = exception.stable_code();
              error["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(error));
              (void)writer("data: [DONE]\n\n");
            } catch (const std::exception& error) {
              if (!started)
                throw;
              stream_log->error_code = "generation_failed";
              json::Value err = json::Value::object();
              json::Value detail = json::Value::object();
              const char* message = error.what();
              detail["message"] =
                  message && *message ? message : "generation failed";
              detail["type"] = "server_error";
              detail["code"] = "generation_failed";
              err["error"] = std::move(detail);
              stream_log->error_event_sent = writer(Sse(err));
              (void)writer("data: [DONE]\n\n");
            }
          },
      .stream_log = std::move(stream_log),
      .defer_stream_headers = !request.chat.return_progress,
  };
}

}  // namespace

bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error) {
  auto converted = item;
  const auto kind = item.member_str("type");
  if (kind == "function_call" || kind == "function_call_output") {
    const auto* id = item.find("call_id");
    if (!id || !id->is_string() || id->str().empty()) {
      *error = "function items require a nonempty call_id";
      return false;
    }
    if (kind == "function_call") {
      tokenization::ChatMessage::ToolCall call;
      call.id = id->str();
      if (!ParseHistoricalFunction(item, &call, error))
        return false;
      message->role = tokenization::ChatRole::kAssistant;
      message->tool_calls.push_back(std::move(call));
      return true;
    }
    converted = json::Value::object();
    converted["role"] = "tool";
    converted["tool_call_id"] = id->str();
    const auto* output = item.find("output");
    if (!output || (!output->is_string() && !output->is_array())) {
      *error = "function output requires a string or content array";
      return false;
    }
    converted["content"] = *output;
  }
  if (const auto* parts = converted.find("content");
      parts && parts->is_array()) {
    auto content = json::Value::array();
    for (const auto& part : parts->items()) {
      auto value = part;
      const auto type = part.member_str("type");
      if (type == "input_image") {
        if (!part.find("image_url") || !part.find("image_url")->is_string() ||
            part.contains("file_id")) {
          *error = "input_image requires an image_url";
          return false;
        }
        value = json::Value::object();
        value["type"] = "image_url";
        value["image_url"] = json::Value::object();
        value["image_url"]["url"] = *part.find("image_url");
        if (const auto* detail = part.find("detail"))
          value["image_url"]["detail"] = *detail;
      } else if (type == "input_text" || type == "output_text")
        value["type"] = "text";
      else if (type != "text") {
        *error = "unsupported Responses input content type";
        return false;
      }
      content.push_back(std::move(value));
    }
    converted["content"] = std::move(content);
  }
  return ParseMessage(converted, message, budget, error);
}

std::optional<HttpResponse> ParseOpenAiResponseControls(const json::Value& body,
                                                        ChatRequest* chat) {
  ParsedChatRequest parsed;
  parsed.chat = *chat;
  // Responses declares host tools (namespace/web_search) that only OpenAI can
  // execute; skip them rather than reject the whole request.
  if (auto error = ParseToolControls(body, &parsed, true, true))
    return error;
  // Clients route namespaced calls by namespace and name, so keep the
  // namespace of each flattened function and echo it on its calls.
  // ParseToolControls has validated the namespace tools arrays.
  if (const auto* tools = body.find("tools"); tools && tools->is_array()) {
    for (const auto& item : tools->items()) {
      const auto name = item.member_str("name");
      if (item.member_str("type") != "namespace" || name.empty())
        continue;
      for (const auto& tool : item.find("tools")->items())
        if (tool.member_str("type") == "function")
          parsed.chat.tool_namespaces[tool.member_str("name")] = name;
    }
  }
  // Responses attempts strict normalization when strict is omitted; Chat
  // Completions keeps its best-effort default. Explicit true/false wins.
  for (auto& tool : parsed.chat.tools) {
    auto definition = json::parse(tool.definition_json);
    auto& function = definition["function"];
    const auto* strict = function.find("strict");
    if (strict && !strict->is_null())
      continue;
    try {
      auto schema = NormalizeToolSchema(json::parse(tool.parameters_json));
      sampling::JsonConstraint::Compile(schema, true);
      function["parameters"] = std::move(schema);
      function["strict"] = true;
      tool.parameters_json = function.find("parameters")->dump();
      parsed.chat.constrained_tools |=
          parsed.chat.tool_choice != ChatRequest::ToolChoice::kNone;
    } catch (const std::invalid_argument&) {
      function["strict"] = false;
    }
    tool.definition_json = definition.dump();
  }
  *chat = std::move(parsed.chat);

  if (const auto* progress = body.find("return_progress");
      progress != nullptr && !progress->is_null()) {
    if (!progress->is_bool())
      return Error(400, "Bad Request", "'return_progress' must be a boolean",
                   "invalid_return_progress");
    chat->return_progress = progress->as_bool();
  }
  if (const auto* cache = body.find("cache_prompt")) {
    if (!cache->is_bool())
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    chat->cache_prompt = cache->as_bool();
  }
  if (const auto* reasoning = body.find("reasoning");
      reasoning && !reasoning->is_null()) {
    if (!reasoning->is_object())
      return Error(400, "Bad Request", "'reasoning' must be an object",
                   "invalid_reasoning");
    for (const auto& [key, value] : reasoning->members()) {
      if (key == "summary")
        continue;  // Requests reasoning text; local reasoning is always sent.
      if (key != "effort")
        return Error(400, "Bad Request", "unsupported reasoning member: " + key,
                     "invalid_reasoning");
      if (value.is_null())
        continue;
      ReasoningOptions options;
      std::string error;
      if (!value.is_string() ||
          !AssignReasoningEffort(&options, value.str(), &error))
        return Error(
            400, "Bad Request",
            error.empty() ? "'reasoning.effort' must be a string" : error,
            "invalid_reasoning");
      if (options.enabled)
        chat->reasoning.enabled = options.enabled;
      if (options.effort)
        chat->reasoning.effort = options.effort;
    }
  }
  if (const auto* text = body.find("text"); text && !text->is_null()) {
    if (!text->is_object())
      return Error(400, "Bad Request", "'text' must be an object",
                   "invalid_response_format");
    for (const auto& [key, value] : text->members()) {
      if (key == "verbosity")
        continue;  // Verbosity has no native equivalent; accept and ignore.
      if (key != "format")
        return Error(400, "Bad Request", "unsupported text member: " + key,
                     "invalid_response_format");
      try {
        chat->response_format = ParseResponseFormat(&value, true);
        if (chat->response_format)
          chat->response_format_description = value.member_str("description");
      } catch (const std::exception& error) {
        return Error(400, "Bad Request", error.what(),
                     "invalid_response_format");
      }
    }
  }
  return {};
}

GeneratedText SplitGeneratedText(
    std::string_view text, TextGenerationBackend::InitialOutputState initial) {
  QuoteTracker quotes;
  auto generated = ParseGeneration(
      text, initial, {}, ChatRequest::ToolChoice::kNone, false, {}, {}, quotes);
  return {.reasoning = std::move(generated.reasoning_content),
          .text = std::move(generated.text)};
}

HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream) {
  const auto initial = backend.initial_output_state(chat);
  auto generation = backend.start_chat(chat, max_tokens, sampling,
                                       request.is_cancelled, stream);
  const auto format = generation->ToolFormat();
  const auto markers = ToolMarkers(format);
  const auto closers = ToolClosers(format);
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  auto timing = std::make_shared<std::string>();
  const auto run = [generation, initial, chat, markers, closers,
                    model = backend.model_id(), stream_log,
                    timing](const HttpResponse::BodyWriter& writer) {
    ResponsesOutput output(model, writer, chat);
    bool started = false;
    const auto begin = [&] {
      if (started)
        return true;
      started = true;
      if (output.Begin())
        return true;
      generation->Cancel();
      return false;
    };
    if (writer && chat.return_progress && !begin())
      return json::Value();
    QuoteTracker quotes;
    StreamingTextFilter filter(
        initial,
        [&](std::string_view piece, bool reasoning) {
          if (output.Append(piece, reasoning))
            return true;
          generation->Cancel();
          return false;
        },
        chat.response_format != nullptr,
        chat.constrained_tools && !chat.response_format,
        !chat.tools.empty() &&
            chat.tool_choice != ChatRequest::ToolChoice::kNone,
        markers, closers, chat.tools);
    try {
      TextGenerationBackend::ProgressCallback on_progress;
      if (writer && chat.return_progress) {
        on_progress =
            [&](const TextGenerationBackend::PromptProgress& progress) {
              return begin() && output.Progress(progress);
            };
      }
      const auto result = writer ? generation->Wait(
                                       [&](std::string_view piece) {
                                         return begin() && filter.Push(piece);
                                       },
                                       on_progress, begin)
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
      if (!begin())
        return json::Value();
      if (!writer)
        filter.Push(result.text);
      if (!filter.Push({}, true)) {
        generation->Cancel();
        return json::Value();
      }
      const auto generated =
          chat.response_format || chat.constrained_tools
              ? ParseStructuredGeneration(
                    filter.raw(), initial, chat.tools, chat.tool_choice,
                    result.finish_reason ==
                        TextGenerationBackend::FinishReason::kStop,
                    chat.constrained_tools && !chat.response_format, markers,
                    closers, quotes)
              : ParseGeneration(filter.raw(), initial, chat.tools,
                                chat.tool_choice,
                                result.finish_reason ==
                                    TextGenerationBackend::FinishReason::kStop,
                                markers, closers, quotes);
      if (!filter.Finish(generated.hide_tool_markup, generated.text,
                         generated.reasoning_content)) {
        generation->Cancel();
        return json::Value();
      }
      for (const auto& call : generated.tool_calls) {
        if (!output.Tool(call)) {
          generation->Cancel();
          return json::Value();
        }
      }
      return output.Complete(result);
    } catch (const std::exception& error) {
      if (!writer || !started)
        throw;
      const auto* generation_error =
          dynamic_cast<const TextGenerationError*>(&error);
      stream_log->error_code = generation_error
                                   ? generation_error->stable_code()
                                   : "generation_failed";
      const char* message = error.what();
      stream_log->error_event_sent =
          output.Fail(message && *message ? message : "generation failed");
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
            .stream_log = std::move(stream_log),
            .defer_stream_headers = !chat.return_progress};
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

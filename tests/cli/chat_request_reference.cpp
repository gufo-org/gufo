// Frozen: do not change these decoders to match the Rust one. They are the
// C++ decoders as #1351 found them, with only their backend, image and schema
// calls routed through the environment (see the header).
#include "tests/cli/chat_request_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/cli/serve/logprobs.hpp"
#include "src/cli/serve/response_format.hpp"
#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/json.hpp"

namespace gufo::server::chat_request_reference {
namespace {

thread_local const ChatDecodeEnvironment* current = nullptr;

const ChatDecodeEnvironment& Env() {
  if (current == nullptr)
    throw std::logic_error("reference decoder without an environment");
  return *current;
}

// Selects the environment for one call.
class Using {
public:
  explicit Using(const ChatDecodeEnvironment& environment)
      : previous_(current) {
    current = &environment;
  }
  ~Using() { current = previous_; }
  Using(const Using&) = delete;
  Using& operator=(const Using&) = delete;

private:
  const ChatDecodeEnvironment* previous_;
};

struct ParsedChatRequest {
  ChatRequest chat;
  std::string model;
  std::size_t max_tokens{0};
  sampling::SamplingConfig sampling;
  bool stream{false};
  bool include_usage{false};
};

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
      if ((message->role != tokenization::ChatRole::kUser &&
           message->role != tokenization::ChatRole::kTool) ||
          url == nullptr || !url->is_string()) {
        *error = "image_url requires a user or tool message and a string URL";
        return false;
      }
      // Resolution is model-owned, except that 'low' bounds it. The other
      // standard values behave as 'auto', since clients such as Codex always
      // send one; anything else is still a malformed request.
      const auto* detail = image->find("detail");
      if (detail != nullptr &&
          (!detail->is_string() ||
           (detail->get_str() != "auto" && detail->get_str() != "low" &&
            detail->get_str() != "high" && detail->get_str() != "original"))) {
        *error = "image_url.detail must be 'auto', 'low', 'high' or 'original'";
        return false;
      }
      const auto image_detail =
          detail != nullptr && detail->get_str() == "low"
              ? tokenization::ChatMessage::ImageDetail::kLow
              : tokenization::ChatMessage::ImageDetail::kAuto;
      try {
        message->images.push_back(
            {output->size(),
             std::make_shared<const std::vector<std::uint8_t>>(
                 Env().read_image(url->get_str(), budget)),
             image_detail});
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
  // Render typed values as the reference chat templates' tojson does, so a
  // replayed call matches the tokens the model generated and its continuation
  // checkpoint is reused. Numbers keep the client's spelling: the one this
  // server returned, as the model generated it. llama.cpp's Jinja runtime
  // writes floats from their double (1.0 as 1) and misses those turns.
  for (const auto& [name, value] : parsed.members()) {
    out->push_back({
        .name = name,
        .value = value.is_string()
                     ? value.get_str()
                     : value.tojson(json::Value::Numbers::kAsParsed),
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

bool ParseToolList(const json::Value* tools,
                   std::vector<tokenization::ChatTool>* output,
                   std::unordered_set<std::string>* names, std::string* error,
                   bool allow_non_function) {
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
      if (!ParseToolList(nested, output, names, error, allow_non_function))
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
    // A set keeps the check linear: only the request body size bounds the
    // number of functions.
    if (!names->insert(tool.name).second) {
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
        Env().compile_schema(*function_obj.find("parameters"));
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
    output->push_back(std::move(tool));
  }
  return true;
}

bool ParseTools(const json::Value* tools,
                std::vector<tokenization::ChatTool>* output, std::string* error,
                bool allow_non_function = false) {
  std::unordered_set<std::string> names;
  for (const auto& tool : *output)
    names.insert(tool.name);
  return ParseToolList(tools, output, &names, error, allow_non_function);
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
        "reasoning_effort must be none, minimal, low, medium, high, xhigh, or "
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

// Match llama.cpp's canonical field and its older compatibility spelling.
// An explicit canonical value takes precedence; -1 keeps the server default.
std::optional<HttpResponse> ParseReasoningBudget(const json::Value& body,
                                                 ReasoningOptions* options) {
  const char* field = "reasoning_budget_tokens";
  const auto* budget = body.find(field);
  if (budget == nullptr || budget->is_null()) {
    field = "thinking_budget_tokens";
    budget = body.find(field);
  }
  if (budget == nullptr || budget->is_null())
    return {};
  const double value = budget->is_number() ? budget->as_double() : -2.0;
  if (!std::isfinite(value) || value < -1.0 ||
      value > std::numeric_limits<std::int32_t>::max() ||
      std::floor(value) != value)
    return Error(400, "Bad Request",
                 "'" + std::string(field) +
                     "' must be an integer between -1 and 2147483647",
                 "invalid_reasoning");
  if (value >= 0.0)
    options->budget_tokens = static_cast<std::size_t>(value);
  return {};
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

std::optional<HttpResponse> ParseRequest(std::string_view request_body,
                                         ParsedChatRequest* output,
                                         bool require_model) {
  json::Value body;
  try {
    body = json::parse(request_body);
  } catch (const std::exception& exception) {
    return Error(400, "Bad Request", exception.what(), "parse_error");
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }

  if (const auto* model = body.find("model");
      model != nullptr && !model->is_null() && !model->is_string()) {
    return Error(400, "Bad Request", "'model' must be a string",
                 "invalid_model");
  }
  output->model = body.member_str("model");
  if (output->model.empty() && !require_model) {
    output->model = Env().model_id;
  }
  if (output->model.empty()) {
    return Error(400, "Bad Request", "'model' is required", "missing_model");
  }
  if (!Env().serves_model(output->model)) {
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
  // As in llama-server, null keeps the default.
  if (const auto* cache_prompt = body.find("cache_prompt");
      cache_prompt != nullptr && !cache_prompt->is_null()) {
    if (!cache_prompt->is_bool()) {
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    }
    output->chat.cache_prompt = cache_prompt->as_bool();
  }
  if (const auto* cache_persist = body.find("cache_persist");
      cache_persist != nullptr && !cache_persist->is_null()) {
    if (!cache_persist->is_bool()) {
      return Error(400, "Bad Request", "'cache_persist' must be a boolean",
                   "invalid_cache_persist");
    }
    output->chat.cache_persist = cache_persist->as_bool();
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
  image_budget.accepts_images = Env().accepts_images;
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
  if (auto error = ParseReasoningBudget(body, &output->chat.reasoning))
    return error;
  if (const auto* kwargs = body.find("chat_template_kwargs")) {
    if (const auto* vision_id = kwargs->find("add_vision_id")) {
      if (!vision_id->is_bool())
        return Error(400, "Bad Request",
                     "'chat_template_kwargs.add_vision_id' must be a boolean",
                     "invalid_template_options");
      output->chat.add_vision_id = vision_id->as_bool();
    }
  }
  const ReasoningOptions defaults = Env().reasoning_defaults;
  if (!output->chat.reasoning.enabled.has_value()) {
    output->chat.reasoning.enabled = defaults.enabled;
  }
  if (!output->chat.reasoning.effort.has_value()) {
    output->chat.reasoning.effort = defaults.effort;
  }
  if (!output->chat.reasoning.preserve_thinking.has_value()) {
    output->chat.reasoning.preserve_thinking = defaults.preserve_thinking;
  }
  if (!output->chat.reasoning.budget_tokens.has_value()) {
    output->chat.reasoning.budget_tokens = defaults.budget_tokens;
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

  const char* max_tokens_field = "max_completion_tokens";
  const json::Value* max_tokens = body.find(max_tokens_field);
  if (max_tokens == nullptr || max_tokens->is_null()) {
    max_tokens_field = "max_tokens";
    max_tokens = body.find(max_tokens_field);
  }
  if (max_tokens != nullptr && !max_tokens->is_null()) {
    const double value =
        max_tokens->is_number() ? max_tokens->as_double() : 0.0;
    if (!max_tokens->is_number() || !std::isfinite(value) ||
        std::floor(value) != value || value < 1.0 ||
        value >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
      return Error(
          400, "Bad Request",
          "'" + std::string(max_tokens_field) + "' must be a positive integer",
          "invalid_max_tokens");
    }
    output->max_tokens = max_tokens->as_size();
  }

  sampling::SamplingConfig parsed_sampling;
  if (const auto sampling_error = ParseSamplingConfig(
          body, Env().sampling_defaults.Resolve(output->chat.reasoning.enabled),
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
  if (const auto error = ParseLogprobs(body, true, &output->sampling))
    return Error(400, "Bad Request", error->message, error->code.c_str());
  for (const std::string_view unsupported : {"modalities", "audio"}) {
    if (const auto* value = body.find(std::string(unsupported));
        value != nullptr && !value->is_null()) {
      if ((unsupported == "modalities" && value->is_array() &&
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

bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error) {
  auto converted = item;
  const auto kind = item.member_str("type");
  if (kind == "function_call" || kind == "function_call_output" ||
      kind == "custom_tool_call" || kind == "custom_tool_call_output") {
    const auto* id = item.find("call_id");
    if (!id || !id->is_string() || id->str().empty()) {
      *error = "function items require a nonempty call_id";
      return false;
    }
    if (kind == "function_call" || kind == "custom_tool_call") {
      tokenization::ChatMessage::ToolCall call;
      call.id = id->str();
      json::Value custom_function;
      const auto* function = &item;
      if (kind == "custom_tool_call") {
        const auto* input = item.find("input");
        if (!input || !input->is_string()) {
          *error = "historical custom tool calls require string input";
          return false;
        }
        // Custom calls carry one free-form input. Preserve it as literal
        // argument data in the model's native tool-history representation.
        auto arguments = json::Value::object();
        arguments["input"] = *input;
        custom_function = item;
        custom_function["arguments"] = arguments.dump();
        function = &custom_function;
      }
      if (!ParseHistoricalFunction(*function, &call, error))
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
  // Like llama.cpp, retain the supplied parameter schema when strict is
  // omitted or null. Promoting optional fields to required ones forces agents
  // to invent values. ParseToolControls still validates explicit strict:true
  // and enables tool constraints regardless of strictness.
  *chat = std::move(parsed.chat);

  if (const auto* progress = body.find("return_progress");
      progress != nullptr && !progress->is_null()) {
    if (!progress->is_bool())
      return Error(400, "Bad Request", "'return_progress' must be a boolean",
                   "invalid_return_progress");
    chat->return_progress = progress->as_bool();
  }
  if (const auto* cache = body.find("cache_prompt");
      cache != nullptr && !cache->is_null()) {
    if (!cache->is_bool())
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    chat->cache_prompt = cache->as_bool();
  }
  if (const auto* persist = body.find("cache_persist");
      persist != nullptr && !persist->is_null()) {
    if (!persist->is_bool())
      return Error(400, "Bad Request", "'cache_persist' must be a boolean",
                   "invalid_cache_persist");
    chat->cache_persist = persist->as_bool();
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
  if (auto error = ParseReasoningBudget(body, &chat->reasoning))
    return error;
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

std::optional<HttpResponse> ParseAnthropicToolControls(const json::Value& body,
                                                       ChatRequest* chat) {
  const auto* tools = body.find("tools");
  const auto* choice = body.find("tool_choice");
  if (tools == nullptr && choice == nullptr)
    return {};
  // Rewrite the Messages declarations as Chat tools so validation, schema
  // constraints and tool framing stay on the single Chat tool path.
  json::Value converted = json::Value::object();
  if (tools != nullptr) {
    if (!tools->is_array())
      return Error(400, "Bad Request", "'tools' must be an array",
                   "invalid_tools");
    auto functions = json::Value::array();
    for (const auto& tool : tools->items()) {
      if (!tool.is_object() || tool.member_str("type", "custom") != "custom" ||
          !tool.contains("input_schema"))
        return Error(400, "Bad Request",
                     "only custom tools with an input_schema are supported",
                     "invalid_tools");
      json::Value function = json::Value::object();
      for (const auto* key : {"name", "description"})
        if (const auto* value = tool.find(key))
          function[key] = *value;
      function["parameters"] = *tool.find("input_schema");
      json::Value wrapped = json::Value::object();
      wrapped["type"] = "function";
      wrapped["function"] = std::move(function);
      functions.push_back(std::move(wrapped));
    }
    converted["tools"] = std::move(functions);
  }
  if (choice != nullptr) {
    const auto type = choice->is_object() ? choice->member_str("type") : "";
    if (type == "auto" || type == "none") {
      converted["tool_choice"] = type;
    } else if (type == "any") {
      converted["tool_choice"] = "required";
    } else if (type == "tool") {
      json::Value forced = json::Value::object();
      forced["type"] = "function";
      forced["function"] = json::Value::object();
      if (const auto* name = choice->find("name"))
        forced["function"]["name"] = *name;
      converted["tool_choice"] = std::move(forced);
    } else {
      return Error(400, "Bad Request",
                   "'tool_choice.type' must be auto, any, tool or none",
                   "invalid_tool_choice");
    }
    if (const auto* single = choice->find("disable_parallel_tool_use")) {
      if (!single->is_bool())
        return Error(400, "Bad Request",
                     "'disable_parallel_tool_use' must be a boolean",
                     "invalid_tools");
      if (type != "tool")
        converted["parallel_tool_calls"] = !single->as_bool();
    }
  }
  ParsedChatRequest parsed;
  parsed.chat = *chat;
  if (auto error = ParseToolControls(converted, &parsed))
    return error;
  *chat = std::move(parsed.chat);
  return {};
}

bool ParseAnthropicImage(const json::Value& block,
                         tokenization::ChatMessage* message,
                         core::ImageReadBudget& budget, std::string* error) {
  const auto* source = block.find("source");
  const auto type = source != nullptr && source->is_object()
                        ? source->member_str("type")
                        : std::string();
  std::string url;
  if (type == "base64") {
    const auto* media = source->find("media_type");
    const auto* data = source->find("data");
    if (media == nullptr || !media->is_string() || data == nullptr ||
        !data->is_string()) {
      *error = "base64 image sources require string media_type and data";
      return false;
    }
    url = "data:" + media->str() + ";base64," + data->str();
  } else if (type == "url") {
    const auto* spelled = source->find("url");
    if (spelled == nullptr || !spelled->is_string()) {
      *error = "url image sources require a string url";
      return false;
    }
    url = spelled->str();
  } else {
    *error = "image blocks require a base64 or url source";
    return false;
  }
  try {
    message->images.push_back(
        {message->content.size(),
         std::make_shared<const std::vector<std::uint8_t>>(
             Env().read_image(url, budget))});
  } catch (const std::exception& exception) {
    *error = exception.what();
    return false;
  }
  return true;
}

bool ParseAnthropicToolMessage(const json::Value& item,
                               std::vector<tokenization::ChatMessage>* messages,
                               core::ImageReadBudget& budget,
                               std::string* error) {
  const auto role = item.member_str("role");
  const auto* content = item.find("content");
  if (!item.is_object() || (role != "user" && role != "assistant") ||
      content == nullptr || !content->is_array()) {
    *error = "tool blocks require a user or assistant content array";
    return false;
  }
  const auto read_text = [&](const json::Value& part, std::string* out) {
    const auto* text = part.find("text");
    if (part.member_str("type") != "text" || text == nullptr ||
        !text->is_string()) {
      *error = "unsupported Messages content block";
      return false;
    }
    *out += text->str();
    return true;
  };
  // User text and tool results may also carry images.
  const auto read_part = [&](const json::Value& part,
                             tokenization::ChatMessage* target) {
    if (part.is_object() && part.member_str("type") == "image")
      return ParseAnthropicImage(part, target, budget, error);
    return read_text(part, &target->content);
  };
  if (role == "assistant") {
    tokenization::ChatMessage message(tokenization::ChatRole::kAssistant, "");
    for (const auto& part : content->items()) {
      const auto type = part.is_object() ? part.member_str("type") : "";
      if (type == "thinking") {
        const auto* thinking = part.find("thinking");
        if (thinking == nullptr || !thinking->is_string()) {
          *error = "thinking blocks require thinking text";
          return false;
        }
        message.thought += thinking->str();
      } else if (type == "tool_use") {
        const auto* id = part.find("id");
        const auto* input = part.find("input");
        if (id == nullptr || !id->is_string() || id->str().empty() ||
            input == nullptr || !input->is_object()) {
          *error = "tool_use blocks require a non-empty id and an object input";
          return false;
        }
        json::Value function = json::Value::object();
        if (const auto* name = part.find("name"))
          function["name"] = *name;
        function["arguments"] = input->dump();
        tokenization::ChatMessage::ToolCall call;
        call.id = id->str();
        if (!ParseHistoricalFunction(function, &call, error))
          return false;
        message.tool_calls.push_back(std::move(call));
      } else if (!read_text(part, &message.content)) {
        return false;
      }
    }
    messages->push_back(std::move(message));
    return true;
  }
  // One user turn carries the results of the previous calls, then any text.
  std::optional<tokenization::ChatMessage> text;
  for (const auto& part : content->items()) {
    if (!part.is_object() || part.member_str("type") != "tool_result") {
      if (!text)
        text.emplace(tokenization::ChatRole::kUser, "");
      if (!read_part(part, &*text))
        return false;
      continue;
    }
    if (text) {
      messages->push_back(std::move(*text));
      text.reset();
    }
    const auto* id = part.find("tool_use_id");
    const auto* failed = part.find("is_error");
    if (id == nullptr || !id->is_string() || id->str().empty() ||
        (failed != nullptr && !failed->is_bool())) {
      *error = "tool_result blocks require a non-empty tool_use_id";
      return false;
    }
    tokenization::ChatMessage result(tokenization::ChatRole::kTool, "");
    result.tool_call_id = id->str();
    if (const auto* output = part.find("content")) {
      if (output->is_string()) {
        result.content = output->str();
      } else if (!output->is_array()) {
        *error = "tool_result content must be a string, text or image blocks";
        return false;
      } else {
        for (const auto& block : output->items())
          if (!read_part(block, &result))
            return false;
      }
    }
    messages->push_back(std::move(result));
  }
  if (text)
    messages->push_back(std::move(*text));
  return true;
}

HttpResponse Err(int status, const char* reason, const char* message,
                 const char* type, const char* code) {
  json::Value e = json::Value::object();
  json::Value obj = json::Value::object();
  obj["message"] = message;
  obj["type"] = type;
  obj["code"] = code;
  e["error"] = std::move(obj);
  return {.status = status, .reason = reason, .body = e.dump()};
}

// Health, readiness and generation share one body once the device is lost:

// Parse a role string into a ChatRole.
tokenization::ChatRole RoleFrom(const std::string& r) {
  if (r == "system")
    return tokenization::ChatRole::kSystem;
  if (r == "developer")
    return tokenization::ChatRole::kDeveloper;
  if (r == "assistant")
    return tokenization::ChatRole::kAssistant;
  if (r == "tool")
    return tokenization::ChatRole::kTool;
  return tokenization::ChatRole::kUser;
}

// Messages replays assistant reasoning as thinking blocks. Restore it as the
// turn's thought so the prompt reproduces the generated tokens.
bool ReadTextContent(const json::Value* content, std::string* out,
                     std::string* thought = nullptr,
                     tokenization::ChatMessage* image_message = nullptr,
                     core::ImageReadBudget* image_budget = nullptr,
                     std::string* error = nullptr) {
  if (content == nullptr)
    return false;
  if (content->is_string()) {
    *out = content->get_str();
  } else if (content->is_array()) {
    for (const auto& part : content->items()) {
      if (thought != nullptr && part.is_object() &&
          part.member_str("type") == "thinking") {
        const auto* thinking = part.find("thinking");
        if (thinking == nullptr || !thinking->is_string())
          return false;
        *thought += thinking->str();
        continue;
      }
      // Messages images ride along with the text of a user turn; `out` is
      // that message's content, so the image lands at its current end.
      if (image_message != nullptr && part.is_object() &&
          part.member_str("type") == "image") {
        std::string ignored;
        if (!ParseAnthropicImage(part, image_message, *image_budget,
                                 error != nullptr ? error : &ignored))
          return false;
        continue;
      }
      const auto* text = part.find("text");
      const auto type = part.member_str("type");
      if (!part.is_object() || text == nullptr || !text->is_string() ||
          (type != "text" && type != "input_text" && type != "output_text")) {
        return false;
      }
      *out += text->str();
    }
  } else {
    return false;
  }
  return true;
}

bool ReadTextMessages(const json::Value* input,
                      std::vector<tokenization::ChatMessage>* messages,
                      bool images_supported, bool responses = false,
                      std::string* parse_error = nullptr) {
  core::ImageReadBudget image_budget;
  image_budget.accepts_images = images_supported;
  if (input == nullptr || !input->is_array() || input->empty())
    return false;
  // Responses replays one assistant turn as several reasoning, message and
  // function_call items. A system/developer item between them would split
  // that turn, so it waits until the next item that starts a new message.
  std::vector<tokenization::ChatMessage> deferred;
  const auto push = [&](tokenization::ChatMessage message) {
    for (auto& system : deferred)
      messages->push_back(std::move(system));
    deferred.clear();
    messages->push_back(std::move(message));
  };
  for (const auto& item : input->items()) {
    if (!responses && item.is_object() && item.find("content") &&
        item.find("content")->is_array() &&
        std::ranges::any_of(item.find("content")->items(),
                            [](const auto& part) {
                              return part.is_object() &&
                                     (part.member_str("type") == "tool_use" ||
                                      part.member_str("type") == "tool_result");
                            })) {
      std::string error;
      if (!ParseAnthropicToolMessage(item, messages, image_budget, &error)) {
        if (parse_error && !error.empty())
          *parse_error = std::move(error);
        return false;
      }
      continue;
    }
    if (responses && (item.member_str("type") == "function_call" ||
                      item.member_str("type") == "function_call_output" ||
                      item.member_str("type") == "custom_tool_call" ||
                      item.member_str("type") == "custom_tool_call_output")) {
      tokenization::ChatMessage message;
      std::string error;
      if (!ParseOpenAiResponseMessage(item, &message, image_budget, &error)) {
        if (parse_error && !error.empty())
          *parse_error = std::move(error);
        return false;
      }
      if (message.role == tokenization::ChatRole::kAssistant &&
          !messages->empty() &&
          messages->back().role == tokenization::ChatRole::kAssistant) {
        auto& calls = messages->back().tool_calls;
        calls.insert(calls.end(), message.tool_calls.begin(),
                     message.tool_calls.end());
      } else {
        push(std::move(message));
      }
      continue;
    }
    if (responses && item.member_str("type") == "reasoning") {
      const auto* summary = item.find("summary");
      const auto* encrypted = item.find("encrypted_content");
      const auto* content = item.find("content");
      // Codex replays null-valued content and encrypted_content echoed by
      // OpenAI; only non-null payload carries reasoning we cannot restore.
      if (summary == nullptr || !summary->is_array() ||
          (encrypted != nullptr && !encrypted->is_null()) ||
          (content != nullptr && !content->is_null()))
        return false;
      tokenization::ChatMessage reasoning;
      reasoning.role = tokenization::ChatRole::kAssistant;
      for (const auto& part : summary->items()) {
        const auto* text = part.find("text");
        if (part.member_str("type") != "summary_text" || text == nullptr ||
            !text->is_string())
          return false;
        reasoning.thought += text->str();
      }
      push(std::move(reasoning));
      continue;
    }
    const auto role = item.member_str("role");
    if (!item.is_object() ||
        (role != "user" && role != "assistant" && role != "system" &&
         role != "developer") ||
        item.member_str("type", "message") != "message" ||
        item.contains("tool_calls")) {
      return false;
    }
    tokenization::ChatMessage message;
    message.role = RoleFrom(role);
    if (responses) {
      std::string error;
      if (!ParseOpenAiResponseMessage(item, &message, image_budget, &error)) {
        if (parse_error && !error.empty())
          *parse_error = std::move(error);
        return false;
      }
    } else {
      std::string error;
      if (!ReadTextContent(item.find("content"), &message.content,
                           message.role == tokenization::ChatRole::kAssistant
                               ? &message.thought
                               : nullptr,
                           message.role == tokenization::ChatRole::kUser
                               ? &message
                               : nullptr,
                           &image_budget, &error)) {
        if (parse_error && !error.empty())
          *parse_error = std::move(error);
        return false;
      }
    }
    if (responses &&
        (message.role == tokenization::ChatRole::kSystem ||
         message.role == tokenization::ChatRole::kDeveloper) &&
        !messages->empty() &&
        messages->back().role == tokenization::ChatRole::kAssistant) {
      deferred.push_back(std::move(message));
      continue;
    }
    if (responses && message.role == tokenization::ChatRole::kAssistant &&
        !messages->empty() &&
        messages->back().role == tokenization::ChatRole::kAssistant &&
        messages->back().content.empty() && !messages->back().thought.empty()) {
      messages->back().content = std::move(message.content);
    } else {
      push(std::move(message));
    }
  }
  for (auto& system : deferred)
    messages->push_back(std::move(system));
  return true;
}

HttpResponse InvalidCompatibilityRequest(std::string_view message) {
  return Err(400, "Bad Request", std::string(message).c_str(),
             "invalid_request_error", "invalid_request");
}

// Per-endpoint allowances for the shared compatibility validator. Every field
// left false rejects its request field on that endpoint.
struct CompatibilityAllowances {
  /// Request field carrying stop sequences, empty when the endpoint has none.
  std::string_view stop_field;
  bool stream{false};
  bool stream_options{false};
  bool ignore_eos{false};
  bool logprobs{false};
  bool response_controls{false};
  bool thinking{false};
  bool output_config{false};
  /// Messages tools and tool_choice; Responses uses response_controls.
  bool tools{false};
};

// Validate the text subset before dispatch so a client never gets an answer
// to a different request. Responses also supports streamed output.
std::optional<HttpResponse> ReadCompatibilityOptions(
    const json::Value& body, std::string_view token_field,
    std::size_t* max_tokens, sampling::SamplingConfig* sampling_config,
    std::optional<bool> thinking,
    const CompatibilityAllowances& allowances = {}) {
  if (!body.is_object())
    return InvalidCompatibilityRequest("request body must be an object");
  if (const auto* model = body.find("model"); model != nullptr) {
    if (!model->is_string())
      return InvalidCompatibilityRequest("'model' must be a string");
    if (!Env().serves_model(model->str()))
      return Err(404, "Not Found", "requested model is not loaded",
                 "invalid_request_error", "model_not_found");
  }
  for (const std::string field : {"stream", "echo", "store", "background"}) {
    const auto* value = body.find(field);
    if (value == nullptr)
      continue;
    if (field == "stream" && allowances.stream) {
      // As on Chat Completions, null leaves streaming off.
      if (!value->is_null() && !value->is_bool())
        return InvalidCompatibilityRequest("'stream' must be a boolean");
      continue;
    }
    if (!value->is_bool() || value->as_bool())
      return InvalidCompatibilityRequest("'" + field + "' must be false");
  }
  for (const std::string field : {"n", "best_of"}) {
    if (const auto* value = body.find(field);
        value != nullptr &&
        (!value->is_number() || value->as_double() != 1.0)) {
      return InvalidCompatibilityRequest("only '" + field + "=1' is supported");
    }
  }
  for (const std::string field : {"stream_options",
                                  "ignore_eos",
                                  "stop",
                                  "stop_sequences",
                                  "logprobs",
                                  "top_logprobs",
                                  "suffix",
                                  "tools",
                                  "tool_choice",
                                  "parallel_tool_calls",
                                  "response_format",
                                  "output_config",
                                  "text",
                                  "reasoning",
                                  "reasoning_effort",
                                  "thinking",
                                  "chat_template_kwargs",
                                  "previous_response_id",
                                  "conversation",
                                  "include",
                                  "truncation",
                                  "modalities",
                                  "audio"}) {
    if (field != allowances.stop_field &&
        !(field == "stream_options" && allowances.stream_options) &&
        !(field == "ignore_eos" && allowances.ignore_eos) &&
        !(field == "logprobs" && allowances.logprobs) &&
        !(field == "thinking" && allowances.thinking) &&
        !(field == "output_config" && allowances.output_config) &&
        !((field == "tools" || field == "tool_choice") && allowances.tools) &&
        body.contains(field) &&
        !(allowances.response_controls &&
          (field == "text" || field == "reasoning" || field == "tools" ||
           field == "tool_choice" || field == "parallel_tool_calls" ||
           field == "include" || field == "top_logprobs"))) {
      return InvalidCompatibilityRequest("request field '" + field +
                                         "' is not supported on this endpoint");
    }
  }
  for (const std::string field : {"max_tokens", "max_completion_tokens",
                                  "max_output_tokens", "n_predict"}) {
    if (field != token_field && body.contains(field)) {
      return InvalidCompatibilityRequest("use '" + std::string(token_field) +
                                         "' on this endpoint");
    }
  }
  const auto& defaults = Env().sampling_defaults;
  *max_tokens = defaults.max_tokens;
  if (const auto error = detail::ReadSamplingInteger(
          body, token_field, std::size_t{1},
          std::size_t{std::numeric_limits<std::uint32_t>::max()}, max_tokens)) {
    return InvalidCompatibilityRequest(error->message);
  }
  if (const auto error = ParseSamplingConfig(body, defaults.Resolve(thinking),
                                             sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }
  return std::nullopt;
}

std::optional<HttpResponse> ReadOpenAiResponseChat(
    const json::Value& body, ChatRequest* chat, std::size_t* max_tokens,
    sampling::SamplingConfig* sampling_config) {
  chat->reasoning = Env().reasoning_defaults;
  if (auto error = ParseOpenAiResponseControls(body, chat))
    return error;
  if (auto error = Env().compatibility(ChatApi::kResponses, body,
                                       chat->reasoning.enabled, max_tokens,
                                       sampling_config)) {
    return error;
  }
  if (const auto error = ParseResponsesLogprobs(body, sampling_config))
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());

  std::vector<tokenization::ChatMessage> messages;
  if (const auto* instructions = body.find("instructions")) {
    if (!instructions->is_string()) {
      return InvalidCompatibilityRequest("'instructions' must be a string");
    }
    messages.push_back(
        {tokenization::ChatRole::kSystem, instructions->str(), "", ""});
  }
  const auto* input = body.find("input");
  std::string input_error =
      "'input' must contain text, message items with text/images, reasoning "
      "items, function calls or function outputs";
  if (input != nullptr && input->is_string() && !input->str().empty()) {
    messages.push_back({tokenization::ChatRole::kUser, input->str(), "", ""});
  } else if (!ReadTextMessages(input, &messages, Env().accepts_images, true,
                               &input_error)) {
    return InvalidCompatibilityRequest(input_error);
  }

  chat->messages = std::move(messages);
  return std::nullopt;
}

// Messages selects reasoning with thinking.type; adaptive leaves the choice to
// the server's default. With enabled thinking, budget_tokens caps the
// reasoning tokens, as llama.cpp does; enabled thinking keeps the server's
// effort unless output_config.effort sets it.
// Reasoning is returned for every display, because replayed thinking blocks
// restore the cached thought.
std::optional<HttpResponse> ReadThinking(const json::Value& body,
                                         ReasoningOptions* reasoning) {
  const auto* thinking = body.find("thinking");
  if (thinking == nullptr)
    return {};
  const auto* type = thinking->is_object() ? thinking->find("type") : nullptr;
  if (type == nullptr || !type->is_string() ||
      (type->str() != "enabled" && type->str() != "adaptive" &&
       type->str() != "disabled"))
    return InvalidCompatibilityRequest(
        "'thinking.type' must be enabled, adaptive, or disabled");
  if (const auto* budget = thinking->find("budget_tokens");
      budget != nullptr &&
      (!budget->is_number() || !std::isfinite(budget->as_double()) ||
       budget->as_double() < 1 ||
       budget->as_double() > std::numeric_limits<std::int32_t>::max() ||
       std::floor(budget->as_double()) != budget->as_double()))
    return InvalidCompatibilityRequest(
        "'thinking.budget_tokens' must be a positive integer at most "
        "2147483647");
  else if (budget != nullptr && type->str() == "enabled")
    reasoning->budget_tokens = static_cast<std::size_t>(budget->as_double());
  if (const auto* display = thinking->find("display");
      display != nullptr &&
      (!display->is_string() ||
       (display->str() != "summarized" && display->str() != "omitted" &&
        display->str() != "updates")))
    return InvalidCompatibilityRequest(
        "'thinking.display' must be summarized, omitted, or updates");
  if (type->str() != "adaptive")
    reasoning->enabled = type->str() == "enabled";
  return {};
}

std::optional<HttpResponse> ReadMessagesOutputConfig(
    const json::Value& body, ReasoningOptions* options) {
  const auto* config = body.find("output_config");
  if (config == nullptr || config->is_null())
    return {};
  if (!config->is_object())
    return InvalidCompatibilityRequest("'output_config' must be an object");
  for (const auto& [key, value] : config->members()) {
    if (key != "effort")
      return InvalidCompatibilityRequest("unsupported output_config member: " +
                                         key);
    if (value.is_null())
      continue;
    // Anthropic effort does not select thinking, so it never enables it;
    // formatters apply it only while thinking is on. Anthropic has no minimal
    // effort.
    const auto effort = value.is_string() && value.str() != "minimal"
                            ? ParseReasoningEffortName(value.str())
                            : std::nullopt;
    if (!effort.has_value())
      return InvalidCompatibilityRequest(
          "'output_config.effort' must be low, medium, high, xhigh, or max");
    options->effort = effort;
  }
  return {};
}

// The request fields Messages and count_tokens share: sampling and reasoning
// controls, tools, system text and the turns.
std::optional<HttpResponse> ReadAnthropicChat(
    const json::Value& body, ChatRequest* chat, std::size_t* max_tokens,
    sampling::SamplingConfig* sampling_config) {
  chat->reasoning = Env().reasoning_defaults;
  if (body.is_object()) {
    if (auto error = ReadThinking(body, &chat->reasoning))
      return error;
    if (auto error = ReadMessagesOutputConfig(body, &chat->reasoning))
      return error;
  }
  if (auto error =
          Env().compatibility(ChatApi::kMessages, body, chat->reasoning.enabled,
                              max_tokens, sampling_config)) {
    return error;
  }
  if (auto error = ParseAnthropicToolControls(body, chat))
    return error;

  std::vector<tokenization::ChatMessage> messages;
  if (const auto* system = body.find("system")) {
    std::string text;
    if (!ReadTextContent(system, &text)) {
      return InvalidCompatibilityRequest("'system' must contain only text");
    }
    messages.push_back(
        {tokenization::ChatRole::kSystem, std::move(text), "", ""});
  }
  std::string messages_error =
      "'messages' must contain text, image, thinking, tool_use or tool_result "
      "blocks";
  if (!ReadTextMessages(body.find("messages"), &messages, Env().accepts_images,
                        false, &messages_error)) {
    return InvalidCompatibilityRequest(messages_error);
  }

  chat->messages = std::move(messages);
  if (const auto error = ParseStopSequences(body.find("stop_sequences"),
                                            StopSequenceFormat::kAnthropic,
                                            &chat->stop_sequences))
    return InvalidCompatibilityRequest(*error);
  return std::nullopt;
}
}  // namespace

std::optional<HttpResponse> Decode(ChatApi api, std::string_view body,
                                   const ChatDecodeEnvironment& environment,
                                   bool require_model,
                                   DecodedChatRequest* output) {
  Using use(environment);
  if (api == ChatApi::kChat) {
    ParsedChatRequest parsed;
    if (auto error = ParseRequest(body, &parsed, require_model))
      return error;
    output->chat = std::move(parsed.chat);
    output->model = std::move(parsed.model);
    if (parsed.max_tokens != 0)
      output->max_tokens = parsed.max_tokens;
    output->sampling = parsed.sampling;
    output->stream = parsed.stream;
    output->include_usage = parsed.include_usage;
    return std::nullopt;
  }
  // The handlers parsed the body and read stream and model after decoding.
  json::Value parsed;
  try {
    parsed = json::parse(body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }
  ChatRequest chat;
  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  auto error =
      api == ChatApi::kResponses
          ? ReadOpenAiResponseChat(parsed, &chat, &max_tokens, &sampling_config)
          : ReadAnthropicChat(parsed, &chat, &max_tokens, &sampling_config);
  if (error)
    return error;
  output->chat = std::move(chat);
  output->max_tokens = max_tokens;
  output->sampling = sampling_config;
  output->stream =
      parsed.find("stream") != nullptr && parsed.find("stream")->as_bool();
  if (const auto* model = parsed.find("model"); model && model->is_string())
    output->model = model->str();
  return std::nullopt;
}

std::optional<HttpResponse> ResponseControls(
    const json::Value& body, ChatRequest* chat,
    const ChatDecodeEnvironment& environment) {
  Using use(environment);
  return ParseOpenAiResponseControls(body, chat);
}

std::optional<HttpResponse> AnthropicToolControls(
    const json::Value& body, ChatRequest* chat,
    const ChatDecodeEnvironment& environment) {
  Using use(environment);
  return ParseAnthropicToolControls(body, chat);
}

bool ResponseItem(const json::Value& item, tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error,
                  const ChatDecodeEnvironment& environment) {
  Using use(environment);
  return ParseOpenAiResponseMessage(item, message, budget, error);
}

std::optional<HttpResponse> Compatibility(
    ChatApi api, const json::Value& body, std::optional<bool> thinking,
    std::size_t* max_tokens, sampling::SamplingConfig* sampling,
    const ChatDecodeEnvironment& environment) {
  Using use(environment);
  if (api == ChatApi::kResponses)
    return ReadCompatibilityOptions(
        body, "max_output_tokens", max_tokens, sampling, thinking,
        {.stream = true, .response_controls = true});
  return ReadCompatibilityOptions(body, "max_tokens", max_tokens, sampling,
                                  thinking,
                                  {.stop_field = "stop_sequences",
                                   .stream = true,
                                   .thinking = true,
                                   .output_config = true,
                                   .tools = true});
}

}  // namespace gufo::server::chat_request_reference

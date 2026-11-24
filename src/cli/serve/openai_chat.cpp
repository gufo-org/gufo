#include "src/cli/serve/openai_chat.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/cli/serve/generation_parse.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/response_format.hpp"
#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"
#include "src/families/qwen/text/chat_template.hpp"

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
  // An append-only JSON prefix; absent on complete calls.
  std::optional<std::string> partial_arguments;
};

struct ParsedGeneration {
  std::string text;
  std::string reasoning_content;
  std::vector<ParsedToolCall> tool_calls;
  bool hide_tool_markup{false};
  std::vector<TextSourceSpan> text_spans;
};

void AppendTextSpans(std::vector<TextSourceSpan>& output,
                     std::span<const TextSourceSpan> source, std::size_t offset,
                     std::size_t size) {
  for (const auto& span : source) {
    if (offset >= span.size) {
      offset -= span.size;
      continue;
    }
    const auto count = std::min(size, span.size - offset);
    if (count)
      output.push_back({span.begin + offset, count});
    size -= count;
    if (!size)
      break;
    offset = 0;
  }
}

void SetParsedText(ParsedGeneration& parsed, std::string_view text,
                   std::string_view source, bool track) {
  parsed.text = std::string(text);
  parsed.text_spans.clear();
  if (track && !text.empty())
    parsed.text_spans.push_back(
        {static_cast<std::size_t>(text.data() - source.data()), text.size()});
}

void RebaseTextSpans(ParsedGeneration& parsed, std::size_t offset) {
  for (auto& span : parsed.text_spans)
    span.begin += offset;
}

constexpr std::array<std::string_view, 7> kToolMarkers{
    tokenization::QwenToolSyntax::kCallOpen,
    "<｜DSML｜tool_calls｜>",
    "<｜DSML｜tool_calls>",
    "<DSML｜tool_calls｜>",
    "<DSML｜tool_calls>",
    "<tool_calls｜>",
    "<tool_calls>",
};
using ReasoningMarkup = ext::ReasoningMarkup;
using ToolMarkerSet = std::span<const std::string_view>;
using ToolCloserSet = std::span<const std::string_view>;

// Closers eligible for removal immediately after an accepted call. Standalone
// closers and XML remain literal; the declared-tool envelope rule is separate.
constexpr std::array<std::string_view, 4> kQwenClosers{
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
    // The client's envelope, whose block rules already strip it whole.
    "</invoke>",
};
constexpr std::array<std::string_view, 9> kToolClosers{
    // Qwen family.
    tokenization::QwenToolSyntax::kParameterClose,
    tokenization::QwenToolSyntax::kFunctionClose,
    tokenization::QwenToolSyntax::kCallClose,
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

// The call markup of one admitted request's tool dialect.
struct ToolMarkup {
  // Openers that start a call; nothing else does.
  ToolMarkerSet markers;
  // Complete native call headers which end a constrained reasoning phase.
  ToolMarkerSet reasoning_markers;
  ToolCloserSet closers;
  // The registered parser of calls the host does not parse itself.
  const ext::ToolDialect* dialect{nullptr};
  // The host parses calls opened by the Qwen marker.
  bool qwen_calls{false};
  // A native grammar frames arguments before a value completes, so partial
  // calls stream. Recovery from JSON fallback text needs a full call.
  bool native{false};
  // Whitespace after an explicit end of reasoning that is framing.
  std::string_view reasoning_separator{"\r\n"};
};

// The markup of the dialect the request admitted. Other dialects' markers and
// closers are ordinary prose under its grammar, not additional calls.
ToolMarkup SelectToolMarkup(const std::optional<std::string>& id) {
  using Format = sampling::JsonConstraint::ToolFormat;
  // The JSON fallback has no native counterpart in llama.cpp. Its grammar
  // leaves text before <tool_call> unconstrained, where models still write
  // their native call syntax; keep recognizing every opener there.
  if (!id || *id == ext::BuiltinToolDialectId(Format::kJson))
    return {
        .markers = kToolMarkers,
        // Mirror the markers: stripping follows the dialect the request
        // admitted, the client's envelope excepted, which no dialect admits
        // and every one of them parses.
        .closers =
            id ? ToolCloserSet(kQwenClosers) : ToolCloserSet(kToolClosers),
        .dialect = ext::FindToolDialect(Format::kDeepSeek),
        .qwen_calls = true,
    };
  if (*id == ext::BuiltinToolDialectId(Format::kQwen)) {
    static constexpr std::array<std::string_view, 1> qwen{
        tokenization::QwenToolSyntax::kCallOpen};
    // A native function header, rather than a bare marker mentioned in prose,
    // permits the implicit reasoning boundary used by llama.cpp's native
    // parsers.
    static constexpr std::array<std::string_view, 1> header{
        tokenization::QwenToolSyntax::kNativeCallOpen};
    return {
        .markers = qwen,
        .reasoning_markers = header,
        .closers = kQwenClosers,
        .qwen_calls = true,
        .native = true,
        // Qwen's llama.cpp PEG uses `reasoning << content`: `<<` consumes
        // space().
        .reasoning_separator = " \t\r\n\f\v",
    };
  }
  const auto* dialect = ext::FindToolDialect(*id);
  if (dialect == nullptr)
    return {};
  return {
      .markers = dialect->markers,
      .reasoning_markers = dialect->reasoning_markers,
      .closers = dialect->closers,
      .dialect = dialect,
      .native = true,
  };
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
    case 507:
      return "Insufficient Storage";
    default:
      return "Internal Server Error";
  }
}

/// Append `text` as the body of a JSON string (without the quotes). The stream
/// writer builds one frame buffer per token instead of a JSON DOM, so it needs
/// the same escaping `json::Value::dump` applies, without its allocation.
void AppendEscapedJson(std::string& out, std::string_view text) {
  for (const unsigned char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char escape[8];
          (void)std::snprintf(escape, sizeof(escape), "\\u%04x", c);
          out += escape;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
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
                 core::ReadImageUrl(url->get_str(), budget)),
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

std::optional<HttpResponse> ParseRequest(const HttpRequest& request,
                                         TextGenerationBackend& backend,
                                         ParsedChatRequest* output,
                                         bool require_model = true) {
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

  if (const auto* model = body.find("model");
      model != nullptr && !model->is_null() && !model->is_string()) {
    return Error(400, "Bad Request", "'model' must be a string",
                 "invalid_model");
  }
  output->model = body.member_str("model");
  if (output->model.empty() && !require_model) {
    output->model = backend.model_id();
  }
  if (output->model.empty()) {
    return Error(400, "Bad Request", "'model' is required", "missing_model");
  }
  if (!backend.serves_model(output->model)) {
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
  image_budget.accepts_images = backend.supports_images();
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

// `value` without its trailing whitespace.
std::string_view TrimTrailing(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

}  // namespace

GenerationMarkup SelectGenerationMarkup(const std::optional<std::string>& id) {
  const auto markup = SelectToolMarkup(id);
  return {
      .markers = markup.markers,
      .reasoning_markers = markup.reasoning_markers,
      .closers = markup.closers,
      .dialect = markup.dialect,
      .qwen_calls = markup.qwen_calls,
      .native = markup.native,
      .reasoning_separator = markup.reasoning_separator,
  };
}

namespace {

// The request fields the generation parser reads.
GenerationSetup ParseSetup(const ChatRequest& chat,
                           TextGenerationBackend::InitialOutputState initial,
                           ext::ReasoningMarkup reasoning,
                           const GenerationMarkup& markup,
                           bool track_spans = false, bool buffered = false) {
  return {
      .initial = initial,
      .reasoning = std::move(reasoning),
      .markup = markup,
      .response_format = chat.response_format != nullptr,
      .constrained_tools = chat.constrained_tools,
      .tool_choice = chat.tool_choice,
      .tools = chat.tools,
      .track_spans = track_spans,
      .buffered = buffered,
  };
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
  json::Value completion_details = json::Value::object();
  completion_details["reasoning_tokens"] = result.reasoning_tokens;
  usage["completion_tokens_details"] = std::move(completion_details);

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
  if (result.component_cache)
    metrics["component_cache"] = ComponentCacheJson(*result.component_cache);
  metrics["cache_hit"] = result.cache_hit;
  if (!result.cache_miss_reason.empty()) {
    metrics["cache_miss_reason"] = result.cache_miss_reason;
    metrics["cache_common_prefix_tokens"] = result.cache_common_prefix_tokens;
    metrics["cache_checkpoint_tokens"] = result.cache_checkpoint_tokens;
  }
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_queued_bytes"] = result.cache_disk_queued_bytes;
  // Deprecated compatibility key; remove at the next breaking release.
  // See src/deprecated-items-to-remove-at-breaking-release.md.
  metrics["cache_shared_bytes"] = std::size_t{0};
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
  metrics["draft_rounds"] = result.draft_rounds;
  metrics["prefill_tokens"] = result.prefill_tokens;
  metrics["prefill_chunks"] = result.prefill_chunks;
  metrics["active_decode_prefill_chunks"] = result.active_decode_prefill_chunks;
  metrics["max_prefill_chunk_tokens"] = result.max_prefill_chunk_tokens;
  metrics["max_prefill_layer_groups"] = result.max_prefill_layer_groups;
  metrics["prefill_yield_rounds"] = result.prefill_yield_rounds;
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

json::Value ToolCallsJson(std::span<const GeneratedCall> calls) {
  json::Value output = json::Value::array();
  for (const auto& call : calls) {
    json::Value item = json::Value::object();
    item["id"] = RandomId("call_");
    item["type"] = "function";
    json::Value function = json::Value::object();
    function["name"] = call.name;
    function["arguments"] = call.arguments;
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
                        const char* finish_reason = nullptr,
                        const json::Value* logprobs = nullptr) {
  json::Value chunk = BaseChunk(id, created, model);
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = std::move(delta);
  if (logprobs)
    choice["logprobs"] = *logprobs;
  if (finish_reason == nullptr) {
    choice["finish_reason"] = json::Value();
  } else {
    choice["finish_reason"] = finish_reason;
  }
  choices.push_back(std::move(choice));
  chunk["choices"] = std::move(choices);
  return chunk;
}

struct ToolDelta {
  std::size_t index;
  std::string id;
  std::string name;
  std::string arguments;
  bool complete{false};
};

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Build the same output items for streaming and buffered responses.
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, HttpResponse::BodyWriter writer,
                  const ChatRequest& chat)
      : writer_(std::move(writer)),
        namespaces_(chat.tool_namespaces),
        cache_persist_(chat.cache_persist) {
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

  void SetLogprobs(std::optional<std::uint32_t> count) {
    if (count)
      response_["top_logprobs"] = static_cast<int>(*count);
  }

  bool Append(std::string_view text, bool reasoning,
              json::Value logprobs = json::Value::array()) {
    if (text.empty())
      return true;
    if (!CloseTool("incomplete"))
      return false;
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
      logprobs_ = json::Value::array();
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
    if (!reasoning) {
      for (const auto& entry : logprobs.items())
        logprobs_.push_back(entry);
      delta["logprobs"] = ResponsesStreamLogprobs(logprobs);
    }
    return Emit(std::move(delta));
  }

  bool Tool(const ToolDelta& call) {
    if (!call.id.empty()) {
      if (!CloseItem("completed"))
        return false;
      tool_ = json::Value::object();
      tool_["id"] = RandomId("fc_");
      tool_["type"] = "function_call";
      tool_["call_id"] = call.id;
      tool_["name"] = call.name;
      if (const auto it = namespaces_.find(call.name); it != namespaces_.end())
        tool_["namespace"] = it->second;
      tool_["arguments"] = "";
      tool_["status"] = "in_progress";
      tool_arguments_.clear();
      auto added = IndexedEvent("response.output_item.added");
      added["item"] = tool_;
      if (!Emit(std::move(added)))
        return false;
    }
    tool_arguments_ += call.arguments;
    if (!call.arguments.empty()) {
      auto delta = IndexedEvent("response.function_call_arguments.delta");
      delta["item_id"] = tool_.member_str("id");
      delta["delta"] = call.arguments;
      if (!Emit(std::move(delta)))
        return false;
    }
    return !call.complete || CloseTool("completed");
  }

  bool CloseTool(const char* status) {
    if (tool_.is_null())
      return connected_;
    unfinished_tool_ |= std::string_view(status) == "incomplete";
    auto done = IndexedEvent("response.function_call_arguments.done");
    done["item_id"] = tool_.member_str("id");
    done["name"] = tool_.member_str("name");
    done["arguments"] = tool_arguments_;
    if (!Emit(std::move(done)))
      return false;
    tool_["arguments"] = tool_arguments_;
    tool_["status"] = status;
    auto closed = IndexedEvent("response.output_item.done");
    closed["item"] = tool_;
    response_["output"].push_back(std::move(tool_));
    tool_ = json::Value();
    ++output_index_;
    return Emit(std::move(closed));
  }

  json::Value Complete(const TextGenerationBackend::Result& result) {
    const bool limited =
        result.finish_reason == TextGenerationBackend::FinishReason::kLength;
    CloseTool("incomplete");
    CloseItem(limited ? "incomplete" : "completed");
    response_["status"] =
        limited || unfinished_tool_ ? "incomplete" : "completed";
    if (limited)
      response_["incomplete_details"]["reason"] = "max_output_tokens";
    else if (unfinished_tool_)
      response_["incomplete_details"] = json::Value::object();
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
    response_["cache_persist"] = cache_persist_;
    Lifecycle(limited || unfinished_tool_ ? "response.incomplete"
                                          : "response.completed");
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
      part["logprobs"] = logprobs_;
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
      done["logprobs"] = ResponsesStreamLogprobs(logprobs_);
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
  // Echoed in the final response, as the applied `cache_persist`.
  bool cache_persist_;
  json::Value response_;
  json::Value item_;
  json::Value tool_;
  std::string tool_arguments_;
  std::string text_;
  json::Value logprobs_ = json::Value::array();
  std::size_t sequence_{0};
  std::size_t output_index_{0};
  bool active_{false};
  bool reasoning_{false};
  bool connected_{true};
  bool unfinished_tool_{false};
};

HttpResponse NonStreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>& generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  const auto result = generation->Wait();
  const auto generated = ParseGeneratedText(
      ParseSetup(request.chat, initial_output_state,
                 generation->ReasoningDelimiters(),
                 SelectGenerationMarkup(generation->ToolDialectId())),
      result.text,
      result.finish_reason == TextGenerationBackend::FinishReason::kStop);

  json::Value response = json::Value::object();
  response["id"] = RandomId("chatcmpl-");
  response["object"] = "chat.completion";
  response["created"] = Now();
  response["model"] = request.model;
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  if (!generated.reasoning.empty()) {
    message["reasoning_content"] = generated.reasoning;
  }
  if (generated.text.empty() && !generated.calls.empty()) {
    message["content"] = json::Value();
  } else {
    message["content"] = generated.text;
  }
  if (!generated.calls.empty()) {
    message["tool_calls"] = ToolCallsJson(generated.calls);
  }
  if (request.sampling.logprobs)
    choice["logprobs"] =
        ChatLogprobs(result.logprobs, *request.sampling.logprobs);
  choice["message"] = std::move(message);
  choice["finish_reason"] = FinishReason(result, !generated.calls.empty());
  choices.push_back(std::move(choice));
  response["choices"] = std::move(choices);
  response["usage"] = Usage(result);
  response["timings"] = GenerationTimings(result);
  response["cache_persist"] = request.chat.cache_persist;
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
  const std::string model = request.model;
  const auto markup = SelectGenerationMarkup(generation->ToolDialectId());
  const auto reasoning = generation->ReasoningDelimiters();
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
           initial_output_state, reasoning, markup,
           stream_log](const HttpResponse::BodyWriter& writer) {
            bool connected = true;
            bool started = false;
            // The id/object/created/model/choices envelope is identical for
            // every token. Keeping it serialized avoids building and dumping
            // a JSON DOM per token; only the delta body changes.
            std::string chunk_prefix = "data: {\"id\":\"";
            AppendEscapedJson(chunk_prefix, id);
            chunk_prefix +=
                "\",\"object\":\"chat.completion.chunk\",\"created\":";
            chunk_prefix += std::to_string(created);
            chunk_prefix += ",\"model\":\"";
            AppendEscapedJson(chunk_prefix, model);
            chunk_prefix += "\",\"choices\":[{\"index\":0,\"delta\":";
            const std::string chunk_suffix = ",\"finish_reason\":null}]}\n\n";
            std::optional<std::string> pending_logprobs;
            const auto write_delta = [&](std::string_view delta,
                                         bool attach_logprobs = true) {
              std::string frame;
              frame.reserve(chunk_prefix.size() + delta.size() + 64);
              frame += chunk_prefix;
              frame += delta;
              if (attach_logprobs && pending_logprobs) {
                frame += ",\"logprobs\":";
                frame += *pending_logprobs;
                pending_logprobs.reset();
              }
              frame += chunk_suffix;
              return writer(frame);
            };
            const auto write_text = [&](std::string_view field,
                                        std::string_view text) {
              std::string delta;
              delta.reserve(field.size() + text.size() + 8);
              delta += "{\"";
              delta += field;
              delta += "\":\"";
              AppendEscapedJson(delta, text);
              delta += "\"}";
              return write_delta(delta);
            };
            const auto begin = [&] {
              if (!started) {
                started = true;
                connected = write_delta("{\"role\":\"assistant\"}", false);
              }
              return connected;
            };
            if (request.chat.return_progress && !begin()) {
              generation->Cancel();
              return;
            }
            const auto write_choice = [&](json::Value delta) {
              return write_delta(delta.dump());
            };
            GenerationStream parser(ParseSetup(
                request.chat, initial_output_state, reasoning, markup));
            std::size_t final_calls = 0;
            const auto on_event = [&](const GenerationEvent& event) {
              using Kind = GenerationEvent::Kind;
              switch (event.kind) {
                case Kind::kReasoningDelta:
                case Kind::kContentDelta:
                  if (event.text.empty())
                    return true;
                  connected = event.kind == Kind::kReasoningDelta
                                  ? write_text("reasoning_content", event.text)
                                  : write_text("content", event.text);
                  return connected;
                case Kind::kToolDelta: {
                  // Completion without additional arguments needs no Chat
                  // delta.
                  if (!event.begin && event.arguments.empty())
                    return connected;
                  auto item = json::Value::object();
                  item["index"] = event.index;
                  if (event.begin) {
                    item["id"] = RandomId("call_");
                    item["type"] = "function";
                    item["function"]["name"] = std::string(event.name);
                  }
                  item["function"]["arguments"] = std::string(event.arguments);
                  auto delta = json::Value::object();
                  delta["tool_calls"] = json::Value::array();
                  delta["tool_calls"].push_back(std::move(item));
                  connected = write_choice(std::move(delta));
                  return connected;
                }
                case Kind::kFinalCall:
                  ++final_calls;
                  return true;
                default:
                  return true;
              }
            };

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
                  [&](const TokenPiece& piece) {
                    if (piece.logprobs && request.sampling.logprobs)
                      pending_logprobs =
                          ChatLogprobs(std::span(piece.logprobs, 1),
                                       *request.sampling.logprobs)
                              .dump();
                    if (!begin() || !parser.Feed(piece.text, on_event))
                      return false;
                    if (pending_logprobs)
                      connected = write_choice(json::Value::object());
                    return connected;
                  },
                  on_progress, begin);
              stream_log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (!connected || result.cancelled) {
                return;
              }
              if (!begin())
                return;
              const bool interrupted =
                  result.finish_reason !=
                  TextGenerationBackend::FinishReason::kStop;
              if (!parser.Finish(!interrupted, interrupted, on_event))
                return;
              json::Value terminal_delta = json::Value::object();
              auto terminal_chunk =
                  ChoiceChunk(id, created, model, std::move(terminal_delta),
                              FinishReason(result, final_calls > 0));
              // llama.cpp reports timings on the terminal choice regardless
              // of the optional OpenAI usage chunk. Proxies need these even
              // when a client does not request stream_options.include_usage.
              terminal_chunk["timings"] = GenerationTimings(result);
              terminal_chunk["cache_persist"] = request.chat.cache_persist;
              if (!writer(Sse(terminal_chunk))) {
                return;
              }
              if (request.include_usage) {
                json::Value usage_chunk = BaseChunk(id, created, model);
                usage_chunk["choices"] = json::Value::array();
                usage_chunk["usage"] = Usage(result);
                usage_chunk["timings"] = GenerationTimings(result);
                usage_chunk["cache_persist"] = request.chat.cache_persist;
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
              // Retain the cause in logs alongside the client's stable code.
              Logger::Error("chat", error.what());
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
             core::ReadImageUrl(url, budget))});
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

namespace {

/// Messages output as Anthropic content blocks. Streamed events follow
/// message_start, content_block_start/delta/stop and message_delta/stop.
class AnthropicOutput {
public:
  AnthropicOutput(std::string model, HttpResponse::BodyWriter writer,
                  const ChatRequest&)
      : writer_(std::move(writer)) {
    message_ = json::Value::object();
    message_["id"] = RandomId("msg_");
    message_["type"] = "message";
    message_["role"] = "assistant";
    message_["model"] = std::move(model);
    message_["content"] = json::Value::array();
    message_["stop_reason"] = json::Value();
    message_["stop_sequence"] = json::Value();
    // Prompt accounting is final only with the result; message_delta carries
    // the complete usage.
    message_["usage"] = json::Value::object();
    message_["usage"]["input_tokens"] = 0;
    message_["usage"]["output_tokens"] = 0;
  }

  bool Begin() {
    auto event = Event("message_start");
    event["message"] = message_;
    return Emit(std::move(event));
  }

  bool Progress(const TextGenerationBackend::PromptProgress&) {
    return connected_;
  }

  bool Append(std::string_view text, bool reasoning) {
    if (!text.empty() && !CloseTool(false))
      return false;
    // Clear the preceding block's held whitespace before collecting this
    // block's first piece; otherwise CloseBlock discards its trailing space.
    if ((!active_ || reasoning_ != reasoning) && !CloseBlock())
      return false;
    std::string piece;
    if (reasoning) {
      // Thinking blocks carry the reasoning trimmed as Chat reports it: drop
      // leading whitespace and hold trailing whitespace until more follows.
      if (!active_ || !reasoning_)
        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.front())) != 0)
          text.remove_prefix(1);
      const auto body = TrimTrailing(text);
      if (body.empty()) {
        held_.append(text);
        return true;
      }
      piece = std::exchange(held_, std::string(text.substr(body.size())));
      piece.append(body);
      text = piece;
    }
    if (text.empty())
      return true;
    if (!active_ || reasoning_ != reasoning) {
      reasoning_ = reasoning;
      active_ = true;
      text_.clear();
      if (!Emit(BlockStart(Block())))
        return false;
    }
    text_.append(text);
    auto delta = json::Value::object();
    delta["type"] = reasoning ? "thinking_delta" : "text_delta";
    delta[reasoning ? "thinking" : "text"] = std::string(text);
    return Emit(BlockDelta(std::move(delta)));
  }

  bool Tool(const ToolDelta& call) {
    if (!call.id.empty()) {
      if (!CloseBlock())
        return false;
      tool_ = json::Value::object();
      tool_["type"] = "tool_use";
      tool_["id"] = call.id;
      tool_["name"] = call.name;
      tool_["input"] = json::Value::object();
      tool_arguments_.clear();
      if (!Emit(BlockStart(tool_)))
        return false;
    }
    tool_arguments_ += call.arguments;
    if (!call.arguments.empty()) {
      auto delta = json::Value::object();
      delta["type"] = "input_json_delta";
      delta["partial_json"] = call.arguments;
      if (!Emit(BlockDelta(std::move(delta))))
        return false;
    }
    return !call.complete || CloseTool(true);
  }

  bool CloseTool(bool complete) {
    if (tool_.is_null())
      return connected_;
    if (complete) {
      tool_["input"] = json::parse(tool_arguments_);
      tool_use_ = true;
    }
    message_["content"].push_back(std::move(tool_));
    tool_ = json::Value();
    return BlockStop();
  }

  json::Value Complete(const TextGenerationBackend::Result& result) {
    CloseTool(false);
    CloseBlock();
    if (message_["content"].empty()) {
      text_.clear();
      reasoning_ = false;
      active_ = true;
      Emit(BlockStart(Block()));
      CloseBlock();
    }
    // Same precedence as Chat finish_reason: a truncated or stopped turn is
    // reported as such even when complete calls precede the cut.
    using Finish = TextGenerationBackend::FinishReason;
    message_["stop_reason"] =
        result.finish_reason == Finish::kStopSequence ? "stop_sequence"
        : result.finish_reason == Finish::kLength     ? "max_tokens"
        : tool_use_                                   ? "tool_use"
                                                      : "end_turn";
    message_["stop_sequence"] = result.finish_reason == Finish::kStopSequence
                                    ? json::Value(result.stop_sequence)
                                    : json::Value();
    auto usage = json::Value::object();
    // Anthropic reports cache reads separately; input_tokens counts only the
    // uncached prompt, so the three input fields add up to the prompt length.
    usage["input_tokens"] =
        result.prompt_tokens -
        std::min(result.cached_prompt_tokens, result.prompt_tokens);
    usage["output_tokens"] = result.completion_tokens;
    usage["cache_creation_input_tokens"] = 0;
    usage["cache_read_input_tokens"] = result.cached_prompt_tokens;
    message_["usage"] = usage;
    message_["timings"] = GenerationTimings(result);
    auto delta = Event("message_delta");
    delta["delta"]["stop_reason"] = message_["stop_reason"];
    delta["delta"]["stop_sequence"] = message_["stop_sequence"];
    delta["usage"] = std::move(usage);
    if (Emit(std::move(delta)))
      Emit(Event("message_stop"));
    return message_;
  }

  bool Fail(std::string_view message) {
    auto event = Event("error");
    event["error"]["type"] = "api_error";
    event["error"]["message"] = std::string(message);
    return Emit(std::move(event));
  }

private:
  static json::Value Event(std::string_view type) {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    return event;
  }

  json::Value Block() const {
    auto block = json::Value::object();
    block["type"] = reasoning_ ? "thinking" : "text";
    block[reasoning_ ? "thinking" : "text"] = text_;
    // Local reasoning is not signed; clients replay the block unchanged.
    if (reasoning_)
      block["signature"] = "";
    return block;
  }

  json::Value BlockStart(json::Value block) const {
    auto event = Event("content_block_start");
    event["index"] = index_;
    event["content_block"] = std::move(block);
    return event;
  }

  json::Value BlockDelta(json::Value delta) const {
    auto event = Event("content_block_delta");
    event["index"] = index_;
    event["delta"] = std::move(delta);
    return event;
  }

  bool BlockStop() {
    auto event = Event("content_block_stop");
    event["index"] = index_++;
    return Emit(std::move(event));
  }

  bool CloseBlock() {
    held_.clear();
    if (!active_)
      return connected_;
    message_["content"].push_back(Block());
    active_ = false;
    return BlockStop();
  }

  bool Emit(json::Value event) {
    if (!writer_)
      return true;
    if (!connected_)
      return false;
    connected_ =
        writer_("event: " + event.member_str("type") + "\n" + Sse(event));
    return connected_;
  }

  HttpResponse::BodyWriter writer_;
  json::Value message_;
  json::Value tool_;
  std::string tool_arguments_;
  std::string text_;
  /// Trailing reasoning whitespace, emitted only if more reasoning follows.
  std::string held_;
  std::size_t index_{0};
  bool active_{false};
  bool reasoning_{false};
  bool tool_use_{false};
  bool connected_{true};
};

/// Runs one Responses or Messages generation through the Chat reasoning,
/// UTF-8 and tool filters, buffered or as SSE events from Output.
template<typename Output>
HttpResponse CreateCompatibilityResponse(
    const HttpRequest& request, TextGenerationBackend& backend,
    const ChatRequest& chat, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling, bool stream,
    std::string_view requested_model) {
  const auto initial = backend.initial_output_state(chat);
  const bool observe =
      std::is_same_v<Output, ResponsesOutput> && sampling.logprobs.has_value();
  auto generation = backend.start_chat(chat, max_tokens, sampling,
                                       request.is_cancelled, stream || observe);
  const auto markup = SelectGenerationMarkup(generation->ToolDialectId());
  const auto reasoning = generation->ReasoningDelimiters();
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  auto timing = std::make_shared<std::string>();
  const auto run = [generation, initial, reasoning, chat, markup,
                    model = requested_model.empty()
                                ? backend.model_id()
                                : std::string(requested_model),
                    stream_log, observe, count = sampling.logprobs,
                    timing](const HttpResponse::BodyWriter& writer) {
    Output output(model, writer, chat);
    std::optional<ResponsesLogprobs> probabilities;
    if (observe)
      probabilities.emplace(*count);
    if constexpr (std::is_same_v<Output, ResponsesOutput>)
      output.SetLogprobs(count);
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
    // A buffered response reads the whole text at the end, so only the
    // final parse produces its calls.
    GenerationStream parser(ParseSetup(chat, initial, reasoning, markup,
                                       observe, !writer && !observe));
    const auto on_event = [&](const GenerationEvent& event) {
      using Kind = GenerationEvent::Kind;
      bool connected = true;
      if (event.kind == Kind::kReasoningDelta ||
          event.kind == Kind::kContentDelta) {
        const bool reasoning = event.kind == Kind::kReasoningDelta;
        if constexpr (std::is_same_v<Output, ResponsesOutput>) {
          connected = output.Append(event.text, reasoning,
                                    probabilities && !reasoning
                                        ? probabilities->Take(event.spans)
                                        : json::Value::array());
        } else {
          connected = output.Append(event.text, reasoning);
        }
      } else if (event.kind == Kind::kToolDelta) {
        connected = output.Tool({
            .index = event.index,
            .id = event.begin ? RandomId("call_") : std::string(),
            .name = std::string(event.name),
            .arguments = std::string(event.arguments),
            .complete = event.complete,
        });
      }
      if (connected)
        return true;
      generation->Cancel();
      return false;
    };
    try {
      TextGenerationBackend::ProgressCallback on_progress;
      if (writer && chat.return_progress) {
        on_progress =
            [&](const TextGenerationBackend::PromptProgress& progress) {
              return begin() && output.Progress(progress);
            };
      }
      const auto result =
          (writer || observe)
              ? generation->Wait(
                    [&](const TokenPiece& piece) {
                      if (probabilities)
                        probabilities->Push(piece);
                      return begin() && parser.Feed(piece.text, on_event);
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
      if (!writer && !observe && !parser.Feed(result.text, on_event))
        return json::Value();
      if (probabilities)
        probabilities->Push(TokenPiece(""), true);
      const bool interrupted =
          result.finish_reason != TextGenerationBackend::FinishReason::kStop;
      if (!parser.Finish(!interrupted, interrupted, on_event)) {
        generation->Cancel();
        return json::Value();
      }
      return output.Complete(result);
    } catch (const std::exception& error) {
      if (!writer || !started)
        throw;
      const auto* generation_error =
          dynamic_cast<const TextGenerationError*>(&error);
      // Retain the cause in logs alongside the client's stable code.
      if (generation_error == nullptr)
        Logger::Error("chat", error.what());
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

}  // namespace

HttpResponse TextGenerationErrorResponse(const TextGenerationError& exception) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = exception.what();
  error["type"] = "server_error";
  error["code"] = exception.stable_code();
  if (exception.required_bytes().has_value())
    error["required_bytes"] = *exception.required_bytes();
  if (exception.remaining_bytes().has_value())
    error["remaining_bytes"] = *exception.remaining_bytes();
  response["error"] = std::move(error);
  if (exception.required_bytes().has_value() ||
      exception.remaining_bytes().has_value()) {
    json::Value gufo = json::Value::object();
    gufo["code"] = exception.stable_code();
    if (exception.required_bytes().has_value())
      gufo["required_bytes"] = *exception.required_bytes();
    if (exception.remaining_bytes().has_value())
      gufo["remaining_bytes"] = *exception.remaining_bytes();
    json::Value usage = json::Value::object();
    usage["gufo"] = std::move(gufo);
    response["usage"] = std::move(usage);
  }
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

HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream, std::string_view model) {
  return CreateCompatibilityResponse<ResponsesOutput>(
      request, backend, chat, max_tokens, sampling, stream, model);
}

HttpResponse CreateAnthropicMessage(const HttpRequest& request,
                                    TextGenerationBackend& backend,
                                    const ChatRequest& chat,
                                    std::size_t max_tokens,
                                    const sampling::SamplingConfig& sampling,
                                    bool stream, std::string_view model) {
  return CreateCompatibilityResponse<AnthropicOutput>(
      request, backend, chat, max_tokens, sampling, stream, model);
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
    return TextGenerationErrorResponse(exception);
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

HttpResponse CountOpenAiInputTokens(const ChatRequest& chat,
                                    TextGenerationBackend& backend) {
  try {
    const auto tokens = backend.count_chat_tokens(chat);
    if (!tokens) {
      json::Value response = json::Value::object();
      json::Value error = json::Value::object();
      error["message"] = "endpoint not implemented on this text model";
      error["type"] = "server_error";
      error["code"] = "not_implemented";
      response["error"] = std::move(error);
      return {
          .status = 501, .reason = "Not Implemented", .body = response.dump()};
    }
    json::Value response = json::Value::object();
    response["object"] = "response.input_tokens";
    response["input_tokens"] = *tokens;
    HttpResponse counted;
    counted.body = response.dump();
    return counted;
  } catch (const TextGenerationError& exception) {
    return TextGenerationErrorResponse(exception);
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

HttpResponse HandleOpenAiChatInputTokens(const HttpRequest& request,
                                         TextGenerationBackend& backend) {
  ParsedChatRequest parsed;
  if (auto error = ParseRequest(request, backend, &parsed))
    return std::move(*error);
  return CountOpenAiInputTokens(parsed.chat, backend);
}

HttpResponse HandleApplyTemplate(const HttpRequest& request,
                                 TextGenerationBackend& backend) {
  ParsedChatRequest parsed;
  if (auto error = ParseRequest(request, backend, &parsed,
                                /*require_model=*/false);
      error.has_value()) {
    return std::move(*error);
  }
  // As in llama-server, `stream` is ignored: the prompt is one JSON response.
  try {
    auto prompt = backend.render_chat_prompt(parsed.chat);
    if (!prompt) {
      json::Value response = json::Value::object();
      json::Value error = json::Value::object();
      error["message"] = "endpoint not implemented on this text model";
      error["type"] = "server_error";
      error["code"] = "not_implemented";
      response["error"] = std::move(error);
      HttpResponse unavailable;
      unavailable.status = 501;
      unavailable.reason = "Not Implemented";
      unavailable.body = response.dump();
      return unavailable;
    }
    json::Value response = json::Value::object();
    response["prompt"] = std::move(*prompt);
    HttpResponse rendered;
    rendered.body = response.dump();
    return rendered;
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

}  // namespace gufo::server

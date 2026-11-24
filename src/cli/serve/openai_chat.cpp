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
#include "src/cli/serve/chat_request.hpp"
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

// Chat Completions bodies decode in Rust (src/core/chat_request); see
// src/cli/serve/chat_request.cpp.
std::optional<HttpResponse> ParseRequest(const HttpRequest& request,
                                         TextGenerationBackend& backend,
                                         ParsedChatRequest* output,
                                         bool require_model = true) {
  DecodedChatRequest decoded;
  if (auto error =
          DecodeChatRequest(ChatApi::kChat, request.body,
                            ChatEnvironment(backend), require_model, &decoded))
    return error;
  output->chat = std::move(decoded.chat);
  output->chat.client_id = request.client_id;
  output->model = decoded.model.value_or(std::string());
  if (decoded.max_tokens)
    output->max_tokens = *decoded.max_tokens;
  output->sampling = std::move(decoded.sampling);
  output->stream = decoded.stream;
  output->include_usage = decoded.include_usage;
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

// Thin shims over the Rust decoder, kept for their callers' json::Value
// interface.
bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error) {
  return DecodeResponseItem(item, message, budget, error);
}

std::optional<HttpResponse> ParseOpenAiResponseControls(const json::Value& body,
                                                        ChatRequest* chat) {
  return DecodeResponseControls(body, chat);
}

std::optional<HttpResponse> ParseAnthropicToolControls(const json::Value& body,
                                                       ChatRequest* chat) {
  return DecodeAnthropicToolControls(body, chat);
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

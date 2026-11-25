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

#include "gufo/chat_response.h"
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

GufoChatBytes CBytes(std::string_view text) noexcept {
  return {text.empty() ? nullptr
                       : reinterpret_cast<const std::uint8_t*>(text.data()),
          text.size()};
}

bool WriteToString(void* ctx, const std::uint8_t* data,
                   std::size_t len) noexcept {
  auto* out = static_cast<std::string*>(ctx);
  if (!out)
    return false;
  try {
    if (len > 0)
      out->append(reinterpret_cast<const char*>(data), len);
    return true;
  } catch (...) {
    return false;
  }
}

bool WriteToBodyWriter(void* ctx, const std::uint8_t* data,
                       std::size_t len) noexcept {
  const auto* writer = static_cast<const HttpResponse::BodyWriter*>(ctx);
  if (!writer || !*writer)
    return true;
  try {
    return (*writer)(
        len == 0 ? std::string_view{}
                 : std::string_view(reinterpret_cast<const char*>(data), len));
  } catch (...) {
    return false;
  }
}

GufoChatFinishReason CFinishReason(
    TextGenerationBackend::FinishReason reason) noexcept {
  switch (reason) {
    case TextGenerationBackend::FinishReason::kLength:
      return GUFO_CHAT_FINISH_LENGTH;
    case TextGenerationBackend::FinishReason::kStopSequence:
      return GUFO_CHAT_FINISH_STOP_SEQUENCE;
    case TextGenerationBackend::FinishReason::kStop:
    default:
      return GUFO_CHAT_FINISH_STOP;
  }
}

struct MetricsHolder {
  std::string component_cache_json;
  std::string timings_json;
  GufoChatMetrics raw{};

  MetricsHolder(const MetricsHolder&) = delete;
  MetricsHolder& operator=(const MetricsHolder&) = delete;
  MetricsHolder(MetricsHolder&&) = delete;
  MetricsHolder& operator=(MetricsHolder&&) = delete;

  explicit MetricsHolder(const TextGenerationBackend::Result& result) {
    if (result.component_cache)
      component_cache_json = ComponentCacheJson(*result.component_cache).dump();
    timings_json = GenerationTimings(result).dump();
    raw = GufoChatMetrics{
        .prompt_tokens = result.prompt_tokens,
        .completion_tokens = result.completion_tokens,
        .cached_prompt_tokens = result.cached_prompt_tokens,
        .reasoning_tokens = result.reasoning_tokens,
        .draft_tokens = result.draft_tokens,
        .draft_accepted_tokens = result.draft_accepted_tokens,
        .draft_rounds = result.draft_rounds,
        .prompt_tokens_per_second = PrefillTokensPerSecond(result),
        .decode_ms = result.decode_ms,
        .cache_hit = result.cache_hit,
        .cache_miss_reason = CBytes(result.cache_miss_reason),
        .cache_common_prefix_tokens = result.cache_common_prefix_tokens,
        .cache_checkpoint_tokens = result.cache_checkpoint_tokens,
        .cache_restore_bytes = result.cache_restore_bytes,
        .cache_snapshot_bytes = result.cache_snapshot_bytes,
        .cache_disk_queued_bytes = result.cache_disk_queued_bytes,
        .cache_restore_ms = result.cache_restore_ms,
        .cache_snapshot_ms = result.cache_snapshot_ms,
        .cache_disk_enqueue_ms = result.cache_disk_enqueue_ms,
        .cache_disk_hit = result.cache_disk_hit,
        .prefill_tokens = result.prefill_tokens,
        .prefill_chunks = result.prefill_chunks,
        .active_decode_prefill_chunks = result.active_decode_prefill_chunks,
        .max_prefill_chunk_tokens = result.max_prefill_chunk_tokens,
        .max_prefill_layer_groups = result.max_prefill_layer_groups,
        .prefill_yield_rounds = result.prefill_yield_rounds,
        .queue_depth_at_submit = result.queue_depth_at_submit,
        .client_queue_depth_at_submit = result.client_queue_depth_at_submit,
        .resident_requests_at_admission = result.resident_requests_at_admission,
        .requested_logical_concurrency = result.requested_logical_concurrency,
        .physical_execution_width = result.physical_execution_width,
        .queue_ms = result.queue_ms,
        .shared_prefix_wait_ms = result.shared_prefix_wait_ms,
        .prefill_ms = result.prefill_ms,
        .ttft_ms = result.ttft_ms,
        .mean_inter_token_ms = result.mean_inter_token_ms,
        .max_inter_token_ms = result.max_inter_token_ms,
        .execution_plan = CBytes(result.execution_plan),
        .component_cache_json = CBytes(component_cache_json),
        .timings_json = CBytes(timings_json),
    };
  }
};

HttpResponse Error(int status, const char* reason, std::string message,
                   const char* code) {
  std::string body;
  (void)gufo_chat_error_response(CBytes(message),
                                 CBytes("invalid_request_error"), CBytes(code),
                                 &WriteToString, &body);
  return {.status = status, .reason = reason, .body = std::move(body)};
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

struct ToolDelta {
  std::size_t index;
  std::string id;
  std::string_view name;
  std::string_view arguments;
  bool complete{false};
};

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Delegates state and JSON/SSE encoding to Rust (gufo-chat-response).
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, HttpResponse::BodyWriter writer,
                  const ChatRequest& chat,
                  std::optional<std::uint32_t> top_logprobs = std::nullopt)
      : writer_(std::move(writer)),
        handle_(nullptr, &gufo_responses_output_free) {
    const std::string id = RandomId("resp_");
    const std::string tool_choice =
        chat.tools.empty() || chat.tool_choice == ChatRequest::ToolChoice::kNone
            ? "none"
        : chat.tool_choice == ChatRequest::ToolChoice::kRequired ? "required"
                                                                 : "auto";
    std::vector<GufoChatBytes> tool_defs;
    tool_defs.reserve(chat.tools.size());
    for (const auto& tool : chat.tools)
      tool_defs.push_back(CBytes(tool.definition_json));
    std::vector<GufoChatNamespaceEntry> ns_entries;
    ns_entries.reserve(chat.tool_namespaces.size());
    for (const auto& [name, ns] : chat.tool_namespaces)
      ns_entries.push_back({CBytes(name), CBytes(ns)});
    const GufoResponsesConfig cfg{
        .id = CBytes(id),
        .created_at = Now(),
        .model = CBytes(model),
        .parallel_tool_calls = !chat.tools.empty() && chat.parallel_tool_calls,
        .tool_choice = CBytes(tool_choice),
        .forced_tool_name = CBytes(chat.forced_tool_name),
        .tool_definitions_json = tool_defs.empty() ? nullptr : tool_defs.data(),
        .tool_definitions_len = tool_defs.size(),
        .namespaces = ns_entries.empty() ? nullptr : ns_entries.data(),
        .namespaces_len = ns_entries.size(),
        .cache_persist = chat.cache_persist,
        .has_top_logprobs = top_logprobs.has_value(),
        .top_logprobs = top_logprobs.value_or(0),
    };
    handle_.reset(gufo_responses_output_new(&cfg, static_cast<bool>(writer_)));
  }

  bool Begin() {
    return gufo_responses_output_begin(handle_.get(), &WriteToBodyWriter,
                                       &writer_) == GUFO_CHAT_RESPONSE_OK;
  }

  bool Progress(const TextGenerationBackend::PromptProgress& progress) {
    const std::string pp = PromptProgressJson(progress).dump();
    return gufo_responses_output_progress(handle_.get(), CBytes(pp),
                                          &WriteToBodyWriter,
                                          &writer_) == GUFO_CHAT_RESPONSE_OK;
  }

  bool Append(std::string_view text, bool reasoning,
              json::Value logprobs = json::Value::array()) {
    if (text.empty())
      return true;
    std::string item_id;
    if (!item_open_ || item_reasoning_ != reasoning) {
      item_id = RandomId(reasoning ? "rs_" : "msg_");
      item_open_ = true;
      item_reasoning_ = reasoning;
    }
    std::string lp_json;
    std::string stream_lp_json;
    GufoChatBytes stream_lp_bytes = CBytes("[]");
    if (!reasoning && !logprobs.items().empty()) {
      lp_json = logprobs.dump();
      stream_lp_json = ResponsesStreamLogprobs(logprobs).dump();
      stream_lp_bytes = CBytes(stream_lp_json);
    }
    return gufo_responses_output_append(handle_.get(), CBytes(text), reasoning,
                                        CBytes(item_id), CBytes(lp_json),
                                        stream_lp_bytes, &WriteToBodyWriter,
                                        &writer_) == GUFO_CHAT_RESPONSE_OK;
  }

  bool Tool(const ToolDelta& call) {
    if (!call.id.empty())
      item_open_ = false;
    const std::string fc_id = call.id.empty() ? std::string() : RandomId("fc_");
    const GufoChatToolDelta d{
        .index = call.index,
        .begin = !call.id.empty(),
        .id = CBytes(call.id),
        .name = CBytes(call.name),
        .arguments = CBytes(call.arguments),
        .complete = call.complete,
    };
    return gufo_responses_output_tool(handle_.get(), &d, CBytes(fc_id),
                                      &WriteToBodyWriter,
                                      &writer_) == GUFO_CHAT_RESPONSE_OK;
  }

  std::string Complete(const TextGenerationBackend::Result& result) {
    const MetricsHolder metrics(result);
    std::string body;
    (void)gufo_responses_output_complete(
        handle_.get(), CFinishReason(result.finish_reason), &metrics.raw,
        &WriteToBodyWriter, &writer_, &WriteToString, &body);
    return body;
  }

  bool Fail(std::string_view message) {
    return gufo_responses_output_fail(handle_.get(), CBytes(message),
                                      &WriteToBodyWriter,
                                      &writer_) == GUFO_CHAT_RESPONSE_OK;
  }

private:
  HttpResponse::BodyWriter writer_;
  std::unique_ptr<GufoResponsesOutput, void (*)(GufoResponsesOutput*)> handle_;
  bool item_open_{false};
  bool item_reasoning_{false};
};

HttpResponse NonStreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>& generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  (void)backend;
  const auto result = generation->Wait();
  const auto generated = ParseGeneratedText(
      ParseSetup(request.chat, initial_output_state,
                 generation->ReasoningDelimiters(),
                 SelectGenerationMarkup(generation->ToolDialectId())),
      result.text,
      result.finish_reason == TextGenerationBackend::FinishReason::kStop);

  const std::string id = RandomId("chatcmpl-");
  std::vector<std::string> call_ids;
  call_ids.reserve(generated.calls.size());
  std::vector<GufoChatToolCall> c_calls;
  c_calls.reserve(generated.calls.size());
  for (const auto& call : generated.calls) {
    call_ids.push_back(RandomId("call_"));
    c_calls.push_back(
        {CBytes(call_ids.back()), CBytes(call.name), CBytes(call.arguments)});
  }
  std::string logprobs_json;
  if (request.sampling.logprobs) {
    logprobs_json =
        ChatLogprobs(result.logprobs, *request.sampling.logprobs).dump();
  }
  const MetricsHolder metrics(result);
  const GufoChatNonStreamingInput input{
      .id = CBytes(id),
      .created = Now(),
      .model = CBytes(request.model),
      .reasoning = CBytes(generated.reasoning),
      .text = CBytes(generated.text),
      .calls = c_calls.empty() ? nullptr : c_calls.data(),
      .calls_len = c_calls.size(),
      .logprobs_json = CBytes(logprobs_json),
      .finish_reason = CFinishReason(result.finish_reason),
      .cache_persist = request.chat.cache_persist,
      .metrics = &metrics.raw,
  };
  std::string body;
  (void)gufo_chat_non_streaming_response(&input, &WriteToString, &body);
  RecordServerMetrics(result);

  std::ostringstream timing;
  timing << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
         << ", inter_token;dur=" << result.mean_inter_token_ms
         << ", max_inter_token;dur=" << result.max_inter_token_ms;

  return {
      .status = 200,
      .reason = "OK",
      .body = std::move(body),
      .headers = {{"Server-Timing", timing.str()}},
      .streaming_body = {},
      .log_details = GenerationLogDetails(result),
  };
}

HttpResponse StreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  (void)backend;
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
            std::unique_ptr<GufoChatStream, void (*)(GufoChatStream*)> stream(
                gufo_chat_stream_new(CBytes(id), created, CBytes(model)),
                &gufo_chat_stream_free);
            if (!stream)
              return;
            bool connected = true;
            bool started = false;
            std::string pending_logprobs;
            auto take_logprobs = [&]() -> std::string {
              std::string out;
              out.swap(pending_logprobs);
              return out;
            };
            const auto begin = [&] {
              if (!started) {
                started = true;
                connected =
                    gufo_chat_stream_role(
                        stream.get(), &WriteToBodyWriter,
                        const_cast<HttpResponse::BodyWriter*>(&writer)) ==
                    GUFO_CHAT_RESPONSE_OK;
              }
              return connected;
            };
            if (request.chat.return_progress && !begin()) {
              generation->Cancel();
              return;
            }
            GenerationStream parser(ParseSetup(
                request.chat, initial_output_state, reasoning, markup));
            std::size_t final_calls = 0;
            auto* wctx = const_cast<HttpResponse::BodyWriter*>(&writer);
            const auto on_event = [&](const GenerationEvent& event) {
              using Kind = GenerationEvent::Kind;
              switch (event.kind) {
                case Kind::kReasoningDelta:
                case Kind::kContentDelta: {
                  if (event.text.empty())
                    return true;
                  const auto lp = take_logprobs();
                  connected =
                      gufo_chat_stream_text_delta(
                          stream.get(), event.kind == Kind::kReasoningDelta,
                          CBytes(event.text), CBytes(lp), &WriteToBodyWriter,
                          wctx) == GUFO_CHAT_RESPONSE_OK;
                  return connected;
                }
                case Kind::kToolDelta: {
                  if (!event.begin && event.arguments.empty())
                    return connected;
                  const std::string call_id =
                      event.begin ? RandomId("call_") : std::string();
                  const GufoChatToolDelta td{
                      .index = event.index,
                      .begin = event.begin,
                      .id = CBytes(call_id),
                      .name = CBytes(event.name),
                      .arguments = CBytes(event.arguments),
                      .complete = event.complete,
                  };
                  const auto lp = take_logprobs();
                  connected =
                      gufo_chat_stream_tool_delta(stream.get(), &td, CBytes(lp),
                                                  &WriteToBodyWriter, wctx) ==
                      GUFO_CHAT_RESPONSE_OK;
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
                    const std::string pp = PromptProgressJson(progress).dump();
                    connected = begin() && gufo_chat_stream_progress(
                                               stream.get(), CBytes(pp),
                                               &WriteToBodyWriter,
                                               wctx) == GUFO_CHAT_RESPONSE_OK;
                    return connected;
                  };
            }

            try {
              const auto result = generation->Wait(
                  [&](const TokenPiece& piece) {
                    if (piece.logprobs && request.sampling.logprobs) {
                      pending_logprobs =
                          ChatLogprobs(std::span(piece.logprobs, 1),
                                       *request.sampling.logprobs)
                              .dump();
                    }
                    if (!begin() || !parser.Feed(piece.text, on_event))
                      return false;
                    if (!pending_logprobs.empty()) {
                      const auto lp = take_logprobs();
                      connected =
                          gufo_chat_stream_empty_delta(
                              stream.get(), CBytes(lp), &WriteToBodyWriter,
                              wctx) == GUFO_CHAT_RESPONSE_OK;
                    }
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
              const MetricsHolder metrics(result);
              (void)gufo_chat_stream_finish(
                  stream.get(), CFinishReason(result.finish_reason),
                  final_calls > 0, request.include_usage,
                  request.chat.cache_persist, &metrics.raw, &WriteToBodyWriter,
                  wctx);
            } catch (const TextGenerationError& exception) {
              if (!started)
                throw;
              stream_log->error_code = exception.stable_code();
              stream_log->error_event_sent =
                  gufo_chat_stream_error(
                      CBytes(exception.what()), CBytes(exception.stable_code()),
                      &WriteToBodyWriter, wctx) == GUFO_CHAT_RESPONSE_OK;
            } catch (const std::exception& error) {
              if (!started)
                throw;
              // Retain the cause in logs alongside the client's stable code.
              Logger::Error("chat", error.what());
              stream_log->error_code = "generation_failed";
              const char* message = error.what();
              stream_log->error_event_sent =
                  gufo_chat_stream_error(
                      CBytes(message && *message ? message
                                                 : "generation failed"),
                      CBytes("generation_failed"), &WriteToBodyWriter,
                      wctx) == GUFO_CHAT_RESPONSE_OK;
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

/// Messages output as Anthropic content blocks. Delegates state and JSON/SSE
/// encoding to Rust (gufo-chat-response).
class AnthropicOutput {
public:
  AnthropicOutput(std::string model, HttpResponse::BodyWriter writer,
                  const ChatRequest&,
                  std::optional<std::uint32_t> = std::nullopt)
      : writer_(std::move(writer)),
        handle_(nullptr, &gufo_anthropic_output_free) {
    const std::string id = RandomId("msg_");
    handle_.reset(gufo_anthropic_output_new(CBytes(id), CBytes(model),
                                            static_cast<bool>(writer_)));
  }

  bool Begin() {
    connected_ = gufo_anthropic_output_begin(handle_.get(), &WriteToBodyWriter,
                                             &writer_) == GUFO_CHAT_RESPONSE_OK;
    return connected_;
  }

  bool Progress(const TextGenerationBackend::PromptProgress&) {
    return connected_;
  }

  bool Append(std::string_view text, bool reasoning) {
    connected_ = gufo_anthropic_output_append(
                     handle_.get(), CBytes(text), reasoning, &WriteToBodyWriter,
                     &writer_) == GUFO_CHAT_RESPONSE_OK;
    return connected_;
  }

  bool Tool(const ToolDelta& call) {
    const GufoChatToolDelta d{
        .index = call.index,
        .begin = !call.id.empty(),
        .id = CBytes(call.id),
        .name = CBytes(call.name),
        .arguments = CBytes(call.arguments),
        .complete = call.complete,
    };
    std::string invalid;
    const auto status =
        gufo_anthropic_output_tool(handle_.get(), &d, &WriteToBodyWriter,
                                   &writer_, &WriteToString, &invalid);
    // Complete arguments become the block's input object, as json::parse
    // reads them; malformed ones fail the response.
    if (status == GUFO_CHAT_RESPONSE_INVALID)
      throw std::runtime_error(invalid);
    connected_ = status == GUFO_CHAT_RESPONSE_OK;
    return connected_;
  }

  std::string Complete(const TextGenerationBackend::Result& result) {
    const MetricsHolder metrics(result);
    std::string body;
    connected_ =
        gufo_anthropic_output_complete(
            handle_.get(), CFinishReason(result.finish_reason),
            CBytes(result.stop_sequence), &metrics.raw, &WriteToBodyWriter,
            &writer_, &WriteToString, &body) == GUFO_CHAT_RESPONSE_OK;
    return body;
  }

  bool Fail(std::string_view message) {
    connected_ = gufo_anthropic_output_fail(handle_.get(), CBytes(message),
                                            &WriteToBodyWriter,
                                            &writer_) == GUFO_CHAT_RESPONSE_OK;
    return connected_;
  }

private:
  HttpResponse::BodyWriter writer_;
  std::unique_ptr<GufoAnthropicOutput, void (*)(GufoAnthropicOutput*)> handle_;
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
  const auto run =
      [generation, initial, reasoning, chat, markup,
       model = requested_model.empty() ? backend.model_id()
                                       : std::string(requested_model),
       stream_log, observe, count = sampling.logprobs,
       timing](const HttpResponse::BodyWriter& writer) -> std::string {
    Output output(model, writer, chat, count);
    std::optional<ResponsesLogprobs> probabilities;
    if (observe)
      probabilities.emplace(*count);
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
      return {};
    // A buffered response reads the whole text at the end, so only the
    // final parse produces its calls.
    GenerationStream parser(ParseSetup(chat, initial, reasoning, markup,
                                       observe, !writer && !observe));
    const auto on_event = [&](const GenerationEvent& event) {
      using Kind = GenerationEvent::Kind;
      bool connected = true;
      if (event.kind == Kind::kReasoningDelta ||
          event.kind == Kind::kContentDelta) {
        const bool is_reasoning = event.kind == Kind::kReasoningDelta;
        if constexpr (std::is_same_v<Output, ResponsesOutput>) {
          connected = output.Append(event.text, is_reasoning,
                                    probabilities && !is_reasoning
                                        ? probabilities->Take(event.spans)
                                        : json::Value::array());
        } else {
          connected = output.Append(event.text, is_reasoning);
        }
      } else if (event.kind == Kind::kToolDelta) {
        connected = output.Tool({
            .index = event.index,
            .id = event.begin ? RandomId("call_") : std::string(),
            .name = event.name,
            .arguments = event.arguments,
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
        return {};
      if (!begin())
        return {};
      if (!writer && !observe && !parser.Feed(result.text, on_event))
        return {};
      if (probabilities)
        probabilities->Push(TokenPiece(""), true);
      const bool interrupted =
          result.finish_reason != TextGenerationBackend::FinishReason::kStop;
      if (!parser.Finish(!interrupted, interrupted, on_event)) {
        generation->Cancel();
        return {};
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
      return {};
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
  auto body = run({});
  return {.status = 200,
          .reason = "OK",
          .body = std::move(body),
          .headers = {{"Server-Timing", *timing}},
          .log_details = stream_log->details};
}

}  // namespace

HttpResponse TextGenerationErrorResponse(const TextGenerationError& exception) {
  std::string body;
  (void)gufo_chat_text_generation_error_response(
      CBytes(exception.what()), CBytes(exception.stable_code()),
      exception.required_bytes().has_value(),
      exception.required_bytes().value_or(0),
      exception.remaining_bytes().has_value(),
      exception.remaining_bytes().value_or(0), &WriteToString, &body);
  HttpResponse output{
      .status = exception.http_status(),
      .reason = StatusReason(exception.http_status()),
      .body = std::move(body),
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
      std::string body;
      (void)gufo_chat_error_response(
          CBytes("endpoint not implemented on this text model"),
          CBytes("server_error"), CBytes("not_implemented"), &WriteToString,
          &body);
      return {
          .status = 501, .reason = "Not Implemented", .body = std::move(body)};
    }
    std::string body;
    (void)gufo_chat_input_tokens_response(*tokens, &WriteToString, &body);
    HttpResponse counted;
    counted.body = std::move(body);
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
      std::string body;
      (void)gufo_chat_error_response(
          CBytes("endpoint not implemented on this text model"),
          CBytes("server_error"), CBytes("not_implemented"), &WriteToString,
          &body);
      HttpResponse unavailable;
      unavailable.status = 501;
      unavailable.reason = "Not Implemented";
      unavailable.body = std::move(body);
      return unavailable;
    }
    std::string body;
    (void)gufo_chat_rendered_prompt_response(CBytes(*prompt), &WriteToString,
                                             &body);
    HttpResponse rendered;
    rendered.body = std::move(body);
    return rendered;
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

}  // namespace gufo::server

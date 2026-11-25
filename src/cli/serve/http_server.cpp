#include "src/cli/serve/http_server.hpp"

#include <gufo/http.h>
#include <gufo/http_server.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <system_error>
#include <thread>
#include <utility>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/chat_request.hpp"
#include "src/cli/serve/decision_service.hpp"
#include "src/cli/serve/embedding_service.hpp"
#include "src/cli/serve/embeddings_api.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/sdapi.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/cli/serve/trace.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/shutdown_signal.hpp"
#include "src/core/rust/callback.hpp"
#include "src/core/utf8.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::server {
namespace {

std::atomic<int> shutdown_signal{0};
static_assert(std::atomic<int>::is_always_lock_free);

void RequestShutdown(int signal) noexcept {
  // A signal may arrive on any model/HTTP worker. Only a lock-free atomic
  // operation belongs here; socket shutdown and joins run in the accept loop.
  shutdown_signal.store(signal, std::memory_order_relaxed);
}

class ShutdownSignals {
public:
  ShutdownSignals() {
    // Reset before installation: a signal landing between the previous
    // guard's uninstall and this store must not be erased afterwards.
    shutdown_signal.store(0, std::memory_order_relaxed);
    guard_ = platform::ShutdownSignalGuard(&RequestShutdown);
  }

private:
  platform::ShutdownSignalGuard guard_;
};

// ---------------------------------------------------------------------------
// Request / response helpers
// ---------------------------------------------------------------------------

std::string ToLower(std::string_view s) {
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

bool HasHeader(const HttpResponse& response, std::string_view name) {
  const std::string lowered = ToLower(name);
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return ToLower(header.first) == lowered;
  });
}

// Text generation routes, whose bodies and replies `--trace` records. Media
// routes carry audio, image and video payloads the trace does not explain.
bool IsTracedRoute(const HttpRequest& req) {
  return req.method == "POST" &&
         (req.path == "/v1/completions" || req.path == "/v1/chat/completions" ||
          req.path == "/v1/responses" || req.path == "/v1/messages" ||
          req.path == "/completion");
}

void TraceRequest(const HttpRequest& req) {
  auto record = Trace::Record("request", req.request_id);
  record["method"] = req.method;
  record["path"] = req.path;
  record["body"] = Trace::Text(req.body);
  Trace::Write(record);
}

void TraceResponse(std::string_view request_id, int status,
                   std::string_view outcome, bool stream,
                   std::string_view body) {
  auto record = Trace::Record("response", request_id);
  record["status"] = status;
  record["outcome"] = std::string(outcome);
  record["stream"] = stream;
  record["body"] = Trace::Text(body);
  Trace::Write(record);
}

bool IsEventStream(const HttpResponse& response) {
  return std::ranges::any_of(response.headers, [](const auto& header) {
    return ToLower(header.first) == "content-type" &&
           ToLower(header.second).starts_with("text/event-stream");
  });
}

std::vector<std::pair<std::string, std::string>> BuildResponseHeaders(
    const HttpResponse& resp) {
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(resp.headers.size() + 4);
  if (resp.status != 101 && !HasHeader(resp, "content-type")) {
    out.emplace_back("Content-Type", "application/json");
  }
  out.emplace_back("Access-Control-Allow-Origin", "*");
  out.emplace_back("Access-Control-Allow-Methods",
                   "GET, POST, DELETE, OPTIONS");
  // Authorization is accepted (and ignored) so browser SDKs that always send
  // it still pass their CORS preflight.
  out.emplace_back("Access-Control-Allow-Headers",
                   "Content-Type, Authorization, Range");
  for (const auto& header : resp.headers) {
    out.push_back(header);
  }
  return out;
}

std::vector<GufoHttpHeaderView> ToHeaderViews(
    const std::vector<std::pair<std::string, std::string>>& headers) {
  std::vector<GufoHttpHeaderView> views;
  views.reserve(headers.size());
  for (const auto& [name, value] : headers) {
    views.push_back(GufoHttpHeaderView{
        .name = {reinterpret_cast<const std::uint8_t*>(name.data()),
                 name.size()},
        .value = {reinterpret_cast<const std::uint8_t*>(value.data()),
                  value.size()},
    });
  }
  return views;
}

// ---------------------------------------------------------------------------
// Response constructors
// ---------------------------------------------------------------------------

long long Now() {
  return static_cast<long long>(std::time(nullptr));
}

std::string RandomId() {
  static constexpr char kChars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> dist(0, 35);
  std::string out;
  out.reserve(12);
  for (int i = 0; i < 12; ++i) {
    out += kChars[dist(gen)];
  }
  return out;
}

HttpResponse Ok(const json::Value& v) {
  return {.status = 200, .reason = "OK", .body = v.dump()};
}

json::Value UsageJson(const TextGenerationBackend::Result& result) {
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
  return usage;
}

HttpResponse WithTiming(HttpResponse response,
                        const TextGenerationBackend::Result& result) {
  RecordServerMetrics(result);

  std::ostringstream value;
  value << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
        << ", inter_token;dur=" << result.mean_inter_token_ms
        << ", max_inter_token;dur=" << result.max_inter_token_ms;
  response.headers.emplace_back("Server-Timing", value.str());

  response.log_details = GenerationLogDetails(result);
  return response;
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
// `status` matches the health/readiness shape and `error` the API error shape.
HttpResponse DeviceLost() {
  json::Value error = json::Value::object();
  error["message"] = kDeviceLostMessage;
  error["type"] = "server_error";
  error["code"] = "device_lost";
  json::Value body = json::Value::object();
  body["status"] = "device_lost";
  body["error"] = std::move(error);
  return {.status = 503, .reason = "Service Unavailable", .body = body.dump()};
}

HttpResponse NotImplemented(const HttpRequest&, TextGenerationBackend&) {
  return Err(501, "Not Implemented",
             "endpoint not implemented on this text model", "server_error",
             "not_implemented");
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

// The served name a validated request selected; responses echo it. A request
// without one is answered under the primary name.
std::string RequestedModel(const json::Value& body,
                           const TextGenerationBackend& backend) {
  const auto* model = body.find("model");
  return model != nullptr && model->is_string() ? model->str()
                                                : backend.model_id();
}

// Validate the text subset before dispatch so a client never gets an answer
// to a different request. Responses also supports streamed output.
std::optional<HttpResponse> ReadCompatibilityOptions(
    const json::Value& body, TextGenerationBackend& backend,
    std::string_view token_field, std::size_t* max_tokens,
    sampling::SamplingConfig* sampling_config, std::optional<bool> thinking,
    const CompatibilityAllowances& allowances = {}) {
  if (!body.is_object())
    return InvalidCompatibilityRequest("request body must be an object");
  if (const auto* model = body.find("model"); model != nullptr) {
    if (!model->is_string())
      return InvalidCompatibilityRequest("'model' must be a string");
    if (!backend.serves_model(model->str()))
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
  const auto defaults = backend.sampling_defaults();
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

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

// The primary ID followed by its accepted aliases, in listing order.
std::vector<std::string> ServedIds(const std::string& primary,
                                   const std::vector<std::string>& aliases) {
  std::vector<std::string> ids{primary};
  ids.insert(ids.end(), aliases.begin(), aliases.end());
  return ids;
}

HttpResponse ListModels(TextGenerationBackend* backend,
                        const VideoJobService* video_jobs,
                        const TtsService* tts, const AsrService* asr,
                        const ImageService* images,
                        const EmbeddingService* embeddings,
                        const DecisionService* decisions) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  json::Value data = json::Value::array();
  for (const auto& id :
       backend != nullptr ? backend->model_ids() : std::vector<std::string>{}) {
    json::Value model = json::Value::object();
    model["id"] = id;
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "gufo";
    if (backend->max_context() > 0)
      model["context_length"] =
          static_cast<std::size_t>(backend->max_context());
    json::Value input_modalities = json::Value::array();
    input_modalities.push_back("text");
    if (backend->supports_images())
      input_modalities.push_back("image");
    json::Value architecture = json::Value::object();
    architecture["input_modalities"] = std::move(input_modalities);
    model["architecture"] = std::move(architecture);
    data.push_back(std::move(model));
  }
  if (video_jobs != nullptr && video_jobs->ready()) {
    json::Value root_model = json::Value::object();
    root_model["id"] = "minimax-h3";
    root_model["object"] = "model";
    root_model["created"] = Now();
    root_model["owned_by"] = "operator-supplied-minimax";
    root_model["capability"] = "video";
    data.push_back(std::move(root_model));
    for (const std::string_view preset :
         {"minimax-h3-exact", "minimax-h3-fast", "minimax-h3-aggressive",
          "minimax-h3-dev", "minimax-h3-fullres"}) {
      json::Value model = json::Value::object();
      model["id"] = std::string(preset);
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-minimax";
      model["root"] = "minimax-h3";
      model["capability"] = "video";
      data.push_back(std::move(model));
    }
    // Configured `--served-model-name` aliases of the root model (#957).
    for (const auto& alias : video_jobs->model_aliases()) {
      json::Value model = json::Value::object();
      model["id"] = alias;
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-minimax";
      model["root"] = "minimax-h3";
      model["capability"] = "video";
      data.push_back(std::move(model));
    }
  }
  if (tts != nullptr && tts->ready()) {
    for (const auto& id : ServedIds(tts->model_id(), tts->model_aliases())) {
      json::Value model = json::Value::object();
      model["id"] = id;
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-qwen";
      model["capability"] = "audio_tts";
      data.push_back(std::move(model));
    }
  }
  if (asr != nullptr && asr->ready()) {
    for (const auto& id : ServedIds(asr->model_id(), asr->model_aliases())) {
      json::Value model = json::Value::object();
      model["id"] = id;
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-qwen";
      model["capability"] = "audio_asr";
      data.push_back(std::move(model));
    }
  }
  if (images != nullptr) {
    for (const auto& id :
         ServedIds(images->model_id(), images->model_aliases())) {
      auto model = json::Value::object();
      model["id"] = id;
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-qwen";
      model["capability"] = "image";
      data.push_back(std::move(model));
    }
  }
  if (embeddings != nullptr && embeddings->ready()) {
    for (const auto& id :
         ServedIds(embeddings->model_id(), embeddings->model_aliases())) {
      json::Value model = json::Value::object();
      model["id"] = id;
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-gemma";
      model["capability"] = "embedding";
      data.push_back(std::move(model));
    }
  }
  for (const auto& decision_id :
       decisions != nullptr
           ? ServedIds(decisions->model_id, decisions->model_aliases)
           : std::vector<std::string>{}) {
    auto model = json::Value::object();
    model["id"] = decision_id;
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "gufo";
    model["capability"] = "decision";
    auto modalities = json::Value::array();
    modalities.push_back("text");
    if (decisions->multimodal) {
      modalities.push_back("image");
      modalities.push_back("audio");
    }
    auto architecture = json::Value::object();
    architecture["input_modalities"] = modalities;
    model["architecture"] = architecture;
    data.push_back(std::move(model));
  }
  resp["data"] = std::move(data);
  return Ok(resp);
}

HttpResponse OpenAiCompletions(const HttpRequest& req,
                               TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(body, b, "max_tokens", &max_tokens,
                                            &sampling_config,
                                            b.reasoning_defaults().enabled,
                                            {
                                                .stop_field = "stop",
                                                .stream = true,
                                                .stream_options = true,
                                                .ignore_eos = true,
                                                .logprobs = true,
                                            })) {
    return std::move(*error);
  }
  if (const auto error = ParseLogprobs(body, false, &sampling_config))
    return InvalidCompatibilityRequest(error->message);
  const auto* stream_field = body.find("stream");
  const bool stream = stream_field != nullptr && stream_field->as_bool();
  bool include_usage = false;
  if (const auto* options = body.find("stream_options")) {
    if (!stream || !options->is_object())
      return InvalidCompatibilityRequest(
          "'stream_options' requires streaming and must be an object");
    for (const auto& [key, value] : options->members()) {
      if (key != "include_usage" || !value.is_bool())
        return InvalidCompatibilityRequest(
            "only boolean 'stream_options.include_usage' is supported");
      include_usage = value.as_bool();
    }
  }
  bool ignore_eos = false;
  if (const auto* value = body.find("ignore_eos")) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'ignore_eos' must be a boolean");
    ignore_eos = value->as_bool();
  }
  bool return_progress = false;
  if (const auto* value = body.find("return_progress");
      value != nullptr && !value->is_null()) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'return_progress' must be a boolean");
    return_progress = value->as_bool();
  }
  bool cache_prompt = true;
  if (const auto* value = body.find("cache_prompt");
      value != nullptr && !value->is_null()) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'cache_prompt' must be a boolean");
    cache_prompt = value->as_bool();
  }
  bool cache_persist = true;
  if (const auto* value = body.find("cache_persist");
      value != nullptr && !value->is_null()) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'cache_persist' must be a boolean");
    cache_persist = value->as_bool();
  }
  std::vector<std::string> stop_sequences;
  if (const auto error = ParseStopSequences(
          body.find("stop"), StopSequenceFormat::kOpenAi, &stop_sequences))
    return InvalidCompatibilityRequest(*error);

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string()) {
    return InvalidCompatibilityRequest("'prompt' must be a single string");
  }
  const std::string prompt = input->str();
  if (prompt.empty()) {
    return Err(400, "Bad Request", "'prompt' is required",
               "invalid_request_error", "missing_prompt");
  }

  auto generation =
      b.start_complete(prompt, max_tokens, sampling_config, req.is_cancelled,
                       stream, ignore_eos, req.client_id, stop_sequences,
                       return_progress, cache_prompt, cache_persist);
  const std::string id = "cmpl-" + RandomId();
  const long long created = Now();
  const std::string model = RequestedModel(body, b);
  if (stream) {
    auto stream_log = std::make_shared<HttpResponse::StreamLog>();
    return {
        .status = 200,
        .reason = "OK",
        .body = {},
        .headers = {{"Content-Type", "text/event-stream"},
                    {"Cache-Control", "no-cache"},
                    {"X-Accel-Buffering", "no"}},
        .streaming_body =
            [generation = std::move(generation), id, created, model,
             include_usage, return_progress, cache_persist,
             logprobs = sampling_config.logprobs,
             stream_log](const HttpResponse::BodyWriter& writer) {
              bool started = return_progress;
              std::optional<json::Value> pending_logprobs;
              std::size_t text_offset = 0;
              const auto write_chunk =
                  [&](std::string_view piece, std::string_view finish_reason,
                      const json::Value* usage = nullptr,
                      const json::Value* timings = nullptr,
                      const json::Value* progress = nullptr) {
                    json::Value chunk = json::Value::object();
                    chunk["id"] = id;
                    chunk["object"] = "text_completion";
                    chunk["created"] = created;
                    chunk["model"] = model;
                    json::Value choices = json::Value::array();
                    if (usage == nullptr) {
                      json::Value choice = json::Value::object();
                      choice["text"] = std::string(piece);
                      choice["index"] = 0;
                      choice["logprobs"] = pending_logprobs
                                               ? std::move(*pending_logprobs)
                                               : json::Value();
                      pending_logprobs.reset();
                      choice["finish_reason"] =
                          finish_reason.empty()
                              ? json::Value()
                              : json::Value(std::string(finish_reason));
                      choices.push_back(std::move(choice));
                    }
                    chunk["choices"] = std::move(choices);
                    if (usage != nullptr)
                      chunk["usage"] = *usage;
                    if (timings != nullptr)
                      chunk["timings"] = *timings;
                    // The finishing and usage chunks echo the applied value.
                    if (usage != nullptr || timings != nullptr)
                      chunk["cache_persist"] = cache_persist;
                    if (progress != nullptr)
                      chunk["prompt_progress"] = *progress;
                    started = true;
                    return writer("data: " + chunk.dump() + "\n\n");
                  };
              core::Utf8Decoder decoder;
              bool connected = true;
              TextGenerationBackend::ProgressCallback on_progress;
              if (return_progress) {
                on_progress =
                    [&](const TextGenerationBackend::PromptProgress& value) {
                      const auto progress = PromptProgressJson(value);
                      connected =
                          write_chunk({}, {}, nullptr, nullptr, &progress);
                      return connected;
                    };
              }
              try {
                const auto result = generation->Wait(
                    [&](const TokenPiece& piece) {
                      if (logprobs && piece.logprobs)
                        pending_logprobs = CompletionLogprobs(
                            std::span(piece.logprobs, 1), text_offset);
                      const auto text = decoder.Push(piece.text, false);
                      text_offset += UnicodeLength(text);
                      connected = write_chunk(text, {});
                      return connected;
                    },
                    on_progress,
                    [&] {
                      // Admitted or long-queued: later failures are SSE.
                      started = true;
                      connected = writer({});
                      return connected;
                    });
                stream_log->details = GenerationLogDetails(result);
                RecordServerMetrics(result);
                if (!connected || result.cancelled)
                  return;
                const auto trailing = decoder.Push({}, true);
                const auto timings = GenerationTimings(result);
                if (!write_chunk(
                        trailing,
                        result.finish_reason ==
                                TextGenerationBackend::FinishReason::kLength
                            ? "length"
                            : "stop",
                        nullptr, &timings))
                  return;
                if (include_usage) {
                  const auto usage = UsageJson(result);
                  if (!write_chunk({}, {}, &usage))
                    return;
                }
                (void)writer("data: [DONE]\n\n");
              } catch (const TextGenerationError& error) {
                if (!started)
                  throw;
                stream_log->error_code = error.stable_code();
                json::Value detail = json::Value::object();
                detail["message"] = error.what();
                detail["type"] = "server_error";
                detail["code"] = error.stable_code();
                json::Value event = json::Value::object();
                event["error"] = std::move(detail);
                stream_log->error_event_sent =
                    writer("data: " + event.dump() + "\n\n");
                (void)writer("data: [DONE]\n\n");
              } catch (const std::exception& error) {
                if (!started)
                  throw;
                // Retain the cause in logs alongside the client's stable code.
                Logger::Error("chat", error.what());
                stream_log->error_code = "generation_failed";
                json::Value detail = json::Value::object();
                const char* message = error.what();
                detail["message"] =
                    message && *message ? message : "generation failed";
                detail["type"] = "server_error";
                detail["code"] = "generation_failed";
                json::Value event = json::Value::object();
                event["error"] = std::move(detail);
                stream_log->error_event_sent =
                    writer("data: " + event.dump() + "\n\n");
                (void)writer("data: [DONE]\n\n");
              }
            },
        .stream_log = std::move(stream_log),
        .defer_stream_headers = !return_progress,
    };
  }

  const auto res = generation->Wait();

  json::Value resp = json::Value::object();
  resp["id"] = id;
  resp["object"] = "text_completion";
  resp["created"] = created;
  resp["model"] = model;
  json::Value choices = json::Value::array();
  json::Value c = json::Value::object();
  c["text"] = core::Utf8Decoder{}.Push(res.text, true);
  c["index"] = 0;
  c["logprobs"] = sampling_config.logprobs ? CompletionLogprobs(res.logprobs)
                                           : json::Value();
  c["finish_reason"] =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "length"
          : "stop";
  choices.push_back(std::move(c));
  resp["choices"] = std::move(choices);
  resp["usage"] = UsageJson(res);
  resp["timings"] = GenerationTimings(res);
  resp["cache_persist"] = cache_persist;
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

// Responses and Messages bodies decode in Rust (src/core/chat_request) from
// the server's reasoning defaults, with this file's compatibility validator
// at the point each API checks it.
std::optional<HttpResponse> DecodeCompatibilityChat(
    ChatApi api, const HttpRequest& req, TextGenerationBackend& b,
    DecodedChatRequest* decoded) {
  auto environment = ChatEnvironment(b);
  environment.compatibility =
      [&b](ChatApi api, const json::Value& body, std::optional<bool> thinking,
           std::size_t* max_tokens, sampling::SamplingConfig* sampling) {
        if (api == ChatApi::kResponses)
          return ReadCompatibilityOptions(
              body, b, "max_output_tokens", max_tokens, sampling, thinking,
              {.stream = true, .response_controls = true});
        return ReadCompatibilityOptions(body, b, "max_tokens", max_tokens,
                                        sampling, thinking,
                                        {.stop_field = "stop_sequences",
                                         .stream = true,
                                         .thinking = true,
                                         .output_config = true,
                                         .tools = true});
      };
  if (auto error = DecodeChatRequest(api, req.body, environment,
                                     /*require_model=*/true, decoded))
    return error;
  return std::nullopt;
}

HttpResponse OpenAiResponses(const HttpRequest& req,
                             TextGenerationBackend& b) try {
  DecodedChatRequest decoded;
  if (auto error =
          DecodeCompatibilityChat(ChatApi::kResponses, req, b, &decoded))
    return std::move(*error);
  decoded.chat.client_id = req.client_id;
  return CreateOpenAiResponse(
      req, b, decoded.chat, decoded.max_tokens.value_or(0), decoded.sampling,
      decoded.stream, decoded.model ? *decoded.model : b.model_id());
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse OpenAiResponsesInputTokens(const HttpRequest& req,
                                        TextGenerationBackend& b) try {
  DecodedChatRequest decoded;
  if (auto error =
          DecodeCompatibilityChat(ChatApi::kResponses, req, b, &decoded))
    return std::move(*error);
  decoded.chat.client_id = req.client_id;
  return CountOpenAiInputTokens(decoded.chat, b);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse AnthropicMessages(const HttpRequest& req,
                               TextGenerationBackend& b) try {
  DecodedChatRequest decoded;
  if (auto error =
          DecodeCompatibilityChat(ChatApi::kMessages, req, b, &decoded))
    return std::move(*error);
  decoded.chat.client_id = req.client_id;
  return CreateAnthropicMessage(
      req, b, decoded.chat, decoded.max_tokens.value_or(0), decoded.sampling,
      decoded.stream, decoded.model ? *decoded.model : b.model_id());
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

// The prompt size a Messages request would have, without generating.
HttpResponse AnthropicCountTokens(const HttpRequest& req,
                                  TextGenerationBackend& b) try {
  DecodedChatRequest decoded;
  if (auto error =
          DecodeCompatibilityChat(ChatApi::kMessages, req, b, &decoded))
    return std::move(*error);
  if (decoded.stream)
    return InvalidCompatibilityRequest("'stream' must be false");
  const auto tokens = b.count_chat_tokens(decoded.chat);
  if (!tokens)
    return NotImplemented(req, b);
  json::Value resp = json::Value::object();
  resp["input_tokens"] = *tokens;
  return Ok(resp);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

// llama-server's /tokenize at its defaults: raw text, control-token spellings
// parsed and nothing prepended, as /completion tokenizes a prompt. The options
// that would change that are refused rather than ignored.
HttpResponse LlamaTokenize(const HttpRequest& req, TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }
  const auto* content = body.find("content");
  if (content == nullptr || !content->is_string()) {
    return InvalidCompatibilityRequest("'content' must be a string");
  }
  for (const auto& [field, supported] :
       {std::pair{"add_special", false}, std::pair{"parse_special", true},
        std::pair{"with_pieces", false}}) {
    const auto* value = body.find(field);
    if (value == nullptr || value->is_null())
      continue;
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'" + std::string(field) +
                                         "' must be a boolean");
    if (value->as_bool() != supported)
      return InvalidCompatibilityRequest("'" + std::string(field) +
                                         "' is supported only as " +
                                         (supported ? "true" : "false"));
  }
  const auto tokens = b.tokenize(content->str());
  if (!tokens)
    return NotImplemented(req, b);
  json::Value ids = json::Value::array();
  for (const auto token : *tokens)
    ids.push_back(static_cast<std::size_t>(token));
  json::Value resp = json::Value::object();
  resp["tokens"] = std::move(ids);
  return Ok(resp);
}

HttpResponse LlamaCompletion(const HttpRequest& req,
                             TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(
          body, b, "n_predict", &max_tokens, &sampling_config,
          b.reasoning_defaults().enabled, {.stop_field = "stop"})) {
    return std::move(*error);
  }

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string() || input->str().empty()) {
    return InvalidCompatibilityRequest("'prompt' must be a nonempty string");
  }
  const std::string prompt = input->str();
  std::vector<std::string> stop_sequences;
  if (const auto error = ParseStopSequences(
          body.find("stop"), StopSequenceFormat::kOpenAi, &stop_sequences))
    return InvalidCompatibilityRequest(*error);
  bool cache_prompt = true;
  if (const auto* value = body.find("cache_prompt");
      value != nullptr && !value->is_null()) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'cache_prompt' must be a boolean");
    cache_prompt = value->as_bool();
  }
  bool cache_persist = true;
  if (const auto* value = body.find("cache_persist");
      value != nullptr && !value->is_null()) {
    if (!value->is_bool())
      return InvalidCompatibilityRequest("'cache_persist' must be a boolean");
    cache_persist = value->as_bool();
  }

  const auto res =
      b.complete(prompt, max_tokens, sampling_config, req.is_cancelled, {},
                 req.client_id, stop_sequences, cache_prompt, cache_persist);

  json::Value resp = json::Value::object();
  resp["content"] = core::Utf8Decoder{}.Push(res.text, true);
  resp["stop"] = true;
  const bool limited =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength;
  const bool matched =
      res.finish_reason == TextGenerationBackend::FinishReason::kStopSequence;
  resp["stopped_eos"] = !limited && !res.cancelled && !matched;
  resp["stopped_length"] = limited;
  resp["stopped_word"] = matched;
  resp["stopped_limit"] = limited;
  resp["stopping_word"] = res.stop_sequence;
  resp["tokens_predicted"] = res.completion_tokens;
  resp["tokens_evaluated"] = res.prompt_tokens;
  resp["tokens_cached"] = res.cached_prompt_tokens;
  resp["timings"] = GenerationTimings(res);
  resp["usage"] = UsageJson(res);
  resp["cache_persist"] = cache_persist;
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

/// llama-server props for a single-model server: what it runs, so a client
/// can record which model, build and settings answered it. `?model=` is
/// accepted and not required, as in llama-server outside router mode. Fields
/// the backend does not report are omitted rather than sent as zero.
HttpResponse LlamaProps(const HttpRequest& req, TextGenerationBackend& b) {
  const std::string requested = req.query_param("model");
  json::Value resp = json::Value::object();
  resp["model"] = requested.empty() ? b.model_id() : requested;
  resp["model_alias"] = b.model_id();
  resp["template"] = "";
  resp["model_info"] = json::Value::object();
  if (auto path = b.model_path(); !path.empty()) {
    resp["model_path"] = std::move(path);
  }
  if (auto build = b.build_info(); !build.empty()) {
    resp["build_info"] = std::move(build);
  }
  if (const auto sessions = b.session_states().size(); sessions > 0) {
    resp["total_slots"] = sessions;
  }
  // Sessions sharing their state (#715) hold prompts up to this length; a
  // longer one is refused with 507 insufficient_memory_permanent.
  if (const auto shared = b.shared_state()) {
    resp["max_prompt_tokens"] = shared->max_prompt_tokens;
  }

  // What a request without sampling fields gets.
  const auto defaults = b.sampling_defaults();
  const auto sampling = defaults.Resolve(b.reasoning_defaults().enabled);
  json::Value params = json::Value::object();
  params["temperature"] = static_cast<double>(sampling.temperature);
  params["top_k"] = static_cast<int>(sampling.top_k);
  params["top_p"] = static_cast<double>(sampling.top_p);
  params["min_p"] = static_cast<double>(sampling.min_p);
  params["seed"] = static_cast<long long>(sampling.seed);
  params["repeat_penalty"] = static_cast<double>(sampling.repeat_penalty);
  params["repeat_last_n"] = sampling.repeat_last_n;
  params["frequency_penalty"] = static_cast<double>(sampling.frequency_penalty);
  params["presence_penalty"] = static_cast<double>(sampling.presence_penalty);
  // -1 means no default limit, as in llama-server, which reports both keys.
  const auto n_predict = defaults.max_tokens > 0
                             ? static_cast<std::int64_t>(defaults.max_tokens)
                             : std::int64_t{-1};
  params["max_tokens"] = n_predict;
  params["n_predict"] = n_predict;
  json::Value generation = json::Value::object();
  if (const std::size_t context = b.max_context(); context > 0) {
    generation["n_ctx"] = context;
  }
  generation["params"] = std::move(params);
  resp["default_generation_settings"] = std::move(generation);

  json::Value modalities = json::Value::object();
  modalities["vision"] = b.supports_images();
  modalities["video"] = false;
  modalities["audio"] = false;
  resp["modalities"] = std::move(modalities);
  return Ok(resp);
}

/// llama-server slots, one per execution session. `task_id`, `state` and an
/// always-empty `prompt` remain for older clients; prompt text is not exposed.
HttpResponse LlamaSlots(const HttpRequest&, TextGenerationBackend& b) {
  auto sessions = b.session_states();
  if (sessions.empty()) {
    sessions.emplace_back();
  }
  const std::size_t context = b.max_context();
  const std::string model = b.model_id();
  json::Value resp = json::Value::array();
  for (std::size_t index = 0; index < sessions.size(); ++index) {
    const auto& session = sessions[index];
    const std::int64_t task =
        session.processing ? static_cast<std::int64_t>(session.request_id) : -1;
    json::Value next_token = json::Value::object();
    next_token["has_next_token"] = session.processing;
    next_token["has_new_line"] = false;
    next_token["n_remain"] =
        session.processing ? static_cast<std::int64_t>(session.remaining_tokens)
                           : std::int64_t{-1};
    next_token["n_decoded"] = session.generated_tokens;
    json::Value slot = json::Value::object();
    slot["id"] = index;
    slot["n_ctx"] = context;
    slot["speculative"] = session.speculative;
    slot["is_processing"] = session.processing;
    slot["id_task"] = task;
    slot["task_id"] = task;
    slot["state"] = session.processing ? 1 : 0;
    slot["n_prompt_tokens"] = session.prompt_tokens;
    slot["n_prompt_tokens_cache"] = session.cached_prompt_tokens;
    slot["n_prompt_tokens_processed"] = session.processed_prompt_tokens;
    slot["prompt"] = "";
    slot["next_token"] = json::Value::array();
    slot["next_token"].push_back(std::move(next_token));
    slot["model"] = model;
    resp.push_back(std::move(slot));
  }
  return Ok(resp);
}

/// Context held by in-flight requests across all sessions.
double KvCacheUsageRatio(const TextGenerationBackend& b) {
  const auto sessions = b.session_states();
  const std::size_t capacity = sessions.size() * b.max_context();
  if (capacity == 0) {
    return 0.0;
  }
  std::size_t used = 0;
  for (const auto& session : sessions) {
    used += session.prompt_tokens + session.generated_tokens;
  }
  return static_cast<double>(used) / static_cast<double>(capacity);
}

HttpResponse LlamaMetrics(const HttpRequest&, TextGenerationBackend& b) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "# HELP gufo_device_lost_total Confirmed GPU context losses\n"
      << "# TYPE gufo_device_lost_total counter\n"
      << "gufo_device_lost_total "
      << detail::DeviceLostTotal().load(std::memory_order_relaxed) << "\n";
  out << "# HELP llamacpp:prompt_tokens_total Total prompt tokens processed, "
         "excluding cache hits\n"
      << "# TYPE llamacpp:prompt_tokens_total counter\n"
      << "llamacpp:prompt_tokens_total "
      << detail::TotalPromptTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:prompt_tokens_cached_total Total prompt tokens "
         "reused from cache\n"
      << "# TYPE llamacpp:prompt_tokens_cached_total counter\n"
      << "llamacpp:prompt_tokens_cached_total "
      << detail::TotalCachedPromptTokens().load(std::memory_order_relaxed)
      << "\n"
      << "# HELP llamacpp:prompt_seconds_total Prompt process time\n"
      << "# TYPE llamacpp:prompt_seconds_total counter\n"
      << "llamacpp:prompt_seconds_total "
      << detail::TotalPromptSeconds().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:tokens_predicted_total Total tokens generated\n"
      << "# TYPE llamacpp:tokens_predicted_total counter\n"
      << "llamacpp:tokens_predicted_total "
      << detail::TotalGenTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:tokens_predicted_seconds_total Predict process "
         "time\n"
      << "# TYPE llamacpp:tokens_predicted_seconds_total counter\n"
      << "llamacpp:tokens_predicted_seconds_total "
      << detail::TotalGenSeconds().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:n_tokens_max Largest observed n_tokens.\n"
      << "# TYPE llamacpp:n_tokens_max counter\n"
      << "llamacpp:n_tokens_max "
      << detail::MaxSequenceTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:spec_decode_num_drafts_total Total speculative "
         "verification rounds\n"
      << "# TYPE llamacpp:spec_decode_num_drafts_total counter\n"
      << "llamacpp:spec_decode_num_drafts_total "
      << detail::TotalDraftRounds().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:spec_decode_num_draft_tokens_total Total draft "
         "tokens proposed\n"
      << "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter\n"
      << "llamacpp:spec_decode_num_draft_tokens_total "
      << detail::TotalDraftTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:spec_decode_num_accepted_tokens_total Total draft "
         "tokens accepted\n"
      << "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter\n"
      << "llamacpp:spec_decode_num_accepted_tokens_total "
      << detail::TotalDraftAcceptedTokens().load(std::memory_order_relaxed)
      << "\n"
      << "# HELP gufo_spec_lookup_tokens_total Total prompt-lookup copies "
         "proposed\n"
      << "# TYPE gufo_spec_lookup_tokens_total counter\n"
      << "gufo_spec_lookup_tokens_total "
      << detail::TotalLookupTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP gufo_spec_lookup_accepted_tokens_total Total prompt-lookup "
         "copies accepted\n"
      << "# TYPE gufo_spec_lookup_accepted_tokens_total counter\n"
      << "gufo_spec_lookup_accepted_tokens_total "
      << detail::TotalLookupAcceptedTokens().load(std::memory_order_relaxed)
      << "\n"
      << "# HELP llamacpp:prompt_tokens_seconds Prompt processing speed in "
         "tokens per second\n"
      << "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
      << "llamacpp:prompt_tokens_seconds "
      << detail::LastPromptSpeed().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:predicted_tokens_seconds Generation speed in tokens "
         "per second\n"
      << "# TYPE llamacpp:predicted_tokens_seconds gauge\n"
      << "llamacpp:predicted_tokens_seconds "
      << detail::LastGenSpeed().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:requests_processing Number of admitted requests, "
         "including cache preparation\n"
      << "# TYPE llamacpp:requests_processing gauge\n"
      << "llamacpp:requests_processing "
      << detail::RequestsProcessing().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:requests_deferred Number of requests waiting for a "
         "session\n"
      << "# TYPE llamacpp:requests_deferred gauge\n"
      << "llamacpp:requests_deferred "
      << detail::RequestsDeferred().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:kv_cache_usage_ratio In-flight prompt and generated "
         "tokens over sessions times context; excludes retained cache\n"
      << "# TYPE llamacpp:kv_cache_usage_ratio gauge\n"
      << "llamacpp:kv_cache_usage_ratio " << KvCacheUsageRatio(b) << "\n";
  if (const auto shared = b.shared_state()) {
    out << "# HELP gufo_shared_state_bytes Session state the configured "
           "sessions share when they do not all fit at full context\n"
        << "# TYPE gufo_shared_state_bytes gauge\n"
        << "gufo_shared_state_bytes " << shared->shared_bytes << "\n"
        << "# HELP gufo_shared_state_charged_bytes Shared session state "
           "charged to sessions\n"
        << "# TYPE gufo_shared_state_charged_bytes gauge\n"
        << "gufo_shared_state_charged_bytes " << shared->charged_bytes << "\n"
        << "# HELP gufo_shared_state_reclaimable_bytes Charged session state "
           "held by idle sessions or mapped past a session's need\n"
        << "# TYPE gufo_shared_state_reclaimable_bytes gauge\n"
        << "gufo_shared_state_reclaimable_bytes " << shared->reclaimable_bytes
        << "\n"
        << "# HELP gufo_shared_state_max_prompt_tokens Longest prompt the "
           "shared session state holds\n"
        << "# TYPE gufo_shared_state_max_prompt_tokens gauge\n"
        << "gufo_shared_state_max_prompt_tokens " << shared->max_prompt_tokens
        << "\n"
        << "# HELP gufo_shared_state_waits_total Requests that waited for "
           "shared session state\n"
        << "# TYPE gufo_shared_state_waits_total counter\n"
        << "gufo_shared_state_waits_total " << shared->waits << "\n"
        << "# HELP gufo_shared_state_pauses_total Decoding requests paused "
           "until shared session state freed\n"
        << "# TYPE gufo_shared_state_pauses_total counter\n"
        << "gufo_shared_state_pauses_total " << shared->pauses << "\n"
        << "# HELP gufo_shared_state_preemptions_total Requests stopped "
           "with memory_pressure so others could continue\n"
        << "# TYPE gufo_shared_state_preemptions_total counter\n"
        << "gufo_shared_state_preemptions_total " << shared->preemptions
        << "\n";
  }
  if (const auto budget = b.prompt_admission_budget()) {
    out << "# HELP gufo_prompt_state_budget_bytes Host RAM prompt state may "
           "claim before admission refuses\n"
        << "# TYPE gufo_prompt_state_budget_bytes gauge\n"
        << "gufo_prompt_state_budget_bytes " << budget->budget_bytes << "\n"
        << "# HELP gufo_prompt_state_reserved_bytes Host prompt state charged "
           "to in-flight requests\n"
        << "# TYPE gufo_prompt_state_reserved_bytes gauge\n"
        << "gufo_prompt_state_reserved_bytes " << budget->used_bytes << "\n"
        << "# HELP gufo_prompt_state_available_bytes Remaining host prompt "
           "state budget\n"
        << "# TYPE gufo_prompt_state_available_bytes gauge\n"
        << "gufo_prompt_state_available_bytes " << budget->available_bytes
        << "\n";
  }
  return {.status = 200,
          .reason = "OK",
          .body = out.str(),
          .headers = {{"Content-Type", "text/plain; version=0.0.4"}}};
}

}  // namespace

bool HasToken(std::string_view value, std::string_view token) {
  const int found = gufo_http_has_token(
      reinterpret_cast<const std::uint8_t*>(value.data()), value.size(),
      reinterpret_cast<const std::uint8_t*>(token.data()), token.size());
  if (found < 0)
    throw std::logic_error("Rust HTTP token check failed internally");
  return found != 0;
}

LoadingResponder::LoadingResponder(const std::string& host, int port) {
  responder_ = gufo_loading_responder_start(
      reinterpret_cast<const std::uint8_t*>(host.data()), host.size(), port);
  listening_.store(responder_ != nullptr, std::memory_order_release);
}

LoadingResponder::~LoadingResponder() {
  stop();
  gufo_loading_responder_free(responder_);
  responder_ = nullptr;
}

void LoadingResponder::stop() {
  if (listening_.exchange(false, std::memory_order_acq_rel) &&
      responder_ != nullptr) {
    gufo_loading_responder_stop(responder_);
  }
}

void LoadingResponder::SetProgress(double progress, std::string_view message) {
  if (responder_ != nullptr) {
    gufo_loading_responder_set_progress(
        responder_, progress,
        reinterpret_cast<const std::uint8_t*>(message.data()), message.size());
  }
}

bool IsListenAddress(const std::string& host) {
  return gufo_http_is_listen_address(
             reinterpret_cast<const std::uint8_t*>(host.data()), host.size()) !=
         0;
}

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

std::string HttpRequest::query_param(const std::string& key) const {
  struct Found {
    std::string value;
    core::ffi::CallbackFailure failure;
  } found;
  const GufoHttpBytesFn keep = [](void* opaque, const std::uint8_t* data,
                                  std::size_t len) noexcept {
    auto& out = *static_cast<Found*>(opaque);
    out.failure.Guard(
        [&] { out.value.assign(reinterpret_cast<const char*>(data), len); });
  };
  const int status = gufo_http_query_param(
      reinterpret_cast<const std::uint8_t*>(query.data()), query.size(),
      reinterpret_cast<const std::uint8_t*>(key.data()), key.size(), keep,
      &found);
  found.failure.Rethrow();
  if (status < 0)
    throw std::logic_error("Rust query parser failed internally");
  return found.value;
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

HttpServer::HttpServer(std::string host, int port,
                       std::shared_ptr<TextGenerationBackend> backend,
                       std::shared_ptr<VideoJobService> video_jobs,
                       std::shared_ptr<TtsService> tts,
                       std::shared_ptr<AsrService> asr,
                       HttpServerOptions options,
                       std::shared_ptr<ImageService> images,
                       std::shared_ptr<EmbeddingService> embeddings,
                       std::shared_ptr<DecisionService> decisions)
    : host_(std::move(host)),
      port_(port),
      backend_(std::move(backend)),
      video_jobs_(std::move(video_jobs)),
      tts_(std::move(tts)),
      asr_(std::move(asr)),
      images_(std::move(images)),
      embeddings_(std::move(embeddings)),
      decisions_(std::move(decisions)),
      options_(std::move(options)) {
  if (options_.max_request_body_bytes == 0 || options_.max_connections == 0) {
    throw std::invalid_argument("HTTP server limits must be positive");
  }
  register_routes();
}

HttpServer::~HttpServer() {
  stop();
  gufo_http_server_free(server_);
  server_ = nullptr;
}

void HttpServer::add(const std::string& method, const std::string& path,
                     Handler handler) {
  routes_.emplace_back(std::make_pair(method, path), std::move(handler));
}

void HttpServer::register_routes() {
  if (backend_ == nullptr) {
    return;
  }
  // ---- OpenAI ----
  add("POST", "/v1/completions", OpenAiCompletions);
  add("POST", "/v1/chat/completions", HandleOpenAiChat);
  add("POST", "/v1/chat/completions/input_tokens", HandleOpenAiChatInputTokens);
  add("POST", "/v1/responses", OpenAiResponses);
  add("POST", "/v1/responses/input_tokens", OpenAiResponsesInputTokens);
  add("POST", "/v1/embeddings", NotImplemented);

  // ---- Anthropic ----
  add("POST", "/v1/messages", AnthropicMessages);
  add("POST", "/v1/messages/count_tokens", AnthropicCountTokens);

  // ---- llama-server ----
  add("POST", "/v1/rerank", NotImplemented);
  add("POST", "/v1/reranking", NotImplemented);
  add("POST", "/rerank", NotImplemented);
  add("POST", "/infill", NotImplemented);
  add("POST", "/completion", LlamaCompletion);
  add("POST", "/apply-template", HandleApplyTemplate);
  add("POST", "/tokenize", LlamaTokenize);
  add("GET", "/props", LlamaProps);
  add("GET", "/slots", LlamaSlots);
  add("GET", "/v1/slots", LlamaSlots);
  add("GET", "/metrics", LlamaMetrics);
  add("GET", "/v1/metrics", LlamaMetrics);

  // ---- sdapi ----
  add("POST", "/sdapi/v1/txt2img", NotImplemented);
  add("POST", "/sdapi/v1/img2img", NotImplemented);
  add("GET", "/sdapi/v1/loras", NotImplemented);
}

bool HttpServer::start(std::string* error) {
  if (server_ != nullptr || stopped_.load(std::memory_order_acquire)) {
    if (error != nullptr) {
      *error = "HTTP server has already been started or stopped";
    }
    return false;
  }
  const auto on_request = [](void* context, const GufoHttpRequestView* view,
                             GufoHttpResponder* responder) noexcept {
    auto* self = static_cast<HttpServer*>(context);
    if (self != nullptr && view != nullptr && responder != nullptr) {
      try {
        self->dispatch_request(*view, responder);
      } catch (const std::exception& e) {
        Logger::Error(
            "http", std::string("event=dispatch_exception reason=") + e.what());
      } catch (...) {
        Logger::Error("http", "event=dispatch_exception reason=unknown");
      }
    }
  };
  const auto on_device_lost = [](void* context) noexcept -> int {
    auto* self = static_cast<HttpServer*>(context);
    if (self == nullptr)
      return 0;
    try {
      return self->device_lost() ? 1 : 0;
    } catch (...) {
      return 0;
    }
  };
  const auto on_error = [](void* context, const std::uint8_t* data,
                           std::size_t len) noexcept {
    if (context == nullptr)
      return;
    auto* out = static_cast<std::string*>(context);
    out->assign(reinterpret_cast<const char*>(data), len);
  };

  GufoHttpServerConfig config{};
  config.host = reinterpret_cast<const std::uint8_t*>(host_.data());
  config.host_len = host_.size();
  config.port = port_;
  config.max_request_body_bytes = options_.max_request_body_bytes;
  config.max_connections = options_.max_connections;
  config.sse_heartbeat_interval_ms =
      options_.sse_heartbeat_interval.count() > 0
          ? static_cast<std::uint64_t>(options_.sse_heartbeat_interval.count())
          : 0;
  config.keep_alive = options_.keep_alive ? 1 : 0;
  config.keep_alive_idle_timeout_ms =
      options_.keep_alive_idle_timeout.count() > 0
          ? static_cast<std::uint64_t>(options_.keep_alive_idle_timeout.count())
          : 0;
  config.max_requests_per_connection = options_.max_requests_per_connection;
  config.handler = on_request;
  config.handler_context = this;
  config.device_lost = on_device_lost;
  config.device_lost_context = this;

  std::string err_msg;
  server_ = gufo_http_server_start(&config, on_error, &err_msg);
  if (server_ == nullptr) {
    if (error != nullptr) {
      *error = err_msg.empty() ? "failed to start HTTP server" : err_msg;
    }
    return false;
  }
  port_ = gufo_http_server_port(server_);
  return true;
}

void HttpServer::run(bool handle_signals) {
  std::optional<ShutdownSignals> signals;
  std::jthread signal_watcher;
  if (handle_signals) {
    signals.emplace();
    signal_watcher = std::jthread([this](std::stop_token stop_token) {
      while (!stop_token.stop_requested() &&
             !stopped_.load(std::memory_order_acquire)) {
        const int signal = shutdown_signal.load(std::memory_order_relaxed);
        if (signal != 0) {
          Logger::Info("server", "event=shutdown_requested signal=" +
                                     std::to_string(signal));
          stop();
          platform::NotifyShutdownHandled();
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  Logger::Info(
      "server",
      "event=listening monotonic_us=" +
          std::to_string(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count()) +
          " address=http://" +
          (host_.find(':') == std::string::npos ? host_ : "[" + host_ + "]") +
          ":" + std::to_string(port_) +
          " max_connections=" + std::to_string(options_.max_connections) +
          " max_body_bytes=" + std::to_string(options_.max_request_body_bytes));
  if (server_ != nullptr) {
    (void)gufo_http_server_run(server_);
  }
  if (signal_watcher.joinable()) {
    signal_watcher.request_stop();
    signal_watcher.join();
  }
}

void HttpServer::stop() {
  const bool was_stopped = stopped_.exchange(true, std::memory_order_acq_rel);
  if (!was_stopped && server_ != nullptr) {
    gufo_http_server_stop(server_);
  }
}

HttpResponse HttpServer::handle_request(const HttpRequest& req) {
  // The voice page holds no data; it opens its WebSocket when the user
  // connects.
  if (req.method == "GET" && req.path == "/voice" && asr_ &&
      asr_->supports_voice()) {
    HttpResponse page{
        .status = 200, .reason = "OK", .body = std::string(VoicePage())};
    page.headers.emplace_back("Content-Type", "text/html; charset=utf-8");
    page.headers.emplace_back("Cache-Control", "no-store");
    return page;
  }
  // `/health` and `/ready` are the native spelling and match llama-server's
  // `/health`. `/healthz` and `/readyz` are aliases so Kubernetes-style probe
  // configuration works unmodified. `GET /` answers like `/health` so
  // llama-swap's Load button and startup preload, which request the upstream
  // root, report success once the loading gate has released (#957).
  if (req.method == "GET" &&
      (req.path == "/" || req.path == "/health" || req.path == "/v1/health" ||
       req.path == "/healthz")) {
    if (device_lost())
      return DeviceLost();
    json::Value body = json::Value::object();
    body["status"] = "ok";
    return Ok(body);
  }
  if (req.method == "GET" && (req.path == "/ready" || req.path == "/v1/ready" ||
                              req.path == "/readyz")) {
    if (device_lost())
      return DeviceLost();
    const bool ready = (backend_ != nullptr && backend_->ready()) ||
                       (video_jobs_ != nullptr && video_jobs_->ready()) ||
                       (tts_ != nullptr && tts_->ready()) ||
                       (asr_ != nullptr && asr_->ready()) ||
                       (embeddings_ != nullptr && embeddings_->ready()) ||
                       images_ != nullptr || decisions_ != nullptr;
    if (!ready) {
      return Err(503, "Service Unavailable", "model service is not ready",
                 "server_error", "not_ready");
    }
    json::Value body = json::Value::object();
    body["status"] = "ready";
    if (backend_ != nullptr && backend_->ready()) {
      body["model"] = backend_->model_id();
    } else if (asr_ != nullptr && asr_->ready()) {
      body["model"] = asr_->model_id();
    } else if (tts_ != nullptr && tts_->ready()) {
      body["model"] = tts_->model_id();
    } else if (embeddings_ != nullptr && embeddings_->ready()) {
      body["model"] = embeddings_->model_id();
    } else if (decisions_ != nullptr) {
      body["model"] = decisions_->model_id;
    } else if (images_ != nullptr) {
      body["model"] = images_->model_id();
    } else {
      body["model"] = "minimax-h3";
    }
    return Ok(body);
  }
  if (req.method == "GET" &&
      (req.path == "/v1/models" || req.path == "/models")) {
    return ListModels(backend_.get(), video_jobs_.get(), tts_.get(), asr_.get(),
                      images_.get(), embeddings_.get(), decisions_.get());
  }
  if (req.path == "/v1/systemone") {
    if (req.method != "POST")
      return Err(405, "Method Not Allowed", "use POST", "invalid_request_error",
                 "method_not_allowed");
    if (!decisions_ || !decisions_->decide)
      return Err(503, "Service Unavailable",
                 "decision service is not configured", "server_error",
                 "decision_service_unavailable");
    json::Value payload;
    try {
      payload = json::parse(req.body);
    } catch (const std::exception& e) {
      return Err(400, "Bad Request", e.what(), "invalid_request_error",
                 "invalid_json");
    }
    try {
      if (const auto* model = payload.find("model");
          model &&
          (!model->is_string() || !ServesModel(*decisions_, model->str())))
        return Err(404, "Not Found", "unknown decision model",
                   "invalid_request_error", "model_not_found");
      auto result = decisions_->decide(payload);
      result["model"] = decisions_->model_id;
      return Ok(result);
    } catch (const std::invalid_argument& e) {
      return Err(400, "Bad Request", e.what(), "invalid_request_error",
                 "invalid_request");
    } catch (const std::exception& e) {
      return Err(500, "Internal Server Error", e.what(), "server_error",
                 "decision_failed");
    }
  }
  if (req.path == "/v1/images/generations" || req.path == "/v1/images/edits") {
    if (!images_)
      return Err(503, "Service Unavailable", "image service is not configured",
                 "server_error", "image_service_unavailable");
    return HandleImageApiRequest(req, *images_);
  }
  // An image server answers every A1111 route; a text server keeps the 501
  // stubs of the route table.
  if (images_ != nullptr && IsSdapiPath(req.path))
    return HandleSdapiRequest(req, *images_);
  if (req.path == "/v1/audio/speech/stream") {
    if (!tts_ || !tts_->ready())
      return Err(503, "Service Unavailable", "TTS is not configured",
                 "server_error", "tts_service_unavailable");
    return HandleTtsWebSocket(req, *tts_);
  }
  if (req.path == "/v1/realtime") {
    if (!asr_ || !asr_->ready())
      return Err(503, "Service Unavailable", "ASR is not configured",
                 "server_error", "asr_service_unavailable");
    return HandleAsrWebSocket(req, *asr_);
  }
  if (IsVideoApiPath(req.path)) {
    if (video_jobs_ == nullptr || !video_jobs_->ready()) {
      return Err(503, "Service Unavailable",
                 "MiniMax H3 video service is not configured", "server_error",
                 "video_service_unavailable");
    }
    return HandleVideoApiRequest(req, *video_jobs_);
  }
  if (IsAudioTtsApiPath(req.path)) {
    if (tts_ == nullptr || !tts_->ready()) {
      return Err(503, "Service Unavailable",
                 "Qwen3-TTS audio service is not configured", "server_error",
                 "tts_service_unavailable");
    }
    return HandleAudioTtsApiRequest(req, *tts_);
  }
  if (IsAudioAsrApiPath(req.path)) {
    if (asr_ == nullptr || !asr_->ready()) {
      return Err(503, "Service Unavailable",
                 "Qwen3-ASR service is not configured", "server_error",
                 "asr_service_unavailable");
    }
    return HandleAudioAsrApiRequest(req, *asr_);
  }
  // Handled before the route table. An embedding-only server has no text
  // backend, and the stub route below must not shadow the service when a
  // backend exists.
  if (embeddings_ != nullptr && IsEmbeddingsApiPath(req.path)) {
    if (device_lost())
      return DeviceLost();
    if (!embeddings_->ready()) {
      return Err(503, "Service Unavailable",
                 "EmbeddingGemma 2 service is not ready", "server_error",
                 "embedding_service_unavailable");
    }
    auto response = HandleEmbeddingsApiRequest(req, *embeddings_);
    // The request that finds the device lost runs the shutdown hook.
    if (device_lost())
      return DeviceLost();
    return response;
  }
  for (const auto& entry : routes_) {
    if (entry.first.first == req.method && entry.first.second == req.path) {
      if (req.method == "POST" && device_lost())
        return DeviceLost();
      return entry.second(req, *backend_);
    }
  }
  return Err(404, "Not Found", "no route for this path",
             "invalid_request_error", "not_found");
}

bool HttpServer::device_lost() {
  const bool lost = (backend_ != nullptr && backend_->device_lost()) ||
                    (embeddings_ != nullptr && embeddings_->device_lost()) ||
                    (images_ != nullptr && images_->device_lost());
  if (!lost)
    return false;
  if (options_.on_device_lost &&
      !device_lost_reported_.exchange(true, std::memory_order_acq_rel)) {
    try {
      options_.on_device_lost();
    } catch (const std::exception& error) {
      Logger::Error("server", "event=device_lost_hook_failed reason=" +
                                  std::string(error.what()));
    }
  }
  return true;
}

void HttpServer::dispatch_request(const GufoHttpRequestView& view,
                                  GufoHttpResponder* responder) {
  static std::atomic<std::uint64_t> next_request{0};
  const auto start_time = std::chrono::steady_clock::now();

  const auto view_str = [](GufoHttpSliceView slice) {
    return slice.data != nullptr && slice.len > 0
               ? std::string(reinterpret_cast<const char*>(slice.data),
                             slice.len)
               : std::string{};
  };

  HttpRequest req;
  req.method = view_str(view.method);
  req.path = view_str(view.path);
  req.query = view_str(view.query);
  req.body = view_str(view.body);
  req.headers.reserve(view.headers_len);
  for (std::size_t i = 0; i < view.headers_len; ++i) {
    req.headers.emplace_back(view_str(view.headers[i].name),
                             view_str(view.headers[i].value));
  }
  if (auto peer = view_str(view.peer_addr); !peer.empty()) {
    req.client_id = std::move(peer);
  }
  req.request_id = "r" + std::to_string(next_request.fetch_add(1) + 1);
  GufoHttpCancel* cancel = view.cancel;
  req.is_cancelled = [cancel] {
    return cancel != nullptr && gufo_http_cancel_is_cancelled(cancel) != 0;
  };

  bool traced = false;
  std::string traced_stream;
  std::mutex traced_mutex;
  bool response_started = false;
  int response_status = 0;

  const auto elapsed_ms = [&] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_time)
        .count();
  };

  const auto send_buffered = [&](const HttpResponse& resp) {
    const auto headers = BuildResponseHeaders(resp);
    const auto header_views = ToHeaderViews(headers);
    return gufo_http_respond_buffered(
               responder, resp.status,
               header_views.empty() ? nullptr : header_views.data(),
               header_views.size(),
               reinterpret_cast<const std::uint8_t*>(resp.body.data()),
               resp.body.size()) != 0;
  };

  const auto trace_failure = [&](const HttpResponse& error) {
    if (traced) {
      std::string captured;
      if (response_started) {
        const std::lock_guard lock(traced_mutex);
        captured = traced_stream;
      }
      TraceResponse(req.request_id,
                    response_started ? response_status : error.status,
                    response_started ? "stream_error" : "failed",
                    response_started, response_started ? captured : error.body);
    }
  };

  const Trace::RequestScope trace_scope(req.request_id);
  try {
    // The transport refuses a request it could not read whole; the reply
    // carries the usual headers and log line.
    const int rejection = view.rejection;
    const bool log_request =
        req.method == "POST" || req.method == "DELETE" ||
        req.path == "/v1/models" || req.path.ends_with("/content") ||
        req.path == "/v1/realtime" || req.path == "/v1/audio/speech/stream";
    const LogLevel request_level =
        log_request ? LogLevel::kInfo : LogLevel::kDebug;
    if (rejection == 0 && Logger::Enabled(request_level)) {
      Logger::Log(request_level, "http",
                  "request=" + req.request_id + " event=received method=" +
                      req.method + " path=" + req.path +
                      " body_bytes=" + std::to_string(req.body.size()));
    }

    HttpResponse resp;
    if (rejection == 413) {
      resp = Err(413, "Payload Too Large", "request body is too large",
                 "invalid_request_error", "payload_too_large");
    } else if (rejection == 431) {
      resp = Err(431, "Request Header Fields Too Large",
                 "request has more than 100 header fields",
                 "invalid_request_error", "request_header_fields_too_large");
    } else if (rejection == 400) {
      resp = Err(400, "Bad Request", "malformed request",
                 "invalid_request_error", "bad_request");
    } else if (rejection != 0) {
      resp = Err(500, "Internal Server Error", "internal server error",
                 "internal_error", "server_exception");
    } else if (req.method == "OPTIONS") {
      resp = {.status = 204, .reason = "No Content"};
    } else {
      traced = Trace::Enabled() && IsTracedRoute(req);
      if (traced) {
        TraceRequest(req);
      }
      resp = handle_request(req);
    }

    resp.headers.emplace_back("X-Request-ID", req.request_id);
    response_status = resp.status;
    bool connected = true;

    if (resp.websocket) {
      response_started = true;
      const auto headers = BuildResponseHeaders(resp);
      const auto header_views = ToHeaderViews(headers);
      // The session runs on this thread inside the call, within this
      // request's trace scope; the request is logged once it ends.
      struct WebSocketRun {
        std::function<void(WebSocket&)>* handler;
        std::exception_ptr failure;
      } run{&resp.websocket, nullptr};
      const auto ws_trampoline = [](void* ctx,
                                    GufoWsSession* session) noexcept {
        auto& run = *static_cast<WebSocketRun*>(ctx);
        try {
          WebSocket socket(session);
          (*run.handler)(socket);
        } catch (...) {
          run.failure = std::current_exception();
        }
      };
      connected = gufo_http_respond_websocket(
                      responder, resp.status,
                      header_views.empty() ? nullptr : header_views.data(),
                      header_views.size(), ws_trampoline, &run) != 0;
      if (run.failure)
        std::rethrow_exception(run.failure);
    } else if (resp.streaming_body) {
      const auto headers = BuildResponseHeaders(resp);
      const auto header_views = ToHeaderViews(headers);
      const bool is_sse = IsEventStream(resp);
      struct HeartbeatTraceCtx {
        std::mutex* mutex;
        std::string* stream;
      } hb_ctx{&traced_mutex, &traced_stream};
      const GufoHttpBytesSink hb_sink =
          traced ? static_cast<GufoHttpBytesSink>([](void* ctx,
                                                     const std::uint8_t* data,
                                                     std::size_t len) noexcept {
            auto* h = static_cast<HeartbeatTraceCtx*>(ctx);
            try {
              const std::lock_guard lock(*h->mutex);
              h->stream->append(reinterpret_cast<const char*>(data), len);
            } catch (...) {
            }
          })
                 : nullptr;

      GufoHttpStream* stream = gufo_http_respond_stream_begin(
          responder, resp.status,
          header_views.empty() ? nullptr : header_views.data(),
          header_views.size(), is_sse ? 1 : 0,
          resp.defer_stream_headers ? 1 : 0, hb_sink, &hb_ctx);
      if (stream == nullptr) {
        connected = false;
      } else {
        if (!resp.defer_stream_headers) {
          response_started = true;
        }
        try {
          const auto send_body = [&](std::string_view chunk) {
            if (!connected)
              return false;
            response_started = true;
            connected =
                gufo_http_stream_write(
                    stream, reinterpret_cast<const std::uint8_t*>(chunk.data()),
                    chunk.size()) != 0;
            if (connected && !chunk.empty() && traced) {
              const std::lock_guard lock(traced_mutex);
              traced_stream.append(chunk);
            }
            return connected;
          };
          resp.streaming_body(send_body);
        } catch (...) {
          if (!response_started) {
            gufo_http_stream_abort_deferred(stream, responder);
          } else {
            (void)gufo_http_stream_end(stream, 0);
          }
          throw;
        }
        (void)device_lost();
        const bool complete = !resp.stream_log ||
                              resp.stream_log->error_code.empty() ||
                              resp.stream_log->error_event_sent;
        connected =
            gufo_http_stream_end(stream, complete ? 1 : 0) != 0 && connected;
      }
    } else {
      (void)device_lost();
      response_started = true;
      connected = send_buffered(resp);
    }

    std::string outcome = connected ? "completed" : "disconnected";
    if (resp.stream_log) {
      resp.log_details = resp.stream_log->details;
      if (!resp.stream_log->error_code.empty()) {
        outcome = "stream_error";
        resp.log_details += " error_code=" + resp.stream_log->error_code;
      }
    }
    if (resp.status >= 400) {
      try {
        const auto body = json::parse(resp.body);
        if (const auto* error = body.find("error"))
          resp.log_details += " error_code=" + error->member_str("code");
      } catch (const std::exception&) {
      }
    }
    if (Logger::Enabled(request_level) || resp.status >= 400 || !connected) {
      Logger::LogRequest(req.request_id, req.method, req.path, resp.status,
                         elapsed_ms(), resp.log_details, outcome,
                         request_level);
    }
    if (traced) {
      const bool stream = static_cast<bool>(resp.streaming_body);
      std::string captured;
      if (stream) {
        const std::lock_guard lock(traced_mutex);
        captured = traced_stream;
      }
      TraceResponse(req.request_id, resp.status, outcome, stream,
                    stream ? captured : resp.body);
    }
  } catch (const TextGenerationError& exception) {
    const auto duration_ms = elapsed_ms();
    HttpResponse resp = TextGenerationErrorResponse(exception);
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms,
                       std::string("error_code=") + exception.stable_code(),
                       response_started ? "stream_error" : "failed");
    trace_failure(resp);
    if (!response_started)
      (void)send_buffered(resp);
  } catch (const std::exception& e) {
    const auto duration_ms = elapsed_ms();
    const char* message = *e.what() ? e.what() : "generation failed";
    Logger::Error("http", "request=" + req.request_id +
                              " event=server_exception reason=" + message);
    HttpResponse resp = Err(500, "Internal Server Error", message,
                            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    trace_failure(resp);
    if (!response_started)
      (void)send_buffered(resp);
  } catch (...) {
    const auto duration_ms = elapsed_ms();
    HttpResponse resp =
        Err(500, "Internal Server Error", "unknown server error",
            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    trace_failure(resp);
    if (!response_started)
      (void)send_buffered(resp);
  }
  (void)device_lost();
}

}  // namespace gufo::server

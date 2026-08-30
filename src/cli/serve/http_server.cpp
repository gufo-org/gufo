#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/json.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {
namespace {

// ---------------------------------------------------------------------------
// Socket I/O helpers
// ---------------------------------------------------------------------------

bool ReadUntil(std::string& out, int fd, std::string_view delim) {
  char buf[4096];
  while (out.find(delim) == std::string::npos) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n <= 0)
      return false;
    out.append(buf, static_cast<std::size_t>(n));
    if (out.size() > (static_cast<std::size_t>(16) * 1024 * 1024))
      return false;
  }
  return true;
}

bool ReadN(std::string& out, int fd, std::size_t n) {
  out.reserve(n);
  std::size_t got = 0;
  char buf[4096];
  while (got < n) {
    const std::size_t want = std::min(sizeof(buf), n - got);
    const ssize_t r = ::read(fd, buf, want);
    if (r <= 0)
      return false;
    out.append(buf, static_cast<std::size_t>(r));
    got += static_cast<std::size_t>(r);
  }
  return true;
}

bool SendAll(int fd, std::string_view data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, flags);
    if (n <= 0)
      return false;
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool IsPeerDisconnected(int fd) noexcept {
  pollfd descriptor{
      .fd = fd,
      .events = POLLERR | POLLHUP,
      .revents = 0,
  };
  const int ready = ::poll(&descriptor, 1, 0);
  return ready > 0 &&
         (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
}

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

std::string UrlDecode(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '+') {
      out += ' ';
    } else if (c == '%' && i + 2 < s.size()) {
      auto hexval = [](char h) {
        if (h >= '0' && h <= '9')
          return h - '0';
        if (h >= 'a' && h <= 'f')
          return h - 'a' + 10;
        if (h >= 'A' && h <= 'F')
          return h - 'A' + 10;
        return 0;
      };
      out += static_cast<char>((hexval(s[i + 1]) << 4) | hexval(s[i + 2]));
      i += 2;
    } else {
      out += c;
    }
  }
  return out;
}

std::size_t ParseContentLength(const HttpRequest& request) {
  const std::string value = request.header("content-length");
  if (value.empty()) {
    return 0;
  }
  return static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
}

bool HasHeader(const HttpResponse& response, std::string_view name) {
  const std::string lowered = ToLower(name);
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return ToLower(header.first) == lowered;
  });
}

std::string BuildResponseHead(const HttpResponse& resp,
                              std::optional<std::size_t> content_length) {
  std::string out;
  out.reserve(256);
  out += "HTTP/1.1 ";
  out += std::to_string(resp.status);
  out += ' ';
  out += resp.reason;
  out += "\r\n";
  if (!HasHeader(resp, "content-type")) {
    out += "Content-Type: application/json\r\n";
  }
  if (content_length.has_value()) {
    out += "Content-Length: " + std::to_string(*content_length) + "\r\n";
  }
  out += "Access-Control-Allow-Origin: *\r\n";
  out += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";
  out +=
      "Access-Control-Allow-Headers: Content-Type, Authorization, Range, "
      "X-Client-ID\r\n";
  for (const auto& [name, value] : resp.headers) {
    out += name;
    out += ": ";
    out += value;
    out += "\r\n";
  }
  out += "Connection: close\r\n\r\n";
  return out;
}

std::string BuildResponse(const HttpResponse& resp) {
  std::string out = BuildResponseHead(resp, resp.body.size());
  out += resp.body;
  return out;
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

json::Value TimingsJson(const TextGenerationBackend::Result& result) {
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
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  return timings;
}

json::Value UsageJson(const TextGenerationBackend::Result& result) {
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
  return usage;
}

json::Value MetricsJson(const TextGenerationBackend::Result& result) {
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
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  return metrics;
}

HttpResponse WithTiming(HttpResponse response,
                        const TextGenerationBackend::Result& result) {
  RecordServerMetrics(result);

  std::ostringstream value;
  value << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
        << ", inter_token;dur=" << result.mean_inter_token_ms
        << ", max_inter_token;dur=" << result.max_inter_token_ms;
  response.headers.emplace_back("Server-Timing", value.str());

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
  response.log_details = details.str();
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

HttpResponse NotImplemented(const HttpRequest&, TextGenerationBackend&) {
  return Err(501, "Not Implemented",
             "endpoint not implemented on this text model", "server_error",
             "not_implemented");
}

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

std::string ContentToString(const json::Value* content) {
  std::string out;
  if (content == nullptr)
    return out;
  if (content->is_string()) {
    out = content->get_str();
  } else if (content->is_array()) {
    for (const auto& part : content->items()) {
      out += part.member_str("text", "");
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

HttpResponse ListModels(TextGenerationBackend* backend,
                        const VideoJobService* video_jobs,
                        const TtsService* tts, const AsrService* asr) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  json::Value data = json::Value::array();
  if (backend != nullptr) {
    json::Value model = json::Value::object();
    model["id"] = backend->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "gufo";
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
  }
  if (tts != nullptr && tts->ready()) {
    json::Value model = json::Value::object();
    model["id"] = tts->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "operator-supplied-qwen";
    model["capability"] = "audio_tts";
    data.push_back(std::move(model));
  }
  if (asr != nullptr && asr->ready()) {
    json::Value model = json::Value::object();
    model["id"] = asr->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "operator-supplied-qwen";
    model["capability"] = "audio_asr";
    data.push_back(std::move(model));
  }
  resp["data"] = std::move(data);
  return Ok(resp);
}

HttpResponse OpenAiCompletions(const HttpRequest& req,
                               TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::string prompt;
  if (const json::Value* p = body.find("prompt")) {
    if (p->is_string()) {
      prompt = p->get_str();
    } else if (p->is_array()) {
      for (const auto& it : p->items())
        prompt += it.get_str();
    }
  }
  if (prompt.empty()) {
    return Err(400, "Bad Request", "'prompt' is required",
               "invalid_request_error", "missing_prompt");
  }

  const auto defaults = b.sampling_defaults();
  const std::size_t max_tokens =
      body.member_size("max_tokens", defaults.max_tokens);
  sampling::SamplingConfig sampling_config;
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, &sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }

  const auto res =
      b.complete(prompt, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "cmpl-" + RandomId();
  resp["object"] = "text_completion";
  resp["created"] = Now();
  resp["model"] = b.model_id();
  json::Value choices = json::Value::array();
  json::Value c = json::Value::object();
  c["text"] = res.text;
  c["index"] = 0;
  c["logprobs"] = json::Value();
  c["finish_reason"] = "stop";
  choices.push_back(std::move(c));
  resp["choices"] = std::move(choices);
  resp["usage"] = UsageJson(res);
  resp["timings"] = TimingsJson(res);
  return WithTiming(Ok(resp), res);
}

HttpResponse OpenAiResponses(const HttpRequest& req, TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::vector<tokenization::ChatMessage> messages;
  if (const json::Value* input = body.find("input")) {
    if (input->is_string()) {
      messages.push_back(
          {tokenization::ChatRole::kUser, input->get_str(), "", ""});
    } else if (input->is_array()) {
      for (const auto& item : input->items()) {
        tokenization::ChatMessage m;
        m.role = RoleFrom(item.member_str("role", "user"));
        m.content = ContentToString(item.find("content"));
        if (m.content.empty())
          m.content = item.member_str("text", "");
        messages.push_back(std::move(m));
      }
    }
  }

  const auto defaults = b.sampling_defaults();
  std::size_t max_tokens = defaults.max_tokens;
  if (body.contains("max_output_tokens")) {
    max_tokens = body.member_size("max_output_tokens", defaults.max_tokens);
  } else if (body.contains("max_tokens")) {
    max_tokens = body.member_size("max_tokens", defaults.max_tokens);
  }
  sampling::SamplingConfig sampling_config;
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, &sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }

  const auto res = b.chat(ChatRequest{std::move(messages)}, max_tokens,
                          sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "resp_" + RandomId();
  resp["object"] = "response";
  resp["status"] = "completed";
  resp["model"] = b.model_id();
  json::Value output = json::Value::array();
  json::Value msg = json::Value::object();
  msg["type"] = "message";
  msg["id"] = "msg_" + RandomId();
  msg["role"] = "assistant";
  json::Value content = json::Value::array();
  json::Value txt = json::Value::object();
  txt["type"] = "output_text";
  txt["text"] = res.text;
  content.push_back(std::move(txt));
  msg["content"] = std::move(content);
  output.push_back(std::move(msg));
  resp["output"] = std::move(output);
  json::Value usage = json::Value::object();
  usage["input_tokens"] = res.prompt_tokens;
  usage["output_tokens"] = res.completion_tokens;
  usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  json::Value input_details = json::Value::object();
  input_details["cached_tokens"] = res.cached_prompt_tokens;
  usage["input_token_details"] = std::move(input_details);
  resp["usage"] = std::move(usage);
  resp["timings"] = TimingsJson(res);
  return WithTiming(Ok(resp), res);
}

HttpResponse AnthropicMessages(const HttpRequest& req,
                               TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::vector<tokenization::ChatMessage> messages;
  if (const json::Value* system = body.find("system")) {
    const std::string sys = system->get_str();
    if (!sys.empty()) {
      messages.push_back({tokenization::ChatRole::kSystem, sys, "", ""});
    }
  }
  if (const json::Value* msgs = body.find("messages")) {
    if (msgs->is_array()) {
      for (const auto& it : msgs->items()) {
        tokenization::ChatMessage m;
        m.role = RoleFrom(it.member_str("role", "user"));
        m.content = ContentToString(it.find("content"));
        messages.push_back(std::move(m));
      }
    }
  }

  const auto defaults = b.sampling_defaults();
  const std::size_t max_tokens =
      body.member_size("max_tokens", defaults.max_tokens);
  sampling::SamplingConfig sampling_config;
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, &sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }

  const auto res = b.chat(ChatRequest{std::move(messages)}, max_tokens,
                          sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "msg_" + RandomId();
  resp["type"] = "message";
  resp["role"] = "assistant";
  resp["model"] = b.model_id();
  json::Value content = json::Value::array();
  json::Value txt = json::Value::object();
  txt["type"] = "text";
  txt["text"] = res.text;
  content.push_back(std::move(txt));
  resp["content"] = std::move(content);
  resp["stop_reason"] = "end_turn";
  resp["stop_sequence"] = json::Value();
  json::Value usage = json::Value::object();
  usage["input_tokens"] = res.prompt_tokens;
  usage["output_tokens"] = res.completion_tokens;
  usage["cache_creation_input_tokens"] = 0;
  usage["cache_read_input_tokens"] = res.cached_prompt_tokens;
  resp["usage"] = std::move(usage);
  resp["timings"] = TimingsJson(res);
  return WithTiming(Ok(resp), res);
}

HttpResponse AnthropicCountTokens(const HttpRequest& req,
                                  TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (...) {
    body = json::Value::object();
  }

  std::size_t total = 0;
  if (const json::Value* system = body.find("system")) {
    total += b.count_tokens(system->get_str());
  }
  if (const json::Value* msgs = body.find("messages")) {
    if (msgs->is_array()) {
      for (const auto& it : msgs->items()) {
        if (const json::Value* content = it.find("content")) {
          if (content->is_string()) {
            total += b.count_tokens(content->get_str());
          } else if (content->is_array()) {
            for (const auto& part : content->items()) {
              total += b.count_tokens(part.member_str("text", ""));
            }
          }
        }
      }
    }
  }

  json::Value resp = json::Value::object();
  resp["input_tokens"] = total;
  return Ok(resp);
}

HttpResponse LlamaCompletion(const HttpRequest& req, TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  const std::string prompt = body.member_str("prompt", "");
  const auto defaults = b.sampling_defaults();
  const std::size_t max_tokens =
      body.member_size("n_predict", defaults.max_tokens);
  sampling::SamplingConfig sampling_config;
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, &sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }

  const auto res =
      b.complete(prompt, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["content"] = res.text;
  resp["stop"] = true;
  resp["stopped_eos"] = false;
  resp["stopped_length"] = false;
  resp["stopped_word"] = false;
  resp["stopped_limit"] = false;
  resp["stopping_word"] = "";
  resp["tokens_predicted"] = res.completion_tokens;
  resp["tokens_evaluated"] = res.prompt_tokens;
  resp["tokens_cached"] = res.cached_prompt_tokens;
  resp["timings"] = TimingsJson(res);
  resp["usage"] = UsageJson(res);
  return WithTiming(Ok(resp), res);
}

HttpResponse LlamaInfill(const HttpRequest& req, TextGenerationBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (...) {
    body = json::Value::object();
  }

  // Best effort: the engine only completes forward, so infill runs a plain
  // completion on the prefix (suffix is ignored by the text model).
  const std::string prompt = body.member_str("input_prefix", "");
  const auto defaults = b.sampling_defaults();
  const std::size_t max_tokens =
      body.member_size("n_predict", defaults.max_tokens);
  sampling::SamplingConfig sampling_config;
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, &sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }

  const auto res =
      b.complete(prompt, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["content"] = res.text;
  resp["tokens_predicted"] = res.completion_tokens;
  resp["tokens_evaluated"] = res.prompt_tokens;
  resp["tokens_cached"] = res.cached_prompt_tokens;
  resp["timings"] = TimingsJson(res);
  resp["usage"] = UsageJson(res);
  return WithTiming(Ok(resp), res);
}

HttpResponse LlamaProps(const HttpRequest& req, TextGenerationBackend&) {
  const std::string model = req.query_param("model");
  if (model.empty()) {
    return Err(400, "Bad Request", "'model' query parameter is required",
               "invalid_request_error", "missing_model");
  }
  json::Value resp = json::Value::object();
  resp["model"] = model;
  resp["template"] = "";
  json::Value model_info = json::Value::object();
  resp["model_info"] = std::move(model_info);
  return Ok(resp);
}

HttpResponse LlamaSlots(const HttpRequest&, TextGenerationBackend& b) {
  json::Value resp = json::Value::array();
  json::Value slot = json::Value::object();
  slot["id"] = 0;
  slot["task_id"] = 0;
  slot["state"] = 0;
  slot["prompt"] = "";
  slot["next_token"] = json::Value();
  slot["model"] = b.model_id();
  resp.push_back(std::move(slot));
  return Ok(resp);
}

HttpResponse LlamaMetrics(const HttpRequest&, TextGenerationBackend&) {
  std::ostringstream out;
  out << "# HELP llamacpp:prompt_tokens_total Total prompt tokens processed\n"
      << "# TYPE llamacpp:prompt_tokens_total counter\n"
      << "llamacpp:prompt_tokens_total "
      << detail::TotalPromptTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:tokens_predicted_total Total tokens generated\n"
      << "# TYPE llamacpp:tokens_predicted_total counter\n"
      << "llamacpp:tokens_predicted_total "
      << detail::TotalGenTokens().load(std::memory_order_relaxed) << "\n"
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
      << "# HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio\n"
      << "# TYPE llamacpp:kv_cache_usage_ratio gauge\n"
      << "llamacpp:kv_cache_usage_ratio 0.0\n";
  return {.status = 200,
          .reason = "OK",
          .body = out.str(),
          .headers = {{"Content-Type", "text/plain; version=0.0.4"}}};
}

}  // namespace

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

std::string HttpRequest::query_param(const std::string& key) const {
  const auto pos = query.find(key + "=");
  if (pos == std::string::npos)
    return "";
  const auto value_start = pos + key.size() + 1;
  const auto end = query.find('&', value_start);
  const std::size_t len = (end == std::string::npos)
                              ? query.size() - value_start
                              : end - value_start;
  return UrlDecode(query.substr(value_start, len));
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

struct HttpServer::ConnectionWorker {
  std::atomic<int> fd{-1};
  std::atomic<bool> done{false};
  std::jthread thread;
};

HttpServer::HttpServer(std::string host, int port,
                       std::shared_ptr<TextGenerationBackend> backend,
                       std::shared_ptr<VideoJobService> video_jobs,
                       std::shared_ptr<TtsService> tts,
                       std::shared_ptr<AsrService> asr, HttpServerLimits limits)
    : host_(std::move(host)),
      port_(port),
      backend_(std::move(backend)),
      video_jobs_(std::move(video_jobs)),
      tts_(std::move(tts)),
      asr_(std::move(asr)),
      limits_(limits) {
  if (limits_.max_request_body_bytes == 0 || limits_.max_connections == 0) {
    throw std::invalid_argument("HTTP server limits must be positive");
  }
  register_routes();
}

HttpServer::~HttpServer() {
  stop();
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
  add("POST", "/v1/responses", OpenAiResponses);
  add("POST", "/v1/embeddings", NotImplemented);
  add("POST", "/v1/images/generations", NotImplemented);
  add("POST", "/v1/images/edits", NotImplemented);

  // ---- Anthropic ----
  add("POST", "/v1/messages", AnthropicMessages);
  add("POST", "/v1/messages/count_tokens", AnthropicCountTokens);

  // ---- llama-server ----
  add("POST", "/v1/rerank", NotImplemented);
  add("POST", "/v1/reranking", NotImplemented);
  add("POST", "/rerank", NotImplemented);
  add("POST", "/infill", LlamaInfill);
  add("POST", "/completion", LlamaCompletion);
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
  (void)::signal(SIGPIPE, SIG_IGN);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    if (error != nullptr)
      *error = "socket() failed";
    return false;
  }
  const int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port_));
  if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  }

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    if (error != nullptr) {
      *error = "bind() failed on " + host_ + ":" + std::to_string(port_);
    }
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::listen(listen_fd_, 16) < 0) {
    if (error != nullptr)
      *error = "listen() failed";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (port_ == 0) {
    sockaddr_in bound{};
    socklen_t length = sizeof(bound);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound),
                      &length) != 0) {
      if (error != nullptr) {
        *error = "getsockname() failed";
      }
      ::close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    port_ = ntohs(bound.sin_port);
  }
  return true;
}

void HttpServer::run() {
  Logger::Info("server",
               "Listening on http://" + host_ + ":" + std::to_string(port_));
  if (backend_ != nullptr) {
    Logger::Info("engine", "Loaded text model: " + backend_->model_id());
  }
  if (video_jobs_ != nullptr && video_jobs_->ready()) {
    Logger::Info("video", "MiniMax H3 video API enabled");
  }
  if (tts_ != nullptr && tts_->ready()) {
    Logger::Info("audio", "Qwen3-TTS audio API enabled");
  }
  if (asr_ != nullptr && asr_->ready()) {
    Logger::Info("audio", "Qwen3-ASR transcription API enabled");
  }
  while (!stopped_.load(std::memory_order_acquire)) {
    const int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (stopped_.load(std::memory_order_acquire)) {
        break;
      }
      continue;
    }
    if (stopped_.load(std::memory_order_acquire)) {
      ::close(client_fd);
      break;
    }

    reap_workers();
    auto worker = std::make_unique<ConnectionWorker>();
    ConnectionWorker* const worker_ptr = worker.get();
    bool overloaded = false;
    {
      const std::lock_guard<std::mutex> lock(workers_mutex_);
      overloaded = stopped_.load(std::memory_order_acquire) ||
                   workers_.size() >= limits_.max_connections;
      if (!overloaded) {
        worker_ptr->fd.store(client_fd, std::memory_order_release);
        worker_ptr->thread = std::jthread([this, worker_ptr, client_fd] {
          handle_connection(client_fd);
          worker_ptr->fd.store(-1, std::memory_order_release);
          worker_ptr->done.store(true, std::memory_order_release);
        });
        workers_.push_back(std::move(worker));
      }
    }
    if (overloaded) {
      HttpResponse response =
          Err(503, "Service Unavailable", "connection limit reached",
              "server_error", "overloaded");
      response.headers.emplace_back("Retry-After", "1");
      (void)SendAll(client_fd, BuildResponse(response));
      ::close(client_fd);
    }
  }
  reap_workers();
}

void HttpServer::stop() {
  const bool was_stopped = stopped_.exchange(true, std::memory_order_acq_rel);
  if (!was_stopped) {
    const int listen_fd = std::exchange(listen_fd_, -1);
    if (listen_fd >= 0) {
      (void)::shutdown(listen_fd, SHUT_RDWR);
      ::close(listen_fd);
    }
  }

  std::vector<std::unique_ptr<ConnectionWorker>> workers;
  {
    const std::lock_guard<std::mutex> lock(workers_mutex_);
    workers = std::move(workers_);
  }
  for (const auto& worker : workers) {
    const int fd = worker->fd.load(std::memory_order_acquire);
    if (fd >= 0) {
      (void)::shutdown(fd, SHUT_RDWR);
    }
  }
}

void HttpServer::reap_workers() {
  std::vector<std::unique_ptr<ConnectionWorker>> finished;
  {
    const std::lock_guard<std::mutex> lock(workers_mutex_);
    auto iterator = workers_.begin();
    while (iterator != workers_.end()) {
      if ((*iterator)->done.load(std::memory_order_acquire)) {
        finished.push_back(std::move(*iterator));
        iterator = workers_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }
}

HttpResponse HttpServer::handle_request(const HttpRequest& req) {
  if (req.method == "GET" && req.path == "/health") {
    json::Value body = json::Value::object();
    body["status"] = "ok";
    return Ok(body);
  }
  if (req.method == "GET" && req.path == "/ready") {
    const bool ready = (backend_ != nullptr && backend_->ready()) ||
                       (video_jobs_ != nullptr && video_jobs_->ready()) ||
                       (tts_ != nullptr && tts_->ready()) ||
                       (asr_ != nullptr && asr_->ready());
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
    } else {
      body["model"] = "minimax-h3";
    }
    return Ok(body);
  }
  if (req.method == "GET" &&
      (req.path == "/v1/models" || req.path == "/models")) {
    return ListModels(backend_.get(), video_jobs_.get(), tts_.get(),
                      asr_.get());
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
  for (const auto& entry : routes_) {
    if (entry.first.first == req.method && entry.first.second == req.path) {
      return entry.second(req, *backend_);
    }
  }
  return Err(404, "Not Found", "no route for this path",
             "invalid_request_error", "not_found");
}

void HttpServer::handle_connection(int client_fd) {
  const struct timeval tv{120, 0};
  ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  const auto start_time = std::chrono::steady_clock::now();
  HttpRequest req;
  try {
    bool ok = false;
    bool payload_too_large = false;
    {
      std::string buffer;
      if (ReadUntil(buffer, client_fd, "\r\n\r\n")) {
        // ReadUntil over-reads: `buffer` holds the headers, the blank line, and
        // possibly some body bytes already. Split at the blank line and carry
        // the over-read body bytes forward so we only read the remainder from
        // the socket.
        const auto delim_pos = buffer.find("\r\n\r\n");
        const std::string headers = buffer.substr(0, delim_pos);
        std::string body = buffer.substr(delim_pos + 4);

        const auto eol = headers.find("\r\n");
        const std::string line =
            (eol == std::string::npos) ? headers : headers.substr(0, eol);
        std::istringstream ls(line);
        std::string target;
        ls >> req.method >> target;
        const auto qpos = target.find('?');
        if (qpos != std::string::npos) {
          req.query = target.substr(qpos + 1);
          req.path = target.substr(0, qpos);
        } else {
          req.path = target;
        }

        std::size_t cursor =
            eol == std::string::npos ? headers.size() : eol + 2;
        while (cursor < headers.size()) {
          const std::size_t next = headers.find("\r\n", cursor);
          const std::size_t line_end =
              next == std::string::npos ? headers.size() : next;
          const std::string_view header_line(headers.data() + cursor,
                                             line_end - cursor);
          const std::size_t colon = header_line.find(':');
          if (colon != std::string_view::npos) {
            std::string name(header_line.substr(0, colon));
            std::string_view raw_value = header_line.substr(colon + 1);
            while (!raw_value.empty() &&
                   std::isspace(
                       static_cast<unsigned char>(raw_value.front())) != 0) {
              raw_value.remove_prefix(1);
            }
            while (!raw_value.empty() &&
                   std::isspace(static_cast<unsigned char>(raw_value.back())) !=
                       0) {
              raw_value.remove_suffix(1);
            }
            req.headers.emplace_back(std::move(name), std::string(raw_value));
          }
          if (next == std::string::npos) {
            break;
          }
          cursor = next + 2;
        }

        const std::size_t content_length = ParseContentLength(req);
        const std::size_t remaining =
            content_length > body.size() ? content_length - body.size() : 0;
        if (content_length > limits_.max_request_body_bytes) {
          payload_too_large = true;
        } else if (content_length > 0) {
          if (remaining > 0) {
            ok = ReadN(body, client_fd, remaining);
          } else {
            ok = true;
          }
          if (body.size() > content_length) {
            body.resize(content_length);
          }
        } else {
          ok = true;
          body.clear();
        }
        if (ok) {
          req.body = std::move(body);
          req.is_cancelled = [client_fd] {
            return IsPeerDisconnected(client_fd);
          };
        }
      }
    }

    HttpResponse resp;
    if (payload_too_large) {
      resp = Err(413, "Payload Too Large", "request body is too large",
                 "invalid_request_error", "payload_too_large");
    } else if (!ok) {
      resp = Err(400, "Bad Request", "malformed request",
                 "invalid_request_error", "bad_request");
    } else if (req.method == "OPTIONS") {
      resp = {.status = 204, .reason = "No Content"};
    } else {
      resp = handle_request(req);
    }

    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    Logger::LogRequest(req.method.empty() ? "UNKNOWN" : req.method,
                       req.path.empty() ? "/" : req.path, resp.status,
                       resp.reason, duration_ms, resp.log_details);

    if (resp.streaming_body) {
      if (SendAll(client_fd, BuildResponseHead(resp, std::nullopt))) {
        resp.streaming_body([client_fd](std::string_view chunk) {
          return SendAll(client_fd, chunk);
        });
      }
    } else {
      (void)SendAll(client_fd, BuildResponse(resp));
    }
  } catch (const TextGenerationError& exception) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    const char* reason = "Service Unavailable";
    if (exception.http_status() == 408) {
      reason = "Request Timeout";
    } else if (exception.http_status() == 429) {
      reason = "Too Many Requests";
    }
    HttpResponse resp = Err(exception.http_status(), reason, exception.what(),
                            "server_error", exception.stable_code());
    if (exception.retryable()) {
      resp.headers.emplace_back("Retry-After", "1");
    }
    Logger::LogRequest(req.method.empty() ? "UNKNOWN" : req.method,
                       req.path.empty() ? "/" : req.path, resp.status,
                       resp.reason, duration_ms, exception.what());
    (void)SendAll(client_fd, BuildResponse(resp));
  } catch (const std::exception& e) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    const HttpResponse resp = Err(500, "Internal Server Error", e.what(),
                                  "internal_error", "server_exception");
    Logger::LogRequest(req.method.empty() ? "UNKNOWN" : req.method,
                       req.path.empty() ? "/" : req.path, resp.status,
                       resp.reason, duration_ms, e.what());
    (void)SendAll(client_fd, BuildResponse(resp));
  } catch (...) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    const HttpResponse resp =
        Err(500, "Internal Server Error", "unknown server error",
            "internal_error", "server_exception");
    Logger::LogRequest(req.method.empty() ? "UNKNOWN" : req.method,
                       req.path.empty() ? "/" : req.path, resp.status,
                       resp.reason, duration_ms, "unknown server error");
    (void)SendAll(client_fd, BuildResponse(resp));
  }
  ::close(client_fd);
}

}  // namespace gufo::server

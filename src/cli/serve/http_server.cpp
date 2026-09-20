#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"
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

bool SendChunk(int fd, std::string_view data) {
  if (data.empty())
    return true;
  char header[2 * sizeof(std::size_t) + 2];
  const auto length =
      std::to_chars(header, header + sizeof(header) - 2, data.size(), 16);
  *length.ptr = '\r';
  *(length.ptr + 1) = '\n';
  return SendAll(fd, std::string_view(header, length.ptr + 2 - header)) &&
         SendAll(fd, data) && SendAll(fd, "\r\n");
}

bool IsPeerDisconnected(int fd) noexcept {
  pollfd descriptor{
      .fd = fd,
      .events = POLLIN | POLLERR | POLLHUP,
      .revents = 0,
  };
#ifdef POLLRDHUP
  descriptor.events |= POLLRDHUP;
#endif
  const int ready = ::poll(&descriptor, 1, 0);
  if (ready <= 0)
    return false;
  if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    return true;
#ifdef POLLRDHUP
  if ((descriptor.revents & POLLRDHUP) != 0)
    return true;
#endif
  if ((descriptor.revents & POLLIN) == 0)
    return false;
  char byte;
  const auto count = ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  return count == 0 || (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                        errno != EINTR);
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
        return -1;
      };
      const int high = hexval(s[i + 1]);
      const int low = hexval(s[i + 2]);
      if (high >= 0 && low >= 0) {
        out += static_cast<char>((high << 4) | low);
        i += 2;
      } else {
        out += c;
      }
    } else {
      out += c;
    }
  }
  return out;
}

std::optional<std::size_t> ParseContentLength(const HttpRequest& request) {
  std::optional<std::size_t> length;
  for (const auto& [name, value] : request.headers) {
    const auto lowered = ToLower(name);
    // Chunked transfer is not implemented. Never interpret its encoded bytes
    // as an empty or partial inference request.
    if (lowered == "transfer-encoding") {
      return std::nullopt;
    }
    if (lowered != "content-length") {
      continue;
    }
    std::size_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        (length.has_value() && *length != parsed)) {
      return std::nullopt;
    }
    length = parsed;
  }
  return length.value_or(0);
}

std::string CredentialHash(std::string_view value) {
  return crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(value.data()), value.size()));
}

bool IsAuthorized(const HttpRequest& request, const std::string& key_hash) {
  if (key_hash.empty()) {
    return true;
  }
  std::string_view credential;
  bool found = false;
  for (const auto& [name, value] : request.headers) {
    if (ToLower(name) == "authorization") {
      if (found) {
        return false;
      }
      found = true;
      credential = value;
    }
  }
  const auto space = credential.find(' ');
  if (space == std::string_view::npos ||
      ToLower(credential.substr(0, space)) != "bearer") {
    return false;
  }
  credential.remove_prefix(space + 1);
  while (credential.starts_with(' ')) {
    credential.remove_prefix(1);
  }
  // Comparing digests does not disclose matching prefixes of the API key.
  return CredentialHash(credential) == key_hash;
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
  if (resp.status != 101 && !HasHeader(resp, "content-type")) {
    out += "Content-Type: application/json\r\n";
  }
  if (content_length.has_value()) {
    out += "Content-Length: " + std::to_string(*content_length) + "\r\n";
  }
  out += "Access-Control-Allow-Origin: *\r\n";
  out += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";
  out += "Access-Control-Allow-Headers: Content-Type, Authorization, Range\r\n";
  for (const auto& [name, value] : resp.headers) {
    out += name;
    out += ": ";
    out += value;
    out += "\r\n";
  }
  if (!HasHeader(resp, "connection"))
    out += "Connection: close\r\n";
  out += "\r\n";
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

bool ReadTextContent(const json::Value* content, std::string* out) {
  if (content == nullptr)
    return false;
  if (content->is_string()) {
    *out = content->get_str();
  } else if (content->is_array()) {
    for (const auto& part : content->items()) {
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
                      std::vector<tokenization::ChatMessage>* messages) {
  if (input == nullptr || !input->is_array() || input->empty())
    return false;
  for (const auto& item : input->items()) {
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
    if (!ReadTextContent(item.find("content"), &message.content))
      return false;
    messages->push_back(std::move(message));
  }
  return true;
}

HttpResponse InvalidCompatibilityRequest(std::string_view message) {
  return Err(400, "Bad Request", std::string(message).c_str(),
             "invalid_request_error", "invalid_request");
}

// Compatibility routes implement a synchronous text subset. Validate options
// before dispatch so a client never gets an answer to a different request.
std::optional<HttpResponse> ReadCompatibilityOptions(
    const json::Value& body, TextGenerationBackend& backend,
    std::string_view token_field, std::size_t* max_tokens,
    sampling::SamplingConfig* sampling_config) {
  if (!body.is_object())
    return InvalidCompatibilityRequest("request body must be an object");
  if (const auto* model = body.find("model"); model != nullptr) {
    if (!model->is_string())
      return InvalidCompatibilityRequest("'model' must be a string");
    if (model->str() != backend.model_id())
      return Err(404, "Not Found", "requested model is not loaded",
                 "invalid_request_error", "model_not_found");
  }
  for (const std::string field : {"stream", "echo", "store", "background"}) {
    if (const auto* value = body.find(field);
        value != nullptr && (!value->is_bool() || value->as_bool())) {
      return InvalidCompatibilityRequest("'" + field + "' must be false");
    }
  }
  for (const std::string field : {"n", "best_of"}) {
    if (const auto* value = body.find(field);
        value != nullptr &&
        (!value->is_number() || value->as_double() != 1.0)) {
      return InvalidCompatibilityRequest("only '" + field + "=1' is supported");
    }
  }
  for (const std::string field : {"stream_options",
                                  "stop",
                                  "stop_sequences",
                                  "logprobs",
                                  "top_logprobs",
                                  "suffix",
                                  "tools",
                                  "tool_choice",
                                  "parallel_tool_calls",
                                  "response_format",
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
    if (body.contains(field)) {
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
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }
  return std::nullopt;
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
                                            &sampling_config)) {
    return std::move(*error);
  }

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string()) {
    return InvalidCompatibilityRequest("'prompt' must be a single string");
  }
  const std::string prompt = input->str();
  if (prompt.empty()) {
    return Err(400, "Bad Request", "'prompt' is required",
               "invalid_request_error", "missing_prompt");
  }

  const auto res = b.complete(prompt, max_tokens, sampling_config,
                              req.is_cancelled, {}, req.client_id);

  json::Value resp = json::Value::object();
  resp["id"] = "cmpl-" + RandomId();
  resp["object"] = "text_completion";
  resp["created"] = Now();
  resp["model"] = b.model_id();
  json::Value choices = json::Value::array();
  json::Value c = json::Value::object();
  c["text"] = core::Utf8Decoder{}.Push(res.text, true);
  c["index"] = 0;
  c["logprobs"] = json::Value();
  c["finish_reason"] =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "length"
          : "stop";
  choices.push_back(std::move(c));
  resp["choices"] = std::move(choices);
  resp["usage"] = UsageJson(res);
  resp["timings"] = GenerationTimings(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse OpenAiResponses(const HttpRequest& req,
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
  if (auto error = ReadCompatibilityOptions(body, b, "max_output_tokens",
                                            &max_tokens, &sampling_config)) {
    return std::move(*error);
  }

  std::vector<tokenization::ChatMessage> messages;
  if (const auto* instructions = body.find("instructions")) {
    if (!instructions->is_string()) {
      return InvalidCompatibilityRequest("'instructions' must be a string");
    }
    messages.push_back(
        {tokenization::ChatRole::kSystem, instructions->str(), "", ""});
  }
  const auto* input = body.find("input");
  if (input != nullptr && input->is_string() && !input->str().empty()) {
    messages.push_back({tokenization::ChatRole::kUser, input->str(), "", ""});
  } else if (!ReadTextMessages(input, &messages)) {
    return InvalidCompatibilityRequest(
        "'input' must be nonempty text or text messages; use "
        "/v1/chat/completions for images and tools");
  }

  ChatRequest chat{std::move(messages)};
  chat.client_id = req.client_id;
  const auto res = b.chat(chat, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "resp_" + RandomId();
  resp["object"] = "response";
  const bool limited =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength;
  resp["status"] = limited ? "incomplete" : "completed";
  resp["incomplete_details"] = json::Value();
  if (limited) {
    resp["incomplete_details"]["reason"] = "max_output_tokens";
  }
  resp["model"] = b.model_id();
  json::Value output = json::Value::array();
  json::Value msg = json::Value::object();
  msg["type"] = "message";
  msg["id"] = "msg_" + RandomId();
  msg["role"] = "assistant";
  json::Value content = json::Value::array();
  json::Value txt = json::Value::object();
  txt["type"] = "output_text";
  txt["text"] = core::Utf8Decoder{}.Push(res.text, true);
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
  usage["input_tokens_details"] = std::move(input_details);
  resp["usage"] = std::move(usage);
  resp["timings"] = GenerationTimings(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse AnthropicMessages(const HttpRequest& req,
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
                                            &sampling_config)) {
    return std::move(*error);
  }

  std::vector<tokenization::ChatMessage> messages;
  if (const auto* system = body.find("system")) {
    std::string text;
    if (!ReadTextContent(system, &text)) {
      return InvalidCompatibilityRequest("'system' must contain only text");
    }
    messages.push_back(
        {tokenization::ChatRole::kSystem, std::move(text), "", ""});
  }
  if (!ReadTextMessages(body.find("messages"), &messages)) {
    return InvalidCompatibilityRequest(
        "'messages' must contain text messages; use /v1/chat/completions "
        "for images and tools");
  }

  ChatRequest chat{std::move(messages)};
  chat.client_id = req.client_id;
  const auto res = b.chat(chat, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "msg_" + RandomId();
  resp["type"] = "message";
  resp["role"] = "assistant";
  resp["model"] = b.model_id();
  json::Value content = json::Value::array();
  json::Value txt = json::Value::object();
  txt["type"] = "text";
  txt["text"] = core::Utf8Decoder{}.Push(res.text, true);
  content.push_back(std::move(txt));
  resp["content"] = std::move(content);
  resp["stop_reason"] =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "max_tokens"
          : "end_turn";
  resp["stop_sequence"] = json::Value();
  json::Value usage = json::Value::object();
  usage["input_tokens"] = res.prompt_tokens;
  usage["output_tokens"] = res.completion_tokens;
  usage["cache_creation_input_tokens"] = 0;
  usage["cache_read_input_tokens"] = res.cached_prompt_tokens;
  resp["usage"] = std::move(usage);
  resp["timings"] = GenerationTimings(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
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
  if (auto error = ReadCompatibilityOptions(body, b, "n_predict", &max_tokens,
                                            &sampling_config)) {
    return std::move(*error);
  }

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string() || input->str().empty()) {
    return InvalidCompatibilityRequest("'prompt' must be a nonempty string");
  }
  const std::string prompt = input->str();

  const auto res = b.complete(prompt, max_tokens, sampling_config,
                              req.is_cancelled, {}, req.client_id);

  json::Value resp = json::Value::object();
  resp["content"] = core::Utf8Decoder{}.Push(res.text, true);
  resp["stop"] = true;
  const bool limited =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength;
  resp["stopped_eos"] = !limited && !res.cancelled;
  resp["stopped_length"] = limited;
  resp["stopped_word"] = false;
  resp["stopped_limit"] = limited;
  resp["stopping_word"] = "";
  resp["tokens_predicted"] = res.completion_tokens;
  resp["tokens_evaluated"] = res.prompt_tokens;
  resp["tokens_cached"] = res.cached_prompt_tokens;
  resp["timings"] = GenerationTimings(res);
  resp["usage"] = UsageJson(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
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
  std::string_view remaining = query;
  while (!remaining.empty()) {
    const auto end = remaining.find('&');
    const auto parameter = remaining.substr(0, end);
    const auto equals = parameter.find('=');
    if (UrlDecode(parameter.substr(0, equals)) == key) {
      return equals == std::string_view::npos
                 ? ""
                 : UrlDecode(parameter.substr(equals + 1));
    }
    if (end == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(end + 1);
  }
  return "";
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

struct HttpServer::ConnectionWorker {
  int fd{-1};  // Protected by workers_mutex_, including shutdown and close.
  std::atomic<bool> done{false};
  std::jthread thread;
};

HttpServer::HttpServer(std::string host, int port,
                       std::shared_ptr<TextGenerationBackend> backend,
                       std::shared_ptr<VideoJobService> video_jobs,
                       std::shared_ptr<TtsService> tts,
                       std::shared_ptr<AsrService> asr,
                       HttpServerOptions options)
    : host_(std::move(host)),
      port_(port),
      backend_(std::move(backend)),
      video_jobs_(std::move(video_jobs)),
      tts_(std::move(tts)),
      asr_(std::move(asr)),
      options_(std::move(options)) {
  if (options_.max_request_body_bytes == 0 || options_.max_connections == 0) {
    throw std::invalid_argument("HTTP server limits must be positive");
  }
  if (!options_.api_key.empty()) {
    api_key_hash_ = CredentialHash(options_.api_key);
  }
  register_routes();
}

HttpServer::~HttpServer() {
  stop();
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
  }
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
  add("POST", "/v1/messages/count_tokens", NotImplemented);

  // ---- llama-server ----
  add("POST", "/v1/rerank", NotImplemented);
  add("POST", "/v1/reranking", NotImplemented);
  add("POST", "/rerank", NotImplemented);
  add("POST", "/infill", NotImplemented);
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
  if (listen_fd_ >= 0 || stopped_.load(std::memory_order_acquire)) {
    if (error != nullptr) {
      *error = "HTTP server has already been started or stopped";
    }
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  if (port_ < 0 || port_ > 65535 ||
      ::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    if (error != nullptr) {
      *error =
          "host must be an IPv4 address and port must be between 0 and 65535";
    }
    return false;
  }
  addr.sin_port = htons(static_cast<unsigned short>(port_));
  (void)::signal(SIGPIPE, SIG_IGN);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    if (error != nullptr)
      *error = "socket() failed";
    return false;
  }
  const int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

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
  Logger::Info(
      "server",
      "event=listening address=http://" + host_ + ":" + std::to_string(port_) +
          " auth=" + (options_.api_key.empty() ? "off" : "bearer") +
          " max_connections=" + std::to_string(options_.max_connections) +
          " max_body_bytes=" + std::to_string(options_.max_request_body_bytes));
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
                   workers_.size() >= options_.max_connections;
      if (!overloaded) {
        worker_ptr->fd = client_fd;
        worker_ptr->thread = std::jthread([this, worker_ptr, client_fd] {
          handle_connection(client_fd);
          {
            const std::lock_guard<std::mutex> lock(workers_mutex_);
            ::close(worker_ptr->fd);
            worker_ptr->fd = -1;
          }
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
  if (!was_stopped && listen_fd_ >= 0) {
    // Keep the descriptor owned until destruction: run() may still be inside
    // accept(). Closing here allows it to observe a reused descriptor.
    (void)::shutdown(listen_fd_, SHUT_RDWR);
  }

  std::vector<std::unique_ptr<ConnectionWorker>> workers;
  {
    const std::lock_guard<std::mutex> lock(workers_mutex_);
    workers = std::move(workers_);
    for (const auto& worker : workers) {
      if (worker->fd >= 0) {
        (void)::shutdown(worker->fd, SHUT_RDWR);
      }
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
  if (!IsAuthorized(req, api_key_hash_)) {
    auto response = Err(401, "Unauthorized", "missing or invalid API key",
                        "authentication_error", "invalid_api_key");
    response.headers.emplace_back("WWW-Authenticate", "Bearer");
    return response;
  }
  // `/health` and `/ready` are the native spelling and match llama-server's
  // `/health`. `/healthz` and `/readyz` are aliases so Kubernetes-style probe
  // configuration works unmodified.
  if (req.method == "GET" &&
      (req.path == "/health" || req.path == "/v1/health" ||
       req.path == "/healthz")) {
    json::Value body = json::Value::object();
    body["status"] = "ok";
    return Ok(body);
  }
  if (req.method == "GET" && (req.path == "/ready" || req.path == "/v1/ready" ||
                              req.path == "/readyz")) {
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
  static std::atomic<std::uint64_t> next_request{0};
  sockaddr_in peer{};
  socklen_t peer_size = sizeof(peer);
  char peer_address[INET_ADDRSTRLEN]{};
  if (::getpeername(client_fd, reinterpret_cast<sockaddr*>(&peer),
                    &peer_size) == 0 &&
      ::inet_ntop(AF_INET, &peer.sin_addr, peer_address, sizeof(peer_address)))
    req.client_id = peer_address;
  req.request_id = "r" + std::to_string(next_request.fetch_add(1) + 1);
  bool response_started = false;
  bool http11 = false;
  int response_status = 0;
  const auto elapsed_ms = [&] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_time)
        .count();
  };
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
        std::string version;
        std::string extra;
        ls >> req.method >> target >> version;
        http11 = version == "HTTP/1.1";
        bool valid_headers = !req.method.empty() && target.starts_with('/') &&
                             (version == "HTTP/1.0" || version == "HTTP/1.1") &&
                             !(ls >> extra);
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
          if (colon != std::string_view::npos && colon > 0) {
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
          } else {
            valid_headers = false;
          }
          if (next == std::string::npos) {
            break;
          }
          cursor = next + 2;
        }

        const auto parsed_length = ParseContentLength(req);
        const std::size_t content_length = parsed_length.value_or(0);
        const std::size_t remaining =
            content_length > body.size() ? content_length - body.size() : 0;
        if (!valid_headers || !parsed_length.has_value()) {
          ok = false;
        } else if (content_length > options_.max_request_body_bytes) {
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
          if (!IsWebSocketUpgrade(req))
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

    // Successful health/metrics polling and video status polling stay quiet.
    const bool log_request =
        req.method == "POST" || req.method == "DELETE" ||
        req.path == "/v1/models" || req.path.ends_with("/content") ||
        req.path == "/v1/realtime" || req.path == "/v1/audio/speech/stream";
    if (ok && log_request) {
      Logger::Info("http", "request=" + req.request_id +
                               " event=received method=" + req.method +
                               " path=" + req.path + " body_bytes=" +
                               std::to_string(req.body.size()));
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

    resp.headers.emplace_back("X-Request-ID", req.request_id);
    response_status = resp.status;
    bool connected = true;
    if (resp.websocket) {
      response_started = true;
      connected = SendAll(client_fd, BuildResponseHead(resp, std::nullopt));
      if (connected) {
        WebSocket socket(client_fd, std::move(req.body));
        resp.websocket(socket);
      }
    } else if (resp.streaming_body) {
      const bool chunked = http11 && !HasHeader(resp, "content-length");
      if (chunked)
        resp.headers.emplace_back("Transfer-Encoding", "chunked");
      const auto head = BuildResponseHead(resp, std::nullopt);
      response_started = true;
      connected = SendAll(client_fd, head);
      if (connected) {
        resp.streaming_body([&](std::string_view chunk) {
          connected = connected && (chunked ? SendChunk(client_fd, chunk)
                                            : SendAll(client_fd, chunk));
          return connected;
        });
        // Without the final chunk, HTTP clients report an incomplete body.
        // Closing an unframed PCM stream would silently look like shorter
        // audio.
        if (connected && chunked &&
            (!resp.stream_log || resp.stream_log->error_code.empty()))
          connected = SendAll(client_fd, "0\r\n\r\n");
      }
    } else {
      const auto payload = BuildResponse(resp);
      response_started = true;
      connected = SendAll(client_fd, payload);
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
        // The status remains useful for an endpoint returning a non-JSON error.
      }
    }
    if (log_request || resp.status >= 400 || !connected) {
      Logger::LogRequest(req.request_id, req.method, req.path, resp.status,
                         elapsed_ms(), resp.log_details, outcome);
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
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    if (exception.retryable()) {
      resp.headers.emplace_back("Retry-After", "1");
    }
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms,
                       std::string("error_code=") + exception.stable_code(),
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  } catch (const std::exception& e) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    HttpResponse resp = Err(500, "Internal Server Error", e.what(),
                            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  } catch (...) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    HttpResponse resp =
        Err(500, "Internal Server Error", "unknown server error",
            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  }
}

}  // namespace gufo::server

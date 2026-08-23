#include "src/server/http_server.hpp"

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
#include <random>
#include <ranges>
#include <sstream>
#include <thread>
#include <utility>

#include "src/server/audio_tts_api.hpp"
#include "src/server/json.hpp"
#include "src/server/tts_service.hpp"
#include "src/server/video_api.hpp"
#include "src/server/video_jobs.hpp"
#include "src/models/qwen/qwen_chat_template.hpp"

namespace strix::server {
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

std::string BuildResponse(const HttpResponse& resp) {
  std::string out;
  out.reserve(resp.body.size() + 256);
  out += "HTTP/1.1 ";
  out += std::to_string(resp.status);
  out += ' ';
  out += resp.reason;
  out += "\r\n";
  if (!HasHeader(resp, "content-type")) {
    out += "Content-Type: application/json\r\n";
  }
  out += "Content-Length: " + std::to_string(resp.body.size()) + "\r\n";
  out += "Access-Control-Allow-Origin: *\r\n";
  out += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";
  out += "Access-Control-Allow-Headers: Content-Type, Authorization, Range\r\n";
  for (const auto& [name, value] : resp.headers) {
    out += name;
    out += ": ";
    out += value;
    out += "\r\n";
  }
  out += "Connection: close\r\n\r\n";
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
  return {200, "OK", v.dump(), {}};
}

HttpResponse WithTiming(HttpResponse response,
                        const InferenceBackend::Result& result) {
  std::ostringstream value;
  value << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
        << ", inter_token;dur=" << result.mean_inter_token_ms;
  response.headers.emplace_back("Server-Timing", value.str());
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
  return {status, reason, e.dump(), {}};
}

HttpResponse NotImplemented(const HttpRequest&, InferenceBackend&) {
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

HttpResponse ListModels(InferenceBackend* backend,
                        const VideoJobService* video_jobs,
                        const TtsService* tts) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  json::Value data = json::Value::array();
  if (backend != nullptr) {
    json::Value model = json::Value::object();
    model["id"] = backend->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "strix";
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
  resp["data"] = std::move(data);
  return Ok(resp);
}

HttpResponse OpenAiCompletions(const HttpRequest& req, InferenceBackend& b) {
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

  const std::size_t max_tokens = body.member_size("max_tokens", 128);
  const float temperature =
      static_cast<float>(body.member_double("temperature", 0.7));

  const auto res =
      b.complete(prompt, max_tokens, temperature, req.is_cancelled);

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
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = res.prompt_tokens;
  usage["completion_tokens"] = res.completion_tokens;
  usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  resp["usage"] = std::move(usage);
  return WithTiming(Ok(resp), res);
}

HttpResponse OpenAiChat(const HttpRequest& req, InferenceBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::vector<tokenization::ChatMessage> messages;
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
  if (messages.empty()) {
    return Err(400, "Bad Request", "'messages' is required",
               "invalid_request_error", "missing_messages");
  }

  const std::size_t max_tokens = body.member_size("max_tokens", 128);
  const float temperature =
      static_cast<float>(body.member_double("temperature", 0.7));

  const auto res = b.chat(messages, max_tokens, temperature, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "chatcmpl-" + RandomId();
  resp["object"] = "chat.completion";
  resp["created"] = Now();
  resp["model"] = b.model_id();
  json::Value choices = json::Value::array();
  json::Value c = json::Value::object();
  c["index"] = 0;
  json::Value msg = json::Value::object();
  msg["role"] = "assistant";
  msg["content"] = res.text;
  c["message"] = std::move(msg);
  c["finish_reason"] = "stop";
  choices.push_back(std::move(c));
  resp["choices"] = std::move(choices);
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = res.prompt_tokens;
  usage["completion_tokens"] = res.completion_tokens;
  usage["total_tokens"] = res.prompt_tokens + res.completion_tokens;
  resp["usage"] = std::move(usage);
  return WithTiming(Ok(resp), res);
}

HttpResponse OpenAiResponses(const HttpRequest& req, InferenceBackend& b) {
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

  std::size_t max_tokens = 256;
  if (body.contains("max_output_tokens")) {
    max_tokens = body.member_size("max_output_tokens", 256);
  } else if (body.contains("max_tokens")) {
    max_tokens = body.member_size("max_tokens", 256);
  }
  const float temperature =
      static_cast<float>(body.member_double("temperature", 0.7));

  const auto res = b.chat(messages, max_tokens, temperature, req.is_cancelled);

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
  resp["usage"] = std::move(usage);
  return WithTiming(Ok(resp), res);
}

HttpResponse AnthropicMessages(const HttpRequest& req, InferenceBackend& b) {
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

  const std::size_t max_tokens = body.member_size("max_tokens", 1024);
  const float temperature =
      static_cast<float>(body.member_double("temperature", 1.0));

  const auto res = b.chat(messages, max_tokens, temperature, req.is_cancelled);

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
  resp["usage"] = std::move(usage);
  return WithTiming(Ok(resp), res);
}

HttpResponse AnthropicCountTokens(const HttpRequest& req, InferenceBackend& b) {
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

HttpResponse LlamaCompletion(const HttpRequest& req, InferenceBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  const std::string prompt = body.member_str("prompt", "");
  const std::size_t max_tokens = body.member_size("n_predict", 128);
  const float temperature =
      static_cast<float>(body.member_double("temperature", 0.8));

  const auto res =
      b.complete(prompt, max_tokens, temperature, req.is_cancelled);

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
  return WithTiming(Ok(resp), res);
}

HttpResponse LlamaInfill(const HttpRequest& req, InferenceBackend& b) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (...) {
    body = json::Value::object();
  }

  // Best effort: the engine only completes forward, so infill runs a plain
  // completion on the prefix (suffix is ignored by the text model).
  const std::string prompt = body.member_str("input_prefix", "");
  const std::size_t max_tokens = body.member_size("n_predict", 128);
  const float temperature =
      static_cast<float>(body.member_double("temperature", 0.7));

  const auto res =
      b.complete(prompt, max_tokens, temperature, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["content"] = res.text;
  return WithTiming(Ok(resp), res);
}

HttpResponse LlamaProps(const HttpRequest& req, InferenceBackend&) {
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

HttpServer::HttpServer(std::string host, int port,
                       std::shared_ptr<InferenceBackend> backend,
                       std::shared_ptr<VideoJobService> video_jobs,
                       std::shared_ptr<TtsService> tts)
    : host_(std::move(host)),
      port_(port),
      backend_(std::move(backend)),
      video_jobs_(std::move(video_jobs)),
      tts_(std::move(tts)) {
  register_routes();
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
  add("POST", "/v1/chat/completions", OpenAiChat);
  add("POST", "/v1/responses", OpenAiResponses);
  add("POST", "/v1/embeddings", NotImplemented);
  add("POST", "/v1/audio/transcriptions", NotImplemented);
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
  return true;
}

void HttpServer::run() {
  std::cout << "strix-server: listening on http://" << host_ << ":" << port_
            << "\n";
  if (backend_ != nullptr) {
    std::cout << "strix-server: text model " << backend_->model_id() << "\n";
  }
  if (video_jobs_ != nullptr && video_jobs_->ready()) {
    std::cout << "strix-server: MiniMax H3 video API enabled\n";
  }
  if (tts_ != nullptr && tts_->ready()) {
    std::cout << "strix-server: Qwen3-TTS audio API enabled\n";
  }
  while (!stopped_.load()) {
    const int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0)
      continue;
    std::thread([this, client_fd] { handle_connection(client_fd); }).detach();
  }
}

void HttpServer::stop() {
  stopped_.store(true);
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

HttpResponse HttpServer::handle_request(const HttpRequest& req) {
  if (req.method == "GET" &&
      (req.path == "/v1/models" || req.path == "/models")) {
    return ListModels(backend_.get(), video_jobs_.get(), tts_.get());
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

  try {
    HttpRequest req;
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
        constexpr std::size_t kMaximumBodyBytes =
            static_cast<std::size_t>(8) * 1024 * 1024;
        if (content_length > kMaximumBodyBytes) {
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
      resp = {204, "No Content", "", {}};
    } else {
      resp = handle_request(req);
    }

    SendAll(client_fd, BuildResponse(resp));
  } catch (const std::exception& e) {
    const HttpResponse resp = Err(500, "Internal Server Error", e.what(),
                                  "internal_error", "server_exception");
    SendAll(client_fd, BuildResponse(resp));
  } catch (...) {
    const HttpResponse resp =
        Err(500, "Internal Server Error", "unknown server error",
            "internal_error", "server_exception");
    SendAll(client_fd, BuildResponse(resp));
  }
  ::close(client_fd);
}

}  // namespace strix::server

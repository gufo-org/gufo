#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cli/serve/logging.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::Logger;
using gufo::server::LogLevel;
using gufo::server::LogLevelFromName;
using gufo::server::LogLevelName;
using gufo::server::TextGenerationBackend;

/// Reports fixed prompt progress before delegating token generation.
class ProgressRequest final : public TextGenerationBackend::GenerationRequest {
public:
  ProgressRequest(std::shared_ptr<GenerationRequest> inner,
                  std::vector<TextGenerationBackend::PromptProgress> progress)
      : inner_(std::move(inner)), progress_(std::move(progress)) {}

  TextGenerationBackend::Result Wait(
      const TextGenerationBackend::TokenCallback& on_token,
      const TextGenerationBackend::ProgressCallback& on_progress) override {
    for (const auto& value : progress_) {
      if (on_progress && !on_progress(value)) {
        inner_->Cancel();
        break;
      }
    }
    return inner_->Wait(on_token, on_progress);
  }
  void Cancel() noexcept override { inner_->Cancel(); }

private:
  std::shared_ptr<GenerationRequest> inner_;
  std::vector<TextGenerationBackend::PromptProgress> progress_;
};

class FakeBackend final : public TextGenerationBackend {
public:
  struct Call {
    std::string prompt;
    gufo::server::ChatRequest chat;
    std::size_t max_tokens = 0;
    gufo::sampling::SamplingConfig sampling;
    std::string client_id;
    std::vector<std::string> stop_sequences;
  };
  Call LastCall() {
    const std::lock_guard lock(mutex_);
    return last_;
  }
  void SetOutput(std::string text) {
    const std::lock_guard lock(mutex_);
    output_ = std::move(text);
  }
  std::string model_id() const override { return "test"; }
  bool ready() const override { return true; }
  bool supports_images() const override { return image_support.load(); }
  std::atomic<bool> image_support{false};
  std::uint32_t max_context() const override { return 65536; }
  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancellation, bool stream, bool ignore_eos,
      std::string_view client_id,
      const std::vector<std::string>& stop_sequences,
      bool return_progress) override {
    last_ignore_eos = ignore_eos;
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_complete(
            prompt, max_tokens, sampling, cancellation, stream, false,
            client_id, stop_sequences, return_progress),
        progress);
  }
  SamplingDefaults sampling_defaults() const override { return defaults; }
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning;
  }
  std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  Result complete(
      std::string_view prompt, std::size_t limit,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancel, const TokenCallback& token,
      std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {}) override {
    ++calls;
    if (failure == 1)
      throw std::length_error("context exceeded");
    if (failure == 2)
      throw std::invalid_argument("invalid prompt");
    Result result;
    if (wait_for_disconnect) {
      entered.release();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!(disconnected = cancel && cancel()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished.release();
      result.cancelled = disconnected;
      return result;
    }
    {
      const std::lock_guard lock(mutex_);
      last_ = {.prompt = std::string(prompt),
               .max_tokens = limit,
               .sampling = sampling,
               .client_id = std::string(client_id),
               .stop_sequences = stop_sequences};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_tokens = 8;
    result.prefill_tokens = 2;
    result.prefill_ms = 4;
    result.completion_tokens = 1;
    result.decode_ms = 2;
    result.finish_reason =
        limit == 1 ? FinishReason::kLength : FinishReason::kStop;
    if (!forced_stop_sequence.empty()) {
      result.finish_reason = FinishReason::kStopSequence;
      result.stop_sequence = forced_stop_sequence;
    }
    if (token)
      (void)token(result.text);
    return result;
  }
  Result chat(const gufo::server::ChatRequest& request, std::size_t limit,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancel,
              const TokenCallback& token) override {
    auto result = complete("", limit, sampling, cancel, token,
                           request.client_id, request.stop_sequences);
    {
      const std::lock_guard lock(mutex_);
      last_.chat = request;
    }
    return result;
  }
  std::atomic<int> calls{0};
  std::atomic<bool> last_ignore_eos{false};
  std::vector<PromptProgress> progress;
  std::atomic<int> failure{0};
  std::string forced_stop_sequence;
  gufo::ReasoningOptions reasoning;
  bool wait_for_disconnect{false};
  std::atomic<bool> disconnected{false};
  std::binary_semaphore entered{0};
  std::binary_semaphore finished{0};

private:
  std::mutex mutex_;
  Call last_;
  std::string output_{"ok"};
};

class RunningServer {
public:
  explicit RunningServer(gufo::server::HttpServerOptions options = {},
                         bool handle_signals = false)
      : backend(std::make_shared<FakeBackend>()),
        server("127.0.0.1", 0, backend, nullptr, nullptr, nullptr,
               std::move(options)) {
    server.add("POST", "/echo", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.body};
    });
    server.add("POST", "/stream-error", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .streaming_body = [](const auto& write) {
            (void)write("first chunk");
            throw std::runtime_error("injected stream failure");
          }};
    });
    server.add("POST", "/stream", [](const auto& request, auto&) {
      auto log = std::make_shared<gufo::server::HttpResponse::StreamLog>();
      return gufo::server::HttpResponse{
          .streaming_body =
              [log, fail = !request.body.empty(),
               reported = request.body == "reported"](const auto& write) {
                (void)write(std::string_view("a\0b", 3));
                (void)write("");
                (void)write("end");
                if (fail)
                  log->error_code = "injected";
                log->error_event_sent = reported;
              },
          .stream_log = log,
      };
    });
    server.add("POST", "/sse-idle", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .headers = {{"Content-Type", "text/event-stream"}},
          .streaming_body =
              [](const auto& write) {
                (void)write("data: first\n\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(90));
                (void)write("data: last\n\n");
                (void)write("data: [DONE]\n\n");
              },
      };
    });
    server.add("POST", "/sse-finish", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{
          .headers = {{"Content-Type", "text/event-stream"}},
          .streaming_body = [fail = !request.body.empty()](const auto& write) {
            (void)write("data: first\n\n");
            if (fail)
              throw std::runtime_error("injected SSE failure");
            (void)write("data: [DONE]\n\n");
          }};
    });
    std::string error;
    assert(server.start(&error));
    worker = std::jthread([this, handle_signals] {
      server.run(handle_signals);
      run_finished.release();
    });
  }
  ~RunningServer() {
    server.stop();
    worker.join();
  }

  int Connect() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    const timeval timeout{3, 0};
    assert(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                        sizeof(timeout)) == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                     sizeof(address)) == 0);
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false) {
    const int fd = Connect();
    while (!request.empty()) {
      const auto count =
          ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      ::shutdown(fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = ::read(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    ::close(fd);
    return response;
  }

  std::string Post(std::string_view path, std::string_view body) {
    return Send("POST " + std::string(path) + " HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  }

  std::shared_ptr<FakeBackend> backend;
  HttpServer server;
  std::binary_semaphore run_finished{0};
  std::jthread worker;
};

void ExpectStatus(const std::string& response, int status) {
  if (!response.starts_with("HTTP/1.1 " + std::to_string(status) + " ")) {
    std::cerr << response << '\n';
    std::abort();
  }
}

void TestAuthorization() {
  RunningServer secured({.api_key = "test-secret"});
  for (const std::string path :
       {"/health", "/ready", "/v1/models", "/v1/chat/completions",
        "/v1/responses", "/v1/messages", "/v1/audio/speech",
        "/v1/audio/transcriptions", "/v1/video/generations", "/echo"}) {
    ExpectStatus(secured.Send("POST " + path + " HTTP/1.1\r\n\r\n"), 401);
  }
  for (const std::string header :
       {"", "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer "
        "test-secret\r\n"}) {
    const auto response =
        secured.Send("GET /v1/models HTTP/1.1\r\n" + header + "\r\n");
    ExpectStatus(response, 401);
    assert(response.find("WWW-Authenticate: Bearer") != std::string::npos);
    assert(response.find("test-secret") == std::string::npos);
  }
  assert(secured.backend->calls == 0);
  ExpectStatus(secured.Send("GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  "
                            "test-secret\r\n\r\n"),
               200);
  ExpectStatus(secured.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"),
               204);
  const std::string body = R"({"prompt":"hi","max_tokens":1})";
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nAuthorization: "
                            "Bearer test-secret\r\n"
                            "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body),
               200);
  assert(secured.backend->calls == 1);
}

struct CapturedLogs {
  std::string text;
  std::string request_id;
};

// Drives one pass of traffic at a fixed verbosity. The process-wide filter is
// shared by every test in this binary, so the previous level is restored.
CapturedLogs CaptureTraffic(LogLevel level) {
  const LogLevel previous = Logger::Level();
  Logger::SetLevel(level);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  CapturedLogs captured;
  {
    RunningServer server;
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
    const auto response = server.Post(
        "/v1/chat/completions?private-query",
        R"({"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true})");
    ExpectStatus(response, 200);
    const auto header = response.find("X-Request-ID: ");
    assert(header != std::string::npos);
    const auto begin = header + std::string("X-Request-ID: ").size();
    captured.request_id =
        response.substr(begin, response.find("\r\n", begin) - begin);

    const auto failed = server.Post("/stream-error", "");
    ExpectStatus(failed, 200);
    assert(failed.find("first chunk") != std::string::npos);
    assert(failed.find("HTTP/1.1", 1) == std::string::npos);

    ExpectStatus(server.Send("GET /not-a-route HTTP/1.1\r\n\r\n"), 404);
  }
  Logger::Info("test", "escaped\n\x1b[31m");
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous);
  captured.text = output.str();
  return captured;
}

void TestRequestLogging() {
  const CapturedLogs info = CaptureTraffic(LogLevel::kInfo);
  const std::string& log = info.text;
  assert(log.find("request=" + info.request_id + " event=received") !=
         std::string::npos);
  assert(log.find("request=" + info.request_id + " event=completed") !=
         std::string::npos);
  assert(log.find("cache=memory") != std::string::npos);
  assert(log.find("acceptance_pct=50.0") != std::string::npos);
  assert(log.find("rss_mib=") != std::string::npos);
  assert(log.find("error_code=server_exception") != std::string::npos);
  // A successful poll is quiet at the default level, and so is the received
  // line for a GET that is not on the inference list.
  assert(log.find("path=/health") == std::string::npos);
  // Statuses never are: 4xx keeps escalating to WARN under any filter.
  assert(log.find("path=/not-a-route status=404") != std::string::npos);
  assert(log.find("[WARN]") != std::string::npos);
  assert(log.find("private-query") == std::string::npos);
  assert(log.find("private-prompt") == std::string::npos);
  assert(log.find("escaped\\x0a\\x1b[31m") != std::string::npos);
  // The startup confirmation is INFO-tier, so the default level reports it.
  // TestQuietTiersSuppressLifecycle covers the tiers that do not.
  assert(log.find("event=listening") != std::string::npos);

  const CapturedLogs debug = CaptureTraffic(LogLevel::kDebug);
  assert(debug.text.find("[DEBUG]") != std::string::npos);
  assert(debug.text.find("event=received method=GET path=/health") !=
         std::string::npos);
  assert(debug.text.find("path=/health status=200") != std::string::npos);
  // Debug adds lines; it must not downgrade an inference completion to DEBUG,
  // nor pull a refused 4xx back under the threshold.
  assert(debug.text.find("[INFO] [http] request=" + debug.request_id +
                         " event=completed") != std::string::npos);
  assert(debug.text.find("path=/not-a-route status=404") != std::string::npos);
  // The higher tier must still hide every byte of the request.
  assert(debug.text.find("private-query") == std::string::npos);
  assert(debug.text.find("private-prompt") == std::string::npos);
  // Options are echoed by `gufo serve`, not by the HTTP layer itself.
  assert(debug.text.find("event=options") == std::string::npos);
}

// The threshold covers the lifecycle lines too: `--log-level=warn|error` boots
// and stops without a word, which docs/SERVER.md states, while an escalation
// keeps its own tier and a filtered INFO receipt line never reaches the log.
void TestQuietTiersSuppressLifecycle() {
  const CapturedLogs warn = CaptureTraffic(LogLevel::kWarn);
  assert(warn.text.find("event=listening") == std::string::npos);
  assert(warn.text.find("event=received") == std::string::npos);
  assert(warn.text.find("path=/not-a-route status=404") != std::string::npos);

  const CapturedLogs error = CaptureTraffic(LogLevel::kError);
  assert(error.text.find("event=listening") == std::string::npos);
  assert(error.text.find("event=received") == std::string::npos);
  assert(error.text.find("path=/not-a-route status=404") == std::string::npos);
}

void TestLogLevelFilter() {
  const LogLevel previous = Logger::Level();
  assert(previous == LogLevel::kInfo);

  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  Logger::SetLevel(LogLevel::kError);
  Logger::Debug("probe", "hidden-debug");
  Logger::Info("probe", "hidden-info");
  Logger::Warn("probe", "hidden-warn");
  Logger::Error("probe", "visible-error");
  // The printf-style entry point the imported DeepSeek runtime uses has to
  // filter before it formats: an INFO loader line stays silent at an absolute
  // threshold while an ERROR line survives.
  Logger::LogFormatted(LogLevel::kInfo, "ds4", "hidden-ds4-info %d", 1);
  Logger::LogFormatted(LogLevel::kError, "ds4", "visible-ds4-error %d", 2);
  Logger::LogRequest("r-quiet", "GET", "/health", 200, 1.0, "", "completed",
                     LogLevel::kDebug);
  Logger::LogRequest("r-404", "GET", "/missing", 404, 1.0);
  Logger::LogRequest("r-500", "POST", "/boom", 500, 1.0);
  Logger::SetLevel(LogLevel::kWarn);
  Logger::Debug("probe", "still-hidden-debug");
  Logger::Warn("probe", "visible-at-warn");
  Logger::LogFormatted(LogLevel::kInfo, "ds4", "still-hidden-ds4-info");
  Logger::LogFormatted(LogLevel::kWarn, "ds4", "visible-ds4-warn");
  Logger::LogRequest("r-404b", "GET", "/missing", 404, 1.0);
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous);

  const std::string log = output.str();
  assert(log.find("hidden-debug") == std::string::npos);
  assert(log.find("hidden-info") == std::string::npos);
  assert(log.find("hidden-warn") == std::string::npos);
  assert(log.find("still-hidden-debug") == std::string::npos);
  assert(log.find("hidden-ds4-info") == std::string::npos);
  assert(log.find("still-hidden-ds4-info") == std::string::npos);
  assert(log.find("visible-error") != std::string::npos);
  assert(log.find("visible-at-warn") != std::string::npos);
  assert(log.find("visible-ds4-error 2") != std::string::npos);
  assert(log.find("visible-ds4-warn") != std::string::npos);
  assert(log.find("[DEBUG]") == std::string::npos);
  // A poll that is quiet by default stays filtered, while an escalation keeps
  // its own tier: `--log-level error` means errors only, not "hide failures".
  assert(log.find("request=r-quiet event=completed") == std::string::npos);
  assert(log.find("request=r-404 event=completed") == std::string::npos);
  assert(log.find("request=r-500 event=completed") != std::string::npos);
  assert(log.find("request=r-404b event=completed") != std::string::npos);
  assert(Logger::Level() == LogLevel::kInfo);
}

void TestLogLevelNames() {
  assert(LogLevelFromName("debug") == LogLevel::kDebug);
  assert(LogLevelFromName("info") == LogLevel::kInfo);
  assert(LogLevelFromName("warn") == LogLevel::kWarn);
  assert(LogLevelFromName("error") == LogLevel::kError);
  assert(!LogLevelFromName("trace").has_value());
  assert(!LogLevelFromName("DEBUG").has_value());
  assert(!LogLevelFromName("").has_value());
  assert(LogLevelName(LogLevel::kDebug) == "debug");
  assert(LogLevelName(LogLevel::kError) == "error");
}

void TestFramingAndMetrics() {
  RunningServer server({.max_request_body_bytes = 8192});
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  for (const std::string header :
       {"Content-Length: nope", "Content-Length: -1", "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3", "Transfer-Encoding: chunked",
        "broken-header"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 400);
  }
  ExpectStatus(server.Send("POST /echo\r\n\r\n"), 400);
  ExpectStatus(server.Send("POST /echo HTTP/1.1 extra\r\n\r\n"), 400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
      400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 8193\r\n\r\n"), 413);
  const std::string payload(5000, 'x');
  const auto echo = server.Send(
      "POST /echo HTTP/1.1\r\nContent-Length: 5000\r\nContent-Length: "
      "5000\r\n\r\n" +
      payload);
  ExpectStatus(echo, 200);
  assert(echo.substr(echo.find("\r\n\r\n") + 4) == payload);

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  const auto response =
      server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body);
  ExpectStatus(response, 200);
  const auto parsed =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  const auto* timings = parsed.find("timings");
  assert(timings != nullptr);
  assert(timings->member_double("prompt_n") == 2);
  assert(timings->member_double("cache_n") == 8);
  assert(timings->member_double("prompt_per_second") == 500);
  assert(timings->member_double("prompt_per_token_ms") == 2);
}

void TestFallbackBackendMetrics() {
  namespace metrics = gufo::server::detail;
  RunningServer server;
  struct Endpoint {
    const char* path;
    const char* body;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})"},
           Endpoint{"/v1/completions", R"({"prompt":"hi"})"},
           Endpoint{"/v1/responses", R"({"input":"hi"})"},
       }) {
    for (const bool stream : {false, true}) {
      const auto prompt_before = metrics::TotalPromptTokens().load();
      const auto generated_before = metrics::TotalGenTokens().load();
      auto body = gufo::json::parse(endpoint.body);
      body["model"] = "test";
      body["stream"] = stream;
      ExpectStatus(server.Post(endpoint.path, body.dump()), 200);
      const auto response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
      ExpectStatus(response, 200);
      // The backend reports 10 prompt tokens, of which 8 were cached.
      assert(response.find("llamacpp:prompt_tokens_total " +
                           std::to_string(prompt_before + 2) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:tokens_predicted_total " +
                           std::to_string(generated_before + 1) + "\n") !=
             std::string::npos);
    }
  }
  struct Case {
    std::size_t prefill, cached;
    bool cancelled;
    std::size_t expected_prompt;
  };
  for (const auto& test : {
           Case{6, 4, false, 6},
           Case{0, 4, false, 6},
           Case{0, 10, false, 0},
           Case{0, 20, false, 0},
           Case{0, 0, true, 0},
           Case{3, 0, true, 3},
       }) {
    const auto prompt_before = metrics::TotalPromptTokens().load();
    const auto generated_before = metrics::TotalGenTokens().load();
    TextGenerationBackend::Result result;
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = test.cached;
    result.prefill_tokens = test.prefill;
    result.completion_tokens = 2;
    result.cancelled = test.cancelled;
    gufo::server::RecordServerMetrics(result);
    assert(metrics::TotalPromptTokens().load() ==
           prompt_before + test.expected_prompt);
    assert(metrics::TotalGenTokens().load() == generated_before + 2);
  }
}

void TestCompatibilityRequests() {
  RunningServer server;
  using gufo::json::parse;
  const auto response_body = [](const std::string& response) {
    ExpectStatus(response, 200);
    return parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  struct Endpoint {
    const char *path, *body, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi"})", "max_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi"})", "max_output_tokens"},
           Endpoint{"/v1/messages",
                    R"({"messages":[{"role":"user","content":"hi"}]})",
                    "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi"})", "n_predict"},
       }) {
    auto body = parse(endpoint.body);
    response_body(server.Post(endpoint.path, body.dump()));
    assert(server.backend->LastCall().max_tokens == 0);
    server.backend->defaults.model = gufo::sampling::TextModelPreset::kQwen38;
    server.backend->defaults.supplied = {};
    server.backend->reasoning.enabled = false;
    response_body(server.Post(endpoint.path, body.dump()));
    const auto preset = server.backend->LastCall().sampling;
    assert(preset.temperature == 0.7F && preset.top_p == 0.8F &&
           preset.top_k == 20 && preset.presence_penalty == 1.5F);
    server.backend->defaults.sampling.top_k = 0;
    server.backend->defaults.supplied.top_k = true;
    body["temperature"] = 0;
    body["presence_penalty"] = 0;
    response_body(server.Post(endpoint.path, body.dump()));
    const auto overridden = server.backend->LastCall().sampling;
    assert(overridden.temperature == 0 && overridden.top_k == 0 &&
           overridden.top_p == 0.8F && overridden.presence_penalty == 0);
    server.backend->defaults = {};
    server.backend->reasoning = {};
    body["model"] = "test";
    body[endpoint.limit] = 1;
    body["temperature"] = 0.6;
    body["top_k"] = 40;
    body["top_p"] = 0.9;
    body["seed"] = 123;
    body["repeat_penalty"] = 1.1;
    const auto output = response_body(server.Post(endpoint.path, body.dump()));
    const auto last = server.backend->LastCall();
    for (int failure : {1, 2}) {
      server.backend->failure = failure;
      const auto rejected = server.Post(endpoint.path, body.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find(failure == 1
                               ? "context_length_exceeded"
                               : "invalid_prompt") != std::string::npos);
    }
    server.backend->failure = 0;
    assert(last.client_id == "127.0.0.1");
    assert(last.max_tokens == 1 && last.sampling.temperature == 0.6F &&
           last.sampling.top_k == 40 && last.sampling.top_p == 0.9F &&
           last.sampling.seed == 123 && last.sampling.repeat_penalty == 1.1F);
    if (std::string_view(endpoint.path) == "/v1/responses") {
      assert(output.member_str("status") == "incomplete");
      assert(output.find("incomplete_details")->member_str("reason") ==
             "max_output_tokens");
      assert(output.find("usage")->contains("input_tokens_details"));
    } else if (std::string_view(endpoint.path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "max_tokens");
    } else if (std::string_view(endpoint.path) == "/completion") {
      assert(output.find("stopped_length")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "length");
    }
    const int calls = server.backend->calls;
    for (const auto& [field, value] :
         {std::pair{"temperature", 2.01}, std::pair{"presence_penalty", 2.01},
          std::pair{"frequency_penalty", -2.01}}) {
      auto invalid = body;
      invalid[field] = value;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto value : {"0", "-1", "1.5", "1e100", "\"1\"", "null"}) {
      auto invalid = body;
      invalid[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto field :
         {"stream", "echo", "store", "background", "tools", "stop", "reasoning",
          "output_config", "logit_bias", "ignore_eos"}) {
      const std::string_view path(endpoint.path);
      const std::string_view name(field);
      if ((path == "/v1/responses" || path == "/v1/completions") &&
          name == "stream")
        continue;
      if (path == "/v1/completions" && name == "ignore_eos")
        continue;
      auto invalid = body;
      invalid[field] = true;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    auto invalid = body;
    invalid["n"] = 1.4;
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    invalid = body;
    invalid["model"] = "wrong";
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 404);
    ExpectStatus(server.Post(endpoint.path, "[]"), 400);
    ExpectStatus(server.Post(endpoint.path, "{"), 400);
    assert(server.backend->calls == calls);
  }
  const int calls = server.backend->calls;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":["one","two"]})"),
               400);
  ExpectStatus(server.Post("/v1/responses",
                           R"({"input":[{"role":"assistant","content":[
                           {"type":"input_text","text":"describe"},
                           {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"tool","content":"hi"}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages", R"({"messages":[]})"), 400);
  ExpectStatus(
      server.Post("/infill", R"({"input_prefix":"one","input_suffix":"two"})"),
      501);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               501);
  assert(server.backend->calls == calls);

  for (const auto& [url, expected] :
       {std::pair{"data:image/gif;base64,AA==", "image/gif"},
        std::pair{"data:image/png,AA==", "base64"},
        std::pair{"data:image/png;base64,A===", "base64"}}) {
    const auto input = parse(
        R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":)" +
        gufo::json::Value(url).dump() +
        R"(},{"type":"input_text","text":"describe"}]}]})");
    const auto rejected = server.Post("/v1/responses", input.dump());
    ExpectStatus(rejected, 400);
    const auto parsed = parse(rejected.substr(rejected.find("\r\n\r\n") + 4));
    const auto& error = *parsed.find("error");
    assert(error.member_str("code") == "invalid_request");
    assert(error.member_str("message").find(expected) != std::string::npos);
  }
  assert(server.backend->calls == calls);

  const auto structured =
      response_body(server.Post("/v1/responses",
                                R"({"input":[{"role":"user","content":[
        {"type":"input_text","text":"describe"},
        {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}],
        "reasoning":{"effort":"none"},
        "text":{"format":{"type":"json_schema","name":"answer","strict":true,
          "schema":{"type":"object","properties":{"score":{"type":"integer","minimum":1,"maximum":5}},
          "required":["score"],"additionalProperties":false}}}})"));
  const auto request = server.backend->LastCall().chat;
  assert(request.response_format && request.reasoning.enabled == false &&
         request.messages.back().images.size() == 1 &&
         request.messages.back().images[0].offset == 8);

  const auto response = response_body(server.Post(
      "/v1/responses",
      R"({"instructions":"Be concise.","input":[{"role":"user","content":[
          {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false})"));
  assert(response.member_str("status") == "completed");
  const auto messages = server.backend->LastCall().chat.messages;
  assert(messages.size() == 2);
  assert(messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         messages[0].content == "Be concise." && messages[1].content == "hi");
  for (const auto limit : {1, 2}) {
    const auto streaming = server.Post(
        "/v1/responses",
        std::string(R"({"input":"hi","stream":true,"max_output_tokens":)") +
            std::to_string(limit) + "}");
    ExpectStatus(streaming, 200);
    assert(streaming.find("text/event-stream") != std::string::npos);
    assert(streaming.find("event: response.created") != std::string::npos);
    assert(streaming.find("event: response.output_text.delta") !=
           std::string::npos);
    assert(streaming.find(limit == 1 ? "event: response.incomplete"
                                     : "event: response.completed") !=
           std::string::npos);
  }
  server.backend->failure = 1;
  const auto failed_stream =
      server.Post("/v1/responses", R"({"input":"hi","stream":true})");
  ExpectStatus(failed_stream, 200);  // Fake backend fails after headers.
  assert(failed_stream.find("event: response.failed") != std::string::npos);
  assert(failed_stream.find("event: response.completed") == std::string::npos);
  const auto failed_data = failed_stream.find(
      "data: ", failed_stream.find("event: response.failed"));
  assert(failed_data != std::string::npos);
  const auto failed_event = gufo::json::parse(
      std::string_view(failed_stream)
          .substr(failed_data + 6,
                  failed_stream.find("\n\n", failed_data) - failed_data - 6));
  assert(failed_event.find("response")->find("error")->member_str("code") ==
         "server_error");
  assert(failed_stream.ends_with("0\r\n\r\n"));
  server.backend->failure = 0;
  const auto continued = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}]},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]},
        {"role":"user","content":"Next question"}]})"));
  assert(continued.member_str("status") == "completed");
  const auto replay_messages = server.backend->LastCall().chat.messages;
  assert(replay_messages.size() == 3 &&
         replay_messages[1].thought == "Thoughts" &&
         replay_messages[1].content == "Answer");

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");
}

void TestModelInputModalities() {
  RunningServer server;
  // Keep the same model ID: capability follows the loaded backend, not its
  // name.
  for (const bool images : {false, true, false}) {
    server.backend->image_support = images;
    const auto response = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 200);
    const auto listing =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(listing.member_str("object") == "list");
    const auto& models = listing.find("data")->items();
    assert(models.size() == 1);
    const auto& model = models[0];
    assert(model.member_str("id") == "test");
    assert(model.member_str("owned_by") == "gufo");
    assert(model.member_size("context_length") == 65536);
    const auto* architecture = model.find("architecture");
    assert(architecture != nullptr);
    const auto* modalities = architecture->find("input_modalities");
    assert(modalities != nullptr);
    assert(modalities->dump() ==
           (images ? R"(["text","image"])" : R"(["text"])"));
  }
}

void TestRawCompletionStreaming() {
  RunningServer server;
  using gufo::json::parse;
  const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
  ExpectStatus(models, 200);
  const auto listing = parse(models.substr(models.find("\r\n\r\n") + 4));
  assert(listing.find("data")->items()[0].member_size("context_length") ==
         65536);

  const auto response = server.Post(
      "/v1/completions",
      R"({"prompt":"hello","max_tokens":256,"stream":true,"stream_options":{"include_usage":true},"ignore_eos":true})");
  ExpectStatus(response, 200);
  assert(response.find("text/event-stream") != std::string::npos);
  std::vector<gufo::json::Value> events;
  std::size_t offset = 0;
  while ((offset = response.find("data: ", offset)) != std::string::npos) {
    const auto begin = offset + 6;
    const auto end = response.find("\n\n", begin);
    assert(end != std::string::npos);
    const auto data = std::string_view(response).substr(begin, end - begin);
    if (data != "[DONE]")
      events.push_back(parse(data));
    offset = end + 2;
  }
  assert(events.size() == 3);
  const auto& content = events[0];
  assert(content.member_str("object") == "text_completion");
  assert(content.find("choices")->items().size() == 1);
  assert(content.find("choices")->items()[0].member_str("text") == "ok");
  assert(content.find("choices")->items()[0].find("finish_reason")->is_null());
  assert(content.find("usage") == nullptr);
  const auto& terminal = events[1];
  assert(terminal.find("choices")->items().size() == 1);
  assert(terminal.find("choices")->items()[0].member_str("text").empty());
  assert(terminal.find("choices")->items()[0].member_str("finish_reason") ==
         "stop");
  assert(terminal.find("usage") == nullptr);
  const auto* timings = terminal.find("timings");
  assert(timings != nullptr);
  assert(timings->member_size("prompt_n") == 2);
  assert(timings->member_double("prompt_ms") == 4);
  assert(timings->member_size("predicted_n") == 1);
  assert(timings->member_size("cache_n") == 8);
  const auto& usage = events[2];
  assert(usage.find("choices")->items().empty());
  assert(usage.find("usage")->member_size("completion_tokens") == 1);
  assert(usage.find("usage")->member_size("cached_tokens") == 8);
  assert(response.find("data: [DONE]\n\n") != std::string::npos);
  assert(server.backend->last_ignore_eos);
  assert(server.backend->LastCall().max_tokens == 256);

  const auto without_usage =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(without_usage, 200);
  assert(without_usage.find("\"timings\":") != std::string::npos);
  assert(without_usage.find("\"usage\":") == std::string::npos);

  server.backend->failure = 1;
  const auto failed = server.Post(
      "/v1/completions",
      R"({"prompt":"hello","stream":true,"stream_options":{"include_usage":true}})");
  ExpectStatus(failed, 200);
  assert(failed.find("\"code\":\"generation_failed\"") != std::string::npos);
  assert(failed.find("\"message\":\"generation failed\"") != std::string::npos);
  assert(failed.find("context exceeded") == std::string::npos);
  assert(failed.find("data: [DONE]\n\n") != std::string::npos);
  server.backend->failure = 0;

  for (
      const auto* body : {
          R"({"prompt":"hello","stream_options":{"include_usage":true}})",
          R"({"prompt":"hello","stream":true,"stream_options":{"include_usage":"true"}})",
          R"({"prompt":"hello","stream":true,"stream_options":{"other":true}})",
          R"({"prompt":"hello","ignore_eos":1})",
          R"({"prompt":"hello","stream":true,"text":{"format":{"type":"json_object"}}})",
          R"({"prompt":"hello","stream":true,"reasoning":{"effort":"none"}})",
      })
    ExpectStatus(server.Post("/v1/completions", body), 400);

  // Raw Completions owns the fixed-length contract; every other text endpoint
  // rejects the field instead of silently generating a shorter run.
  ExpectStatus(
      server.Post(
          "/v1/chat/completions",
          R"({"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"ignore_eos":true})"),
      400);
}

void TestInvalidBindSettings() {
  for (const int port : {-1, 65536}) {
    HttpServer server("127.0.0.1", port, nullptr);
    std::string error;
    assert(!server.start(&error));
    assert(!error.empty());
  }
  HttpServer invalid("bad.address", 0, nullptr);
  std::string error;
  assert(!invalid.start(&error));
  assert(!error.empty());
}

void TestCompatibilityUtf8() {
  RunningServer server;
  server.backend->SetOutput("é中😀\xE2\x94!\xF0\x9F");
  const std::string expected = "é中😀\xEF\xBF\xBD!\xEF\xBF\xBD";
  for (const auto& [path, input] : {
           std::pair{"/v1/completions", R"({"prompt":"hi"})"},
           std::pair{"/v1/responses", R"({"input":"hi"})"},
           std::pair{"/v1/messages",
                     R"({"messages":[{"role":"user","content":"hi"}]})"},
           std::pair{"/completion", R"({"prompt":"hi"})"},
       }) {
    const auto response = server.Post(path, input);
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    std::string text;
    if (std::string_view(path) == "/v1/completions")
      text = output.find("choices")->items()[0].member_str("text");
    else if (std::string_view(path) == "/v1/responses")
      text = output.find("output")
                 ->items()[0]
                 .find("content")
                 ->items()[0]
                 .member_str("text");
    else if (std::string_view(path) == "/v1/messages")
      text = output.find("content")->items()[0].member_str("text");
    else
      text = output.member_str("content");
    assert(text == expected);
  }
}

void TestQueryParameters() {
  gufo::server::HttpRequest request;
  request.query = "notafter=wrong&note=after=wrong&after=right+value%26x";
  assert(request.query_param("after") == "right value&x");
  request.query = "notafter=wrong&note=after=wrong";
  assert(request.query_param("after").empty());
  request.query = "%61fter=encoded&broken=%xz&empty";
  assert(request.query_param("after") == "encoded");
  assert(request.query_param("broken") == "%xz");
  assert(request.query_param("empty").empty());
}

void TestPeerDisconnect() {
  RunningServer server;
  server.backend->wait_for_disconnect = true;
  const int fd = server.Connect();
  const std::string body = R"({"prompt":"hi","max_tokens":128})";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
  ::close(fd);
  assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
  assert(server.backend->disconnected);
}

void TestCompatibilityStopSequences() {
  RunningServer server;
  server.backend->forced_stop_sequence = "END";
  for (
      const auto& [path, body] : {
          std::pair{"/v1/completions", R"({"prompt":"hello","stop":"END"})"},
          std::pair{"/completion", R"({"prompt":"hello","stop":["END"]})"},
          std::pair{
              "/v1/messages",
              R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"]})"},
      }) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    assert(server.backend->LastCall().stop_sequences ==
           std::vector<std::string>{"END"});
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    if (std::string_view(path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "stop_sequence");
      assert(output.member_str("stop_sequence") == "END");
    } else if (std::string_view(path) == "/completion") {
      assert(output.find("stopped_word")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
      assert(output.member_str("stopping_word") == "END");
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "stop");
    }
  }
  const auto before = server.backend->calls.load();
  for (const auto* stop : {"null", "\"END\"", "[null]", "[\"\"]", "true"}) {
    auto body = gufo::json::parse(
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})");
    body["stop_sequences"] = gufo::json::parse(stop);
    ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  }
  ExpectStatus(server.Post("/v1/responses", R"({"input":"hi","stop":"END"})"),
               400);
  assert(server.backend->calls == before);
  auto body = gufo::json::parse(
      R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":[]})");
  for (int i = 0; i < 65; ++i)
    body["stop_sequences"].push_back(std::to_string(i));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  body["stop_sequences"] = gufo::json::Value::array();
  for (int i = 0; i < 5; ++i)
    body["stop_sequences"].push_back(std::string(4096, 'x'));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  server.backend->forced_stop_sequence.clear();
  for (const int limit : {1, 32}) {
    body["stop_sequences"] = gufo::json::Value::array();
    body["max_tokens"] = limit;
    const auto response = server.Post("/v1/messages", body.dump());
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(output.member_str("stop_reason") ==
           (limit == 1 ? "max_tokens" : "end_turn"));
    assert(output.find("stop_sequence")->is_null());
  }
}

void TestCompatibilityThinkingDefaults() {
  RunningServer server;
  for (const bool enabled : {false, true}) {
    server.backend->reasoning = {
        .enabled = enabled,
        .effort = gufo::ReasoningEffort::kHigh,
        .preserve_thinking = true,
    };
    for (
        const auto& [path, body] : {
            std::pair{"/v1/responses", R"({"input":"hello"})"},
            std::pair{
                "/v1/messages",
                R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})"},
        }) {
      ExpectStatus(server.Post(path, body), 200);
      const auto reasoning = server.backend->LastCall().chat.reasoning;
      assert(reasoning.enabled == enabled);
      assert(reasoning.effort == gufo::ReasoningEffort::kHigh);
      assert(reasoning.preserve_thinking == true);
    }
  }
}

void TestResponseSamplingDefaults() {
  RunningServer server;
  using gufo::sampling::TextModelPreset;
  for (const auto model :
       {TextModelPreset::kQwen38, TextModelPreset::kDeepSeekV4Flash}) {
    server.backend->defaults.model = model;
    for (const bool server_thinking : {false, true}) {
      server.backend->reasoning.enabled = server_thinking;
      for (const char* effort :
           {"null", "\"none\"", "\"minimal\"", "\"low\"", "\"medium\"",
            "\"high\"", "\"xhigh\"", "\"max\""}) {
        auto body = gufo::json::parse(
            R"({"input":"hello","max_output_tokens":1,"temperature":null,"top_p":null,"presence_penalty":null,"reasoning":{},"text":{"format":{"type":"json_object"}}})");
        body["reasoning"]["effort"] = gufo::json::parse(effort);
        const bool thinking = std::string_view(effort) == "null"
                                  ? server_thinking
                                  : std::string_view(effort) != "\"none\"";
        const bool qwen_off = model == TextModelPreset::kQwen38 && !thinking;
        server.backend->defaults.supplied = {};
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        auto call = server.backend->LastCall();
        assert(call.chat.reasoning.enabled == thinking);
        assert(call.chat.response_format);
        assert(call.sampling.temperature == (qwen_off ? 0.7F : 1.0F));
        assert(call.sampling.top_p == (qwen_off ? 0.8F : 0.95F));
        assert(call.sampling.presence_penalty == (qwen_off ? 1.5F : 0.0F));
        assert(call.sampling.top_k ==
               (model == TextModelPreset::kQwen38 ? 20 : 0));
        // Explicit CLI values survive reasoning changes and SDK nulls.
        server.backend->defaults.sampling.temperature = 0.25F;
        server.backend->defaults.sampling.top_k = 0;
        server.backend->defaults.supplied = {.temperature = true,
                                             .top_k = true};
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        call = server.backend->LastCall();
        assert(call.sampling.temperature == 0.25F && call.sampling.top_k == 0);
        assert(call.sampling.top_p == (qwen_off ? 0.8F : 0.95F));
        // Per-request zero overrides both CLI and model values, also streamed.
        body["temperature"] = 0;
        body["presence_penalty"] = 0;
        body["top_k"] = 3;
        body["stream"] = true;
        ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
        call = server.backend->LastCall();
        assert(call.sampling.temperature == 0 && call.sampling.top_k == 3);
        assert(call.sampling.presence_penalty == 0);
      }
    }
  }
  // Responses must preserve the same request-owned cache control as Chat.
  auto body = gufo::json::parse(R"({"input":"hello","max_output_tokens":1})");
  ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
  assert(server.backend->LastCall().chat.cache_prompt);
  for (const bool stream : {false, true}) {
    body["stream"] = stream;
    for (const bool cache : {false, true}) {
      body["cache_prompt"] = cache;
      ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
      assert(server.backend->LastCall().chat.cache_prompt == cache);
    }
  }
  for (const char* invalid : {"null", "0", "\"false\""}) {
    body["cache_prompt"] = gufo::json::parse(invalid);
    const int calls = server.backend->calls;
    ExpectStatus(server.Post("/v1/responses", body.dump()), 400);
    assert(server.backend->calls == calls);
  }
}

void TestStreamingFraming() {
  RunningServer server;
  const std::string chunks = std::string("3\r\na\0b\r\n", 8) + "3\r\nend\r\n";
  for (const std::string body : {"", "fail", "reported"}) {
    const auto response = server.Post("/stream", body);
    ExpectStatus(response, 200);
    assert(response.find("Transfer-Encoding: chunked\r\n") !=
           std::string::npos);
    assert(response.substr(response.find("\r\n\r\n") + 4) ==
           chunks + (body == "fail" ? "" : "0\r\n\r\n"));
  }
  const auto thrown = server.Post("/stream-error", "");
  assert(thrown.substr(thrown.find("\r\n\r\n") + 4) == "b\r\nfirst chunk\r\n");
  // Existing HTTP/1.0 clients retain close-delimited framing.
  const auto legacy = server.Send("POST /stream HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.substr(legacy.find("\r\n\r\n") + 4) ==
         std::string("a\0bend", 6));
}

void TestSseHeartbeat() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(20)});
  const auto response = server.Post("/sse-idle", "");
  ExpectStatus(response, 200);
  const auto first = response.find("data: first\n\n");
  const auto ping = response.find(": ping\n\n", first);
  const auto second_ping = response.find(": ping\n\n", ping + 1);
  const auto last = response.find("data: last\n\n", second_ping);
  assert(first != std::string::npos);
  assert(ping != std::string::npos);
  assert(second_ping != std::string::npos);
  assert(last != std::string::npos);
  assert(response.find("data: [DONE]\n\n", last) != std::string::npos);
  assert(response.ends_with("0\r\n\r\n"));

  const auto legacy = server.Send("POST /sse-idle HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.find(": ping\n\n") != std::string::npos);
  assert(legacy.ends_with("data: [DONE]\n\n"));

  RunningServer disabled({.sse_heartbeat_interval = {}});
  assert(disabled.Post("/sse-idle", "").find(": ping") == std::string::npos);
}

void TestSseHeartbeatShutdown() {
  // Quick completion and exceptions can request stop while the heartbeat
  // thread is entering its wait. Cleanup must not wait for this deadline:
  // Send() has a three-second socket timeout.
  RunningServer server({.sse_heartbeat_interval = std::chrono::seconds(30)});
  std::vector<std::jthread> clients;
  for (int client = 0; client < 8; ++client) {
    clients.emplace_back([&] {
      for (int request = 0; request < 16; ++request) {
        const bool fail = request % 2 != 0;
        const auto response = server.Post("/sse-finish", fail ? "fail" : "");
        ExpectStatus(response, 200);
        assert(response.find("data: first\n\n") != std::string::npos);
        assert(response.find(": ping") == std::string::npos);
        assert(response.ends_with("0\r\n\r\n") == !fail);
        assert((response.find("data: [DONE]\n\n") != std::string::npos) ==
               !fail);
      }
    });
  }
}

void TestSignalShutdown() {
  // Process signals must never terminate the test runner itself. Prove that
  // both idle listeners and active generation return through normal cleanup.
  for (const int signal : {SIGINT, SIGTERM}) {
    for (const bool active : {false, true}) {
      const pid_t child = ::fork();
      assert(child >= 0);
      if (child == 0) {
        ::alarm(5);
        {
          RunningServer server({}, true);
          // Accepting a request proves run() installed its handlers.
          ExpectStatus(server.Post("/echo", "ready"), 200);
          int fd = -1;
          if (active) {
            server.backend->wait_for_disconnect = true;
            fd = server.Connect();
            const std::string body = R"({"prompt":"hello"})";
            const std::string request =
                "POST /v1/completions HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body;
            assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(request.size()));
            assert(server.backend->entered.try_acquire_for(
                std::chrono::seconds(2)));
          }
          assert(::kill(::getpid(), signal) == 0);
          assert(server.run_finished.try_acquire_for(std::chrono::seconds(2)));
          if (active) {
            assert(server.backend->disconnected);
            ::close(fd);
          }
        }
        ::_exit(0);
      }
      int status = 0;
      assert(::waitpid(child, &status, 0) == child);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
  }
}

}  // namespace

void TestRawCompletionPromptProgress() {
  RunningServer server;
  server.backend->progress = {
      {.total = 4, .cache = 1, .processed = 4, .time_ms = 2}};
  const auto response =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":true})");
  ExpectStatus(response, 200);
  const auto progress = response.find(
      R"("choices":[{"text":"","index":0,"logprobs":null,"finish_reason":null}],)"
      R"("prompt_progress":{"total":4,"cache":1,"processed":4,"time_ms":2}})");
  assert(progress != std::string::npos);
  assert(progress < response.find(R"("text":"ok")"));

  const auto plain =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(plain, 200);
  assert(plain.find("prompt_progress") == std::string::npos);
  const auto null_progress =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":null})");
  ExpectStatus(null_progress, 200);
  assert(null_progress.find("prompt_progress") == std::string::npos);
  const auto buffered = server.Post(
      "/v1/completions", R"({"prompt":"hello","return_progress":true})");
  ExpectStatus(buffered, 200);
  assert(buffered.find("prompt_progress") == std::string::npos);
  ExpectStatus(server.Post("/v1/completions",
                           R"({"prompt":"hello","return_progress":1})"),
               400);
}

int main() {
  // The log assertions below match "[LEVEL] [component]" text written to a
  // redirected stderr sink, so the real stderr's TTY state must not add ANSI
  // tint around the level tag.
  ::setenv("NO_COLOR", "1", 1);
  TestRequestLogging();
  TestQuietTiersSuppressLifecycle();
  TestLogLevelFilter();
  TestLogLevelNames();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestFramingAndMetrics();
  TestFallbackBackendMetrics();
  TestCompatibilityRequests();
  TestModelInputModalities();
  TestRawCompletionStreaming();
  TestRawCompletionPromptProgress();
  TestCompatibilityStopSequences();
  TestCompatibilityThinkingDefaults();
  TestResponseSamplingDefaults();
  TestCompatibilityUtf8();
  TestPeerDisconnect();
  TestStreamingFraming();
  TestSseHeartbeat();
  TestSseHeartbeatShutdown();
  TestSignalShutdown();
  std::cout << "HTTP transport checks passed.\n";
}

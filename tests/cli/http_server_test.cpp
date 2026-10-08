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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/trace.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::Logger;
using gufo::server::LogLevel;
using gufo::server::LogLevelFromName;
using gufo::server::LogLevelName;
using gufo::server::TextGenerationBackend;

/// Reports an optional admission and fixed prompt progress before delegating
/// token generation.
class ProgressRequest final : public TextGenerationBackend::GenerationRequest {
public:
  ProgressRequest(std::shared_ptr<GenerationRequest> inner,
                  std::vector<TextGenerationBackend::PromptProgress> progress,
                  bool admit = false)
      : inner_(std::move(inner)),
        progress_(std::move(progress)),
        admit_(admit) {}

  TextGenerationBackend::Result Wait(
      const TextGenerationBackend::TokenCallback& on_token,
      const TextGenerationBackend::ProgressCallback& on_progress,
      const TextGenerationBackend::StartCallback& on_start) override {
    if (admit_ && on_start && !on_start())
      inner_->Cancel();
    for (const auto& value : progress_) {
      if (on_progress && !on_progress(value)) {
        inner_->Cancel();
        break;
      }
    }
    return inner_->Wait(on_token, on_progress, on_start);
  }
  void Cancel() noexcept override { inner_->Cancel(); }

private:
  std::shared_ptr<GenerationRequest> inner_;
  std::vector<TextGenerationBackend::PromptProgress> progress_;
  bool admit_;
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
    std::string trace_request;
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
  bool device_lost() const override { return lost.load(); }
  bool supports_images() const override { return image_support.load(); }
  std::atomic<bool> image_support{false};
  std::uint32_t max_context() const override { return 65536; }
  std::vector<SessionState> session_states() const override { return sessions; }
  std::vector<SessionState> sessions;
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
        progress, admit.load());
  }
  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancellation, bool stream) override {
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                          cancellation, stream),
        std::vector<PromptProgress>{}, admit.load());
  }
  SamplingDefaults sampling_defaults() const override { return defaults; }
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning;
  }
  InitialOutputState initial_output_state(
      const gufo::server::ChatRequest& request) const override {
    return initial_output_state_override.value_or(
        TextGenerationBackend::initial_output_state(request));
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
    // A failed probe: the scheduler reports device loss from then on.
    if (failure == 3) {
      lost = true;
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kDeviceLost,
          gufo::server::kDeviceLostMessage);
    }
    // A failure on a device that still accepts work.
    if (failure == 4)
      throw std::runtime_error("unspecified launch failure");
    if (failure == 5)
      throw std::runtime_error("");
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
               .stop_sequences = stop_sequences,
               .trace_request = gufo::server::Trace::CurrentRequest()};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_rounds = 2;
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
    if (token && failure != 8) {
      std::this_thread::sleep_for(token_delay);
      (void)token(result.text);
    }
    if (failure == 6)
      throw std::runtime_error("context exceeded");
    if (failure == 7) {
      lost = true;
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kDeviceLost,
          gufo::server::kDeviceLostMessage);
    }
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
  // Report a scheduler admission before generation, as the HIP backend does.
  std::atomic<bool> admit{false};
  std::chrono::milliseconds token_delay{0};
  std::atomic<int> failure{0};
  std::atomic<bool> lost{false};
  std::string forced_stop_sequence;
  gufo::ReasoningOptions reasoning;
  std::optional<InitialOutputState> initial_output_state_override;
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

std::string ResponseRequestId(const std::string& response) {
  const auto header = response.find("X-Request-ID: ");
  assert(header != std::string::npos);
  const auto begin = header + std::string("X-Request-ID: ").size();
  return response.substr(begin, response.find("\r\n", begin) - begin);
}

// The body of a chunked HTTP/1.1 response, as the client reassembles it.
std::string ChunkedBody(const std::string& response) {
  std::size_t cursor = response.find("\r\n\r\n");
  assert(cursor != std::string::npos);
  cursor += 4;
  std::string body;
  for (;;) {
    const auto line_end = response.find("\r\n", cursor);
    assert(line_end != std::string::npos);
    const auto size =
        std::stoul(response.substr(cursor, line_end - cursor), nullptr, 16);
    if (size == 0) {
      return body;
    }
    body += response.substr(line_end + 2, size);
    cursor = line_end + 2 + size + 2;
  }
}

// `--trace` keeps what a text request carried in and out under the id the
// logs name. Polls, other routes and unauthenticated requests stay out.
void TestContentTrace() {
  namespace fs = std::filesystem;
  using gufo::server::Trace;
  const auto path =
      fs::temp_directory_path() /
      ("gufo-trace-test-" + std::to_string(::getpid()) + ".jsonl");
  fs::remove(path);
  assert(!Trace::Open(path.string()).has_value());
  std::string plain_id;
  std::string stream_id;
  std::string stream_wire;
  std::string invalid_id;
  {
    RunningServer server(
        {.api_key = "test-secret",
         .sse_heartbeat_interval = std::chrono::milliseconds(5)});
    server.backend->SetOutput("<tool_call>leak");
    const auto post = [&](std::string_view route, std::string_view body,
                          bool authorized = true) {
      return server.Send(
          "POST " + std::string(route) + " HTTP/1.1\r\n" +
          (authorized ? "Authorization: Bearer test-secret\r\n" : "") +
          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" +
          std::string(body));
    };
    const auto plain = post("/v1/completions",
                            R"({"prompt":"private-prompt","max_tokens":8})");
    ExpectStatus(plain, 200);
    plain_id = ResponseRequestId(plain);
    // The id is bound where the backend generates, so the scheduler's record
    // can name it.
    assert(server.backend->LastCall().trace_request == plain_id);
    // Admission commits the stream before a slow first token, so keepalives
    // reach the client and must reach the trace too.
    server.backend->admit = true;
    server.backend->token_delay = std::chrono::milliseconds(40);
    const auto streamed =
        post("/v1/completions",
             R"({"prompt":"streamed-prompt","max_tokens":8,"stream":true})");
    ExpectStatus(streamed, 200);
    stream_id = ResponseRequestId(streamed);
    stream_wire = ChunkedBody(streamed);
    assert(stream_wire.find(": ping\n\n") != std::string::npos);
    assert(server.backend->LastCall().trace_request == stream_id);
    const auto invalid = post("/v1/completions", "{");
    ExpectStatus(invalid, 400);
    invalid_id = ResponseRequestId(invalid);
    ExpectStatus(
        post("/v1/completions", R"({"prompt":"unauthorized-prompt"})", false),
        401);
    ExpectStatus(post("/echo", "echo-body"), 200);
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\nAuthorization: Bearer "
                             "test-secret\r\n\r\n"),
                 200);
  }
  Trace::Close();

  assert((fs::status(path).permissions() & fs::perms::all) ==
         (fs::perms::owner_read | fs::perms::owner_write));
  std::vector<gufo::json::Value> records;
  {
    std::ifstream input(path);
    for (std::string line; std::getline(input, line);) {
      assert(line.find("unauthorized-prompt") == std::string::npos);
      assert(line.find("echo-body") == std::string::npos);
      records.push_back(gufo::json::parse(line));
    }
  }
  fs::remove(path);
  assert(records.size() == 6);
  const auto record = [&](std::string_view event,
                          const std::string& id) -> const gufo::json::Value& {
    for (const auto& candidate : records) {
      if (candidate.member_str("event") == event &&
          candidate.member_str("request") == id) {
        assert(!candidate.member_str("time").empty());
        return candidate;
      }
    }
    std::abort();
  };

  const auto& request = record("request", plain_id);
  assert(request.member_str("method") == "POST");
  assert(request.member_str("path") == "/v1/completions");
  assert(request.member_str("body") ==
         R"({"prompt":"private-prompt","max_tokens":8})");
  const auto& reply = record("response", plain_id);
  assert(reply.member_size("status") == 200);
  assert(reply.member_str("outcome") == "completed");
  assert(!reply.find("stream")->as_bool());
  const auto body = gufo::json::parse(reply.member_str("body"));
  assert(body.find("choices")->items()[0].member_str("text") ==
         "<tool_call>leak");

  assert(
      record("request", stream_id).member_str("body").find("streamed-prompt") !=
      std::string::npos);
  const auto& streamed = record("response", stream_id);
  assert(streamed.find("stream")->as_bool());
  assert(streamed.member_str("body") == stream_wire);
  assert(stream_wire.find("<tool_call>leak") != std::string::npos);

  assert(record("request", invalid_id).member_str("body") == "{");
  assert(record("response", invalid_id).member_size("status") == 400);
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
      const auto cached_before = metrics::TotalCachedPromptTokens().load();
      const auto rounds_before = metrics::TotalDraftRounds().load();
      const auto drafts_before = metrics::TotalDraftTokens().load();
      const auto accepted_before = metrics::TotalDraftAcceptedTokens().load();
      const auto prompt_seconds_before = metrics::TotalPromptSeconds().load();
      const auto gen_seconds_before = metrics::TotalGenSeconds().load();
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
      assert(response.find("llamacpp:prompt_tokens_cached_total " +
                           std::to_string(cached_before + 8) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_drafts_total " +
                           std::to_string(rounds_before + 2) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_draft_tokens_total " +
                           std::to_string(drafts_before + 8) + "\n") !=
             std::string::npos);
      assert(response.find("llamacpp:spec_decode_num_accepted_tokens_total " +
                           std::to_string(accepted_before + 4) + "\n") !=
             std::string::npos);
      assert(metrics::TotalPromptSeconds().load() - prompt_seconds_before >
                 0.0039 &&
             metrics::TotalGenSeconds().load() - gen_seconds_before > 0.0019);
      assert(metrics::MaxSequenceTokens().load() >= 11);
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

void TestLlamaSlotsAndMetrics() {
  RunningServer server;
  const auto get = [&](const std::string& path) {
    const auto response = server.Send("GET " + path + " HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 200);
    return response.substr(response.find("\r\n\r\n") + 4);
  };
  const auto expect_slot = [](const gufo::json::Value& slot, std::size_t id,
                              bool processing) {
    assert(slot.member_size("id", 99) == id);
    assert(slot.member_size("n_ctx") == 65536);
    assert(slot.find("speculative")->is_bool());
    assert(slot.find("is_processing")->as_bool(!processing) == processing);
    assert(slot.find("state")->as_double() == (processing ? 1 : 0));
    assert(slot.find("task_id")->as_double() ==
           slot.find("id_task")->as_double());
    for (const char* key : {"n_prompt_tokens", "n_prompt_tokens_cache",
                            "n_prompt_tokens_processed"}) {
      assert(slot.find(key)->is_number());
    }
    assert(slot.find("prompt")->is_string() &&
           slot.member_str("prompt").empty());
    assert(slot.member_str("model") == "test");
    const auto* next = slot.find("next_token");
    assert(next != nullptr && next->is_array() && next->size() == 1);
    const auto& token = next->items().front();
    assert(token.find("has_next_token")->as_bool(!processing) == processing);
    assert(token.find("has_new_line")->is_bool());
    assert(token.find("n_remain")->is_number() &&
           token.find("n_decoded")->is_number());
  };

  // Backends without a session pool report one idle slot.
  auto slots = gufo::json::parse(get("/slots"));
  assert(slots.is_array() && slots.size() == 1);
  const auto& fallback = slots.items().front();
  expect_slot(fallback, 0, false);
  assert(fallback.find("id_task")->as_double() == -1);
  assert(fallback.member_size("n_prompt_tokens", 99) == 0);
  assert(fallback.find("next_token")
             ->items()
             .front()
             .find("n_remain")
             ->as_double() == -1);
  assert(get("/metrics").find("llamacpp:kv_cache_usage_ratio 0\n") !=
         std::string::npos);

  server.backend->sessions.resize(2);
  server.backend->sessions[1] = {
      .processing = true,
      .speculative = true,
      .request_id = 42,
      .prompt_tokens = 10,
      .cached_prompt_tokens = 8,
      .processed_prompt_tokens = 2,
      .generated_tokens = 3,
      .remaining_tokens = 5,
  };
  slots = gufo::json::parse(get("/v1/slots"));
  assert(slots.size() == 2);
  expect_slot(slots.items()[0], 0, false);
  const auto& busy = slots.items()[1];
  expect_slot(busy, 1, true);
  assert(busy.find("speculative")->as_bool() &&
         busy.member_size("id_task") == 42 &&
         busy.member_size("n_prompt_tokens") == 10 &&
         busy.member_size("n_prompt_tokens_cache") == 8 &&
         busy.member_size("n_prompt_tokens_processed") == 2);
  const auto& token = busy.find("next_token")->items().front();
  assert(token.member_size("n_remain") == 5 &&
         token.member_size("n_decoded") == 3);

  const auto metrics = get("/metrics");
  assert(metrics.find("# HELP gufo_device_lost_total ") != std::string::npos);
  assert(metrics.find("# TYPE gufo_device_lost_total counter\n") !=
         std::string::npos);
  assert(metrics.find("gufo_device_lost_total 0\n") != std::string::npos);
  const std::string ratio_prefix = "llamacpp:kv_cache_usage_ratio ";
  const auto ratio_pos = metrics.find("\n" + ratio_prefix);
  assert(ratio_pos != std::string::npos);
  assert(std::stod(metrics.substr(ratio_pos + 1 + ratio_prefix.size())) ==
         13.0 / (2.0 * 65536));
  for (const std::string counter :
       {"prompt_tokens_total", "prompt_tokens_cached_total",
        "prompt_seconds_total", "tokens_predicted_total",
        "tokens_predicted_seconds_total", "n_tokens_max",
        "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total"}) {
    assert(metrics.find("# HELP llamacpp:" + counter + " ") !=
           std::string::npos);
    assert(metrics.find("# TYPE llamacpp:" + counter + " counter\n") !=
           std::string::npos);
  }
  assert(metrics.find("# TYPE llamacpp:kv_cache_usage_ratio gauge\n") !=
         std::string::npos);
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
      if ((path == "/v1/responses" || path == "/v1/completions" ||
           path == "/v1/messages") &&
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
  server.backend->failure = 6;
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

  // Codex replays reasoning items with the null content and encrypted_content
  // fields it serialized from the response; only non-null payload is rejected.
  const auto null_replay = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}],
         "content":null,"encrypted_content":null},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]}]})"));
  assert(null_replay.member_str("status") == "completed");
  const auto null_replay_messages = server.backend->LastCall().chat.messages;
  assert(null_replay_messages.size() == 2 &&
         null_replay_messages[1].thought == "Thoughts" &&
         null_replay_messages[1].content == "Answer");
  const auto opaque_replay =
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"question"},
        {"type":"reasoning","id":"rs_1",
         "summary":[],"encrypted_content":"gAAAA"}]})");
  ExpectStatus(opaque_replay, 400);

  // Codex re-sends developer items mid-conversation. They stay in place, so
  // the prompt before them is unchanged and reusable.
  const auto in_place =
      response_body(server.Post("/v1/responses",
                                R"({"instructions":"Base rules.","input":[
        {"role":"user","content":"first"},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"noted","annotations":[]}]},
        {"type":"message","role":"developer","content":[
          {"type":"input_text","text":"Compacted state"}]},
        {"role":"user","content":"second"}]})"));
  assert(in_place.member_str("status") == "completed");
  const auto in_place_messages = server.backend->LastCall().chat.messages;
  assert(in_place_messages.size() == 5 &&
         in_place_messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         in_place_messages[0].content == "Base rules." &&
         in_place_messages[1].content == "first" &&
         in_place_messages[2].content == "noted" &&
         in_place_messages[3].role ==
             gufo::tokenization::ChatRole::kDeveloper &&
         in_place_messages[3].content == "Compacted state" &&
         in_place_messages[4].content == "second");

  // A developer item inside a replayed reasoning/function_call group waits
  // for the group to end instead of splitting the assistant turn.
  const auto replay_items = gufo::json::parse(R"([
      {"role":"user","content":"Check state."},
      {"type":"reasoning","summary":[
        {"type":"summary_text","text":"I will inspect."}]},
      {"type":"function_call","call_id":"call_1","name":"lookup",
       "arguments":"{\"key\":\"state\"}"},
      {"type":"function_call_output","call_id":"call_1","output":"OK"}])");
  const auto developer =
      gufo::json::parse(R"({"role":"developer","content":"Follow policy."})");
  for (const std::size_t position : {2U, 3U}) {
    auto input = gufo::json::Value::array();
    for (std::size_t i = 0; i < replay_items.size(); ++i) {
      if (i == position)
        input.push_back(developer);
      input.push_back(replay_items.items()[i]);
    }
    auto body = gufo::json::Value::object();
    body["input"] = std::move(input);
    const auto replay =
        response_body(server.Post("/v1/responses", body.dump()));
    assert(replay.member_str("status") == "completed");
    const auto grouped = server.backend->LastCall().chat.messages;
    assert(grouped.size() == 4);
    assert(grouped[0].role == gufo::tokenization::ChatRole::kUser &&
           grouped[1].role == gufo::tokenization::ChatRole::kAssistant &&
           grouped[1].content.empty() &&
           grouped[1].thought == "I will inspect." &&
           grouped[1].tool_calls.size() == 1 &&
           grouped[1].tool_calls[0].id == "call_1" &&
           grouped[2].role == gufo::tokenization::ChatRole::kDeveloper &&
           grouped[2].content == "Follow policy." &&
           grouped[3].role == gufo::tokenization::ChatRole::kTool &&
           grouped[3].tool_call_id == "call_1" && grouped[3].content == "OK");
  }

  // The Responses API carries request-only fields with no native effect here
  // (hosted tool types, include, reasoning.summary, text.verbosity). Accept
  // them and keep every executable function tool, flattening the client-side
  // namespace grouping. Codex is one such client.
  const auto hosted = response_body(server.Post(
      "/v1/responses",
      R"({"model":"test","instructions":"You are a coding agent.","input":[
          {"type":"message","role":"developer","content":[
            {"type":"input_text","text":"AGENTS instructions"}]},
          {"type":"message","role":"user","content":[
            {"type":"input_text","text":"say hi"}]}],
        "reasoning":{"effort":"medium","summary":"auto"},
        "text":{"verbosity":"low"},
        "tool_choice":"auto","parallel_tool_calls":true,
        "store":false,"stream":false,
        "include":["reasoning.encrypted_content"],
        "prompt_cache_key":"cache-1","client_metadata":{"thread_id":"t-1"},
        "tools":[
          {"type":"function","name":"exec_command","strict":false,
           "parameters":{"type":"object",
             "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
          {"type":"namespace","name":"multi_agent_v1","tools":[
            {"type":"function","name":"close_agent","strict":false,
             "parameters":{"type":"object","properties":{}}}]},
          {"type":"web_search","external_web_access":false}]})"));
  assert(hosted.member_str("status") == "completed");
  const auto hosted_call = server.backend->LastCall();
  assert(hosted_call.chat.tools.size() == 2 &&
         hosted_call.chat.tools[0].name == "exec_command" &&
         hosted_call.chat.tools[1].name == "close_agent");
  assert(hosted_call.chat.reasoning.enabled == true &&
         hosted_call.chat.reasoning.effort == gufo::ReasoningEffort::kMedium);
  assert(hosted_call.chat.messages.size() == 3 &&
         hosted_call.chat.messages[0].role ==
             gufo::tokenization::ChatRole::kSystem &&
         hosted_call.chat.messages[0].content == "You are a coding agent." &&
         hosted_call.chat.messages[2].content == "say hi");

  // Responses replays function calls and their outputs between turns under the
  // same call_id; the adapter folds them back into the prompt.
  const auto tool_replay = response_body(server.Post("/v1/responses",
                                                     R"({"input":[
          {"type":"message","role":"user","content":[
            {"type":"input_text","text":"list files"}]},
          {"type":"function_call","id":"fc_1","call_id":"call_1",
           "name":"exec_command","arguments":"{\"cmd\":\"ls\"}",
           "status":"completed"},
          {"type":"function_call_output","call_id":"call_1","output":"file.txt"},
          {"type":"message","role":"assistant","content":[
            {"type":"output_text","text":"Here are the files.",
             "annotations":[]}]}]})"));
  assert(tool_replay.member_str("status") == "completed");
  const auto tool_messages = server.backend->LastCall().chat.messages;
  assert(tool_messages.size() == 4);
  assert(tool_messages[1].role == gufo::tokenization::ChatRole::kAssistant &&
         tool_messages[1].tool_calls.size() == 1 &&
         tool_messages[1].tool_calls[0].id == "call_1" &&
         tool_messages[1].tool_calls[0].name == "exec_command");
  assert(tool_messages[2].content == "file.txt" &&
         tool_messages[3].content == "Here are the files.");

  // The include leniency is Responses-only: endpoints that run the shared
  // compatibility validator (Messages) still reject it.
  ExpectStatus(
      server.Post(
          "/v1/messages",
          R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":2,
                      "include":["x"]})"),
      400);

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");

  // Messages reports reasoning in its own block and restores replayed blocks
  // as the assistant thought.
  server.backend->SetOutput("<think>plan</think>answer");
  const auto thinking = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"enabled","budget_tokens":1024}})"));
  const auto blocks = thinking.find("content")->items();
  assert(blocks.size() == 2 && blocks[0].member_str("type") == "thinking" &&
         blocks[0].member_str("thinking") == "plan" &&
         blocks[0].contains("signature") &&
         blocks[1].member_str("type") == "text" &&
         blocks[1].member_str("text") == "answer");
  assert(server.backend->LastCall().chat.reasoning.enabled == true);
  response_body(server.Post("/v1/messages", R"({"messages":[
      {"role":"user","content":"hi"},
      {"role":"assistant","content":[
        {"type":"thinking","thinking":"plan","signature":""},
        {"type":"text","text":"answer"}]},
      {"role":"user","content":"next"}],
    "thinking":{"type":"disabled"}})"));
  const auto replayed = server.backend->LastCall().chat;
  assert(replayed.reasoning.enabled == false);
  assert(replayed.messages.size() == 3 &&
         replayed.messages[1].thought == "plan" &&
         replayed.messages[1].content == "answer");

  // An explicit content phase preserves requested literal reasoning tags.
  // The default automatic phase above still recognizes reasoning blocks.
  server.backend->initial_output_state_override =
      FakeBackend::InitialOutputState::kContent;
  const std::string literal_thinking = "<think>literal example</think>";
  server.backend->SetOutput(literal_thinking);
  const auto literal = response_body(
      server.Post("/v1/messages", R"({"messages":[{"role":"user","content":
        "Copy this XML exactly: <think>literal example</think>"}],
        "thinking":{"type":"disabled"}})"));
  assert(server.backend->LastCall().chat.reasoning.enabled == false);
  const auto literal_blocks = literal.find("content")->items();
  assert(literal_blocks.size() == 1 &&
         literal_blocks[0].member_str("type") == "text" &&
         literal_blocks[0].member_str("text") == literal_thinking &&
         "disabled thinking preserves literal tags as one text block");
  server.backend->initial_output_state_override.reset();

  server.backend->SetOutput("answer");
  const auto plain = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}]})"));
  assert(plain.find("content")->items().size() == 1 &&
         plain.find("content")->items()[0].member_str("text") == "answer");
  for (const auto* invalid :
       {R"({"messages":[{"role":"user","content":[
           {"type":"thinking","thinking":"plan"}]}]})",
        R"({"messages":[{"role":"user","content":"hi"}],"thinking":true})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"enabled","budget_tokens":0}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"adaptive","display":"full"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"adaptive"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"minimal"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"format":{"type":"json_schema"}}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "reasoning_effort":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":true}})"})
    ExpectStatus(server.Post("/v1/messages", invalid), 400);

  // The reasoning fields Claude Code sends on every request. Adaptive keeps
  // the server's thinking default and the reasoning is still returned.
  server.backend->SetOutput("<think>plan</think>answer");
  for (const bool server_thinking : {false, true}) {
    server.backend->reasoning.enabled = server_thinking;
    server.backend->reasoning.effort = gufo::ReasoningEffort::kLow;
    const auto adaptive = response_body(
        server.Post("/v1/messages", R"({"max_tokens":256,"stream":false,
            "thinking":{"type":"adaptive","display":"omitted"},
            "output_config":{"effort":"xhigh"},
            "messages":[{"role":"user","content":"hi"}]})"));
    if (server_thinking)
      assert(adaptive.find("content")->items()[0].member_str("thinking") ==
             "plan");
    const auto reasoning = server.backend->LastCall().chat.reasoning;
    assert(reasoning.enabled == server_thinking);
    assert(reasoning.effort == gufo::ReasoningEffort::kXHigh);
  }
  server.backend->reasoning = {};
  const auto updates = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"enabled","display":"updates"}})"));
  assert(updates.find("content")->items()[0].member_str("thinking") == "plan");
  // Effort never enables thinking.
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "thinking":{"type":"disabled"},"output_config":{"effort":"low"}})"));
  assert(server.backend->LastCall().chat.reasoning.enabled == false);
  assert(server.backend->LastCall().chat.reasoning.effort ==
         gufo::ReasoningEffort::kLow);

  // Messages tools map onto the Chat tool path: declarations, tool_choice,
  // tool_use output and replayed tool_use/tool_result history.
  const std::string weather_tool = R"([{"name":"get_weather",
      "description":"Get weather","input_schema":{"type":"object",
      "properties":{"city":{"type":"string"}},"required":["city"]}}])";
  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto called = response_body(server.Post(
      "/v1/messages", R"({"max_tokens":64,"tool_choice":{"type":"any"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}"));
  assert(called.member_str("stop_reason") == "tool_use");
  const auto call_blocks = called.find("content")->items();
  assert(call_blocks.size() == 1 &&
         call_blocks[0].member_str("type") == "tool_use" &&
         !call_blocks[0].member_str("id").empty() &&
         call_blocks[0].member_str("name") == "get_weather" &&
         call_blocks[0].find("input")->member_str("city") == "Rome");
  const auto declared = server.backend->LastCall().chat;
  assert(declared.tools.size() == 1 &&
         declared.tools[0].name == "get_weather" &&
         declared.tools[0].description == "Get weather" &&
         declared.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kRequired &&
         declared.constrained_tools);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"tool","name":"get_weather"},
          "tools":)" + weather_tool +
                                "}"));
  assert(server.backend->LastCall().chat.forced_tool_name == "get_weather" &&
         !server.backend->LastCall().chat.parallel_tool_calls);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"auto","disable_parallel_tool_use":true},
          "tools":)" + weather_tool +
                                "}"));
  assert(server.backend->LastCall().chat.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kAuto &&
         !server.backend->LastCall().chat.parallel_tool_calls);
  response_body(server.Post("/v1/messages",
                            R"({"messages":[{"role":"user","content":"hi"}],
          "tool_choice":{"type":"none"},"tools":)" +
                                weather_tool + "}"));
  assert(server.backend->LastCall().chat.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kNone &&
         !server.backend->LastCall().chat.constrained_tools);

  server.backend->SetOutput("It is sunny.");
  const auto answered = response_body(server.Post("/v1/messages", R"({
      "tools":)" + weather_tool + R"(,"messages":[
        {"role":"user","content":"weather in Rome?"},
        {"role":"assistant","content":[
          {"type":"thinking","thinking":"look it up","signature":""},
          {"type":"text","text":"Checking."},
          {"type":"tool_use","id":"toolu_1","name":"get_weather",
           "input":{"city":"Rome"}}]},
        {"role":"user","content":[
          {"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
          {"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
           "content":[{"type":"text","text":"timeout"}]},
          {"type":"text","text":"thanks"}]}]})"));
  assert(answered.member_str("stop_reason") == "end_turn" &&
         answered.find("content")->items()[0].member_str("text") ==
             "It is sunny.");
  const auto history = server.backend->LastCall().chat.messages;
  assert(history.size() == 5);
  assert(history[1].role == gufo::tokenization::ChatRole::kAssistant &&
         history[1].thought == "look it up" &&
         history[1].content == "Checking." &&
         history[1].tool_calls.size() == 1 &&
         history[1].tool_calls[0].id == "toolu_1" &&
         history[1].tool_calls[0].name == "get_weather" &&
         history[1].tool_calls[0].arguments.size() == 1 &&
         history[1].tool_calls[0].arguments[0].name == "city" &&
         history[1].tool_calls[0].arguments[0].value == "Rome");
  assert(history[2].role == gufo::tokenization::ChatRole::kTool &&
         history[2].tool_call_id == "toolu_1" && history[2].content == "sunny");
  assert(history[3].role == gufo::tokenization::ChatRole::kTool &&
         history[3].tool_call_id == "toolu_2" &&
         history[3].content == "timeout");
  assert(history[4].role == gufo::tokenization::ChatRole::kUser &&
         history[4].content == "thanks");

  const auto calls_before = server.backend->calls.load();
  for (const auto& invalid : std::vector<std::string>{
           R"({"messages":[{"role":"user","content":"hi"}],"tools":{}})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tools":[{"type":"web_search_20250305","name":"web_search"}]})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tools":[{"name":"f"}]})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":{"type":"any"}})",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":"auto","tools":)" +
               weather_tool + "}",
           R"({"messages":[{"role":"user","content":"hi"}],
               "tool_choice":{"type":"auto","disable_parallel_tool_use":1},
               "tools":)" +
               weather_tool + "}",
           R"({"messages":[{"role":"assistant","content":[{"type":"tool_use",
               "id":"toolu_1","name":"f","input":"{}"}]}]})",
           R"({"messages":[{"role":"assistant","content":[{"type":"tool_use",
               "name":"f","input":{}}]}]})",
           R"({"messages":[{"role":"user","content":[{"type":"tool_result",
               "content":"x"}]}]})",
           R"({"messages":[{"role":"user","content":[{"type":"tool_result",
               "tool_use_id":"toolu_1","content":[{"type":"image"}]}]}]})",
       })
    ExpectStatus(server.Post("/v1/messages", invalid), 400);
  assert(server.backend->calls == calls_before);
  // A turn cut by max_tokens reports the cut, as Chat finish_reason does,
  // even when a complete call precedes it.
  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto truncated = response_body(server.Post(
      "/v1/messages", R"({"max_tokens":1,"tool_choice":{"type":"auto"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}"));
  assert(truncated.member_str("stop_reason") == "max_tokens");
  // Without declared tools, call markup stays visible text as before.
  server.backend->SetOutput("<tool_call>leak");
  const auto undeclared = response_body(server.Post(
      "/v1/messages", R"({"messages":[{"role":"user","content":"hi"}]})"));
  assert(undeclared.member_str("stop_reason") == "end_turn" &&
         undeclared.find("content")->items().size() == 1 &&
         undeclared.find("content")->items()[0].member_str("text") ==
             "<tool_call>leak");

  // Streamed Messages use Anthropic SSE events over the same filters.
  const auto sse_events = [](const std::string& wire) {
    std::vector<gufo::json::Value> events;
    for (auto data = wire.find("data: "); data != std::string::npos;
         data = wire.find("data: ", data + 6)) {
      const auto end = wire.find("\n\n", data);
      events.push_back(gufo::json::parse(
          std::string_view(wire).substr(data + 6, end - data - 6)));
    }
    return events;
  };
  server.backend->SetOutput("<think>plan</think>answer");
  const auto streamed_text =
      server.Post("/v1/messages", R"({"stream":true,"max_tokens":64,
          "thinking":{"type":"enabled","budget_tokens":1024},
          "messages":[{"role":"user","content":"hi"}]})");
  ExpectStatus(streamed_text, 200);
  assert(streamed_text.find("text/event-stream") != std::string::npos &&
         streamed_text.find("event: message_start\n") != std::string::npos);
  std::vector<std::string> types;
  std::string thought, answer;
  const auto text_events = sse_events(streamed_text);
  for (const auto& event : text_events) {
    types.push_back(event.member_str("type"));
    if (const auto* delta = event.find("delta");
        delta && delta->member_str("type") == "thinking_delta")
      thought += delta->member_str("thinking");
    else if (delta && delta->member_str("type") == "text_delta")
      answer += delta->member_str("text");
  }
  assert(thought == "plan" && answer == "answer");
  // Thinking blocks keep the trimmed reasoning Messages reported before
  // streaming, buffered and streamed alike.
  server.backend->SetOutput("<think>\nplan\n\nmore\n</think>\n\nanswer");
  for (const bool stream : {false, true}) {
    const auto framed =
        server.Post("/v1/messages",
                    std::string(R"({"max_tokens":64,"stream":)") +
                        (stream ? "true" : "false") +
                        R"(,"thinking":{"type":"enabled","budget_tokens":1024},
            "messages":[{"role":"user","content":"hi"}]})");
    ExpectStatus(framed, 200);
    std::string framed_thought;
    if (stream) {
      for (const auto& event : sse_events(framed))
        if (const auto* delta = event.find("delta");
            delta && delta->member_str("type") == "thinking_delta")
          framed_thought += delta->member_str("thinking");
    } else {
      framed_thought =
          response_body(framed).find("content")->items()[0].member_str(
              "thinking");
    }
    assert(framed_thought == "plan\n\nmore");
  }
  assert(types.front() == "message_start" && types.back() == "message_stop");
  assert(text_events[1].find("content_block")->member_str("type") ==
             "thinking" &&
         text_events[1].member_size("index") == 0);
  const auto& text_done = text_events[text_events.size() - 2];
  assert(text_done.member_str("type") == "message_delta" &&
         text_done.find("delta")->member_str("stop_reason") == "end_turn" &&
         text_done.find("usage")->member_size("input_tokens") == 10 &&
         text_done.find("usage")->member_size("cache_read_input_tokens") == 8);
  assert(std::ranges::count(types, "content_block_start") == 2 &&
         std::ranges::count(types, "content_block_stop") == 2);

  server.backend->SetOutput(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>");
  const auto streamed_tool = server.Post(
      "/v1/messages", R"({"stream":true,"tool_choice":{"type":"any"},
          "messages":[{"role":"user","content":"weather in Rome?"}],
          "tools":)" + weather_tool +
                          "}");
  ExpectStatus(streamed_tool, 200);
  assert(streamed_tool.find("<tool_call>") == std::string::npos);
  bool tool_started = false;
  std::string partial_json, stop_reason;
  for (const auto& event : sse_events(streamed_tool)) {
    if (const auto* block = event.find("content_block"))
      tool_started |= block->member_str("type") == "tool_use" &&
                      block->member_str("name") == "get_weather";
    if (const auto* delta = event.find("delta")) {
      if (delta->member_str("type") == "input_json_delta")
        partial_json += delta->member_str("partial_json");
      if (event.member_str("type") == "message_delta")
        stop_reason = delta->member_str("stop_reason");
    }
  }
  assert(tool_started && stop_reason == "tool_use" &&
         gufo::json::parse(partial_json).member_str("city") == "Rome");

  server.backend->failure = 6;
  const auto failed_messages = server.Post(
      "/v1/messages",
      R"({"stream":true,"messages":[{"role":"user","content":"hi"}]})");
  ExpectStatus(failed_messages, 200);  // Fake backend fails after headers.
  const auto failed_events = sse_events(failed_messages);
  assert(failed_events.back().member_str("type") == "error" &&
         failed_events.back().find("error")->member_str("type") ==
             "api_error" &&
         failed_messages.find("event: message_stop") == std::string::npos);
  server.backend->failure = 0;
  server.backend->SetOutput("ok");
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

  for (const int failure : {1, 5, 6}) {
    server.backend->failure = failure;
    const std::string message =
        failure == 5 ? "generation failed" : "context exceeded";
    for (
        const auto& [path, body] : {
            std::pair{
                "/v1/completions",
                R"({"prompt":"hello","stream":true,"stream_options":{"include_usage":true}})"},
            std::pair{
                "/v1/chat/completions",
                R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
            std::pair{"/v1/responses", R"({"input":"hello","stream":true})"},
        }) {
      const auto failed = server.Post(path, body);
      ExpectStatus(failed, failure == 6 ? 200 : 500);
      assert(failed.find("\"message\":\"" + message + "\"") !=
             std::string::npos);
      if (failure != 6) {
        assert(failed.find("Content-Type: application/json") !=
               std::string::npos);
        assert(failed.find("data: ") == std::string::npos);
        assert(failed.find("\"code\":\"server_exception\"") !=
               std::string::npos);
        continue;
      }
      const bool responses = std::string_view(path) == "/v1/responses";
      assert(failed.find(responses ? "\"code\":\"server_error\""
                                   : "\"code\":\"generation_failed\"") !=
             std::string::npos);
      assert(failed.find(responses ? "event: response.failed"
                                   : "data: [DONE]\n\n") != std::string::npos);
      assert(failed.find("event: response.completed") == std::string::npos);
      assert(failed.ends_with("0\r\n\r\n"));
    }
  }
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

void TestDeviceLoss() {
  std::atomic<int> hook_calls{0};
  RunningServer server({.on_device_lost = [&] { ++hook_calls; }});
  const std::string chat =
      R"({"model":"test","messages":[{"role":"user","content":"hi"}]})";

  // A failure while the device still accepts work keeps the existing contract.
  server.backend->failure = 4;
  const auto failed = server.Post("/v1/completions", R"({"prompt":"hello"})");
  ExpectStatus(failed, 500);
  assert(failed.find("\"code\":\"server_exception\"") != std::string::npos);
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  ExpectStatus(server.Send("GET /ready HTTP/1.1\r\n\r\n"), 200);
  assert(hook_calls == 0);

  // A stream has committed its status; the terminal event names the loss.
  server.backend->failure = 7;
  const auto stream =
      server.Post("/v1/completions", R"({"prompt":"hello","stream":true})");
  ExpectStatus(stream, 200);
  assert(stream.find("\"code\":\"device_lost\"") != std::string::npos);
  assert(stream.find(gufo::server::kDeviceLostMessage) != std::string::npos);
  assert(hook_calls == 1);

  server.backend->failure = 0;
  for (const std::string path : {"/health", "/v1/health", "/healthz", "/ready",
                                 "/v1/ready", "/readyz"}) {
    const auto response = server.Send("GET " + path + " HTTP/1.1\r\n\r\n");
    ExpectStatus(response, 503);
    assert(response.find("\"status\":\"device_lost\"") != std::string::npos);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
  }
  const int calls = server.backend->calls;
  for (const auto& [path, body] :
       {std::pair<std::string, std::string>{"/v1/chat/completions", chat},
        {"/v1/completions", R"({"prompt":"hello","stream":true})"},
        {"/v1/embeddings", R"({"input":"hello"})"}}) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find("Retry-After") == std::string::npos);
  }
  assert(server.backend->calls == calls);
  // Routes that never reach the device keep answering.
  ExpectStatus(server.Send("GET /v1/models HTTP/1.1\r\n\r\n"), 200);
  ExpectStatus(server.Send("GET /metrics HTTP/1.1\r\n\r\n"), 200);
  assert(hook_calls == 1);

  // Before a response starts, the failing request itself gets the 503.
  std::atomic<int> first_hook_calls{0};
  RunningServer first({.on_device_lost = [&] { ++first_hook_calls; }});
  first.backend->failure = 3;
  for (const auto& [path, body] :
       {std::pair<std::string, std::string>{"/v1/completions",
                                            R"({"prompt":"hello"})"},
        {"/v1/chat/completions", chat}}) {
    const auto response = first.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find(gufo::server::kDeviceLostMessage) !=
           std::string::npos);
    assert(response.find("Retry-After") == std::string::npos);
  }
  assert(first.backend->calls == 1);
  assert(first_hook_calls == 1);
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions", R"({"prompt":"hello","stream":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
       std::pair{"/v1/responses", R"({"input":"hello","stream":true})"}}) {
    RunningServer early;
    early.backend->failure = 3;
    const auto response = early.Post(path, body);
    ExpectStatus(response, 503);
    assert(response.find("\"code\":\"device_lost\"") != std::string::npos);
    assert(response.find("data: ") == std::string::npos);
  }
}

void TestDeviceLossWhileWriterBlocked() {
  std::atomic<int> hook_calls{0};
  std::binary_semaphore hook_called{0};
  std::binary_semaphore writing{0};
  std::atomic<bool> write_finished{false};
  RunningServer server({.sse_heartbeat_interval = std::chrono::milliseconds(0),
                        .on_device_lost = [&] {
                          ++hook_calls;
                          hook_called.release();
                        }});
  server.server.add("POST", "/blocked", [&](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body = [&](const auto& write) {
          const std::string chunk(8 * 1024 * 1024, 'x');
          writing.release();
          (void)write(chunk);
          write_finished = true;
        }};
  });
  const int fd = server.Connect();
  const int receive_buffer = 1024;
  assert(::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer,
                      sizeof(receive_buffer)) == 0);
  const std::string request = "POST /blocked HTTP/1.1\r\n\r\n";
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(writing.try_acquire_for(std::chrono::seconds(2)));
  // The peer never drains the response. No health request or completed write
  // is available to trigger shutdown; the listener must observe the loss.
  server.backend->lost = true;
  assert(hook_called.try_acquire_for(std::chrono::seconds(2)));
  assert(hook_calls == 1 && !write_finished);
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
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

void TestDeferredStreamHeaders() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(5)});
  server.server.add("POST", "/early-failure", [](const auto&, auto&) {
    return gufo::server::HttpResponse{
        .headers = {{"Content-Type", "text/event-stream"}},
        .streaming_body =
            [](const auto&) {
              std::this_thread::sleep_for(std::chrono::milliseconds(30));
              throw std::runtime_error("early generation failure");
            },
        .defer_stream_headers = true};
  });
  const auto failed = server.Post("/early-failure", "");
  ExpectStatus(failed, 500);
  assert(failed.find(": ping") == std::string::npos);
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions", R"({"prompt":"hello","stream":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
       std::pair{"/v1/responses", R"({"input":"hello","stream":true})"}}) {
    server.backend->failure =
        8;  // Successful termination without a token callback.
    server.backend->SetOutput("");
    const auto empty = server.Post(path, body);
    ExpectStatus(empty, 200);
    assert(empty.ends_with("0\r\n\r\n"));
  }
  // Opt-in progress deliberately starts the stream before tokens exist.
  server.backend->progress = {{.total = 100, .processed = 10}};
  server.backend->failure = 4;
  const auto progress =
      server.Post("/v1/completions",
                  R"({"prompt":"hello","stream":true,"return_progress":true})");
  ExpectStatus(progress, 200);
  assert(progress.find("prompt_progress") != std::string::npos);
  assert(progress.find("\"code\":\"generation_failed\"") != std::string::npos);
  server.backend->progress.clear();
  for (
      const auto& [path, body] :
      {std::pair{"/v1/completions",
                 R"({"prompt":"hello","stream":true,"return_progress":true})"},
       std::pair{
           "/v1/chat/completions",
           R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true,"return_progress":true})"},
       std::pair{
           "/v1/responses",
           R"({"input":"hello","stream":true,"return_progress":true})"}}) {
    const auto failure = server.Post(path, body);
    ExpectStatus(failure, 200);
    assert(failure.find(std::string_view(path) == "/v1/responses"
                            ? "\"code\":\"server_error\""
                            : "\"code\":\"generation_failed\"") !=
           std::string::npos);
    assert(failure.ends_with("0\r\n\r\n"));
  }
}

void TestAdmittedStreamHeaders() {
  RunningServer server(
      {.sse_heartbeat_interval = std::chrono::milliseconds(5)});
  server.backend->admit = true;
  const std::pair<const char*, const char*> requests[] = {
      {"/v1/completions", R"({"prompt":"hello","stream":true})"},
      {"/v1/chat/completions",
       R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})"},
      {"/v1/responses", R"({"input":"hello","stream":true})"}};
  // Admission commits headers, so keepalives cover prefill before a token.
  server.backend->token_delay = std::chrono::milliseconds(40);
  for (const auto& [path, body] : requests) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    const auto ping = response.find(": ping\n\n");
    assert(ping != std::string::npos);
    const auto token = response.find(R"("ok")");
    assert(token != std::string::npos);
    assert(ping < token);
    assert(response.ends_with("0\r\n\r\n"));
  }
  // After admission, failures before any token are terminal SSE errors.
  for (const auto& [path, body] : requests) {
    RunningServer lost;
    lost.backend->admit = true;
    lost.backend->failure = 3;
    const auto failed = lost.Post(path, body);
    ExpectStatus(failed, 200);
    assert(failed.find(gufo::server::kDeviceLostMessage) != std::string::npos);
    assert(failed.find("data: ") != std::string::npos);
    assert(failed.ends_with("0\r\n\r\n"));
  }
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
  TestContentTrace();
  TestQuietTiersSuppressLifecycle();
  TestLogLevelFilter();
  TestLogLevelNames();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestFramingAndMetrics();
  TestFallbackBackendMetrics();
  TestLlamaSlotsAndMetrics();
  TestCompatibilityRequests();
  TestModelInputModalities();
  TestRawCompletionStreaming();
  TestDeviceLoss();
  TestDeviceLossWhileWriterBlocked();
  TestRawCompletionPromptProgress();
  TestCompatibilityStopSequences();
  TestCompatibilityThinkingDefaults();
  TestResponseSamplingDefaults();
  TestCompatibilityUtf8();
  TestPeerDisconnect();
  TestStreamingFraming();
  TestSseHeartbeat();
  TestSseHeartbeatShutdown();
  TestDeferredStreamHeaders();
  TestAdmittedStreamHeaders();
  TestSignalShutdown();
  std::cout << "HTTP transport checks passed.\n";
}

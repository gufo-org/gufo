#ifndef GUFO_TESTS_CLI_HTTP_SERVER_TEST_SUPPORT_HPP_
#define GUFO_TESTS_CLI_HTTP_SERVER_TEST_SUPPORT_HPP_

// The HTTP server test harness, shared by the main suite and the POSIX-only
// signal-shutdown suite.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/trace.hpp"
#include "src/core/platform/net.hpp"

namespace {
namespace net = gufo::net;

using gufo::server::HttpServer;
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
  std::vector<std::string> model_ids() const override { return names; }
  std::vector<std::string> names{"test"};
  bool ready() const override { return true; }
  bool device_lost() const override { return lost.load(); }
  bool supports_images() const override { return image_support.load(); }
  std::atomic<bool> image_support{false};
  std::uint32_t max_context() const override { return 65536; }
  std::vector<SessionState> session_states() const override { return sessions; }
  std::vector<SessionState> sessions;
  std::string model_path() const override { return served_path; }
  std::string build_info() const override { return served_build; }
  std::string served_path;
  std::string served_build;
  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancellation, bool stream, bool ignore_eos,
      std::string_view client_id,
      const std::vector<std::string>& stop_sequences, bool return_progress,
      bool cache_prompt, bool cache_persist) override {
    last_ignore_eos = ignore_eos;
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_complete(
            prompt, max_tokens, sampling, cancellation, stream, false,
            client_id, stop_sequences, return_progress, cache_prompt,
            cache_persist),
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
  std::optional<std::size_t> count_chat_tokens(
      const gufo::server::ChatRequest& request) const override {
    counted_chat = request;
    counted_messages = request.messages.size();
    counted_tools = request.tools.size();
    if (count_failure == 1)
      throw std::invalid_argument("invalid counting prompt");
    if (count_failure == 2)
      throw std::length_error("counting prompt too large");
    return chat_token_count;
  }
  std::optional<std::size_t> chat_token_count;
  int count_failure{0};
  mutable gufo::server::ChatRequest counted_chat;
  mutable std::size_t counted_messages{0};
  mutable std::size_t counted_tools{0};
  std::optional<std::string> render_chat_prompt(
      const gufo::server::ChatRequest& request) const override {
    rendered_chat = request;
    if (render_error)
      throw std::invalid_argument(*render_error);
    return rendered_prompt;
  }
  std::optional<std::string> rendered_prompt;
  std::optional<std::string> render_error;
  mutable gufo::server::ChatRequest rendered_chat;
  Result complete(std::string_view prompt, std::size_t limit,
                  const gufo::sampling::SamplingConfig& sampling,
                  const CancellationCheck& cancel, const TokenCallback& token,
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true,
                  bool cache_persist = true) override {
    ++calls;
    last_cache_prompt = cache_prompt;
    last_cache_persist = cache_persist;
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
    if (sampling.logprobs) {
      gufo::server::DecodedLogprobs probabilities;
      probabilities.entries.push_back({3, result.text, -3.0});
      if (*sampling.logprobs > 0)
        probabilities.entries.push_back({1, "alternative", -0.5});
      result.logprobs.push_back(std::move(probabilities));
    }
    if (token && failure != 8) {
      std::this_thread::sleep_for(token_delay);
      (void)token(gufo::server::TokenPiece(
          result.text,
          result.logprobs.empty() ? nullptr : &result.logprobs.front()));
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
  std::atomic<bool> last_cache_prompt{true};
  std::atomic<bool> last_cache_persist{true};
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
                         bool handle_signals = false,
                         std::string host = "127.0.0.1")
      : backend(std::make_shared<FakeBackend>()),
        server(std::move(host), 0, backend, nullptr, nullptr, nullptr,
               std::move(options)) {
    server.add("GET", "/peer", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.client_id};
    });
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

  net::Socket Connect(const std::string& peer = "127.0.0.1") {
    sockaddr_storage address{};
    net::SocklenT length = 0;
    auto* ipv4 = reinterpret_cast<sockaddr_in*>(&address);
    auto* ipv6 = reinterpret_cast<sockaddr_in6*>(&address);
    if (::inet_pton(AF_INET, peer.c_str(), &ipv4->sin_addr) == 1) {
      ipv4->sin_family = AF_INET;
      ipv4->sin_port = htons(server.port());
      length = sizeof(*ipv4);
    } else {
      assert(::inet_pton(AF_INET6, peer.c_str(), &ipv6->sin6_addr) == 1);
      ipv6->sin6_family = AF_INET6;
      ipv6->sin6_port = htons(server.port());
      length = sizeof(*ipv6);
    }
    const net::Socket fd = net::CreateStreamSocket(address.ss_family);
    assert(fd != net::kInvalidSocket);
    assert(net::SetSocketTimeouts(fd, 3'000));
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address), length) ==
           0);
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false,
                   const std::string& peer = "127.0.0.1") {
    const net::Socket fd = Connect(peer);
    while (!request.empty()) {
      const auto count = net::SendNoSignal(fd, request.data(), request.size());
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      net::ShutdownSocket(fd, net::ShutdownHow::kWrite);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = net::Receive(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    net::CloseSocket(fd);
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
}  // namespace

#endif  // GUFO_TESTS_CLI_HTTP_SERVER_TEST_SUPPORT_HPP_

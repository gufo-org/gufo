#ifndef GUFO_SERVER_HTTP_SERVER_HPP_
#define GUFO_SERVER_HTTP_SERVER_HPP_

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

struct GufoHttpServer;
struct GufoHttpRequestView;
struct GufoHttpResponder;
struct GufoLoadingResponder;

namespace gufo::server {

class VideoJobService;
class TtsService;
class AsrService;
class ImageService;
class EmbeddingService;
struct DecisionService;
class WebSocket;

/// Whether a comma-separated header value lists the lowercase token.
bool HasToken(std::string_view value, std::string_view token);

struct HttpRequest {
  std::string method;  // "GET" / "POST" / ...
  std::string path;    // "/v1/chat/completions" (no query)
  std::string query;   // raw query string (no leading '?')
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
  TextGenerationBackend::CancellationCheck is_cancelled;
  std::string request_id{};
  /// Transport-derived peer address, never read from client headers.
  std::string client_id{"anonymous"};

  /// URL-decoded value of a query param, or "" if absent.
  std::string query_param(const std::string& key) const;
  /// Case-insensitive request-header lookup, or "" if absent.
  std::string header(std::string_view name) const {
    const auto equal_case_insensitive = [](std::string_view lhs,
                                           std::string_view rhs) {
      if (lhs.size() != rhs.size()) {
        return false;
      }
      for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(lhs[index])) !=
            std::tolower(static_cast<unsigned char>(rhs[index]))) {
          return false;
        }
      }
      return true;
    };
    for (const auto& [header_name, value] : headers) {
      if (equal_case_insensitive(header_name, name)) {
        return value;
      }
    }
    return "";
  }
};

struct HttpResponse {
  using BodyWriter = std::function<bool(std::string_view)>;
  using StreamingBody = std::function<void(const BodyWriter&)>;

  int status = 200;
  std::string reason = "OK";
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
  StreamingBody streaming_body;
  std::string log_details;
  struct StreamLog {
    std::string details;
    std::string error_code;
    /// A terminal error was successfully written in the stream's protocol.
    bool error_event_sent{false};
  };
  std::shared_ptr<StreamLog> stream_log{};
  std::function<void(WebSocket&)> websocket{};
  /// Text streams commit headers when generation starts or has queued for a
  /// bounded time, so failures before that return a real 5xx while
  /// keepalives still cover prefill and queue waits.
  bool defer_stream_headers{false};
};

using Handler =
    std::function<HttpResponse(const HttpRequest&, TextGenerationBackend&)>;

struct HttpServerOptions {
  std::size_t max_request_body_bytes{static_cast<std::size_t>(8) * 1024 * 1024};
  std::size_t max_connections{16};
  std::chrono::milliseconds sse_heartbeat_interval{std::chrono::seconds(15)};
  /// Reuse HTTP/1.1 connections for sequential requests. The CLI enables it;
  /// embedders that assume one request per socket keep the old behavior.
  bool keep_alive{false};
  /// A reused connection waiting for its next request holds a worker slot for
  /// at most this long.
  std::chrono::milliseconds keep_alive_idle_timeout{std::chrono::seconds(5)};
  /// Bound the requests one connection may serve, so a pooled client rotates
  /// sockets and idle clients cannot pin a worker indefinitely.
  std::size_t max_requests_per_connection{100};
  /// Called once, on the listener or a request thread, when the text backend or
  /// the embedding service reports a lost device. Health, readiness and
  /// generation answer 503 `device_lost` from then on; the hook decides how the
  /// process exits.
  std::function<void()> on_device_lost{};
};

/// True for a numeric IPv4 or IPv6 literal `HttpServer` can bind. Host names
/// are not resolved; IPv6 link-local literals may carry a `%<interface>` scope.
bool IsListenAddress(const std::string& host);

/// Minimal bounded HTTP/1.1 server for trusted-LAN model serving.
class HttpServer {
public:
  HttpServer(std::string host, int port,
             std::shared_ptr<TextGenerationBackend> backend,
             std::shared_ptr<VideoJobService> video_jobs = nullptr,
             std::shared_ptr<TtsService> tts = nullptr,
             std::shared_ptr<AsrService> asr = nullptr,
             HttpServerOptions options = {},
             std::shared_ptr<ImageService> images = nullptr,
             std::shared_ptr<EmbeddingService> embeddings = nullptr,
             std::shared_ptr<DecisionService> decisions = nullptr);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  void add(const std::string& method, const std::string& path, Handler handler);

  /// Bind + listen. Returns false and sets *error on failure.
  bool start(std::string* error);

  /// Blocking accept loop. The CLI handles SIGINT/SIGTERM; embedded callers
  /// retain their own process signal handlers and call stop() explicitly.
  void run(bool handle_signals = false);

  void stop();

  [[nodiscard]] int port() const noexcept { return port_; }

private:
  HttpResponse handle_request(const HttpRequest& req);
  bool device_lost();
  void dispatch_request(const GufoHttpRequestView& view,
                        GufoHttpResponder* responder);
  void register_routes();

  std::string host_;
  int port_;
  std::shared_ptr<TextGenerationBackend> backend_;
  std::shared_ptr<VideoJobService> video_jobs_;
  std::shared_ptr<TtsService> tts_;
  std::shared_ptr<AsrService> asr_;
  std::shared_ptr<ImageService> images_;
  std::shared_ptr<EmbeddingService> embeddings_;
  std::shared_ptr<DecisionService> decisions_;
  HttpServerOptions options_;
  std::function<HttpResponse()> metrics_;
  GufoHttpServer* server_ = nullptr;
  std::atomic<bool> stopped_{false};
  std::atomic<bool> device_lost_reported_{false};
  std::vector<std::pair<std::pair<std::string, std::string>, Handler>> routes_;
};

/// Holds the listen port while a model loads and answers every request with
/// llama-server's loading reply: 503 `Loading model`, `/health` included. A
/// poller can tell a loading server from a dead one, and a client that treats
/// a 200 health check as readiness (llama-swap, the functional runner) keeps
/// waiting instead of sending work to a server that cannot serve it.
///
/// Best effort: when the address cannot be bound, or for port 0, nothing
/// listens and the caller proceeds exactly as without it. stop() closes the
/// socket so `HttpServer::start` can bind the same address.
///
/// Beside the `error` object the body reports the load stage as top-level
/// `progress` (0 to 1) and `message`, which llama-swap v269 and later show
/// and use to restart `healthCheckTimeout` whenever `progress` changes.
class LoadingResponder {
public:
  LoadingResponder(const std::string& host, int port);
  ~LoadingResponder();

  LoadingResponder(const LoadingResponder&) = delete;
  LoadingResponder& operator=(const LoadingResponder&) = delete;
  LoadingResponder(LoadingResponder&&) = delete;
  LoadingResponder& operator=(LoadingResponder&&) = delete;

  /// True while the responder owns the listen socket.
  [[nodiscard]] bool listening() const noexcept {
    return listening_.load(std::memory_order_acquire);
  }

  /// Stops answering and releases the port. Returns within one poll interval
  /// plus the bounded waits for a request that is still arriving and for the
  /// answered client to close.
  void stop();

  /// Reports a load stage in later replies. `progress` is clamped to [0, 1].
  /// Safe to call from any thread, before or after stop().
  void SetProgress(double progress, std::string_view message);

private:
  GufoLoadingResponder* responder_ = nullptr;
  std::atomic<bool> listening_{false};
};

/// Scheduled requests are recorded by the scheduler when they finish. Other
/// backends contribute here, at completion.
inline void RecordServerMetrics(const TextGenerationBackend::Result& result) {
  if (result.token_metrics_recorded)
    return;
  // Older backends may only report prompt/cache totals. Infer their completed
  // uncached prompt, but never infer work for a partially cancelled request.
  const auto prompt_tokens =
      result.prefill_tokens > 0 || result.cancelled
          ? result.prefill_tokens
          : result.prompt_tokens -
                std::min(result.prompt_tokens, result.cached_prompt_tokens);
  Metrics().prompt_tokens.Add(prompt_tokens);
  Metrics().gen_tokens.Add(result.completion_tokens);
  RecordRequestMetrics(result);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_HTTP_SERVER_HPP_

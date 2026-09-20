#ifndef GUFO_SERVER_HTTP_SERVER_HPP_
#define GUFO_SERVER_HTTP_SERVER_HPP_

#include <atomic>
#include <cctype>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

class VideoJobService;
class TtsService;
class AsrService;
class ImageService;
class WebSocket;

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
  };
  std::shared_ptr<StreamLog> stream_log{};
  std::function<void(WebSocket&)> websocket{};
};

using Handler =
    std::function<HttpResponse(const HttpRequest&, TextGenerationBackend&)>;

struct HttpServerOptions {
  std::size_t max_request_body_bytes{static_cast<std::size_t>(8) * 1024 * 1024};
  std::size_t max_connections{16};
  std::string api_key;
};

/// Minimal bounded HTTP/1.1 server for trusted-LAN model serving.
class HttpServer {
public:
  HttpServer(std::string host, int port,
             std::shared_ptr<TextGenerationBackend> backend,
             std::shared_ptr<VideoJobService> video_jobs = nullptr,
             std::shared_ptr<TtsService> tts = nullptr,
             std::shared_ptr<AsrService> asr = nullptr,
             HttpServerOptions options = {},
             std::shared_ptr<ImageService> images = nullptr);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  void add(const std::string& method, const std::string& path, Handler handler);

  /// Bind + listen. Returns false and sets *error on failure.
  bool start(std::string* error);

  /// Blocking accept loop.
  void run();

  void stop();

  [[nodiscard]] int port() const noexcept { return port_; }

private:
  struct ConnectionWorker;

  HttpResponse handle_request(const HttpRequest& req);
  void handle_connection(int client_fd);
  void reap_workers();
  void register_routes();

  std::string host_;
  int port_;
  std::shared_ptr<TextGenerationBackend> backend_;
  std::shared_ptr<VideoJobService> video_jobs_;
  std::shared_ptr<TtsService> tts_;
  std::shared_ptr<AsrService> asr_;
  std::shared_ptr<ImageService> images_;
  HttpServerOptions options_;
  std::string api_key_hash_;
  int listen_fd_ = -1;
  std::atomic<bool> stopped_{false};
  std::mutex workers_mutex_;
  std::vector<std::unique_ptr<ConnectionWorker>> workers_;
  std::vector<std::pair<std::pair<std::string, std::string>, Handler>> routes_;
};

namespace detail {
inline std::atomic<std::uint64_t>& TotalPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalGenTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& LastPromptSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
inline std::atomic<double>& LastGenSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
}  // namespace detail

inline void RecordServerMetrics(const TextGenerationBackend::Result& result) {
  detail::TotalPromptTokens().fetch_add(result.prompt_tokens,
                                        std::memory_order_relaxed);
  detail::TotalGenTokens().fetch_add(result.completion_tokens,
                                     std::memory_order_relaxed);

  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  if (prompt_per_second > 0.0) {
    detail::LastPromptSpeed().store(prompt_per_second,
                                    std::memory_order_relaxed);
  }
  if (tok_per_sec > 0.0) {
    detail::LastGenSpeed().store(tok_per_sec, std::memory_order_relaxed);
  }
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_HTTP_SERVER_HPP_

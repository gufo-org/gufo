#ifndef GUFO_SERVER_RERANK_METRICS_HPP_
#define GUFO_SERVER_RERANK_METRICS_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "gufo/ext/rerank.hpp"
#include "src/core/metrics.hpp"

namespace gufo::server {

struct RerankMetrics {
  RerankMetrics();

  core::metrics::Registry registry;
  core::metrics::Counter device_lost;
  core::metrics::Counter ok;
  core::metrics::Counter error;
  core::metrics::Counter cancelled;
  core::metrics::Counter invalid;
  core::metrics::Counter lost;
  core::metrics::Counter documents;
  core::metrics::Counter input_tokens;
  core::metrics::Histogram request_seconds;
  core::metrics::IntGauge in_flight;

  [[nodiscard]] core::metrics::Counter& Requests(ext::RerankStatus status) {
    switch (status) {
      case ext::RerankStatus::kOk:
        return ok;
      case ext::RerankStatus::kCancelled:
        return cancelled;
      case ext::RerankStatus::kDeviceLost:
        return lost;
      case ext::RerankStatus::kInputTooLong:
      case ext::RerankStatus::kInvalidInput:
      case ext::RerankStatus::kUnsupportedInput:
        return invalid;
      case ext::RerankStatus::kFailed:
        return error;
    }
    return error;
  }
};

inline RerankMetrics::RerankMetrics() {
  using namespace core::metrics;
  device_lost = registry.Add<Counter>("gufo_device_lost_total",
                                      "Confirmed GPU context losses");
  const auto requests = registry.AddFamily<Counter>(
      "gufo_rerank_requests_total",
      "Rerank requests by outcome, where invalid means the runner rejected "
      "the request as malformed or too long");
  ok = requests.Add({{"status", "ok"}});
  error = requests.Add({{"status", "error"}});
  cancelled = requests.Add({{"status", "cancelled"}});
  invalid = requests.Add({{"status", "invalid"}});
  lost = requests.Add({{"status", "device_lost"}});
  documents = registry.Add<Counter>(
      "gufo_rerank_documents_total",
      "Documents received by the service, including those later rejected");
  input_tokens = registry.Add<Counter>(
      "gufo_rerank_input_tokens_total",
      "Tokens evaluated by successful requests, after truncation");
  static constexpr std::array<double, 12> kRequestSeconds = {
      0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0};
  request_seconds = registry.Add<Histogram>(
      "gufo_rerank_request_seconds",
      "Time from admission to the end of a rerank request", kRequestSeconds);
  in_flight =
      registry.Add<IntGauge>("gufo_rerank_requests_in_flight",
                             "Rerank requests admitted and not finished");
}

class RerankRequestScope {
public:
  explicit RerankRequestScope(RerankMetrics& metrics)
      : metrics_(metrics), start_(std::chrono::steady_clock::now()) {
    metrics_.in_flight.Add(1);
  }
  ~RerankRequestScope() {
    metrics_.in_flight.Add(-1);
    metrics_.Requests(status_).Add(1);
    metrics_.request_seconds.Observe(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count());
  }
  RerankRequestScope(const RerankRequestScope&) = delete;
  RerankRequestScope& operator=(const RerankRequestScope&) = delete;

  void Accepted(std::size_t documents) {
    metrics_.documents.Add(static_cast<std::uint64_t>(documents));
  }
  void Finish(ext::RerankStatus status, const ext::RerankScores& scores) {
    status_ = status;
    if (status == ext::RerankStatus::kOk)
      metrics_.input_tokens.Add(std::accumulate(scores.token_counts.begin(),
                                                scores.token_counts.end(),
                                                std::uint64_t{0}));
  }

private:
  RerankMetrics& metrics_;
  std::chrono::steady_clock::time_point start_;
  ext::RerankStatus status_{ext::RerankStatus::kFailed};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_RERANK_METRICS_HPP_

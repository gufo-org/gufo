#ifndef GUFO_SERVER_EMBEDDING_METRICS_HPP_
#define GUFO_SERVER_EMBEDDING_METRICS_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <span>

#include "gufo/ext/embeddings.hpp"
#include "src/core/metrics.hpp"

namespace gufo::server {

struct EmbeddingMetrics {
  EmbeddingMetrics();

  core::metrics::Registry registry;
  core::metrics::Counter device_lost;
  core::metrics::Counter ok;
  core::metrics::Counter error;
  core::metrics::Counter cancelled;
  core::metrics::Counter invalid;
  core::metrics::Counter lost;
  core::metrics::Counter text_inputs;
  core::metrics::Counter image_inputs;
  core::metrics::Counter audio_inputs;
  core::metrics::Counter video_inputs;
  core::metrics::Counter input_tokens;
  core::metrics::Histogram request_seconds;
  core::metrics::IntGauge in_flight;

  [[nodiscard]] core::metrics::Counter& Requests(ext::EmbeddingStatus status) {
    switch (status) {
      case ext::EmbeddingStatus::kOk:
        return ok;
      case ext::EmbeddingStatus::kCancelled:
        return cancelled;
      case ext::EmbeddingStatus::kDeviceLost:
        return lost;
      case ext::EmbeddingStatus::kInputTooLong:
      case ext::EmbeddingStatus::kInvalidInput:
      case ext::EmbeddingStatus::kInvalidAudioInput:
      case ext::EmbeddingStatus::kInvalidVideoInput:
        return invalid;
      case ext::EmbeddingStatus::kFailed:
        return error;
    }
    return error;
  }

  [[nodiscard]] core::metrics::Counter& Inputs(ext::EmbeddingPart::Type type) {
    switch (type) {
      case ext::EmbeddingPart::Type::kText:
        return text_inputs;
      case ext::EmbeddingPart::Type::kImage:
        return image_inputs;
      case ext::EmbeddingPart::Type::kAudio:
        return audio_inputs;
      case ext::EmbeddingPart::Type::kVideo:
        return video_inputs;
    }
    return text_inputs;
  }
};

inline EmbeddingMetrics::EmbeddingMetrics() {
  using namespace core::metrics;
  device_lost = registry.Add<Counter>("gufo_device_lost_total",
                                      "Confirmed GPU context losses");
  const auto requests = registry.AddFamily<Counter>(
      "gufo_embedding_requests_total",
      "Embedding requests by outcome, where invalid means the service "
      "rejected the request as malformed or too long");
  ok = requests.Add({{"status", "ok"}});
  error = requests.Add({{"status", "error"}});
  cancelled = requests.Add({{"status", "cancelled"}});
  invalid = requests.Add({{"status", "invalid"}});
  lost = requests.Add({{"status", "device_lost"}});
  const auto parts = registry.AddFamily<Counter>(
      "gufo_embedding_inputs_total",
      "Embedding input parts received, by modality, including parts the "
      "service later rejects");
  text_inputs = parts.Add({{"modality", "text"}});
  image_inputs = parts.Add({{"modality", "image"}});
  audio_inputs = parts.Add({{"modality", "audio"}});
  video_inputs = parts.Add({{"modality", "video"}});
  input_tokens = registry.Add<Counter>(
      "gufo_embedding_input_tokens_total",
      "Tokens embedded by successful requests, after truncation");
  static constexpr std::array<double, 12> kRequestSeconds = {
      0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0};
  request_seconds = registry.Add<Histogram>(
      "gufo_embedding_request_seconds",
      "Time from admission to the end of an embedding request",
      kRequestSeconds);
  in_flight =
      registry.Add<IntGauge>("gufo_embedding_requests_in_flight",
                             "Embedding requests admitted and not finished");
}

class EmbeddingRequestScope {
public:
  explicit EmbeddingRequestScope(EmbeddingMetrics& metrics)
      : metrics_(metrics), start_(std::chrono::steady_clock::now()) {
    metrics_.in_flight.Add(1);
  }
  ~EmbeddingRequestScope() {
    metrics_.in_flight.Add(-1);
    metrics_.Requests(status_).Add(1);
    metrics_.request_seconds.Observe(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count());
  }
  EmbeddingRequestScope(const EmbeddingRequestScope&) = delete;
  EmbeddingRequestScope& operator=(const EmbeddingRequestScope&) = delete;

  void Accepted(std::span<const ext::EmbeddingInput> inputs) {
    for (const auto& input : inputs)
      for (const auto& part : input.content)
        metrics_.Inputs(part.type).Add(1);
  }
  void Finish(ext::EmbeddingStatus status, const ext::EmbeddingBatch& batch) {
    status_ = status;
    if (status == ext::EmbeddingStatus::kOk)
      metrics_.input_tokens.Add(std::accumulate(batch.token_counts.begin(),
                                                batch.token_counts.end(),
                                                std::uint64_t{0}));
  }

private:
  EmbeddingMetrics& metrics_;
  std::chrono::steady_clock::time_point start_;
  ext::EmbeddingStatus status_{ext::EmbeddingStatus::kFailed};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_EMBEDDING_METRICS_HPP_

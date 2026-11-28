#ifndef GUFO_SERVER_ASR_METRICS_HPP_
#define GUFO_SERVER_ASR_METRICS_HPP_

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <utility>

#include "gufo/ext/asr.hpp"
#include "src/core/metrics.hpp"

namespace gufo::server {

struct AsrMetrics {
  struct Sessions {
    core::metrics::IntGauge open;
    core::metrics::Counter opened;
  };

  AsrMetrics();

  core::metrics::Registry registry;
  core::metrics::Counter ok;
  core::metrics::Counter error;
  core::metrics::Counter cancelled;
  core::metrics::Counter invalid;
  core::metrics::FloatCounter batch_audio_seconds;
  core::metrics::FloatCounter live_audio_seconds;
  core::metrics::Histogram request_seconds;
  core::metrics::Histogram real_time_factor;
  core::metrics::IntGauge in_flight;
  Sessions live_sessions;
  Sessions voice_sessions;
};

inline AsrMetrics::AsrMetrics() {
  using namespace core::metrics;
  const auto requests = registry.AddFamily<Counter>(
      "gufo_asr_requests_total", "Transcription requests by outcome");
  ok = requests.Add({{"status", "ok"}});
  error = requests.Add({{"status", "error"}});
  cancelled = requests.Add({{"status", "cancelled"}});
  invalid = requests.Add({{"status", "invalid"}});
  const auto audio = registry.AddFamily<FloatCounter>(
      "gufo_asr_audio_seconds_total",
      "Seconds of audio transcribed, by request kind");
  batch_audio_seconds = audio.Add({{"kind", "batch"}});
  live_audio_seconds = audio.Add({{"kind", "live"}});
  static constexpr std::array<double, 12> kRequestSeconds = {
      0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0};
  request_seconds = registry.Add<Histogram>(
      "gufo_asr_request_seconds",
      "Wall seconds from admission to the end of a transcription request, "
      "including time queued behind other requests",
      kRequestSeconds);
  static constexpr std::array<double, 8> kRealTimeFactor = {
      0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0};
  real_time_factor = registry.Add<Histogram>(
      "gufo_asr_real_time_factor",
      "Processing seconds per second of audio in successful requests, from "
      "the runner's own timing, or wall time when it reports none",
      kRealTimeFactor);
  in_flight = registry.Add<IntGauge>("gufo_asr_requests_in_flight",
                                     "Transcription requests admitted and "
                                     "not finished");
  live_sessions.open = registry.Add<IntGauge>(
      "gufo_asr_live_sessions", "Open live transcription sessions");
  voice_sessions.open = registry.Add<IntGauge>(
      "gufo_asr_voice_sessions", "Open voice conversation sessions");
  const auto opened = registry.AddFamily<Counter>(
      "gufo_asr_sessions_total", "Realtime sessions opened, by kind");
  live_sessions.opened = opened.Add({{"kind", "live"}});
  voice_sessions.opened = opened.Add({{"kind", "voice"}});
}

class AsrRequestScope {
public:
  explicit AsrRequestScope(AsrMetrics& metrics)
      : metrics_(metrics),
        outcome_(metrics.error),
        start_(std::chrono::steady_clock::now()) {
    metrics_.in_flight.Add(1);
  }
  ~AsrRequestScope() {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count();
    metrics_.in_flight.Add(-1);
    outcome_.Add(1);
    metrics_.request_seconds.Observe(seconds);
    if (audio_seconds_ > 0.0) {
      metrics_.batch_audio_seconds.Add(audio_seconds_);
      metrics_.real_time_factor.Observe(processing_seconds_.value_or(seconds) /
                                        audio_seconds_);
    }
  }
  AsrRequestScope(const AsrRequestScope&) = delete;
  AsrRequestScope& operator=(const AsrRequestScope&) = delete;

  void Succeeded(const ext::TranscriptionResult* result) {
    outcome_ = metrics_.ok;
    if (result == nullptr)
      return;
    if (result->timings.total_ms > 0.0)
      processing_seconds_ = result->timings.total_ms / 1000.0;
    if (result->sample_rate != 0)
      audio_seconds_ = static_cast<double>(result->audio_samples) /
                       static_cast<double>(result->sample_rate);
  }
  void Cancelled() { outcome_ = metrics_.cancelled; }
  void InvalidAudio() { outcome_ = metrics_.invalid; }

private:
  AsrMetrics& metrics_;
  core::metrics::Counter outcome_;
  std::chrono::steady_clock::time_point start_;
  double audio_seconds_{0.0};
  std::optional<double> processing_seconds_;
};

class AsrSessionScope {
public:
  AsrSessionScope(std::shared_ptr<AsrMetrics> metrics,
                  const AsrMetrics::Sessions& sessions)
      : metrics_(std::move(metrics)), open_(sessions.open) {
    sessions.opened.Add(1);
    open_.Add(1);
  }
  ~AsrSessionScope() { open_.Add(-1); }
  AsrSessionScope(const AsrSessionScope&) = delete;
  AsrSessionScope& operator=(const AsrSessionScope&) = delete;

private:
  // Keeps `open_` valid when the session outlives its service.
  std::shared_ptr<AsrMetrics> metrics_;
  core::metrics::IntGauge open_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_ASR_METRICS_HPP_

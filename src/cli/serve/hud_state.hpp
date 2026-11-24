#ifndef GUFO_SERVER_HUD_STATE_HPP_
#define GUFO_SERVER_HUD_STATE_HPP_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

/// One finished request as shown in the HUD request table. Times in
/// milliseconds, speeds in tokens per second. The ring is newest first.
struct HudRequestRecord {
  std::int64_t ts_ms{0};
  std::string client_id{"anonymous"};
  std::string status{"stop"};
  std::uint64_t prompt_tokens{0};
  std::uint64_t cached_tokens{0};
  std::uint64_t completion_tokens{0};
  std::uint64_t queue_depth{0};
  double queue_ms{0.0};
  double ttft_ms{0.0};
  double prefill_tps{0.0};
  double decode_tps{0.0};
  double duration_ms{0.0};
};

namespace hud_detail {
constexpr std::size_t kHudRingSize = 200;

inline std::int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline std::mutex& RequestsMutex() {
  static std::mutex mutex;
  return mutex;
}

/// Process-wide ring of the most recent finished requests, like the
/// process-wide `/metrics` counters. Function-local statics keep it available
/// to every target that records requests (scheduler or blocking backends)
/// without a shared library symbol.
inline std::vector<HudRequestRecord>& RequestsRing() {
  static std::vector<HudRequestRecord> ring;
  return ring;
}

inline double TokensPerSecond(std::uint64_t tokens, double ms) {
  if (tokens == 0 || ms <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(tokens) / (ms / 1000.0);
}
}  // namespace hud_detail

/// Records one finished request into the process-wide HUD ring. Called from
/// RecordRequestMetrics, which every completion path reaches exactly once.
inline void RecordHudRequest(const TextGenerationBackend::Result& result) {
  HudRequestRecord record;
  record.ts_ms = hud_detail::NowMs();
  record.client_id = result.client_id;
  switch (result.finish_reason) {
    case TextGenerationBackend::FinishReason::kLength:
      record.status = "length";
      break;
    case TextGenerationBackend::FinishReason::kCancelled:
      record.status = "abort";
      break;
    case TextGenerationBackend::FinishReason::kStop:
    case TextGenerationBackend::FinishReason::kStopSequence:
      record.status = "stop";
      break;
  }
  record.prompt_tokens = result.prompt_tokens;
  record.cached_tokens = result.cached_prompt_tokens;
  record.completion_tokens = result.completion_tokens;
  record.queue_depth = result.queue_depth_at_submit;
  record.queue_ms = result.queue_ms;
  record.ttft_ms = result.ttft_ms;
  record.prefill_tps =
      hud_detail::TokensPerSecond(result.prefill_tokens, result.prefill_ms);
  record.decode_tps =
      hud_detail::TokensPerSecond(result.completion_tokens, result.decode_ms);
  record.duration_ms = result.queue_ms + result.prefill_ms + result.decode_ms;
  std::lock_guard lock(hud_detail::RequestsMutex());
  auto& ring = hud_detail::RequestsRing();
  ring.insert(ring.begin(), std::move(record));
  if (ring.size() > hud_detail::kHudRingSize) {
    ring.resize(hud_detail::kHudRingSize);
  }
}

/// Builds the full `/hud/api/state` document: model and session context, the
/// live queue gauges, the recent request ring, engine-native node telemetry
/// read from sysfs (GPU busy, GTT, temperature, power, PCIe, RAM, CPU, disk)
/// and the sampled decode/CPU/disk series for the sparklines. Everything
/// comes from the inference engine itself; no external scrape source.
json::Value HudStateJson(const TextGenerationBackend& backend);

/// The embedded browser client served at `/hud`.
std::string_view HudPage();

}  // namespace gufo::server

#endif  // GUFO_SERVER_HUD_STATE_HPP_

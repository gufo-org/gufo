#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

// Process-wide `/metrics` values. Token counters advance per prefill chunk and
// per generated token, so scrapes see work while requests are still running.
namespace detail {
inline std::atomic<std::uint64_t>& DeviceLostTotal() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalGenTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Request totals below are added once per request when it finishes.
inline std::atomic<std::uint64_t>& TotalCachedPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& TotalPromptSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<double>& TotalGenSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<std::uint64_t>& MaxSequenceTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftRounds() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftAcceptedTokens() {
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
// Schedulers add their own deltas, so a draining scheduler and its
// replacement can both contribute during a model reload.
inline std::atomic<std::int64_t>& RequestsProcessing() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
inline std::atomic<std::int64_t>& RequestsDeferred() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
}  // namespace detail

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
inline void RecordRequestMetrics(const TextGenerationBackend::Result& result) {
  const auto add_seconds = [](std::atomic<double>& total, double ms) {
    double current = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(current, current + ms / 1000.0,
                                        std::memory_order_relaxed)) {
    }
  };
  detail::TotalCachedPromptTokens().fetch_add(
      std::min(result.cached_prompt_tokens, result.prompt_tokens),
      std::memory_order_relaxed);
  add_seconds(detail::TotalPromptSeconds(), result.prefill_ms);
  add_seconds(detail::TotalGenSeconds(), result.decode_ms);
  const std::uint64_t sequence_tokens =
      result.prompt_tokens + result.completion_tokens;
  std::uint64_t max_tokens =
      detail::MaxSequenceTokens().load(std::memory_order_relaxed);
  while (max_tokens < sequence_tokens &&
         !detail::MaxSequenceTokens().compare_exchange_weak(
             max_tokens, sequence_tokens, std::memory_order_relaxed)) {
  }
  detail::TotalDraftRounds().fetch_add(result.draft_rounds,
                                       std::memory_order_relaxed);
  detail::TotalDraftTokens().fetch_add(result.draft_tokens,
                                       std::memory_order_relaxed);
  detail::TotalDraftAcceptedTokens().fetch_add(result.draft_accepted_tokens,
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

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
          : result.finish_reason ==
                  TextGenerationBackend::FinishReason::kStopSequence
              ? "stop_sequence"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " queue_ms=" << result.queue_ms
      << " shared_prefix_wait_ms=" << result.shared_prefix_wait_ms
      << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens
      << " draft_rounds=" << result.draft_rounds;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
  if (result.lookup_tokens > 0)
    out << " lookup_accepted=" << result.lookup_accepted_tokens
        << " lookup_proposed=" << result.lookup_tokens;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_rounds"] = result.draft_rounds;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  if (result.lookup_tokens > 0) {
    timings["lookup_n"] = result.lookup_tokens;
    timings["lookup_n_accepted"] = result.lookup_accepted_tokens;
  }
  return timings;
}

/// llama-server `prompt_progress` object for streamed `return_progress`.
inline json::Value PromptProgressJson(
    const TextGenerationBackend::PromptProgress& progress) {
  json::Value value = json::Value::object();
  value["total"] = progress.total;
  value["cache"] = progress.cache;
  value["processed"] = progress.processed;
  value["time_ms"] = progress.time_ms;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_

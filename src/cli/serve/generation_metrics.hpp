#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <iomanip>
#include <sstream>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
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
      << " queue_ms=" << result.queue_ms << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
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
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  return timings;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_

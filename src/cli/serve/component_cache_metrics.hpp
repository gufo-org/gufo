#ifndef GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_
#define GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"
#include "src/cache/streaming.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
struct ComponentCacheMetrics {
  std::string reuse{"none"};
  std::uint32_t selection_reason{};
  std::size_t deepest_live{}, deepest_ram{}, deepest_disk{}, selected{};
  std::size_t restored_bytes{}, prefilled{}, captures{}, captured_bytes{};
  std::size_t borrowed_capacity_bytes{}, copied_payload_bytes{};
  std::size_t disk_queued_bytes{};
  double disk_enqueue_ms{};
  cache::ResourceSnapshot ledger;
  std::optional<cache::PersistenceQueueStats> persistence;
  cache::SpillMetrics spill;
};
inline json::Value ComponentCacheJson(const ComponentCacheMetrics& value) {
  auto result = json::Value::object();
  result["schema"] = "component-cache-v1";
  result["reuse"] = value.reuse;
  result["selection_reason"] = static_cast<std::size_t>(value.selection_reason);
  result["deepest_live_tokens"] = value.deepest_live;
  result["deepest_ram_tokens"] = value.deepest_ram;
  result["deepest_disk_tokens"] = value.deepest_disk;
  result["selected_tokens"] = value.selected;
  result["restored_bytes"] = value.restored_bytes;
  result["prefilled_tokens"] = value.prefilled;
  result["captures"] = value.captures;
  result["captured_capacity_bytes"] = value.captured_bytes;
  result["borrowed_capacity_bytes"] = value.borrowed_capacity_bytes;
  result["copied_payload_bytes"] = value.copied_payload_bytes;
  auto spill = json::Value::object();
  spill["idle_bytes"] = value.spill.idle_bytes;
  spill["foreground_bytes"] = value.spill.foreground_bytes;
  spill["idle_copy_ms"] = value.spill.idle_copy_ns / 1e6;
  spill["foreground_copy_ms"] = value.spill.foreground_copy_ns / 1e6;
  spill["stream_wait_ms"] = value.spill.stream_wait_ns / 1e6;
  spill["residual_wait_ms"] = value.spill.residual_wait_ns / 1e6;
  spill["reassignments"] = value.spill.reassignments;
  spill["idle_failures"] = value.spill.idle_failures;
  result["global_spill"] = std::move(spill);
  // These are contemporaneous global gauges, not request-local attribution.
  result["global_ledger_bytes"] = value.ledger.total_bytes;
  result["global_ram_bytes"] = value.ledger.ram_bytes;
  result["global_peak_ledger_bytes"] = value.ledger.peak_total_bytes;
  result["global_peak_ram_bytes"] = value.ledger.peak_ram_bytes;
  result["global_persistence_pinned_bytes"] =
      value.ledger.persistence_pinned_bytes;
  result["global_peak_persistence_pinned_bytes"] =
      value.ledger.peak_persistence_pinned_bytes;
  auto categories = json::Value::object();
  constexpr const char* names[]{"backing_free",         "backing_assigned",
                                "backing_materialized", "private_state",
                                "private_tail",         "metadata",
                                "transfer_staging"};
  for (std::size_t i = 0; i < cache::kResourceCategoryCount; ++i) {
    auto gauge = json::Value::object();
    gauge["bytes"] = value.ledger.bytes[i];
    gauge["reserved_bytes"] = value.ledger.reserved_bytes[i];
    gauge["peak_bytes"] = value.ledger.peak_bytes[i];
    categories[names[i]] = std::move(gauge);
  }
  result["global_categories"] = std::move(categories);
  if (value.persistence) {
    const auto& stats = *value.persistence;
    auto queue = json::Value::object();
    queue["accepted_jobs"] = stats.accepted;
    queue["coalesced_jobs"] = stats.coalesced;
    queue["superseded_jobs"] = stats.superseded;
    queue["skipped_jobs"] = stats.skipped;
    queue["published_jobs"] = stats.completed;
    queue["failed_jobs"] = stats.failed;
    queue["pending_jobs"] = stats.pending;
    queue["peak_pending_jobs"] = stats.peak_pending;
    // Queue capacities sum source ownership per job. The ledger above counts
    // shared physical allocations once across all persistence pins.
    queue["pinned_source_capacity_bytes"] = stats.pinned_bytes;
    queue["peak_pinned_source_capacity_bytes"] = stats.peak_pinned_bytes;
    queue["transferred_bytes"] = stats.timing.bytes;
    queue["yield_ms"] = stats.timing.yield_ns / 1e6;
    queue["filesystem_ms"] = stats.timing.filesystem_ns / 1e6;
    queue["fsync_ms"] = stats.timing.fsync_ns / 1e6;
    queue["checksum_ms"] = stats.timing.checksum_ns / 1e6;
    result["global_persistence"] = std::move(queue);
  }
  return result;
}
}  // namespace gufo::server
#endif

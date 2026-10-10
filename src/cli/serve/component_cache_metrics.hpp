#ifndef GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_
#define GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/cache/ledger.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
struct ComponentCacheMetrics {
  std::string reuse{"none"};
  std::uint32_t selection_reason{};
  std::size_t deepest_live{}, deepest_ram{}, deepest_disk{}, selected{};
  std::size_t restored_bytes{}, prefilled{}, captures{}, captured_bytes{};
  std::size_t disk_queued_bytes{};
  double disk_enqueue_ms{};
  cache::ResourceSnapshot ledger;
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
  // These are contemporaneous global gauges, not request-local attribution.
  result["global_ledger_bytes"] = value.ledger.total_bytes;
  result["global_ram_bytes"] = value.ledger.ram_bytes;
  result["global_persistence_pinned_bytes"] =
      value.ledger.persistence_pinned_bytes;
  auto categories = json::Value::object();
  constexpr const char* names[]{"backing_free",         "backing_assigned",
                                "backing_materialized", "private_state",
                                "private_tail",         "metadata",
                                "transfer_staging"};
  for (std::size_t i = 0; i < cache::kResourceCategoryCount; ++i) {
    auto gauge = json::Value::object();
    gauge["bytes"] = value.ledger.bytes[i];
    gauge["reserved_bytes"] = value.ledger.reserved_bytes[i];
    categories[names[i]] = std::move(gauge);
  }
  result["global_categories"] = std::move(categories);
  return result;
}
}  // namespace gufo::server
#endif

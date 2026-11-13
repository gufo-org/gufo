#ifndef GUFO_CACHE_COST_HPP_
#define GUFO_CACHE_COST_HPP_

#include <cmath>
#include <stdexcept>

#include "src/cache/types.hpp"

namespace gufo::cache {
// Diagnostic calibration for one model/mode on gfx1151. Coefficients are
// seconds; depth is normalized to 100,000 tokens. This does not change policy.
struct CostEstimate {
  double d{}, e{};
  double ram_bytes_per_second{}, disk_bytes_per_second{};
  [[nodiscard]] double PrefillSeconds(Rows first, Rows end) const {
    if (end < first || !std::isfinite(d) || !std::isfinite(e) || d < 0 || e < 0)
      throw std::invalid_argument(
          "invalid prefill cost interval or calibration");
    return double(end - first) *
           (d + e * (double(end) + double(first)) / 200000.0);
  }
  [[nodiscard]] double RestoreSeconds(std::size_t bytes, bool disk) const {
    const auto rate = disk ? disk_bytes_per_second : ram_bytes_per_second;
    if (!std::isfinite(rate) || rate <= 0)
      throw std::invalid_argument("invalid restore cost calibration");
    return double(bytes) / rate;
  }
};
}  // namespace gufo::cache
#endif

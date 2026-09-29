#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, pinned Flash-Next Q4 target/Q8 MTP, 2026-09-20. Milliseconds per
// cohort, including catch-up, recursive proposals and target verification.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary decode).
// Reproduce with qwen38_flash_next_gpu_probe --cost-audit 0. Allocation and C1
// graph capture are warmed outside the measurement; each row is the median
// of three complete cycles. Use --depth N to audit one depth. See benchmarks.
// C1/d0 was remeasured with invariant MTP cache projections for issue #300.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {38.5F, 47.3F, 56.1F, 65.7F, 74.1F, 82.2F, 89.7F, 96.8F},         // C1
        {46.1F, 59.6F, 74.0F, 90.5F, 109.0F, 121.2F, 135.9F, 145.8F},     // C2
        {57.6F, 80.6F, 108.6F, 131.4F, 161.9F, 183.5F, 213.5F, 234.3F},   // C4
        {68.4F, 104.7F, 140.2F, 175.5F, 211.6F, 254.2F, 293.7F, 325.8F},  // C6
        {78.7F, 121.9F, 167.4F, 215.7F, 264.7F, 311.2F, 364.1F, 411.1F},  // C8
    },
    {
        // Context 4096
        {39.7F, 49.3F, 59.8F, 69.7F, 79.1F, 89.1F, 98.6F, 108.8F},        // C1
        {49.1F, 64.3F, 80.5F, 97.1F, 116.6F, 130.9F, 145.3F, 157.2F},     // C2
        {62.3F, 87.7F, 118.6F, 142.4F, 173.0F, 197.9F, 228.4F, 255.4F},   // C4
        {75.1F, 115.4F, 156.5F, 193.1F, 233.5F, 278.9F, 320.1F, 358.1F},  // C6
        {87.9F, 134.3F, 189.7F, 238.6F, 296.0F, 342.8F, 402.1F, 456.0F},  // C8
    },
    {
        // Context 32768
        {42.5F, 51.9F, 61.5F, 71.1F, 82.5F, 92.5F, 103.6F, 112.6F},        // C1
        {54.9F, 69.0F, 83.2F, 99.6F, 120.5F, 135.3F, 151.4F, 163.7F},      // C2
        {73.6F, 96.9F, 125.9F, 150.1F, 185.3F, 212.3F, 243.9F, 266.5F},    // C4
        {92.0F, 128.6F, 167.0F, 206.3F, 249.5F, 298.3F, 339.1F, 370.7F},   // C6
        {110.1F, 152.5F, 201.8F, 256.0F, 321.9F, 376.1F, 432.0F, 490.9F},  // C8
    },
};

// Sampled policy uses fixed configured capacity for reproducibility; greedy
// batch policy uses these curves to bootstrap each physical occupancy.
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = kMtpCycleMilliseconds[interval][cohort][i];
    const auto high = kMtpCycleMilliseconds[interval + 1][cohort][i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal =
          high - kMtpCycleMilliseconds[interval + 1][cohort][i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

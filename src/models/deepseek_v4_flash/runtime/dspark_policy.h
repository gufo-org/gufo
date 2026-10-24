#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_DSPARK_POLICY_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_DSPARK_POLICY_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

// C1 draft + verification + commit in hundredths of one target step.
// Measured costs at d0/d16K are about 145/165/185/210/230 for tails 1..5.
// The linear estimate stays within three units of those measurements.
// Use eight cycles to avoid suspending a productive request after one poor
// four-cycle window. Clearly unprofitable cold starts can still stop at four.
inline bool ds4_dspark_serial_backoff(uint64_t steps, uint64_t drafted,
                                      uint64_t accepted) {
  if (steps < 4u || drafted == 0u)
    return false;
  const uint64_t cost = 125u * steps + 21u * drafted;
  const uint64_t emitted = steps + accepted;
  return cost > emitted * (steps < 8u ? 125u : 110u);
}

// Hundredths of an ordinary batch decode step, measured on gfx1151.
// See docs/models/deepseek-v4-flash/artifacts/cost-calibration.json.
inline uint32_t ds4_dspark_cycle_cost(uint32_t depth, uint32_t tail,
                                      size_t concurrency) {
  using Costs = std::array<uint32_t, 3>;
  static constexpr std::array<Costs, 4> short_context{
      {{165, 185, 215}, {195, 260, 430}, {225, 275, 375}, {255, 330, 445}}};
  static constexpr std::array<Costs, 4> medium_context{
      {{160, 190, 220}, {185, 245, 295}, {215, 275, 360}, {230, 305, 410}}};
  static constexpr std::array<Costs, 4> long_context{
      {{160, 190, 225}, {185, 240, 300}, {210, 270, 370}, {220, 305, 390}}};
  const size_t group = std::min((concurrency - 1u) / 2u, size_t{3});
  const size_t column = std::clamp(tail, 1u, 3u) - 1u;
  const bool deep = depth > 4096u;
  const uint32_t span = deep ? 12288u : 3968u;
  const uint32_t weight =
      deep ? std::min(depth - 4096u, span) : (depth > 128u ? depth - 128u : 0u);
  const uint32_t low = (deep ? medium_context : short_context)[group][column];
  const uint32_t high = (deep ? long_context : medium_context)[group][column];
  return (low * (span - weight) + high * weight) / span;
}

// Inspect the current position's confidence before sampling it. Never drop an
// earlier proposal based on a later token. Remaining positions use the same
// conditional-confidence estimate; observed acceptance still supplies the
// request's existing width/backoff policy.
struct ds4_dspark_confidence_policy {
  uint32_t depth = 0;
  size_t concurrency = 1;
  uint32_t maximum = 3;
  uint32_t included = 0;
  double survival = 1;
  double expected = 1;  // The target-known anchor is always emitted.

  uint32_t Cost(uint32_t width) const {
    return ds4_dspark_cycle_cost(depth, width, concurrency);
  }

  bool Include(float confidence) {
    if (!std::isfinite(confidence) || confidence < 0 || confidence > 1 ||
        included >= maximum)
      return false;
    const uint32_t current_cost = included ? Cost(included) : 100;
    double projected_survival = survival, projected_expected = expected;
    bool profitable = false;
    for (uint32_t width = included + 1; width <= maximum; ++width) {
      projected_survival *= confidence;
      projected_expected += projected_survival;
      profitable |= projected_expected * current_cost > expected * Cost(width);
    }
    if (profitable) {
      survival *= confidence;
      expected += survival;
      ++included;
    }
    return profitable;
  }
};

#endif

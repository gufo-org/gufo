#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_SPLIT_PLAN_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_SPLIT_PLAN_HPP_

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::test {

/// Where a reference session splits a prompt so that each checkpoint it is
/// compared with ends a pass, and every other pass is the one the checked
/// session runs: the ascending union of `boundaries` and the pass ends
/// Session::Feed plans from `start` to `end`. A prefill pass takes up to
/// PrefillCapacity rows (all of them when at most PrefillThroughCapacity
/// remain), less what puts the next pass off a selection block, and ends early
/// at the kMaxPassCheckpoints-th boundary it covers, or, with `in_pass` false
/// (no device room for the checkpoint arena), at the first one. Sparse query
/// groups depend on where a pass starts, so a split only at the boundaries is
/// not the same arithmetic once a prompt spans several passes.
inline std::vector<std::uint32_t> SplitPlan(
    const models::qwen38_flash_next::Model& model, std::uint32_t start,
    std::uint32_t end, std::span<const std::uint32_t> boundaries,
    bool in_pass = true) {
  const std::uint32_t capacity = model.PrefillCapacity();
  const std::uint32_t whole = model.PrefillThroughCapacity();
  const std::uint32_t ratio =
      std::max<std::uint32_t>(1, model.config().compress_ratio);
  const std::size_t per_pass =
      in_pass ? models::qwen38_flash_next::rocm::kMaxPassCheckpoints : 1;
  std::vector<std::uint32_t> splits;
  std::size_t next = 0;
  for (std::uint32_t pos = start; pos < end;) {
    const std::uint32_t remaining = end - pos;
    std::uint32_t n =
        remaining <= whole ? remaining : std::min(capacity, remaining);
    // A following pass starts on a selection block, as in Session::Feed.
    if (const std::uint32_t over = (pos + n) % ratio;
        pos + n < end && over != 0 && n > over)
      n -= over;
    std::size_t last = next;
    while (last < boundaries.size() && boundaries[last] <= pos + n)
      ++last;
    if (last - next > per_pass || (!in_pass && last > next)) {
      last = std::min(last, next + per_pass);
      n = boundaries[last - 1] - pos;
    }
    for (std::size_t i = next; i < last; ++i)
      splits.push_back(boundaries[i]);
    pos += n;
    splits.push_back(pos);
    next = last;
  }
  std::sort(splits.begin(), splits.end());
  splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
  return splits;
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_SPLIT_PLAN_HPP_

#ifndef GUFO_MODELS_QWEN_PREFILL_GROUPS_HPP_
#define GUFO_MODELS_QWEN_PREFILL_GROUPS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::qwen {

/// The estimated time of one prompt pass of a prefill step.
struct PrefillPassTime {
  /// All its layers, spread evenly over them.
  double layers{};
  /// Work after its last layer, such as the draft's context injection,
  /// before the next pass can yield.
  double end{};
};

/// Layers before which a prefill step of consecutive passes yields, one
/// ascending list per pass, so that `groups` layer groups take about equal
/// time. A group may end inside one pass or span the end of one and the start
/// of the next. Layer zero yields before a pass starts; the first pass never
/// yields there.
[[nodiscard]] inline std::vector<std::vector<std::uint32_t>> PrefillYieldLayers(
    std::span<const PrefillPassTime> passes, std::uint32_t layers,
    std::size_t groups) {
  std::vector<std::vector<std::uint32_t>> result(passes.size());
  double total = 0;
  for (const auto& pass : passes)
    total += pass.layers + pass.end;
  if (groups < 2 || layers == 0 || !(total > 0))
    return result;
  std::size_t pass = 0;
  double start = 0;
  for (std::size_t group = 1; group < groups; ++group) {
    const double target =
        total * static_cast<double>(group) / static_cast<double>(groups);
    while (pass < passes.size() &&
           target >= start + passes[pass].layers + passes[pass].end) {
      start += passes[pass].layers + passes[pass].end;
      ++pass;
    }
    if (pass == passes.size())
      break;
    // The nearest yield: before one of this pass's layers, or at the next
    // pass's start, after this pass's end.
    const double unit = passes[pass].layers / layers;
    const double into = target - start;
    const double next = passes[pass].layers + passes[pass].end;
    const auto layer =
        unit > 0 ? std::min<std::uint32_t>(
                       static_cast<std::uint32_t>(std::lround(into / unit)),
                       layers - 1)
                 : 0U;
    std::size_t owner = pass;
    std::uint32_t at = layer;
    if (!(unit > 0) || std::abs(into - layer * unit) > next - into) {
      if (++owner == passes.size())
        continue;
      at = 0;
    }
    if (owner == 0 && at == 0)
      continue;
    auto& list = result[owner];
    if (list.empty() || list.back() < at)
      list.push_back(at);
  }
  return result;
}

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_PREFILL_GROUPS_HPP_

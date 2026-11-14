// Layer groups spread over a prefill step of several passes
// (PrefillYieldLayers): ascending yield layers in range, never before the
// step's first layer, and groups of about equal time.

#include "src/models/qwen/prefill_groups.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using gufo::models::qwen::PrefillPassTime;
using gufo::models::qwen::PrefillYieldLayers;
using Layers = std::vector<std::vector<std::uint32_t>>;
using Passes = std::vector<PrefillPassTime>;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string Name(const Passes& passes, std::size_t groups) {
  std::string name = "passes";
  for (const auto& pass : passes)
    name += " " + std::to_string(pass.layers) + "+" + std::to_string(pass.end);
  return name + ", " + std::to_string(groups) + " groups";
}

// The time between consecutive yields, over the passes in order. A pass's end
// belongs to the group running its last layer.
std::vector<double> GroupTimes(const Passes& passes, std::uint32_t layers,
                               const Layers& yields) {
  std::vector<double> times{0};
  for (std::size_t pass = 0; pass < passes.size(); ++pass) {
    std::size_t next = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      if (next < yields[pass].size() && yields[pass][next] == layer) {
        times.push_back(0);
        ++next;
      }
      times.back() += passes[pass].layers / layers;
    }
    times.back() += passes[pass].end;
  }
  return times;
}

void CheckShape(const Passes& passes, std::uint32_t layers,
                std::size_t groups) {
  const auto name = Name(passes, groups);
  const auto yields = PrefillYieldLayers(passes, layers, groups);
  Require(yields.size() == passes.size(), name + ": one list per pass");
  std::size_t count = 0;
  for (std::size_t pass = 0; pass < yields.size(); ++pass) {
    const auto& list = yields[pass];
    Require(std::ranges::adjacent_find(list, std::ranges::greater_equal{}) ==
                list.end(),
            name + ": layers must ascend");
    Require(list.empty() || list.back() < layers, name + ": layer in range");
    Require(pass != 0 || list.empty() || list.front() != 0,
            name + ": no yield before the step");
    count += list.size();
  }
  Require(count < std::max<std::size_t>(groups, 1),
          name + ": more yields than groups");
  double total = 0, step = 0;
  for (const auto& pass : passes) {
    total += pass.layers + pass.end;
    // The widest stretch without a yield point: a layer, or the last layer
    // and the pass's end.
    step = std::max(step, pass.layers / layers + pass.end);
  }
  // Every group boundary lands on the yield point nearest its share.
  const auto share =
      total / static_cast<double>(std::max<std::size_t>(groups, 1));
  for (const auto time : GroupTimes(passes, layers, yields))
    Require(time <= share + step * 1.000001,
            name + ": a group exceeds its share by more than a stretch");
}

}  // namespace

int main() {
  try {
    // The issue's prompt, priced as the runner prices it on UD-Q4_K_XL: a
    // 2,048-row FP16 pass, the 505 Q8_1 rows up to the stable boundary and
    // the five-row tail. The short passes cost more per row, so the fifth
    // group takes part of the 505-row pass, not all of it plus the tail.
    Require(PrefillYieldLayers(Passes{{3806.4, 0}, {1256.25, 0}, {131.25, 0}},
                               64, 5) == Layers{{17, 35, 52}, {18}, {}},
            "2048 + 505 + 5 rows in five groups");
    // DFlash2 injects each pass's rows into its draft after the pass; the
    // groups before that end shrink to make room for it.
    Require(PrefillYieldLayers(
                Passes{{3806.4, 245.76}, {1256.25, 60.6}, {131.25, 0.6}}, 64,
                5) == Layers{{18, 37, 55}, {18}, {}},
            "2048 + 505 + 5 rows with draft injection in five groups");
    // One pass of 64 layers in sixteen groups: a yield every four layers.
    Layers single{{}};
    for (std::uint32_t layer = 4; layer < 64; layer += 4)
      single.front().push_back(layer);
    Require(PrefillYieldLayers(Passes{{1, 0}}, 64, 16) == single,
            "one pass in sixteen groups");
    Require(PrefillYieldLayers(Passes{{3806.4, 0}, {1268, 0}}, 64, 1) ==
                Layers{{}, {}},
            "one group never yields");
    Require(PrefillYieldLayers(Passes{{0, 0}, {0, 0}}, 64, 4) == Layers{{}, {}},
            "a step without cost never yields");
    for (const auto& passes : std::vector<Passes>{
             {{131.25, 0.6}, {3797.4, 245.2}, {1256.25, 60.6}, {131.25, 0.6}},
             {{1, 0}, {1, 0}},
             {{7492.8, 0}, {122.25, 0}},
             {{122.25, 0}, {7492.8, 0}},
             {{122.25, 500}, {7492.8, 0}},
             {{1470, 72}, {1470, 72}, {1470, 72}, {1470, 72}, {3172.8, 203}},
             {{3806.4, 245.76}, {3806.4, 245.76}}}) {
      for (std::size_t groups = 0; groups <= 16; ++groups)
        CheckShape(passes, 64, groups);
    }
    std::cout << "Qwen27B prefill group checks passed.\n";
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << "qwen27b_prefill_groups_test: " << failure.what() << '\n';
    return 1;
  }
}

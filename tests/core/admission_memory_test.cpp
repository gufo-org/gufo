#include "src/core/hip/admission_memory.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>

namespace {

void Expect(bool condition, const char* message) {
  if (condition)
    return;
  std::cerr << message << '\n';
  std::abort();
}

}  // namespace

int main() {
  using gufo::hip::AdmissionFreeBytes;
  using gufo::hip::DxgiNodeFromMask;
  using gufo::platform::GpuLocalBudget;
  constexpr std::uint64_t kGiB = 1ull << 30;
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  // The 4 GiB the snapshot cache leaves the OS, plus 2.5 GiB.
  constexpr std::uint64_t kMargin = 6 * kGiB + kGiB / 2;
  // 23 GiB of budget left, as measured when a 262144-token Flash-Next
  // session was refused on a Strix Halo.
  constexpr GpuLocalBudget kRoomy{.budget_bytes = 104 * kGiB,
                                  .usage_bytes = 81 * kGiB};

  // No budget reading (Linux, a discrete GPU, a failed query): HIP stands.
  Expect(AdmissionFreeBytes(8 * kGiB, std::nullopt, 40 * kGiB, kMargin) ==
             8 * kGiB,
         "without a budget HIP's free bytes stand");

  // No RAM reading: the budget alone is not trusted.
  Expect(
      AdmissionFreeBytes(8 * kGiB, kRoomy, std::nullopt, kMargin) == 8 * kGiB,
      "without a RAM reading HIP's free bytes stand");

  // A budget at or under its usage leaves nothing, and must not wrap around.
  constexpr GpuLocalBudget kOver{.budget_bytes = 80 * kGiB,
                                 .usage_bytes = 90 * kGiB};
  constexpr GpuLocalBudget kSpent{.budget_bytes = 90 * kGiB,
                                  .usage_bytes = 90 * kGiB};
  Expect(AdmissionFreeBytes(8 * kGiB, kOver, 40 * kGiB, kMargin) == 8 * kGiB,
         "a budget under its usage adds nothing");
  Expect(AdmissionFreeBytes(8 * kGiB, kSpent, 40 * kGiB, kMargin) == 8 * kGiB,
         "a budget equal to its usage adds nothing");

  // RAM at or under the margin leaves nothing, and must not wrap around.
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, kMargin - 1, kMargin) == 8 * kGiB,
         "RAM one byte under the margin adds nothing");
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, kMargin, kMargin) == 8 * kGiB,
         "RAM equal to the margin adds nothing");
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, 0, kMargin) == 8 * kGiB,
         "no RAM at all adds nothing");

  // A model whose weights fit the HIP heap: HIP's larger figure is kept,
  // 72 GiB against 69 - 6.5 GiB of RAM.
  Expect(AdmissionFreeBytes(72 * kGiB,
                            GpuLocalBudget{.budget_bytes = 107 * kGiB,
                                           .usage_bytes = 17 * kGiB},
                            69 * kGiB, kMargin) == 72 * kGiB,
         "HIP's larger figure is kept");

  // Weights that left HIP almost nothing: the host RAM above the margin
  // bounds the 23 GiB of budget left.
  Expect(AdmissionFreeBytes(7 * kGiB, kRoomy, 14 * kGiB, kMargin) ==
             14 * kGiB - kMargin,
         "the RAM above the margin bounds the budget left");
  // The budget left binds when it is the smaller of the two.
  Expect(AdmissionFreeBytes(1 * kGiB,
                            GpuLocalBudget{.budget_bytes = 100 * kGiB,
                                           .usage_bytes = 95 * kGiB},
                            40 * kGiB, kMargin) == 5 * kGiB,
         "the budget left binds when it is the smaller bound");

  // The RAM bound reaches HIP's figure when available RAM is the margin plus
  // that figure.
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, kMargin + 8 * kGiB - 1,
                            kMargin) == 8 * kGiB,
         "RAM one byte under the threshold keeps HIP's figure");
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, kMargin + 8 * kGiB, kMargin) ==
             8 * kGiB,
         "RAM at the threshold gives HIP's figure");
  Expect(AdmissionFreeBytes(8 * kGiB, kRoomy, kMargin + 8 * kGiB + 1,
                            kMargin) == 8 * kGiB + 1,
         "RAM one byte over the threshold raises the result by one byte");

  // DXGI node: one set bit names it.
  Expect(DxgiNodeFromMask(1, "gfx1100") == 0u, "bit 0 is node 0");
  Expect(DxgiNodeFromMask(4, "gfx1100") == 2u, "bit 2 is node 2");
  Expect(DxgiNodeFromMask(1u << 31, "") == 31u, "bit 31 is node 31");
  Expect(DxgiNodeFromMask(2, "gfx1151") == 1u, "the mask wins over the arch");
  // Several set bits name no single node.
  Expect(!DxgiNodeFromMask(3, "gfx1151").has_value(),
         "several bits name no node, even on gfx1151");
  Expect(!DxgiNodeFromMask((1u << 31) | 1u, "gfx1100").has_value(),
         "the highest and lowest bits together name no node");
  // A zero mask is node 0 on Strix Halo only.
  Expect(DxgiNodeFromMask(0, "gfx1151") == 0u,
         "a zero mask is node 0 on gfx1151");
  Expect(DxgiNodeFromMask(0, "gfx1151:sramecc+:xnack-") == 0u,
         "feature suffixes do not change the arch");
  Expect(!DxgiNodeFromMask(0, "gfx1100").has_value(),
         "a zero mask on another arch names no node");
  Expect(!DxgiNodeFromMask(0, "gfx115").has_value(),
         "a zero mask on a shorter arch name names no node");
  Expect(!DxgiNodeFromMask(0, "").has_value(),
         "a zero mask with no arch name names no node");

  // Whatever the readings, the result is never below HIP's figure and never
  // above the larger of that figure and either bound.
  const std::uint64_t hip_values[] = {0, 1, 8 * kGiB, 72 * kGiB, kMax};
  const std::optional<GpuLocalBudget> locals[] = {
      std::nullopt,
      GpuLocalBudget{},
      kRoomy,
      kOver,
      GpuLocalBudget{.budget_bytes = kMax, .usage_bytes = 0},
      GpuLocalBudget{.budget_bytes = kMax, .usage_bytes = kMax},
  };
  const std::optional<std::uint64_t> hosts[] = {
      std::nullopt, std::uint64_t{0}, kMargin, 14 * kGiB, 70 * kGiB, kMax,
  };
  const std::uint64_t margins[] = {0, kMargin, kMax};
  for (const std::uint64_t hip : hip_values) {
    for (const auto& local : locals) {
      for (const auto& host : hosts) {
        for (const std::uint64_t margin : margins) {
          const std::uint64_t result =
              AdmissionFreeBytes(hip, local, host, margin);
          Expect(result >= hip, "the result is never below HIP's free bytes");
          if (!local.has_value() || !host.has_value()) {
            Expect(result == hip, "a missing reading leaves HIP's figure");
            continue;
          }
          const std::uint64_t budget_left =
              local->budget_bytes > local->usage_bytes
                  ? local->budget_bytes - local->usage_bytes
                  : 0;
          const std::uint64_t host_left = *host > margin ? *host - margin : 0;
          Expect(result <= std::max(hip, budget_left),
                 "the result never exceeds the budget left");
          Expect(result <= std::max(hip, host_left),
                 "the result never exceeds the RAM above the margin");
        }
      }
    }
  }

  std::cout << "Admission memory checks passed.\n";
}

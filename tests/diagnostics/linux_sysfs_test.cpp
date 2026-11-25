#include "src/core/diagnostics/linux_sysfs.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/platform/os_info.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void Write(const std::filesystem::path& path, std::string_view content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

}  // namespace

int main() {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("gufo-sysfs-" + std::to_string(gufo::platform::ProcessId()));
  std::filesystem::remove_all(root);
  const auto sys = root / "sys";
  const auto nodes = sys / "class" / "kfd" / "kfd" / "topology" / "nodes";

  // Strix Halo layout: a CPU node without SIMDs and the gfx1151 node at
  // 0000:c5:00.0 (location_id 0xc500) with 80 SIMDs, two per CU.
  Write(nodes / "0" / "properties",
        "cpu_cores_count 32\nsimd_count 0\nlocation_id 0\ndomain 0\n");
  Write(nodes / "1" / "properties",
        "cpu_cores_count 0\nsimd_count 80\nsimd_per_cu 2\n"
        "location_id 50432\ndomain 0\n");
  const auto card = sys / "class" / "drm" / "card1" / "device";
  Write(card / "uevent",
        "DRIVER=amdgpu\nPCI_ID=1002:1586\nPCI_SLOT_NAME=0000:c5:00.0\n");

  const gufo::diagnostics::LinuxSysfs sysfs(sys, root / "proc");
  Expect(sysfs.QueryKfdComputeUnits("0000:c5:00.0") == 40u,
         "KFD node at the GPU's PCI slot reports 40 CUs");
  Expect(!sysfs.QueryKfdComputeUnits("0000:c6:00.0"),
         "no KFD node matches another PCI slot");
  Expect(!sysfs.QueryKfdComputeUnits("0001:c5:00.0"),
         "no KFD node matches another PCI domain");
  Expect(!sysfs.QueryKfdComputeUnits("not-a-slot"),
         "a malformed PCI slot matches nothing");

  const auto gpus = sysfs.QueryDrmGpuDevices();
  Expect(gpus.size() == 1 && gpus[0].compute_units == 40,
         "the DRM card carries its KFD compute units");

  Expect(!gpus[0].gtt && !sysfs.QueryGtt(),
         "a card without GTT counters reports no GTT");

  // The amdgpu default GTT on a 128 GB Strix Halo (62.5 GiB), and a card
  // without counters listed first that QueryGtt skips.
  Write(card / "mem_info_gtt_total", "67147419648\n");
  Write(card / "mem_info_gtt_used", "19386368\n");
  Write(sys / "class" / "drm" / "card0" / "device" / "uevent",
        "DRIVER=simpledrm\n");
  const auto gtt = sysfs.QueryGtt();
  Expect(gtt && gtt->total_bytes == 67147419648ull &&
             gtt->used_bytes == 19386368ull,
         "QueryGtt reads the first card with GTT counters");
  bool card_gtt = false;
  for (const auto& gpu : sysfs.QueryDrmGpuDevices())
    card_gtt |= gpu.card_name == "card1" && gpu.gtt &&
                gpu.gtt->total_bytes == 67147419648ull;
  Expect(card_gtt, "the DRM card carries its GTT counters");
  Write(card / "mem_info_gtt_used", "not-a-number\n");
  Expect(!sysfs.QueryGtt(), "unparsable GTT counters report no GTT");

  // MemTotal 128073 MiB: the recommended GTT is 123977 MiB.
  constexpr std::uint64_t kMiB = 1ull << 20;
  constexpr std::uint64_t ram = 128073 * kMiB;
  static_assert(gufo::diagnostics::RecommendedGttBytes(ram) == 123977 * kMiB);
  static_assert(gufo::diagnostics::GttBelowRecommended(67147419648ull, ram));
  static_assert(!gufo::diagnostics::GttBelowRecommended(123977 * kMiB, ram));
  static_assert(!gufo::diagnostics::GttBelowRecommended(126976 * kMiB, ram));
  static_assert(!gufo::diagnostics::GttBelowRecommended(
      123977 * kMiB - 1000 * kMiB, ram));
  static_assert(gufo::diagnostics::RecommendedGttBytes(1ull << 30) == 0);

  std::filesystem::remove_all(root);
  std::cout << "linux_sysfs_test: all checks passed\n";
  return 0;
}

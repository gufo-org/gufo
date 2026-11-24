#include "src/ext/host_budget.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "src/core/platform/os_info.hpp"
#include "src/mem/ledger.hpp"

namespace gufo::ext {

std::uint64_t CgroupAvailableBytes(const std::filesystem::path& root,
                                   std::string_view membership_text,
                                   std::uint64_t host_available) {
  std::uint64_t available = host_available;
  // A cgroup limit can be much smaller than the host's available memory.
  // Walk parents too: a child may say "max" beneath a limited ancestor.
  std::istringstream membership{std::string(membership_text)};
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used)) {
        // memory.current includes page cache the kernel reclaims under
        // pressure. Count inactive file pages as free; the active ones may
        // still be in use, so they stay charged. Dirty and writeback pages
        // cannot be dropped until written, so they stay charged too. A
        // malformed memory.stat is ignored.
        std::ifstream stat_file(path / "memory.stat");
        std::uint64_t inactive_file = 0, unreclaimable = 0;
        bool found = false, valid = true;
        for (std::string key; stat_file >> key;) {
          std::uint64_t value = 0;
          if (!(stat_file >> value)) {
            valid = false;
            break;
          }
          if (key == "inactive_file") {
            inactive_file = value;
            found = true;
          } else if (key == "file_dirty" || key == "file_writeback") {
            unreclaimable += value;
          }
        }
        if (found && valid)
          used -= std::min(
              used, inactive_file - std::min(inactive_file, unreclaimable));
        available = std::min(available, limit > used ? limit - used : 0);
      }
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

namespace {

struct MeminfoKiB {
  std::optional<std::uint64_t> available;
  std::uint64_t cma_free = 0;
};

MeminfoKiB ParseMeminfoKiB(std::string_view meminfo) {
  MeminfoKiB result;
  std::istringstream lines{std::string(meminfo)};
  for (std::string line; std::getline(lines, line);) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0;
    if (!(fields >> key >> kib))
      continue;
    if (key == "MemAvailable:")
      result.available = kib;
    else if (key == "CmaFree:")
      result.cma_free = kib;
  }
  return result;
}

std::uint64_t KiBToBytes(std::uint64_t kib) {
  constexpr auto max_bytes = std::numeric_limits<std::uint64_t>::max();
  return kib > max_bytes / 1024 ? max_bytes : kib * 1024;
}

std::uint64_t SaturatingSubtract(std::uint64_t value, std::uint64_t amount) {
  return value > amount ? value - amount : 0;
}

}  // namespace

std::optional<std::uint64_t> MeminfoAvailableBytes(std::string_view meminfo) {
  const auto kib = ParseMeminfoKiB(meminfo);
  if (!kib.available)
    return std::nullopt;
  // Both fields can exceed the byte range while their difference still fits.
  return KiBToBytes(SaturatingSubtract(*kib.available, kib.cma_free));
}

std::optional<HostMemorySample> MeminfoSample(std::string_view meminfo) {
  const auto kib = ParseMeminfoKiB(meminfo);
  if (!kib.available)
    return std::nullopt;
  return HostMemorySample{.available = KiBToBytes(*kib.available),
                          .cma_free = KiBToBytes(kib.cma_free)};
}

HostMemorySample SampleHostMemory() {
  HostMemorySample sample{.available = platform::AvailableMemoryBytes()};
  std::ifstream meminfo("/proc/meminfo");
  std::ostringstream meminfo_text;
  meminfo_text << meminfo.rdbuf();
  if (const auto parsed = MeminfoSample(meminfo_text.str()))
    sample = *parsed;
  std::ifstream membership("/proc/self/cgroup");
  std::ostringstream membership_text;
  membership_text << membership.rdbuf();
  sample.cgroup_available =
      CgroupAvailableBytes("/sys/fs/cgroup", membership_text.str(),
                           std::numeric_limits<std::uint64_t>::max());
  sample.outstanding = mem::StateLedger::Process().OutstandingBytes();
  return sample;
}

namespace {

std::size_t ClampToSize(std::uint64_t bytes) {
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(bytes, std::numeric_limits<std::size_t>::max()));
}

}  // namespace

std::uint64_t PinnableBytes(const HostMemorySample& sample) {
  return SaturatingSubtract(
      std::min(SaturatingSubtract(sample.available, sample.cma_free),
               sample.cgroup_available),
      sample.outstanding);
}

std::uint64_t SnapshotBudgetBytes(const HostMemorySample& sample) {
  return PinnableBytes(sample) / 2;
}

std::uint64_t SnapshotCeilingBytes(const HostMemorySample& sample) {
  const auto host = SaturatingSubtract(
      sample.available, std::max(sample.cma_free, kHostSnapshotHeadroomBytes));
  const auto cgroup =
      SaturatingSubtract(sample.cgroup_available, kHostSnapshotHeadroomBytes);
  return SaturatingSubtract(std::min(host, cgroup), sample.outstanding);
}

std::uint64_t HostAvailableBytes() {
  return PinnableBytes(SampleHostMemory());
}

std::uint64_t HostReserveShortfallBytes(const HostMemorySample& sample) {
  return SaturatingSubtract(
      std::max(sample.cma_free, kHostSnapshotHeadroomBytes), sample.available);
}

std::string DescribeHostMemory(const HostMemorySample& sample) {
  std::string text =
      "host_available_bytes=" + std::to_string(sample.available) +
      " cma_free_bytes=" + std::to_string(sample.cma_free);
  if (sample.cgroup_available != std::numeric_limits<std::uint64_t>::max())
    text +=
        " cgroup_available_bytes=" + std::to_string(sample.cgroup_available);
  return text +
         " outstanding_state_bytes=" + std::to_string(sample.outstanding) +
         " host_reserve_bytes=" + std::to_string(kHostSnapshotHeadroomBytes) +
         " host_ceiling_bytes=" + std::to_string(SnapshotCeilingBytes(sample));
}

std::size_t HostSnapshotBudgetBytes() {
  return ClampToSize(SnapshotBudgetBytes(SampleHostMemory()));
}

std::size_t HostSnapshotCeilingBytes() {
  return ClampToSize(SnapshotCeilingBytes(SampleHostMemory()));
}
}  // namespace gufo::ext

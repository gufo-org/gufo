#include "src/ext/host_budget.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "src/core/platform/os_info.hpp"

namespace gufo::ext {
namespace {

struct MeminfoMemory {
  std::uint64_t available_bytes;
  std::uint64_t cma_free_bytes;
};

std::optional<MeminfoMemory> ReadMeminfo(std::string_view meminfo) {
  std::optional<std::uint64_t> available_kib;
  std::uint64_t cma_free_kib = 0;
  std::istringstream lines{std::string(meminfo)};
  for (std::string line; std::getline(lines, line);) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0;
    if (!(fields >> key >> kib))
      continue;
    if (key == "MemAvailable:")
      available_kib = kib;
    else if (key == "CmaFree:")
      cma_free_kib = kib;
  }
  if (!available_kib)
    return std::nullopt;
  if (*available_kib > std::numeric_limits<std::uint64_t>::max() / 1024)
    return std::nullopt;
  return MeminfoMemory{
      *available_kib * 1024,
      cma_free_kib > std::numeric_limits<std::uint64_t>::max() / 1024
          ? std::numeric_limits<std::uint64_t>::max()
          : cma_free_kib * 1024};
}

std::optional<std::uint64_t> CgroupHeadroomBytes(
    const std::filesystem::path& root, std::string_view membership_text) {
  std::optional<std::uint64_t> available;
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
        const auto headroom = limit > used ? limit - used : 0;
        available = available ? std::min(*available, headroom) : headroom;
      }
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

std::size_t ClampToSize(std::uint64_t bytes) {
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(bytes, std::numeric_limits<std::size_t>::max()));
}

}  // namespace

std::optional<std::uint64_t> MeminfoAvailableBytes(std::string_view meminfo) {
  const auto parsed = ReadMeminfo(meminfo);
  if (!parsed)
    return std::nullopt;
  return parsed->available_bytes -
         std::min(parsed->available_bytes, parsed->cma_free_bytes);
}

std::uint64_t CgroupAvailableBytes(const std::filesystem::path& root,
                                   std::string_view membership_text,
                                   std::uint64_t host_available) {
  const auto headroom = CgroupHeadroomBytes(root, membership_text);
  return headroom ? std::min(host_available, *headroom) : host_available;
}

HostSnapshotMemory PlanHostSnapshotMemory(
    std::uint64_t raw_available_bytes, std::uint64_t cma_free_bytes,
    std::optional<std::uint64_t> cgroup_headroom_bytes) {
  HostSnapshotMemory memory;
  memory.raw_available_bytes = raw_available_bytes;
  memory.cma_free_bytes = cma_free_bytes;
  memory.cma_excluded_bytes = std::min(raw_available_bytes, cma_free_bytes);
  memory.cgroup_headroom_bytes = cgroup_headroom_bytes;
  memory.available_bytes = raw_available_bytes - memory.cma_excluded_bytes;
  memory.cgroup_limited =
      cgroup_headroom_bytes && *cgroup_headroom_bytes < memory.available_bytes;
  if (memory.cgroup_limited)
    memory.available_bytes = *cgroup_headroom_bytes;
  memory.ceiling_bytes =
      ClampToSize(memory.available_bytes > memory.reserve_bytes
                      ? memory.available_bytes - memory.reserve_bytes
                      : 0);
  memory.automatic_budget_bytes = ClampToSize(memory.available_bytes / 2);
  return memory;
}

/// Host RAM still available, after cgroup limits. Unified-memory device
/// allocations draw from the same RAM, so HIP's device capacity is not used.
HostSnapshotMemory QueryHostSnapshotMemory() {
  std::ifstream meminfo("/proc/meminfo");
  std::ostringstream meminfo_text;
  meminfo_text << meminfo.rdbuf();
  const auto parsed = ReadMeminfo(meminfo_text.str());
  std::ifstream membership("/proc/self/cgroup");
  std::ostringstream membership_text;
  membership_text << membership.rdbuf();
  auto memory = PlanHostSnapshotMemory(
      parsed ? parsed->available_bytes : platform::AvailableMemoryBytes(),
      parsed ? parsed->cma_free_bytes : 0,
      CgroupHeadroomBytes("/sys/fs/cgroup", membership_text.str()));
  memory.from_meminfo = parsed.has_value();
  return memory;
}

std::size_t HostSnapshotBudgetBytes() {
  return QueryHostSnapshotMemory().automatic_budget_bytes;
}

std::size_t HostSnapshotCeilingBytes() {
  return QueryHostSnapshotMemory().ceiling_bytes;
}
}  // namespace gufo::ext

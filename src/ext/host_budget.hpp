#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace gufo::ext {

/// MemAvailable from /proc/meminfo text minus CmaFree, in bytes. Free CMA
/// pages count as available but only hold movable pages, not GPU
/// allocations; the kernel's KHO scratch area can make that several GiB.
/// Nullopt when MemAvailable is missing.
[[nodiscard]] std::optional<std::uint64_t> MeminfoAvailableBytes(
    std::string_view meminfo);
/// `host_available` bounded by every cgroup v2 memory limit from the process's
/// cgroup (named in `membership_text`, the /proc/self/cgroup contents) up to
/// `root`. Inactive file pages in memory.stat are reclaimable page cache and
/// are not counted as used; without that line, memory.current is used as is.
[[nodiscard]] std::uint64_t CgroupAvailableBytes(
    const std::filesystem::path& root, std::string_view membership_text,
    std::uint64_t host_available);
/// Automatic snapshot budget: half the host RAM available after loading
/// (excluding free CMA pages), after cgroup limits.
[[nodiscard]] std::size_t HostSnapshotBudgetBytes();
/// RAM always left to the OS and other processes by an explicit cache limit.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;
/// One host-memory observation and its unchanged cache/admission policy.
/// Availability already reflects committed allocations; reservations not yet
/// physically allocated must be subtracted separately by their owner.
struct HostSnapshotMemory {
  std::uint64_t raw_available_bytes{0};
  std::uint64_t cma_free_bytes{0};
  std::uint64_t cma_excluded_bytes{0};
  /// Minimum finite cgroup headroom, before bounding it by host availability.
  /// Missing when no applicable finite cgroup limit could be read.
  std::optional<std::uint64_t> cgroup_headroom_bytes;
  std::uint64_t available_bytes{0};
  std::uint64_t reserve_bytes{kHostSnapshotHeadroomBytes};
  std::size_t ceiling_bytes{0};
  std::size_t automatic_budget_bytes{0};
  /// A tie leaves the host as limiter; the cgroup bound is still reported.
  bool cgroup_limited{false};
  /// False means the platform's AvailableMemoryBytes() fallback was used.
  bool from_meminfo{false};
};

/// Pure calculation shared by the live query and deterministic CPU checks.
/// Applies CMA exclusion, cgroup headroom and the existing 4 GiB reserve once.
[[nodiscard]] HostSnapshotMemory PlanHostSnapshotMemory(
    std::uint64_t raw_available_bytes, std::uint64_t cma_free_bytes,
    std::optional<std::uint64_t> cgroup_headroom_bytes);
/// Collects meminfo and cgroup inputs once, then calculates both cache limits.
/// The fields belong to one sampling pass, not an atomic kernel snapshot.
[[nodiscard]] HostSnapshotMemory QueryHostSnapshotMemory();

/// Most an explicit --cache-ram-bytes may claim: available host RAM minus
/// kHostSnapshotHeadroomBytes.
[[nodiscard]] std::size_t HostSnapshotCeilingBytes();

}  // namespace gufo::ext

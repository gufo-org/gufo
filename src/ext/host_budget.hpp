#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
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
/// Host RAM still available (MemAvailable minus free CMA, or the
/// platform's figure), after cgroup limits and less the state the host has
/// promised to sessions but not committed: PinnableBytes(SampleHostMemory()).
/// Unified-memory device allocations draw from the same RAM, so HIP's device
/// capacity is not used.
[[nodiscard]] std::uint64_t HostAvailableBytes();
/// RAM always left to the OS and other processes by an explicit cache limit.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;

/// The host memory figures the snapshot budgets derive from, in bytes.
struct HostMemorySample {
  /// MemAvailable, or the platform's available figure without /proc/meminfo.
  std::uint64_t available{};
  /// CmaFree: free CMA pages, KHO scratch included. MemAvailable counts them,
  /// but they hold only movable pages (page cache, anonymous memory), never
  /// pinned or GPU memory.
  std::uint64_t cma_free{};
  /// Room left under the process's cgroup v2 limits; the maximum without one.
  std::uint64_t cgroup_available{std::numeric_limits<std::uint64_t>::max()};
  /// State promised to sessions but not yet committed (mem::LessOutstanding).
  std::uint64_t outstanding{};
};

/// MemAvailable and CmaFree from /proc/meminfo text; nullopt when MemAvailable
/// is missing. The cgroup and outstanding fields keep their defaults.
[[nodiscard]] std::optional<HostMemorySample> MeminfoSample(
    std::string_view meminfo);
/// This host's sample now: /proc/meminfo, /proc/self/cgroup and the state
/// ledger's outstanding promises.
[[nodiscard]] HostMemorySample SampleHostMemory();
/// RAM that pinned and GPU memory may still take: available less free CMA
/// pages, within the cgroup limit, less outstanding state.
[[nodiscard]] std::uint64_t PinnableBytes(const HostMemorySample& sample);
/// Automatic snapshot budget: half of PinnableBytes().
[[nodiscard]] std::uint64_t SnapshotBudgetBytes(const HostMemorySample& sample);
/// Most an explicit cache limit may pin: available less the larger of free
/// CMA pages and kHostSnapshotHeadroomBytes, within the cgroup limit less the
/// headroom, less outstanding state. The reserve and free CMA pages overlap:
/// the OS and other processes keep their headroom in pages pins cannot use,
/// so it is not subtracted twice (#1198). Pins still never count CMA pages.
[[nodiscard]] std::uint64_t SnapshotCeilingBytes(
    const HostMemorySample& sample);
/// How far the host's available memory is below its reserve:
/// max(CmaFree, kHostSnapshotHeadroomBytes) minus MemAvailable, or 0. Memory
/// actually in use, not state promised to sessions: a page the RAM cache
/// borrowed from that promise and gives back raises the promise as much as
/// MemAvailable, so only real memory measures what a release achieves. Gufo's
/// own growth stops well above it (mem::HostFreeFloorBytes), so a shortfall
/// means memory another process took (#932). The cgroup limit is not used:
/// state pages are not charged to it.
[[nodiscard]] std::uint64_t HostReserveShortfallBytes(
    const HostMemorySample& sample);
/// The sample and the ceiling it implies, as space-separated key=value pairs
/// for refusal logs.
[[nodiscard]] std::string DescribeHostMemory(const HostMemorySample& sample);

/// Automatic snapshot budget for this host: SnapshotBudgetBytes() of a fresh
/// sample.
[[nodiscard]] std::size_t HostSnapshotBudgetBytes();
/// Most an explicit --cache-ram-bytes may claim on this host:
/// SnapshotCeilingBytes() of a fresh sample.
[[nodiscard]] std::size_t HostSnapshotCeilingBytes();

}  // namespace gufo::ext

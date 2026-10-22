#ifndef GUFO_CORE_PLATFORM_OS_INFO_HPP_
#define GUFO_CORE_PLATFORM_OS_INFO_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::platform {

/// OS page size in bytes; never zero.
std::size_t PageSizeBytes();

/// Best-effort physically available memory in bytes. Linux reports
/// sysconf(_SC_AVPHYS_PAGES) times the page size; Windows reports
/// GlobalMemoryStatusEx available physical memory. Callers layer their own
/// refinements (MemAvailable parsing, cgroup limits) on top. Returns 0
/// when the OS cannot answer.
std::uint64_t AvailableMemoryBytes();

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_OS_INFO_HPP_

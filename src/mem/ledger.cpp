#include "src/mem/ledger.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "src/core/platform/os_info.hpp"

namespace gufo::mem {

namespace {

// `key` minus CmaFree, in bytes: free CMA pages count as free or available
// but hold only movable pages, never GPU memory.
std::optional<std::uint64_t> ParseMeminfo(std::string_view meminfo,
                                          std::string_view wanted) {
  std::optional<std::uint64_t> value_kib;
  std::uint64_t cma_free_kib = 0;
  std::istringstream lines{std::string(meminfo)};
  for (std::string line; std::getline(lines, line);) {
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0;
    if (!(fields >> key >> kib))
      continue;
    if (key == wanted)
      value_kib = kib;
    else if (key == "CmaFree:")
      cma_free_kib = kib;
  }
  if (!value_kib)
    return std::nullopt;
  return (*value_kib > cma_free_kib ? *value_kib - cma_free_kib : 0) * 1024;
}

std::optional<std::uint64_t> ReadMeminfo(std::string_view wanted) {
  std::ifstream file("/proc/meminfo");
  if (!file)
    return std::nullopt;
  std::ostringstream text;
  text << file.rdbuf();
  return ParseMeminfo(text.str(), wanted);
}

}  // namespace

std::optional<std::uint64_t> ParseMeminfoAvailable(std::string_view meminfo) {
  return ParseMeminfo(meminfo, "MemAvailable:");
}

std::optional<std::uint64_t> ParseMeminfoFree(std::string_view meminfo) {
  return ParseMeminfo(meminfo, "MemFree:");
}

std::uint64_t HostFreeBytes() {
  if (const auto free = ReadMeminfo("MemFree:"))
    return *free;
  return HostAvailableBytes();
}

std::optional<std::uint64_t> ParseBuddyinfoPageBlocks(
    std::string_view buddyinfo) {
  // "Node 0, zone   Normal  c0 c1 ... c10": counts of free blocks by order.
  constexpr std::size_t kOrder = 9;  // 2 MiB with 4 KiB pages
  std::istringstream lines{std::string(buddyinfo)};
  std::uint64_t bytes = 0;
  bool any = false;
  for (std::string line; std::getline(lines, line);) {
    const auto zone = line.find("zone");
    if (zone == std::string::npos)
      continue;
    std::istringstream fields(line.substr(zone + 4));
    std::string name;
    fields >> name;
    std::uint64_t count = 0;
    for (std::size_t order = 0; fields >> count; ++order)
      if (order >= kOrder)
        bytes += count * (std::uint64_t{4096} << order);
    any = true;
  }
  if (!any)
    return std::nullopt;
  return bytes;
}

std::uint64_t HostFreeFloorBytes() {
  static const std::uint64_t floor =
      HostFloorBytes() + (std::uint64_t{4} << 30);
  return floor;
}

bool CanCreatePagesWithoutReclaim(std::uint64_t bytes) {
  const std::uint64_t floor = HostFreeFloorBytes();
  // Blocks to spare beyond the request: 512 MiB.
  constexpr std::uint64_t kMargin = std::uint64_t{256} * kPageBytes;
  if (HostFreeBytes() < floor + bytes)
    return false;
  std::ifstream file("/proc/buddyinfo");
  if (!file)
    return true;
  std::ostringstream text;
  text << file.rdbuf();
  const auto blocks = ParseBuddyinfoPageBlocks(text.str());
  return !blocks || *blocks >= bytes + kMargin;
}

std::uint64_t HostAvailableBytes() {
  if (const auto available = ReadMeminfo("MemAvailable:"))
    return *available;
  return platform::AvailableMemoryBytes();
}

std::uint64_t BackgroundFloorBytes(std::uint64_t total_bytes) noexcept {
  constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
  return std::clamp<std::uint64_t>(total_bytes / 16, 4 * kGiB, 8 * kGiB);
}

std::uint64_t HostFloorBytes() {
  const auto memory = platform::QueryHostMemory();
  return BackgroundFloorBytes(memory ? memory->total_bytes : 0);
}

bool HeapCoversHost(std::uint64_t heap_total_bytes,
                    std::uint64_t host_total_bytes) noexcept {
  return host_total_bytes != 0 &&
         heap_total_bytes + BackgroundFloorBytes(host_total_bytes) >=
             host_total_bytes;
}

}  // namespace gufo::mem

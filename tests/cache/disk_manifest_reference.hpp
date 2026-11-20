#ifndef GUFO_TESTS_CACHE_DISK_MANIFEST_REFERENCE_HPP_
#define GUFO_TESTS_CACHE_DISK_MANIFEST_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/cache/disk.hpp"

// The disk manifest codec and CRC-64 as C++ implemented them before the Rust
// port, frozen for comparison.
namespace gufo::cache::reference {
struct Plan {
  std::size_t retained{}, references{};
};
// Both throw std::invalid_argument for a rejected manifest.
[[nodiscard]] std::vector<std::uint8_t> Encode(const DiskManifest&);
[[nodiscard]] DiskManifest Decode(std::span<const std::uint8_t>);
// The metadata a decoded manifest retains, and its payload references.
[[nodiscard]] Plan PlanManifest(std::span<const std::uint8_t>);
// DiskChecksum, and each of its two paths.
[[nodiscard]] std::uint64_t Checksum(std::span<const std::uint8_t>,
                                     std::uint64_t previous = 0);
[[nodiscard]] std::uint64_t ChecksumTableOnly(std::span<const std::uint8_t>,
                                              std::uint64_t previous);
[[nodiscard]] bool CanFold();
[[nodiscard]] std::uint64_t ChecksumFoldOnly(std::span<const std::uint8_t>,
                                             std::uint64_t previous);
}  // namespace gufo::cache::reference
#endif

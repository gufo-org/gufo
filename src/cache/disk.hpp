#ifndef GUFO_CACHE_DISK_HPP_
#define GUFO_CACHE_DISK_HPP_

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/cache/checkpoint.hpp"

namespace gufo::cache {
inline constexpr std::uint32_t kDiskFormatVersion = 2;
inline constexpr std::size_t kMaxManifestBytes = std::size_t{16} * 1024 * 1024;
using DiskFileId = std::array<std::uint8_t, 16>;
// SHA-256 of the adapter's canonical compatibility identity, supplied by the
// caller. Display names never participate in the persistent compatibility key.
using CompatibilityDigest = std::array<std::uint8_t, 32>;

// CRC-64/ECMA-182, initial value zero, no reflection or final xor. Incremental
// so card 12 can verify arbitrarily large payloads using bounded buffers.
[[nodiscard]] std::uint64_t DiskChecksum(std::span<const std::uint8_t>,
                                         std::uint64_t previous = 0);
struct DiskPayload {
  DiskFileId file{};
  std::uint64_t bytes{}, checksum{};
  bool operator==(const DiskPayload&) const = default;
};
struct DiskComponent {
  ComponentDescriptor descriptor;
  ComponentPosition position;
  std::vector<DiskPayload> chunks;
  std::optional<DiskPayload> tail, private_state;
};
struct DiskManifest {
  CheckpointId checkpoint;
  LineageId lineage;
  CompatibilityDigest compatibility{};
  Identity input;
  std::vector<Token> tokens;
  std::vector<DiskComponent> components;
  CheckpointPurpose purpose{CheckpointPurpose::kPrompt};
  std::uint32_t rank{};
};
// Explicit little-endian, length-delimited binary encoding with CRC-64 over
// every preceding byte, including the magic/version. No native struct images.
// Both operations reject incoherent geometry and excessive metadata. Encoding
// is a pure metadata operation, not publication (card 11).
[[nodiscard]] std::vector<std::uint8_t> EncodeManifest(const DiskManifest&);
[[nodiscard]] DiskManifest DecodeManifest(std::span<const std::uint8_t>);
[[nodiscard]] std::string DiskFileName(DiskFileId);

class DurableEntry {
public:
  DurableEntry() = default;
  DurableEntry(DurableEntry&&) noexcept = default;
  DurableEntry& operator=(DurableEntry&&) noexcept;
  DurableEntry(const DurableEntry&) = delete;
  DurableEntry& operator=(const DurableEntry&) = delete;
  ~DurableEntry() = default;
  [[nodiscard]] DiskFileId File() const { return file_; }
  [[nodiscard]] const DiskManifest& Manifest() const { return manifest_; }
  // Startup attests metadata and dependency sizes only. Streaming restore must
  // validate payload checksums before making a destination executable.
  [[nodiscard]] bool Durable() const { return durable_; }
  [[nodiscard]] bool PayloadVerified() const { return payload_verified_; }

private:
  friend class DiskStore;
  ResourceCharge metadata_;
  DiskFileId file_{};
  DiskManifest manifest_;
  bool durable_{true};
  bool payload_verified_{false};
};
struct DiskStartupStats {
  std::uint64_t manifest_bytes_read{}, legacy_probe_bytes_read{};
  std::uint64_t dependency_stats{}, rejected_manifests{};
  std::uint64_t managed_bytes{}, removed_legacy_bytes{};
};
// Linux process ownership: LOCK lives at the configured root, covering v2 and
// legacy cleanup together. The lock file is never unlinked. Managed namespaces
// must be real directories; symlinks and nonregular dependencies are rejected.
// Unknown files/directories are never removed or descended into. Startup never
// opens a v2 payload. Retained and transient metadata are admitted through the
// supplied ledger, which must outlive construction. No writes/eviction/lookup
// yet; excess v2 usage fails startup after legacy cleanup, before the caller
// can publish any new bytes.
class DiskStore {
public:
  DiskStore(ResourceLedger&, const std::filesystem::path& directory,
            std::uint64_t budget_bytes);
  ~DiskStore();
  DiskStore(const DiskStore&) = delete;
  DiskStore& operator=(const DiskStore&) = delete;
  DiskStore(DiskStore&&) = delete;
  DiskStore& operator=(DiskStore&&) = delete;
  [[nodiscard]] std::span<const DurableEntry> Entries() const;
  [[nodiscard]] const DiskStartupStats& Stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif

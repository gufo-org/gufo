#ifndef GUFO_CACHE_DISK_HPP_
#define GUFO_CACHE_DISK_HPP_

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stop_token>
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
  std::shared_ptr<const ResourceCharge> pin_;
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
// Host-buffer publication only; caller retains and admits immutable buffers
// until Publish returns. Full chunks already referenced by a checkpoint can be
// omitted.
struct DiskWriteBuffer {
  bool private_file{};
  DiskFileId file{};
  std::span<const std::uint8_t> bytes;
};
// Produces exactly one requested piece, settling device work before return.
// Source ownership and immutability remain the caller's responsibility. New
// payload checksums may be zero in the manifest and are computed during write.
struct DiskWriteSource {
  bool private_file{};
  DiskFileId file{};
  std::uint64_t bytes{};
  std::function<void(std::uint64_t, std::span<std::uint8_t>)> read;
  std::size_t alignment{1};
};
struct DiskReadStats {
  std::uint64_t bytes{}, filesystem_ns{}, checksum_ns{};
  bool verification_cached{};
};
// Optional piece lease. acquire runs before touching staging; release runs on
// every exit, including acquire/read failure, and must not throw. Neither runs
// under index locks.
struct DiskStagingAccess {
  std::function<void()> acquire, release;
};
struct DiskPublicationStats {
  std::uint64_t fsync_calls{}, fsync_ns{};
  std::uint64_t filesystem_ns{}, checksum_ns{}, metadata_lock_ns{};
  std::uint64_t io_lock_ns{};
};
// Retains a checkpoint's dependency pin while the store is alive.
class DiskReadPin {
public:
  explicit operator bool() const { return bool(token_); }

private:
  friend class DiskStore;
  std::shared_ptr<const ResourceCharge> token_;
};
// Owned metadata plus a dependency pin. The store must outlive this snapshot
// and every read using it. It remains stable across concurrent index changes.
class DiskSnapshot {
public:
  DiskSnapshot(DiskSnapshot&&) noexcept = default;
  DiskSnapshot& operator=(DiskSnapshot&&) = delete;
  DiskSnapshot(const DiskSnapshot&) = delete;
  [[nodiscard]] const DiskManifest& Manifest() const { return manifest_; }

private:
  friend class DiskStore;
  DiskSnapshot() = default;
  ResourceCharge metadata_;
  DiskReadPin pin_;
  DiskManifest manifest_;
  const void* store_{};
};
#ifdef GUFO_CACHE_TESTING
enum class DiskPublicationStep {
  kPayloadWritten,
  kPayloadSynced,
  kPayloadRenamed,
  kTemporaryDirectorySynced,
  kChunkDirectorySynced,
  kDependenciesSynced,
  kManifestWritten,
  kManifestSynced,
  kManifestRenamed,
  kManifestDirectorySynced,
  kFinalTemporaryDirectorySynced,
  kIndexed,
};
#endif
// Linux directory ownership; no payload reads or reclamation at startup.
// Mutating operations serialize through one I/O lock (including writers),
// protecting their dependencies from eviction. Pin/Open/Stats use short locks
// without waiting for I/O, streams or optional-write yielding.
// Entries() is a quiescent view:
// do not retain it across publication/retirement or background reclamation.
// Pin() is thread-safe and blocks retirement until all copies are released.
// The supplied ledger must outlive the store and its worker.
// Excess preexisting bytes still fail startup before new writes are admitted.
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
  [[nodiscard]] DiskStartupStats Stats() const;
  [[nodiscard]] DiskPublicationStats PublicationStats() const;
  void Publish(DiskFileId manifest_file, const DiskManifest&,
               std::span<const DiskWriteBuffer>);
  // Reuses one caller-admitted staging piece. No payload-sized allocation.
  [[nodiscard]] DiskPublicationStats PublishStream(
      DiskFileId, const DiskManifest&, std::span<const DiskWriteSource>,
      std::span<std::uint8_t> staging,
      DiskPublicationStats* observation = nullptr,
      const DiskStagingAccess* access = nullptr, std::stop_token stop = {});
  [[nodiscard]] std::optional<DiskSnapshot> Open(CheckpointId) const;
  // Reads one dependency, invokes consume only with complete pieces, and
  // checks its CRC and stable file identity before returning. Consumers must
  // keep their destination non-executable until every dependency has passed.
  [[nodiscard]] DiskReadStats ReadPayload(
      const DiskSnapshot&, bool private_file, const DiskPayload&,
      std::span<std::uint8_t> staging,
      const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>&
          consume,
      DiskReadStats* observation = nullptr);
  [[nodiscard]] DiskReadPin Pin(CheckpointId) const;
  // Returns false for a missing or pinned checkpoint. Unlinks and fsyncs its
  // manifest before releasing any dependency; shared chunks remain referenced.
  bool Retire(CheckpointId);
  // Explicitly schedule a worker after complete startup discovery. Errors are
  // delivered by WaitForReclamation(); destruction joins the worker.
  void ReclaimOrphans();
  void WaitForReclamation();
#ifdef GUFO_CACHE_TESTING
  void SetCrashHook(std::function<void(DiskPublicationStep)>);
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif

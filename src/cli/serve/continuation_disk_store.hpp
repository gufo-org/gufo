#ifndef GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_
#define GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class ContinuationDiskEventAction : std::uint8_t {
  kStored,
  kRestored,
  kMiss,
  kRemoved,
  kSkipped,
};

enum class ContinuationDiskEventReason : std::uint8_t {
  kSaved,
  kHit,
  kNotFound,
  kUnsupported,
  kByteCapacity,
  kStagingCapacity,
  kLru,
  kExactReplacement,
  kCorrupt,
  kChecksumMismatch,
  kUnsafeFile,
  kIoFailure,
  kSerializationFailure,
  kRestoreFailure,
  kBusy,
};

/// Sanitized disk-cache event. Prompt contents, token values, paths, model
/// identities, and compatibility fingerprints are deliberately absent.
struct ContinuationDiskEvent {
  ContinuationDiskEventAction action{ContinuationDiskEventAction::kMiss};
  ContinuationDiskEventReason reason{ContinuationDiskEventReason::kNotFound};
  std::size_t file_bytes{0};
  std::size_t payload_bytes{0};
  std::size_t token_count{0};
  std::size_t retained_bytes{0};
  std::size_t capacity_bytes{0};
  std::size_t staging_capacity_bytes{0};
  std::size_t staging_used_bytes{0};
  double elapsed_ms{0.0};
};

struct ContinuationDiskStoreOptions {
  std::filesystem::path directory;
  std::size_t capacity_bytes{TextRunnerDiskCacheOptions::kDefaultCapacityBytes};
  /// Zero resolves to the host snapshot budget, capped by capacity_bytes.
  std::size_t staging_capacity_bytes{0};
};

/// Restart-safe, provider-neutral exact-prefix snapshot store.
///
/// Model code owns compatibility identity and payload serialization. The store
/// owns checksums, exact token verification, byte limits, atomic publication,
/// startup indexing, LRU replacement, permissions, and collision-safe lookup.
///
/// Persistence uses one bounded worker. Payload writes and fsync run outside
/// the metadata gate, so existing entries remain readable during persistence.
class ContinuationDiskStore {
public:
  using EventSink = std::function<void(const ContinuationDiskEvent&)>;
  using KeyHashFunction =
      std::function<std::string(std::span<const std::uint8_t>)>;

  struct SaveResult {
    bool stored{false};
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
  };

  struct RestoreResult {
    bool restored{false};
    std::size_t token_count{0};
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
  };

  explicit ContinuationDiskStore(ContinuationDiskStoreOptions options,
                                 EventSink event_sink = {},
                                 KeyHashFunction key_hash = {});
  ~ContinuationDiskStore();

  ContinuationDiskStore(const ContinuationDiskStore&) = delete;
  ContinuationDiskStore& operator=(const ContinuationDiskStore&) = delete;
  ContinuationDiskStore(ContinuationDiskStore&&) = delete;
  ContinuationDiskStore& operator=(ContinuationDiskStore&&) = delete;

  [[nodiscard]] SaveResult Save(
      const TextModelRunner& runner,
      std::span<const TextRunnerToken> checkpoint_tokens,
      const TextRunnerSnapshot& snapshot,
      std::span<const std::uint8_t> input_identity = {});

  /// Holds disk-only capture memory against the staging budget until it is
  /// queued or discarded. Reservations can safely outlive the store.
  class CaptureReservation {
  public:
    ~CaptureReservation();
    CaptureReservation(const CaptureReservation&) = delete;
    CaptureReservation& operator=(const CaptureReservation&) = delete;

  private:
    friend class ContinuationDiskStore;
    CaptureReservation(std::shared_ptr<std::atomic<std::size_t>> counter,
                       std::size_t bytes)
        : counter_(std::move(counter)), bytes_(bytes) {}
    std::shared_ptr<std::atomic<std::size_t>> counter_;
    std::size_t bytes_;
  };

  [[nodiscard]] std::unique_ptr<CaptureReservation> ReserveCapture(
      const TextModelRunner& runner, std::size_t token_count,
      std::size_t snapshot_bytes,
      std::span<const std::uint8_t> input_identity = {});

  /// Conservative admission before capturing a snapshot. Includes queued and
  /// active snapshot bytes; failure means a harmless cache skip.
  [[nodiscard]] bool CanSave(
      const TextModelRunner& runner, std::size_t token_count,
      std::size_t snapshot_bytes,
      std::span<const std::uint8_t> input_identity = {}) const;

  /// Enqueues immutable storage, retaining both model and snapshot until done.
  /// The returned byte count is queued work, not completed disk I/O.
  [[nodiscard]] std::size_t SaveAsync(
      std::shared_ptr<const TextModelRunner> runner,
      std::vector<TextRunnerToken> checkpoint_tokens,
      std::shared_ptr<const TextRunnerSnapshot> snapshot,
      std::vector<std::uint8_t> input_identity = {},
      std::unique_ptr<CaptureReservation> reservation = {});

  /// Drains accepted writes. Shutdown also drains automatically.
  void Flush();

  /// Restores the longest exact saved prefix of prompt into state.
  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity = {});

  /// Prefix lengths that prompt shares with stored entries but that no entry
  /// holds exactly.
  ///
  /// For each stored entry of the same compatibility identity, the common
  /// prefix length with prompt is a candidate. Lengths below min_tokens, equal
  /// to prompt.size(), or already stored exactly are dropped. Returns at most
  /// max_boundaries distinct lengths in ascending order. Callers snapshot at
  /// those positions so a prefix shared across conversations (system prompt,
  /// tool schemas) becomes restorable by the next one instead of being
  /// prefilled again.
  [[nodiscard]] std::vector<std::size_t> SharedPrefixBoundaries(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::size_t min_tokens, std::size_t max_boundaries,
      std::span<const std::uint8_t> input_identity = {});

  /// Marks the exact entry for tokens as recently used without reading it.
  /// Returns false when no such entry exists.
  [[nodiscard]] bool Touch(const TextModelRunner& runner,
                           std::span<const TextRunnerToken> tokens,
                           std::span<const std::uint8_t> input_identity = {});

  [[nodiscard]] std::size_t entry_count() const noexcept;
  [[nodiscard]] std::size_t retained_bytes() const noexcept;
  [[nodiscard]] std::size_t capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t staging_capacity_bytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_

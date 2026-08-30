#ifndef GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_
#define GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_

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
};

struct ContinuationDiskStoreOptions {
  std::filesystem::path directory;
  std::size_t capacity_bytes{0};
  std::size_t staging_capacity_bytes{0};
};

/// Restart-safe, provider-neutral exact-prefix snapshot store.
///
/// Model code owns compatibility identity and payload serialization. The store
/// owns checksums, exact token verification, byte limits, atomic publication,
/// startup indexing, LRU replacement, permissions, and collision-safe lookup.
///
/// Operations are synchronous and serialized. This deliberately bounds both
/// staging memory and outstanding disk I/O to one operation.
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
      const TextRunnerSnapshot& snapshot);

  /// Restores the longest exact saved prefix of prompt into state.
  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt);

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

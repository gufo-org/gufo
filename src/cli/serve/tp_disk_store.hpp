#ifndef GUFO_SERVER_TP_DISK_STORE_HPP_
#define GUFO_SERVER_TP_DISK_STORE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

/// Rank 1's half of the TP2 disk cache.
///
/// Rank 0's continuation disk store decides what is saved, restored and
/// evicted, and writes its own half of each snapshot. This store keeps rank 1's
/// half under the file key rank 0 chose, one file per key, and bounds itself by
/// least-recent use. It never has to agree with rank 0's index: an entry
/// missing here only turns rank 0's restore into a cache miss.
class TpDiskStore {
public:
  struct Options {
    std::filesystem::path directory;
    std::size_t capacity_bytes{0};
    /// Snapshot bytes queued for writing at most; a persist beyond it is
    /// skipped.
    std::size_t staging_capacity_bytes{0};
  };

  /// `identity` and `payload_version` are the runner's persistence
  /// descriptor: a file written under another one is never restored.
  TpDiskStore(Options options, std::vector<std::uint8_t> identity,
              std::uint32_t payload_version);
  /// Finishes queued writes.
  ~TpDiskStore();

  TpDiskStore(const TpDiskStore&) = delete;
  TpDiskStore& operator=(const TpDiskStore&) = delete;

  /// Queues `snapshot` for writing under `key`. The snapshot is immutable and
  /// kept alive until written. False when it was skipped (staging full).
  bool PersistAsync(std::uint64_t key,
                    std::shared_ptr<const TextModelRunner> runner,
                    std::shared_ptr<const TextRunnerSnapshot> snapshot);

  /// Restores entry `key` into `state`. Throws when the entry is missing,
  /// corrupt or written for another model; a corrupt entry is removed.
  void Restore(std::uint64_t key, const TextModelRunner& runner,
               TextRunnerState& state);

  /// Waits for queued writes.
  void Flush();

  [[nodiscard]] std::size_t entry_count() const;
  [[nodiscard]] std::size_t retained_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_DISK_STORE_HPP_

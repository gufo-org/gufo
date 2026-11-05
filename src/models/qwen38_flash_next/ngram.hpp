#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_HPP_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Newest-first tail of the previous tokens of one sequence, the complete
/// hash state of the PLE window. Copying it snapshots the state.
struct NgramHistory {
  static constexpr std::int32_t kNone = -1;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> prev{kNone, kNone};

  void Reset() noexcept { prev.fill(kNone); }
  bool operator==(const NgramHistory&) const = default;
};

/// Computes the table rows every token gathers: `ple_heads` rows per token,
/// head-major, exactly as the reference hashes them. An EOS anywhere in the
/// window cuts every older token; a missing predecessor also reads as EOS.
/// `history` is advanced past `tokens`.
void HashNgramRows(const Config& config, NgramHistory& history,
                   std::span<const std::int32_t> tokens,
                   std::span<std::uint32_t> rows);

/// Rows a sequence gathered. Every row is a deduplicated copy of another row
/// in the same gather, a row-cache hit or a disk read, so
/// rows == dedup_rows + cache_hits + disk_reads once its gathers complete.
struct NgramTotals {
  std::uint64_t rows{0};
  std::uint64_t dedup_rows{0};
  std::uint64_t cache_hits{0};
  std::uint64_t disk_reads{0};
};

/// Per-sequence counters. Gathers and table readers update them concurrently.
class NgramCounters {
public:
  [[nodiscard]] NgramTotals Load() const noexcept {
    return {rows_.load(std::memory_order_relaxed),
            dedup_rows_.load(std::memory_order_relaxed),
            cache_hits_.load(std::memory_order_relaxed),
            disk_reads_.load(std::memory_order_relaxed)};
  }

private:
  friend class NgramTable;
  std::atomic<std::uint64_t> rows_{0};
  std::atomic<std::uint64_t> dedup_rows_{0};
  std::atomic<std::uint64_t> cache_hits_{0};
  std::atomic<std::uint64_t> disk_reads_{0};
};

/// The next `rows` rows of a gather belong to the sequence counted by
/// `counters`, which may be null.
struct NgramGatherOwner {
  std::size_t rows{0};
  NgramCounters* counters{nullptr};
};

/// Reads n-gram embedding rows from the GGUF shard with direct I/O and a
/// bounded cache of compressed rows. The table stays out of the page cache.
/// Rows are dequantized to F32 and laid out [token][head][ple_head_dim].
class NgramTable {
public:
  ~NgramTable();
  NgramTable(const NgramTable&) = delete;
  NgramTable& operator=(const NgramTable&) = delete;

  /// Borrows the bound GGUF descriptor during Open; retains its own handle,
  /// unbuffered where the platform allows it and a buffered duplicate
  /// otherwise.
  [[nodiscard]] static std::unique_ptr<NgramTable> Open(
      platform::PlatformFile file, std::uint64_t file_offset,
      std::uint64_t rows, std::uint32_t row_dim, core::GgmlType type,
      std::string* error_msg = nullptr);

  /// Gathers `rows.size()` rows into `out` (rows.size() * row_dim floats).
  /// Duplicate ids are read once. Returns false on any failed read.
  [[nodiscard]] bool Read(std::span<const std::uint32_t> rows,
                          std::span<float> out);

  /// Prepare scratch before serving requests; does not read table rows.
  /// Only valid while no gather is outstanding.
  void ReserveGather(std::size_t rows);

  /// Starts a gather on the resident readers. The output must stay alive
  /// until WaitRead; only one gather may be outstanding on this table.
  /// `owners` partitions `rows` in order; empty leaves the gather uncounted.
  [[nodiscard]] bool StartRead(std::span<const std::uint32_t> rows,
                               std::span<float> out,
                               std::span<const NgramGatherOwner> owners = {});
  [[nodiscard]] bool WaitRead();

  [[nodiscard]] std::uint32_t RowDim() const noexcept { return row_dim_; }
  [[nodiscard]] std::uint64_t Rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t RowBytes() const noexcept { return row_bytes_; }

private:
  NgramTable() = default;

  struct Job {
    std::uint32_t row;
    float* dst;
    NgramCounters* owner;
  };
  struct Copy {
    const float* src;
    float* dst;
  };
  // Reuse deduplication storage across gathers. Entries index the first
  // occurrence in the current input, so no row-cache state is retained here.
  std::vector<std::size_t> first_rows_;
  void DecodeRow(const std::uint8_t* src, float* dst) const;
  bool ReadCached(std::uint32_t row, float* dst);
  bool ReadOne(std::uint32_t row, float* dst, std::vector<std::uint8_t>& buf,
               NgramCounters* owner);
  void Worker();

  struct CacheEntry {
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    std::uint32_t row{0};
    bool valid{false};
  };
  std::unique_ptr<CacheEntry[]> cache_entries_;
  std::vector<std::uint8_t> cache_rows_;
  std::size_t cache_count_{0};

  platform::PlatformFile file_{platform::kInvalidFile};
  bool direct_{false};
  std::uint64_t base_offset_{0};
  std::uint64_t rows_{0};
  std::uint32_t row_dim_{0};
  std::size_t row_bytes_{0};
  core::GgmlType type_{core::GgmlType::kIQ4_NL};

  // A small resident pool keeps decode-time reads from paying thread
  // creation; prefill batches fan out over the same workers.
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::vector<Job> jobs_;
  std::vector<Copy> copies_;
  std::size_t next_job_{0};
  std::size_t pending_{0};
  bool active_{false};
  bool failed_{false};
  bool stop_{false};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_HPP_

#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NGRAM_HPP_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
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

/// Reads n-gram embedding rows from the GGUF shard with direct I/O and a
/// bounded cache of compressed rows. The table stays out of the page cache.
/// Rows are dequantized to F32 and laid out [token][head][ple_head_dim].
class NgramTable {
public:
  ~NgramTable();
  NgramTable(const NgramTable&) = delete;
  NgramTable& operator=(const NgramTable&) = delete;

  /// Borrows the bound GGUF descriptor during Open; retains its own handle.
  [[nodiscard]] static std::unique_ptr<NgramTable> Open(
      int file_descriptor, std::uint64_t file_offset, std::uint64_t rows,
      std::uint32_t row_dim, core::GgmlType type,
      std::string* error_msg = nullptr);

  /// Gathers `rows.size()` rows into `out` (rows.size() * row_dim floats).
  /// Duplicate ids are read once. Returns false on any failed read.
  [[nodiscard]] bool Read(std::span<const std::uint32_t> rows,
                          std::span<float> out);

  /// Starts a gather on the resident readers. The output must stay alive
  /// until WaitRead; only one gather may be outstanding on this table.
  [[nodiscard]] bool StartRead(std::span<const std::uint32_t> rows,
                               std::span<float> out);
  [[nodiscard]] bool WaitRead();

  [[nodiscard]] std::uint32_t RowDim() const noexcept { return row_dim_; }
  [[nodiscard]] std::uint64_t Rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t RowBytes() const noexcept { return row_bytes_; }

private:
  NgramTable() = default;

  struct Job {
    std::uint32_t row;
    float* dst;
  };
  struct Copy {
    const float* src;
    float* dst;
  };
  void DecodeRow(const std::uint8_t* src, float* dst) const;
  bool ReadCached(std::uint32_t row, float* dst);
  bool ReadOne(std::uint32_t row, float* dst, std::vector<std::uint8_t>& buf);
  void Worker();

  struct CacheEntry {
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    std::uint32_t row{0};
    bool valid{false};
  };
  std::unique_ptr<CacheEntry[]> cache_entries_;
  std::vector<std::uint8_t> cache_rows_;
  std::size_t cache_count_{0};

  int fd_{-1};
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

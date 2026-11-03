#include "src/models/qwen38_flash_next/ngram.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::size_t kPage = 4096;
// Next-chunk prefetch hides later gathers behind the current transformer
// pass, but a prefill's first chunk waits for its own gather. Readers block
// in `pread`, so the pool sets that gather's queue depth.
constexpr std::size_t kWorkers = 128;
constexpr std::size_t kReadBatch = 8;
constexpr std::size_t kBatchJobs = 1024;
constexpr std::size_t kCacheBytes = 128 * 1024 * 1024;

}  // namespace

void HashNgramRows(const Config& c, NgramHistory& history,
                   std::span<const std::int32_t> tokens,
                   std::span<std::uint32_t> rows) {
  const std::uint32_t n = c.ple_ngram_size;
  const std::int64_t eos = c.ple_eos_token;
  std::array<std::uint64_t, Config::kMaxPleNgram> ctx{};
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    // The token's own EOS does not cut its context, only an older one does.
    ctx[0] = static_cast<std::uint64_t>(tokens[i]);
    bool cut = false;
    for (std::uint32_t s = 1; s < n; ++s) {
      const std::int32_t t = cut ? NgramHistory::kNone : history.prev[s - 1];
      cut = cut || t < 0 || t == eos;
      ctx[s] = static_cast<std::uint64_t>(cut ? eos : t);
    }
    std::uint32_t* out = rows.data() + i * c.ple_heads;
    for (std::uint32_t order = 2; order <= n; ++order) {
      std::uint64_t mixed = ctx[0] * c.ple_multipliers[0];
      for (std::uint32_t j = 1; j < order; ++j) {
        mixed ^= ctx[j] * c.ple_multipliers[j];
      }
      const std::uint32_t base = (order - 2) * c.ple_heads_per_ngram;
      for (std::uint32_t g = 0; g < c.ple_heads_per_ngram; ++g) {
        const std::uint32_t h = base + g;
        out[h] = static_cast<std::uint32_t>(mixed % c.ple_head_vocab[h]) +
                 c.ple_head_offsets[h];
      }
    }
    for (std::size_t s = history.prev.size(); s-- > 1;) {
      history.prev[s] = history.prev[s - 1];
    }
    history.prev[0] = tokens[i];
  }
}

NgramTable::~NgramTable() {
  (void)WaitRead();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
  if (file_ != platform::kInvalidFile) {
    platform::CloseFile(file_);
  }
}

std::unique_ptr<NgramTable> NgramTable::Open(
    platform::PlatformFile file, std::uint64_t file_offset, std::uint64_t rows,
    std::uint32_t row_dim, core::GgmlType type, std::string* error_msg) {
  std::unique_ptr<NgramTable> t(new NgramTable());
  const bool blocked =
      type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kQ8_0;
  const std::size_t block = blocked ? 32 : 1;
  const std::size_t block_bytes = type == core::GgmlType::kIQ4_NL ? 18
                                  : type == core::GgmlType::kQ8_0 ? 34
                                                                  : 2;
  if ((!blocked && type != core::GgmlType::kBF16) || row_dim == 0 ||
      row_dim % block != 0) {
    if (error_msg != nullptr) {
      *error_msg = "unsupported n-gram table format";
    }
    return nullptr;
  }
  t->type_ = type;
  t->row_dim_ = row_dim;
  t->row_bytes_ = row_dim / block * block_bytes;
  t->cache_count_ = std::bit_floor(kCacheBytes / t->row_bytes_);
  t->cache_entries_ = std::make_unique<CacheEntry[]>(t->cache_count_);
  t->cache_rows_.resize(t->cache_count_ * t->row_bytes_);
  t->rows_ = rows;
  t->base_offset_ = file_offset;
  // Linux direct I/O avoids retaining every touched row in the page cache;
  // Windows uses cached concurrent reads of the otherwise mapped model.
  // When the platform cannot provide it, rows read buffered through a
  // duplicate: the table always owns the handle it closes.
  t->file_ = platform::ReopenRandomReadFile(file, t->direct_);
  if (!platform::FileValid(t->file_)) {
    t->file_ = platform::DuplicateFile(file);
  }
  if (t->file_ == platform::kInvalidFile) {
    if (error_msg != nullptr) {
      *error_msg = "cannot open bound n-gram table";
    }
    return nullptr;
  }
  for (std::size_t i = 0; i < kWorkers; ++i) {
    t->workers_.emplace_back([raw = t.get()] { raw->Worker(); });
  }
  return t;
}

void NgramTable::DecodeRow(const std::uint8_t* src, float* dst) const {
  if (type_ == core::GgmlType::kIQ4_NL) {
    gufo::quant::DequantizeIQ4_NL(src, dst, row_dim_);
  } else if (type_ == core::GgmlType::kQ8_0) {
    gufo::quant::DequantizeQ8_0(src, dst, row_dim_);
  } else {
    for (std::uint32_t i = 0; i < row_dim_; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src + 2 * i, 2);
      const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
      std::memcpy(dst + i, &f, sizeof(float));
    }
  }
}

bool NgramTable::ReadCached(std::uint32_t row, float* dst) {
  if (cache_count_ == 0) {
    return false;
  }
  const std::size_t slot = (row * 2654435761U) & (cache_count_ - 1);
  CacheEntry& entry = cache_entries_[slot];
  if (entry.busy.test_and_set(std::memory_order_acquire)) {
    return false;
  }
  const bool hit = entry.valid && entry.row == row;
  if (hit) {
    DecodeRow(cache_rows_.data() + slot * row_bytes_, dst);
  }
  entry.busy.clear(std::memory_order_release);
  return hit;
}

bool NgramTable::ReadOne(std::uint32_t row, float* dst,
                         std::vector<std::uint8_t>& buf, NgramCounters* owner) {
  if (row >= rows_) {
    return false;
  }
  if (ReadCached(row, dst)) {
    if (owner != nullptr)
      owner->cache_hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  const std::uint64_t offset = base_offset_ + row * row_bytes_;
  const std::uint64_t begin = direct_ ? offset & ~(kPage - 1) : offset;
  const std::uint64_t end =
      direct_ ? (offset + row_bytes_ + kPage - 1) & ~(kPage - 1)
              : offset + row_bytes_;
  const std::size_t length = end - begin;
  if (buf.size() < length + kPage) {
    buf.resize(length + kPage);
  }
  // O_DIRECT needs a page-aligned buffer; align inside the vector.
  auto* base =
      reinterpret_cast<std::uintptr_t>(buf.data()) % kPage == 0
          ? buf.data()
          : buf.data() +
                (kPage - reinterpret_cast<std::uintptr_t>(buf.data()) % kPage);
  // The aligned window may run past the end of the file: only the row's own
  // bytes have to arrive.
  const std::size_t needed = (offset - begin) + row_bytes_;
  const std::size_t slot =
      cache_count_ != 0 ? (row * 2654435761U) & (cache_count_ - 1) : 0;
  CacheEntry* entry = cache_count_ != 0 ? &cache_entries_[slot] : nullptr;
  std::uint8_t* cached =
      entry != nullptr ? cache_rows_.data() + slot * row_bytes_ : nullptr;
  std::size_t got = 0;
  std::error_code ec;
  while (got < needed) {
    const std::size_t n =
        platform::ReadAt(file_, begin + got, base + got, length - got, ec);
    if (n == 0 && ec == std::errc::interrupted) {
      continue;
    }
    if (n == 0) {
      return false;
    }
    got += n;
  }
  if (owner != nullptr)
    owner->disk_reads_.fetch_add(1, std::memory_order_relaxed);
  // Cache only a complete row, still in its original quantized format.
  // A busy slot is skipped; readers never wait for another row's I/O.
  if (entry != nullptr &&
      !entry->busy.test_and_set(std::memory_order_acquire)) {
    std::memcpy(cached, base + (offset - begin), row_bytes_);
    entry->row = row;
    entry->valid = true;
    entry->busy.clear(std::memory_order_release);
  }
  DecodeRow(base + (offset - begin), dst);
  return true;
}

void NgramTable::Worker() {
  std::vector<std::uint8_t> buf;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    wake_.wait(lock,
               [&] { return stop_ || (active_ && next_job_ < jobs_.size()); });
    if (stop_) {
      return;
    }
    // Large gathers amortize the queue lock. Small gathers keep one read
    // per worker, so decode does not serialize its few rows into a batch.
    std::array<Job, kReadBatch> batch;
    const std::size_t count = std::min(
        jobs_.size() >= kBatchJobs ? kReadBatch : 1, jobs_.size() - next_job_);
    std::copy_n(jobs_.data() + next_job_, count, batch.data());
    next_job_ += count;
    lock.unlock();
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i) {
      ok = ReadOne(batch[i].row, batch[i].dst, buf, batch[i].owner) && ok;
    }
    lock.lock();
    failed_ = failed_ || !ok;
    pending_ -= count;
    if (pending_ == 0) {
      done_.notify_all();
    }
  }
}

bool NgramTable::Read(std::span<const std::uint32_t> rows,
                      std::span<float> out) {
  return StartRead(rows, out) && WaitRead();
}

void NgramTable::ReserveGather(std::size_t rows) {
  const std::lock_guard lock(mutex_);
  if (active_)
    throw std::logic_error("cannot reserve an outstanding n-gram gather");
  if (rows > std::numeric_limits<std::size_t>::max() / 4)
    throw std::length_error("n-gram gather is too large");
  const auto capacity = std::bit_ceil(std::max<std::size_t>(2, rows * 2));
  if (first_rows_.size() < capacity)
    first_rows_.resize(capacity);
  // Touch the reserved pages too: later, larger prefills should not fault
  // their scratch while cache backing is being populated.
  jobs_.resize(rows);
  jobs_.clear();
  copies_.resize(rows);
  copies_.clear();
}

bool NgramTable::StartRead(std::span<const std::uint32_t> rows,
                           std::span<float> out,
                           std::span<const NgramGatherOwner> owners) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t owned = 0;
  for (const auto& owner : owners)
    owned += owner.rows;
  if (active_ || rows.size() > out.size() / row_dim_ ||
      rows.size() > std::numeric_limits<std::size_t>::max() / 4 ||
      (!owners.empty() && owned != rows.size()) ||
      std::any_of(rows.begin(), rows.end(),
                  [this](std::uint32_t row) { return row >= rows_; })) {
    return false;
  }
  for (const auto& owner : owners) {
    if (owner.counters != nullptr)
      owner.counters->rows_.fetch_add(owner.rows, std::memory_order_relaxed);
  }
  // A node-based map allocates on every gather. Those allocations can wait
  // behind background cache page registration before any GPU work is queued.
  // Keep a flat table at most half full and clear only this gather's size;
  // decode after a large prefill must not scan the whole retained capacity.
  constexpr auto empty = std::numeric_limits<std::size_t>::max();
  const auto capacity =
      std::bit_ceil(std::max<std::size_t>(2, rows.size() * 2));
  if (first_rows_.size() < capacity)
    first_rows_.resize(capacity);
  auto first = std::span(first_rows_).first(capacity);
  std::ranges::fill(first, empty);
  jobs_.clear();
  copies_.clear();
  jobs_.reserve(rows.size());
  copies_.reserve(rows.size());
  auto owner = owners.begin();
  std::size_t owner_end = owner != owners.end() ? owner->rows : rows.size();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    while (i == owner_end) {
      ++owner;
      owner_end += owner->rows;
    }
    NgramCounters* const counters =
        owner != owners.end() ? owner->counters : nullptr;
    const auto hash = std::uint64_t{rows[i]} * 0x9e3779b97f4a7c15ULL;
    auto slot = static_cast<std::size_t>(hash ^ (hash >> 32)) & (capacity - 1);
    while (first[slot] != empty && rows[first[slot]] != rows[i])
      slot = (slot + 1) & (capacity - 1);
    if (first[slot] == empty) {
      first[slot] = i;
      jobs_.push_back({rows[i], out.data() + i * row_dim_, counters});
    } else {
      copies_.push_back(
          {out.data() + first[slot] * row_dim_, out.data() + i * row_dim_});
      if (counters != nullptr)
        counters->dedup_rows_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (jobs_.size() >= kBatchJobs) {
    std::sort(jobs_.begin(), jobs_.end(),
              [](const Job& a, const Job& b) { return a.row < b.row; });
  } else {
    // Small cached gathers avoid waking the I/O pool for every decode
    // token. Large prefills still parallelize their row conversions.
    std::erase_if(jobs_, [this](const Job& job) {
      if (!ReadCached(job.row, job.dst))
        return false;
      if (job.owner != nullptr)
        job.owner->cache_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    });
  }
  next_job_ = 0;
  pending_ = jobs_.size();
  failed_ = false;
  active_ = true;
  // A decode gather has a handful of reads: wake one reader per read, not
  // the whole pool. Readers keep taking jobs until the queue is empty.
  if (pending_ >= workers_.size()) {
    wake_.notify_all();
  } else {
    for (std::size_t i = 0; i < pending_; ++i)
      wake_.notify_one();
  }
  return true;
}

bool NgramTable::WaitRead() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!active_) {
    return false;
  }
  done_.wait(lock, [&] { return pending_ == 0; });
  if (!failed_) {
    for (const Copy& copy : copies_) {
      std::copy_n(copy.src, row_dim_, copy.dst);
    }
  }
  jobs_.clear();
  copies_.clear();
  active_ = false;
  return !failed_;
}

}  // namespace gufo::models::qwen38_flash_next

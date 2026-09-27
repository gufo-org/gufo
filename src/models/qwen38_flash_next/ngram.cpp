#include "src/models/qwen38_flash_next/ngram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <unordered_map>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::size_t kPage = 4096;
// Keep enough direct reads outstanding while layer 0 runs.
constexpr std::size_t kWorkers = 32;
constexpr std::size_t kReadBatch = 8;
constexpr std::size_t kBatchJobs = 1024;
constexpr std::size_t kCacheBytes = 8 * 1024 * 1024;

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
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

std::unique_ptr<NgramTable> NgramTable::Open(
    int file_descriptor, std::uint64_t file_offset, std::uint64_t rows,
    std::uint32_t row_dim, core::GgmlType type, std::string* error_msg) {
  std::unique_ptr<NgramTable> t(new NgramTable());
  const bool supported = type == core::GgmlType::kIQ4_NL ||
                         type == core::GgmlType::kBF16 ||
                         type == core::GgmlType::kQ8_0;
  const std::size_t row_bytes =
      type == core::GgmlType::kBF16 ? static_cast<std::size_t>(row_dim) * 2
      : (type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kQ8_0)
          ? gufo::quant::QuantizedRowBytes(type, row_dim)
          : 0;
  if (!supported || row_bytes == 0) {
    if (error_msg != nullptr) {
      *error_msg = "unsupported n-gram table format";
    }
    return nullptr;
  }
  t->type_ = type;
  t->row_dim_ = row_dim;
  t->row_bytes_ = row_bytes;
  t->cache_count_ = std::bit_floor(kCacheBytes / t->row_bytes_);
  t->cache_entries_ = std::make_unique<CacheEntry[]>(t->cache_count_);
  t->cache_rows_.resize(t->cache_count_ * t->row_bytes_);
  t->rows_ = rows;
  t->base_offset_ = file_offset;
  // Direct I/O bypasses the page cache; the mapping used for the rest of the
  // model must not be used here or every touched row would stay resident.
  const auto path = "/proc/self/fd/" + std::to_string(file_descriptor);
  t->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  t->direct_ = t->fd_ >= 0;
  if (t->fd_ < 0) {
    t->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  }
  if (t->fd_ < 0) {
    if (error_msg != nullptr) {
      *error_msg = std::string("cannot open bound n-gram table: ") +
                   std::strerror(errno);
    }
    return nullptr;
  }
  const std::size_t workers = std::min(
      kWorkers, static_cast<std::size_t>(std::thread::hardware_concurrency()));
  for (std::size_t i = 0; i < std::max<std::size_t>(1, workers); ++i) {
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
                         std::vector<std::uint8_t>& buf) {
  if (row >= rows_) {
    return false;
  }
  if (ReadCached(row, dst)) {
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
  while (got < needed) {
    const ssize_t n = ::pread(fd_, base + got, length - got, begin + got);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }
    got += static_cast<std::size_t>(n);
  }
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
      ok = ReadOne(batch[i].row, batch[i].dst, buf) && ok;
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

bool NgramTable::StartRead(std::span<const std::uint32_t> rows,
                           std::span<float> out) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_ || rows.size() > out.size() / row_dim_ ||
      std::any_of(rows.begin(), rows.end(),
                  [this](std::uint32_t row) { return row >= rows_; })) {
    return false;
  }
  // Read each distinct row once, then copy it to its other slots.
  std::unordered_map<std::uint32_t, std::size_t> first;
  first.reserve(rows.size());
  jobs_.clear();
  copies_.clear();
  jobs_.reserve(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto [it, inserted] = first.emplace(rows[i], i);
    if (inserted) {
      jobs_.push_back({rows[i], out.data() + i * row_dim_});
    } else {
      copies_.push_back(
          {out.data() + it->second * row_dim_, out.data() + i * row_dim_});
    }
  }
  if (jobs_.size() >= kBatchJobs) {
    std::sort(jobs_.begin(), jobs_.end(),
              [](const Job& a, const Job& b) { return a.row < b.row; });
  } else {
    // Small cached gathers avoid waking the I/O pool for every decode
    // token. Large prefills still parallelize their row conversions.
    std::erase_if(
        jobs_, [this](const Job& job) { return ReadCached(job.row, job.dst); });
  }
  next_job_ = 0;
  pending_ = jobs_.size();
  failed_ = false;
  active_ = true;
  if (pending_ != 0) {
    wake_.notify_all();
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

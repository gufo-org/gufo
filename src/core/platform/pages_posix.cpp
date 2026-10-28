#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "src/core/platform/pages.hpp"

namespace gufo::platform {
namespace {

// Ordinary anonymous pages, populated before use. Opting out of THP before
// population prevents later huge-page promotion from invalidating GPU
// mappings of the same range.
class CommittedRange {
public:
  static void* Commit(std::size_t bytes) noexcept {
    void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED)
      return nullptr;
    // Kernels built without THP reject this advice with EINVAL; their pages
    // cannot be promoted. All other failures must release the mapping.
    if (madvise(data, bytes, MADV_NOHUGEPAGE) != 0 && errno != EINVAL) {
      munmap(data, bytes);
      return nullptr;
    }
    if (!Populate(data, bytes)) {
      munmap(data, bytes);
      return nullptr;
    }
    return data;
  }

private:
  static bool Populate(void* data, std::size_t bytes) noexcept {
    constexpr std::size_t page = 4096;
    const auto workers = std::min<std::size_t>(4, bytes / (128ULL << 20));
    std::atomic<bool> populated{true};
    const auto populate = [&](std::size_t begin, std::size_t end) {
      // A long population call holds the process's mapping lock while
      // faulting pages. Bound each call so request-side heap growth can make
      // progress.
      constexpr std::size_t piece = 64ULL << 10;
      for (auto offset = begin; offset < end; offset += piece) {
        if (!populated.load(std::memory_order_relaxed))
          break;
        if (madvise(static_cast<std::byte*>(data) + offset,
                    std::min(piece, end - offset), MADV_POPULATE_WRITE) != 0) {
          populated.store(false, std::memory_order_relaxed);
          break;
        }
      }
    };
    if (workers >= 2) {
      // Large serial page faults can hold up inference even when growth runs
      // on a background thread. Populate disjoint ordinary-page ranges, then
      // join before exposing any of the backing.
      std::array<std::jthread, 3> faults;
      std::size_t next = 0;
      for (std::size_t i = 0; i + 1 < workers; ++i) {
        const auto end = (bytes / page) * (i + 1) / workers * page;
        try {
          faults[i] = std::jthread(populate, next, end);
        } catch (...) {
          // Thread exhaustion falls back to populating the remaining range
          // here. Every started worker still joins.
          break;
        }
        next = end;
      }
      populate(next, bytes);
    } else {
      populate(0, bytes);
    }
    return populated.load(std::memory_order_relaxed);
  }
};

}  // namespace

void* CommitZeroedPages(std::size_t bytes) noexcept {
  return CommittedRange::Commit(bytes);
}

void ReleaseCommittedPages(void* base, std::size_t bytes) noexcept {
  if (base != nullptr && bytes > 0) {
    munmap(base, bytes);
  }
}

bool CanVisitMappedChunks() noexcept {
  return true;
}

void VisitMappedChunks(
    const void* source_data, std::size_t bytes, std::size_t chunk_bytes,
    std::size_t workers,
    const std::function<bool(std::size_t worker, std::size_t offset,
                             const void* chunk, std::size_t chunk_size)>&
        visit) {
  if (source_data == nullptr || bytes == 0 || chunk_bytes == 0)
    return;
  const auto chunks = bytes / chunk_bytes + (bytes % chunk_bytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  std::atomic<bool> stopped{false};
  auto visit_chunks = [&](std::size_t worker) {
    while (failure.load(std::memory_order_relaxed) == 0 &&
           !stopped.load(std::memory_order_relaxed)) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * chunk_bytes;
      const auto size = std::min(chunk_bytes, bytes - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source_data)) +
                    offset;
      if (madvise(input, size, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      const bool keep_going = visit(worker, offset, input, size);
      // The visitor owns a copy of these bytes now. Keep the mapping valid
      // for the reader without retaining its populated PTEs.
      (void)madvise(input, size, MADV_DONTNEED);
      if (!keep_going) {
        stopped.store(true, std::memory_order_relaxed);
        break;
      }
    }
  };
  {
    std::vector<std::jthread> threads;
    const auto count = std::clamp<std::size_t>(workers, 1, chunks);
    for (std::size_t worker = 1; worker < count; ++worker)
      threads.emplace_back(visit_chunks, worker);
    visit_chunks(0);
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

void* CloneMappedWeightsToAnonymousPages(const void* source_data,
                                         std::size_t bytes) {
  if (source_data == nullptr || bytes == 0) {
    return nullptr;
  }
  void* host_copy = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED) {
    return nullptr;
  }
  (void)madvise(host_copy, bytes, MADV_HUGEPAGE);
  try {
    VisitMappedChunks(source_data, bytes, 16ULL << 20, 16,
                      [&](std::size_t, std::size_t offset, const void* chunk,
                          std::size_t size) {
                        std::memcpy(
                            static_cast<std::uint8_t*>(host_copy) + offset,
                            chunk, size);
                        return true;
                      });
  } catch (...) {
    munmap(host_copy, bytes);
    throw;
  }
  return host_copy;
}

void PopulateWritablePages(void* raw_data, std::size_t bytes,
                           std::size_t worker_limit) noexcept {
  auto* data = static_cast<std::uint8_t*>(raw_data);
  if (data == nullptr || bytes == 0)
    return;
  const long page = sysconf(_SC_PAGESIZE);
  if (page <= 0)
    return;
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto skip = (page - address % page) % page;
  if (bytes <= skip)
    return;
  const auto length = (bytes - skip) / page * page;
  if (length == 0)
    return;
  // Populate before huge-page advice to avoid synchronous UMA compaction on
  // first touch. Borrowed checkpoints prepare host pages only when exported.
  const auto workers = std::max<std::size_t>(
      1, std::min<std::size_t>(worker_limit, length / (32ULL << 20)));
  const auto pages = length / page;
  {
    std::vector<std::jthread> faults;
    for (std::size_t worker = 1; worker < workers; ++worker) {
      const auto begin = pages * worker / workers * page;
      const auto end = pages * (worker + 1) / workers * page;
      faults.emplace_back([=] {
        (void)madvise(data + skip + begin, end - begin, MADV_POPULATE_WRITE);
      });
    }
    (void)madvise(data + skip, pages / workers * page, MADV_POPULATE_WRITE);
  }
  (void)madvise(data + skip, length, MADV_HUGEPAGE);
}

std::string DescribeResidentPages(void* pointer, std::size_t size) {
  if (pointer == nullptr || size == 0) {
    return "mincore unavailable: no host mapping";
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return "mincore unavailable: page size query failed";
  }
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto page = static_cast<std::uintptr_t>(page_size);
  const auto aligned = address & ~(page - 1U);
  const std::size_t prefix = address - aligned;
  if (size > std::numeric_limits<std::size_t>::max() - prefix) {
    return "mincore unavailable: range overflow";
  }
  const std::size_t page_count = (prefix + size + page - 1U) / page;
  constexpr std::size_t kChunkPages = 1ULL << 20U;
  std::vector<unsigned char> states(std::min(page_count, kChunkPages), 0U);
  std::size_t resident = 0;
  std::size_t completed = 0;
  while (completed < page_count) {
    const std::size_t pages = std::min(states.size(), page_count - completed);
    void* chunk = reinterpret_cast<void*>(aligned + completed * page);
    if (mincore(chunk, pages * page, states.data()) != 0) {
      return "mincore unavailable for this allocation";
    }
    resident += static_cast<std::size_t>(
        std::count_if(states.begin(), states.begin() + pages,
                      [](unsigned char value) { return (value & 1U) != 0U; }));
    completed += pages;
  }
  return std::to_string(resident) + "/" + std::to_string(page_count) +
         " pages resident";
}

SharedTempMapping CreateSharedTempMapping(std::size_t bytes) noexcept {
  const int fd = memfd_create("gufo-probe", MFD_CLOEXEC);
  if (fd < 0) {
    return {
        .unavailable_reason =
            std::string("memfd_create failed: ") + std::strerror(errno),
    };
  }
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const int err = errno;
    close(fd);
    return {
        .unavailable_reason =
            std::string("ftruncate failed: ") + std::strerror(err),
    };
  }
  void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    const int err = errno;
    close(fd);
    return {
        .unavailable_reason = std::string("mmap failed: ") + std::strerror(err),
    };
  }
  return {
      .data = ptr,
      .size = bytes,
      .handle = fd,
  };
}

void ReleaseSharedTempMapping(SharedTempMapping& mapping) noexcept {
  if (mapping.data != nullptr && mapping.size > 0) {
    munmap(mapping.data, mapping.size);
  }
  if (mapping.handle >= 0) {
    close(static_cast<int>(mapping.handle));
  }
  mapping = {};
}

}  // namespace gufo::platform

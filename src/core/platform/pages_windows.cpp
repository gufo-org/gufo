#include "src/core/platform/pages.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gufo::platform {

void* CommitZeroedPages(std::size_t bytes) noexcept {
  // Reserving and committing together hands out zero-filled ordinary pages;
  // the OS maps them on first touch.
  return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void ReleaseCommittedPages(void* base, std::size_t) noexcept {
  if (base != nullptr) {
    VirtualFree(base, 0, MEM_RELEASE);
  }
}

void* CloneMappedWeightsToAnonymousPages(const void*, std::size_t) {
  return nullptr;
}

bool CanVisitMappedChunks() noexcept {
  return false;
}

void VisitMappedChunks(const void*, std::size_t, std::size_t, std::size_t,
                       const std::function<bool(std::size_t, std::size_t,
                                                const void*, std::size_t)>&) {
  throw std::logic_error("mapped chunk visits are not supported on Windows");
}

void PopulateWritablePages(void* data, std::size_t bytes,
                           std::size_t worker_limit) noexcept {
  if (data == nullptr || bytes == 0) {
    return;
  }
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const std::size_t page_size =
      info.dwPageSize > 0 ? static_cast<std::size_t>(info.dwPageSize) : 4096U;
  auto* base = static_cast<volatile std::uint8_t*>(data);
  constexpr std::size_t kMinParallelBytes = 64ULL << 20;
  const unsigned hw = std::thread::hardware_concurrency();
  const std::size_t max_workers = std::max<std::size_t>(1, worker_limit);
  const std::size_t workers =
      bytes >= kMinParallelBytes
          ? std::min<std::size_t>(max_workers, hw > 0 ? hw : max_workers)
          : 1;
  const std::size_t pages = (bytes + page_size - 1U) / page_size;
  const std::size_t pages_per_worker = (pages + workers - 1U) / workers;
  std::vector<std::thread> pool;
  if (workers > 1) {
    pool.reserve(workers - 1U);
  }
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin_page = worker * pages_per_worker;
    if (begin_page >= pages) {
      break;
    }
    const std::size_t end_page = std::min(pages, begin_page + pages_per_worker);
    auto touch = [base, begin_page, end_page, page_size, bytes]() {
      for (std::size_t page = begin_page; page < end_page; ++page) {
        const std::size_t offset = page * page_size;
        if (offset < bytes) {
          base[offset] = base[offset];
        }
      }
    };
    if (worker + 1U == workers || end_page >= pages) {
      touch();
    } else {
      pool.emplace_back(touch);
    }
  }
  for (auto& thread : pool) {
    thread.join();
  }
}

std::string DescribeResidentPages(void*, std::size_t) {
  return "mincore unavailable on Windows";
}

SharedTempMapping CreateSharedTempMapping(std::size_t) noexcept {
  return {
      .unavailable_reason = "memfd_create unavailable on Windows",
  };
}

void ReleaseSharedTempMapping(SharedTempMapping& mapping) noexcept {
  mapping = {};
}

}  // namespace gufo::platform

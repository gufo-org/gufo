#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

#include "src/core/platform/pages.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

int main() {
#if defined(_WIN32)
  // Windows keeps a visited mapping in the working set, so callers take the
  // file pipeline instead.
  assert(!gufo::platform::CanVisitMappedChunks());
#else
  assert(gufo::platform::CanVisitMappedChunks());

  // An odd size leaves a short last chunk.
  constexpr std::size_t bytes = (5ULL << 20) + 12345;
  std::vector<std::uint8_t> expected(bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    expected[i] = static_cast<std::uint8_t>((i * 131) ^ (i >> 9));
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo_platform_pages_test_" + std::to_string(::getpid()));
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(expected.data()),
               static_cast<std::streamsize>(bytes));
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  assert(fd >= 0);
  void* mapping = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(mapping != MAP_FAILED);

  // Every byte arrives once, at its offset, from a worker below the limit.
  constexpr std::size_t chunk = 1ULL << 20;
  constexpr std::size_t workers = 3;
  std::vector<std::uint8_t> copy(bytes);
  std::vector<int> seen(bytes / chunk + 1);
  std::mutex mutex;
  gufo::platform::VisitMappedChunks(
      mapping, bytes, chunk, workers,
      [&](std::size_t worker, std::size_t offset, const void* data,
          std::size_t size) {
        assert(worker < workers);
        assert(offset % chunk == 0);
        assert(size == std::min(chunk, bytes - offset));
        std::memcpy(copy.data() + offset, data, size);
        const std::lock_guard lock(mutex);
        ++seen[offset / chunk];
        return true;
      });
  assert(copy == expected);
  for (const auto count : seen)
    assert(count == 1);

  // A false return stops the remaining chunks.
  std::size_t visited = 0;
  gufo::platform::VisitMappedChunks(
      mapping, bytes, chunk, 1,
      [&](std::size_t, std::size_t, const void*, std::size_t) {
        ++visited;
        return false;
      });
  assert(visited == 1);

  // The anonymous clone used for registration matches the file.
  void* clone =
      gufo::platform::CloneMappedWeightsToAnonymousPages(mapping, bytes);
  assert(clone != nullptr);
  assert(std::memcmp(clone, expected.data(), bytes) == 0);
  gufo::platform::ReleaseCommittedPages(clone, bytes);

  assert(::munmap(mapping, bytes) == 0);
  assert(::close(fd) == 0);
  std::filesystem::remove(path);
#endif
}

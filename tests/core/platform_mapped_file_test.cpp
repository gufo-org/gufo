#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/platform/mapped_file.hpp"

namespace {

// Checks stay active under NDEBUG: several conditions carry the call under
// test.
void Require(bool ok, const char* condition, int line) {
  if (!ok) {
    std::fprintf(stderr, "FAIL line %d: %s\n", line, condition);
    std::exit(1);
  }
}
#define CHECK(condition) Require((condition), #condition, __LINE__)

std::filesystem::path WriteFile(const std::filesystem::path& path,
                                const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  CHECK(out.good());
  return path;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  namespace platform = gufo::platform;

  const auto dir =
      fs::temp_directory_path() /
      ("gufo_mapped_file_test_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(dir);

  const std::size_t page = platform::SystemPageSize();
  CHECK(page > 0);
  std::vector<std::uint8_t> bytes(3 * page + 123);
  // Distinct page contents expose positional reads that use the wrong offset.
  std::uint32_t state = 0x12345678;
  for (auto& byte : bytes) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    byte = static_cast<std::uint8_t>(state);
  }
  const auto file = WriteFile(dir / "data.bin", bytes);
  const auto empty = WriteFile(dir / "empty.bin", {});

  {
    std::string error;
    auto mapped = platform::ReadOnlyMappedFile::Open(file, &error);
    CHECK(mapped.valid());
    CHECK(mapped.size() == bytes.size());
    CHECK(std::equal(bytes.begin(), bytes.end(), mapped.data()));
    CHECK(mapped.PlatformHandle() != platform::kInvalidFile);
    auto moved = std::move(mapped);
    CHECK(!mapped.valid());
    CHECK(mapped.PlatformHandle() == platform::kInvalidFile);
    CHECK(moved.valid() && moved.size() == bytes.size());

    CHECK(!platform::ReadOnlyMappedFile::Open(empty, &error).valid());
    CHECK(!error.empty());
    error.clear();
    CHECK(!platform::ReadOnlyMappedFile::Open(dir / "missing.bin", &error)
               .valid());
    CHECK(!error.empty());
  }

  {
    platform::RawMappedFile whole;
    CHECK(whole.Open(file));
    CHECK(whole.size == bytes.size());
    CHECK(std::equal(bytes.begin(), bytes.end(),
                     static_cast<const std::uint8_t*>(whole.data)));
    whole.Close();
    CHECK(whole.data == nullptr && whole.size == 0);

    platform::RawMappedFile none;
    CHECK(!none.Open(empty));
    CHECK(none.data == nullptr);
  }

  {
    // Unaligned offsets must work on every platform.
    const std::uint64_t offsets[] = {0, 1, page - 1, page, page + 17};
    for (const std::uint64_t offset : offsets) {
      const std::size_t length = page + 5;
      platform::RawMappedFile range;
      CHECK(range.OpenRange(file, offset, length));
      CHECK(range.size == length);
      const auto* data = static_cast<const std::uint8_t*>(range.data);
      CHECK(std::equal(data, data + length, bytes.begin() + offset));
      range.Close();
      CHECK(range.data == nullptr && range.view_base == nullptr);
    }
    platform::RawMappedFile zero;
    CHECK(!zero.OpenRange(file, page, 0));
  }

  {
    const auto handle = platform::OpenReadFile(file);
    CHECK(platform::FileValid(handle));
    CHECK(platform::FileSizeBytes(handle) == bytes.size());
    std::vector<std::uint8_t> out(64);
    CHECK(platform::ReadFileAt(handle, out.data(), out.size(), page + 3) ==
          out.size());
    CHECK(std::equal(out.begin(), out.end(), bytes.begin() + page + 3));
    // Short read at the tail, 0 at and past EOF.
    CHECK(platform::ReadFileAt(handle, out.data(), out.size(),
                               bytes.size() - 10) == 10);
    CHECK(platform::ReadFileAt(handle, out.data(), out.size(), bytes.size()) ==
          0);
    const auto reopened = platform::ReopenUnbuffered(handle);
    if (platform::FileValid(reopened)) {
      fs::resize_file(file, bytes.size());
      platform::CloseFile(reopened);
    }
    platform::CloseFile(handle);
    CHECK(!platform::FileValid(platform::OpenReadFile(dir / "missing.bin")));
    // A failing read reports 0, never a wrapped negative count.
    CHECK(platform::ReadFileAt(platform::kInvalidFile, out.data(), out.size(),
                               0) == 0);
  }

  {
    auto mapped = platform::ReadOnlyMappedFile::Open(file);
    CHECK(mapped.valid());
    bool direct = false;
    auto handle =
        platform::ReopenRandomReadFile(mapped.PlatformHandle(), direct);
    // A platform without direct I/O keeps the existing buffered fallback.
    if (!platform::FileValid(handle))
      handle = platform::DuplicateFile(mapped.PlatformHandle());
    CHECK(platform::FileValid(handle));
    const std::size_t stride = direct ? page : 1;
    std::atomic<bool> correct{true};
    std::vector<std::thread> readers;
    for (std::size_t worker = 0; worker < 8; ++worker) {
      readers.emplace_back([&, worker] {
        std::vector<std::uint8_t> storage(2 * page);
        void* buffer = storage.data();
        std::size_t space = storage.size();
        buffer = std::align(page, page, buffer, space);
        if (buffer == nullptr) {
          correct = false;
          return;
        }
        for (std::size_t i = 0; i < 64; ++i) {
          const auto offset =
              ((worker * 37 + i * 19) % (2 * page / stride)) * stride;
          if (platform::ReadFileAt(handle, buffer, page, offset) != page ||
              std::memcmp(buffer, bytes.data() + offset, page) != 0)
            correct = false;
          std::error_code ec;
          if (platform::ReadAt(handle, offset, buffer, page, ec) != page ||
              ec || std::memcmp(buffer, bytes.data() + offset, page) != 0)
            correct = false;
        }
      });
    }
    for (auto& reader : readers)
      reader.join();
    CHECK(correct);
    platform::CloseFile(handle);
  }

  fs::remove_all(dir);
}

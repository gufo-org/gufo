#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "tests/support/temporary_directory.hpp"

using gufo::hip::WeightUpload;

namespace {

void Require(bool ok, const std::string& message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

struct Device {
  void* data{nullptr};
  explicit Device(std::size_t size) {
    Require(hipMalloc(&data, size) == hipSuccess, "hipMalloc");
  }
  ~Device() { (void)hipFree(data); }
};

}  // namespace

int main() {
  TemporaryDirectory files("gufo-weight-upload");
  std::array<std::filesystem::path, 2> paths{files.path() / "first",
                                             files.path() / "second"};
  std::array<std::vector<std::uint8_t>, 2> source;
  source[0].resize((35U << 20) + 4079);
  source[1].resize((2U << 20) + 39);
  for (std::size_t shard = 0; shard < source.size(); ++shard) {
    for (std::size_t i = 0; i < source[shard].size(); ++i) {
      source[shard][i] = static_cast<std::uint8_t>((i * 1315423911ULL) ^
                                                   (i >> 13) ^ (shard * 79));
    }
    std::ofstream out(paths[shard], std::ios::binary);
    out.write(reinterpret_cast<const char*>(source[shard].data()),
              static_cast<std::streamsize>(source[shard].size()));
    Require(out.good(), "write shard");
  }

  std::array<gufo::core::GgufMappedRegion, 2> regions;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    regions[i] = {source[i].data(), source[i].size(),
                  gufo::platform::OpenReadFile(paths[i])};
    Require(gufo::platform::FileValid(regions[i].file_descriptor),
            "open original shard");
    std::filesystem::rename(paths[i], paths[i].string() + ".old");
    std::ofstream replacement(paths[i], std::ios::binary);
    replacement << "replacement must never supply model bytes";
  }
  constexpr std::size_t guard = 64;
  std::vector<std::uint8_t> expected(guard, 0xa5);
  struct Slice {
    std::uint32_t shard;
    std::size_t offset;
    std::size_t size;
    std::size_t destination;
  };
  std::vector<Slice> slices;
  const auto slice = [&](std::uint32_t shard, std::size_t offset,
                         std::size_t size) {
    slices.push_back({shard, offset, size, expected.size()});
    expected.insert(expected.end(), source[shard].begin() + offset,
                    source[shard].begin() + offset + size);
    expected.insert(expected.end(), guard, 0xa5);
  };
  // Several staging chunks, unaligned starts, and the partial last page at
  // EOF exercise aligned direct reads without copying surrounding bytes.
  slice(0, 19, source[0].size() - 19);
  slice(1, 33, source[1].size() - 33);
  for (std::size_t i = 0; i < 100; ++i) {
    slice(static_cast<std::uint32_t>(i % 2), i * 4093 + 7, i * 71 + 1);
  }
  Device device(expected.size());
  Require(hipMemset(device.data, 0xa5, expected.size()) == hipSuccess,
          "initialize guards");
  // The uploads run on non-blocking streams, which do not wait for the null
  // stream; a late guard fill would overwrite them.
  Require(hipDeviceSynchronize() == hipSuccess, "wait for guards");
  std::string error;
  auto upload = WeightUpload::Create(regions, &error);
  Require(upload != nullptr, error);
  // These addresses identify file offsets; their contents must never supply
  // upload bytes or enter HIP's pageable host-registration cache.
  for (auto& buffer : source)
    std::fill(buffer.begin(), buffer.end(), 0);
  for (const auto& s : slices) {
    Require(upload->CopyMapped(source[s.shard].data() + s.offset, s.size,
                               static_cast<char*>(device.data) + s.destination,
                               &error),
            error);
  }
  Require(!upload->CopyMapped(&error, 1, device.data, &error),
          "reject a pointer outside the source mappings");
  Require(upload->Copy(1, source[1].size(), 0, nullptr, &error),
          "empty range at EOF");
  Require(upload->Finish(&error), error);
  std::vector<std::uint8_t> actual(expected.size());
  Require(hipMemcpy(actual.data(), device.data, actual.size(),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "download uploaded bytes");
  Require(actual == expected, "all shard bytes and destination guards match");

  // A rejected range cancels queued reads and still drains active transfers.
  for (int i = 0; i < 32; ++i) {
    Require(upload->Copy(0, 0, 4096, static_cast<char*>(device.data) + i * 4096,
                         &error),
            error);
  }
  Require(!upload->Copy(1, std::numeric_limits<std::uint64_t>::max(), 1,
                        device.data, &error),
          "reject overflowing file offset");
  Require(!upload->Finish(&error), "failed upload reports failure after drain");
  upload.reset();

  upload = WeightUpload::Create(regions, &error);
  Require(upload != nullptr, "reopen shards");
  std::filesystem::resize_file(paths[1].string() + ".old", 17);
  Require(upload->Copy(1, 4096, 8192, device.data, &error),
          "queue a range from the original file extent");
  Require(!upload->Finish(&error), "a truncated shard fails its pending read");
  std::puts(
      "PASS: byte-exact upload, EOF, guards, bounds and failed-read drain");
  upload.reset();
  for (const auto& region : regions)
    gufo::platform::CloseFile(region.file_descriptor);
}

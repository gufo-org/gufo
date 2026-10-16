#include "src/models/qwen38_flash_next/kernels/rocm/snapshot_copy.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next::rocm;
namespace {
void Check(hipError_t result) {
  assert(result == hipSuccess);
}
void Copies(std::span<std::byte> host, void* device, hipStream_t stream,
            const std::vector<std::byte>& expected) {
  constexpr std::array<std::size_t, 10> sizes{
      0, 1, 15, 16, 4095, 4096, 4097, 65535, 1048576 + 13, 3 * 1048576};
  for (const auto size : sizes) {
    for (const auto offset : {0U, 1U, 3U, 15U, 16U}) {
      const auto destination = host.size() - size - offset;
      const auto source = expected.size() - size - offset;
      std::ranges::fill(host, std::byte{0xcd});
      Check(qfn::CopySnapshotToHost(host.data() + destination,
                                    static_cast<std::byte*>(device) + source,
                                    size, stream));
      Check(hipStreamSynchronize(stream));
      assert(std::equal(expected.begin() + source,
                        expected.begin() + source + size,
                        host.begin() + destination));
      assert(std::all_of(host.begin(), host.begin() + destination,
                         [](auto b) { return b == std::byte{0xcd}; }));
      assert(std::all_of(host.begin() + destination + size, host.end(),
                         [](auto b) { return b == std::byte{0xcd}; }));
      assert(hipGetLastError() == hipSuccess);
    }
  }
}
}  // namespace

int main() {
  constexpr std::size_t bytes = 4 * 1048576;
  std::vector<std::byte> expected(bytes);
  for (std::size_t i = 0; i < bytes; ++i)
    expected[i] = std::byte((i * 37 + (i >> 11) + (i >> 19)) & 255);
  void* device{};
  Check(hipMalloc(&device, bytes));
  Check(hipMemcpy(device, expected.data(), bytes, hipMemcpyHostToDevice));
  hipStream_t stream{};
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Production registration, including interior device aliases and copies
  // ending at the allocation boundary. Every untouched destination byte is
  // independently checked, as are odd/unaligned fallback ranges.
  auto* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(mapping != MAP_FAILED);
  assert(madvise(mapping, bytes, MADV_NOHUGEPAGE) == 0);
  assert(madvise(mapping, bytes, MADV_POPULATE_WRITE) == 0);
  Check(hipHostRegister(mapping, bytes, hipHostRegisterDefault));
  Copies({static_cast<std::byte*>(mapping), bytes}, device, stream, expected);
  Check(hipHostUnregister(mapping));
  // The same address must fall back once its registration is removed.
  Copies({static_cast<std::byte*>(mapping), bytes}, device, stream, expected);
  assert(munmap(mapping, bytes) == 0);

  std::vector<std::byte> pageable(bytes);
  Copies(pageable, device, stream, expected);
  Check(hipStreamDestroy(stream));
  Check(hipFree(device));
  std::cout << "PASS: registered/pageable copies, exact bytes, all guards, "
               "aligned/unaligned offsets and registration removal\n";
}

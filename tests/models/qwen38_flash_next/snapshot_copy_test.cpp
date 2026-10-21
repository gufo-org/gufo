#include "src/core/hip/snapshot_copy.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <span>
#include <thread>
#include <vector>

namespace gpu = gufo::hip;
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
      Check(gpu::CopySnapshotToHost(host.data() + destination,
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
void PeerCapture(std::span<std::byte> host, void* device,
                 hipStream_t copy_stream,
                 const std::vector<std::byte>& expected, bool upload = false) {
  hipStream_t compute{};
  Check(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));
  void* scratch{};
  Check(hipMalloc(&scratch, 4096));
  Check(hipStreamBeginCapture(compute, hipStreamCaptureModeThreadLocal));
  Check(hipMemsetAsync(scratch, 0x39, 4096, compute));
  // A frozen slot can download while another worker records a decode graph.
  // Its alias lookup and kernel belong only to the independent copy stream.
  std::thread worker([&] {
    if (upload)
      Check(gpu::CopySnapshotToDevice(device, host.data(), host.size(),
                                      copy_stream));
    else
      Check(gpu::CopySnapshotToHost(host.data(), device, host.size(),
                                    copy_stream));
    Check(hipStreamSynchronize(copy_stream));
  });
  worker.join();
  hipGraph_t graph{};
  Check(hipStreamEndCapture(compute, &graph));
  assert(std::ranges::equal(host, expected));
  std::vector<std::byte> actual_device(host.size());
  Check(hipMemcpy(actual_device.data(), device, actual_device.size(),
                  hipMemcpyDeviceToHost));
  assert(actual_device == expected);
  hipGraphExec_t replay{};
  Check(hipGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
  std::ranges::fill(host, std::byte{0xcd});
  if (upload)
    Check(hipMemset(device, 0xcd, host.size()));
  for (int i = 0; i < 2; ++i) {
    Check(hipMemsetAsync(scratch, 0, 4096, compute));
    Check(hipGraphLaunch(replay, compute));
    Check(hipStreamSynchronize(compute));
    assert(
        std::ranges::all_of(host, [](auto b) { return b == std::byte{0xcd}; }));
    if (upload) {
      Check(hipMemcpy(actual_device.data(), device, actual_device.size(),
                      hipMemcpyDeviceToHost));
      assert(std::ranges::all_of(actual_device,
                                 [](auto b) { return b == std::byte{0xcd}; }));
    }
    std::array<std::byte, 4096> actual;
    Check(hipMemcpy(actual.data(), scratch, actual.size(),
                    hipMemcpyDeviceToHost));
    assert(std::ranges::all_of(actual,
                               [](auto b) { return b == std::byte{0x39}; }));
  }
  Check(hipGraphExecDestroy(replay));
  Check(hipGraphDestroy(graph));
  Check(hipFree(scratch));
  Check(hipStreamDestroy(compute));
}
void Uploads(std::span<std::byte> host, void* device, hipStream_t stream,
             const std::vector<std::byte>& expected) {
  std::ranges::copy(expected, host.begin());
  std::vector<std::byte> actual(host.size());
  for (const std::size_t size :
       {0, 1, 15, 16, 4095, 4096, 4097, 65535, 1048576 + 13, 3 * 1048576}) {
    for (const auto offset : {0U, 1U, 3U, 15U, 16U}) {
      const auto first = host.size() - size - offset;
      Check(hipMemsetAsync(device, 0xcd, host.size(), stream));
      Check(gpu::CopySnapshotToDevice(static_cast<std::byte*>(device) + first,
                                      host.data() + first, size, stream));
      Check(hipStreamSynchronize(stream));
      Check(hipMemcpy(actual.data(), device, actual.size(),
                      hipMemcpyDeviceToHost));
      assert(std::equal(expected.begin() + first,
                        expected.begin() + first + size,
                        actual.begin() + first));
      assert(std::all_of(actual.begin(), actual.begin() + first,
                         [](auto b) { return b == std::byte{0xcd}; }));
      assert(std::all_of(actual.begin() + first + size, actual.end(),
                         [](auto b) { return b == std::byte{0xcd}; }));
      assert(hipGetLastError() == hipSuccess);
    }
  }
  Check(hipMemcpy(device, expected.data(), expected.size(),
                  hipMemcpyHostToDevice));
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
  Uploads({static_cast<std::byte*>(mapping), bytes}, device, stream, expected);
  PeerCapture({static_cast<std::byte*>(mapping), bytes}, device, stream,
              expected, true);
  Check(hipMemcpy(device, expected.data(), expected.size(),
                  hipMemcpyHostToDevice));
  PeerCapture({static_cast<std::byte*>(mapping), bytes}, device, stream,
              expected);
  Check(hipHostUnregister(mapping));
  // The same address must fall back once its registration is removed.
  Copies({static_cast<std::byte*>(mapping), bytes}, device, stream, expected);
  Uploads({static_cast<std::byte*>(mapping), bytes}, device, stream, expected);
  assert(munmap(mapping, bytes) == 0);

  std::vector<std::byte> pageable(bytes);
  Copies(pageable, device, stream, expected);
  Uploads(pageable, device, stream, expected);
  void* borrowed{};
  Check(hipMalloc(&borrowed, bytes));
  Check(gpu::CopySnapshotToDevice(borrowed, device, bytes, stream));
  Check(hipStreamSynchronize(stream));
  Check(hipMemcpy(pageable.data(), borrowed, bytes, hipMemcpyDeviceToHost));
  assert(pageable == expected);
  Check(hipFree(borrowed));
  Check(hipStreamDestroy(stream));
  Check(hipFree(device));
  std::cout << "PASS: registered/pageable copies, exact bytes, all guards, "
               "aligned/unaligned offsets, registration removal and peer graph "
               "capture/replay\n";
}

// Runtime memory policy must retain managed fallback and registered cache
// storage, including direct CPU access after GPU completion.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/managed_alloc.hpp"

namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

__global__ void Transform(std::uint32_t* values, std::size_t count) {
  const auto i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count)
    values[i] = values[i] * 1664525U + 1013904223U;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool restricted =
        argc == 2 && std::string_view(argv[1]) == "--deny-long-term-pins";
    if (restricted) {
      rlimit limit{};
      Require(getrlimit(RLIMIT_MEMLOCK, &limit) == 0, "read memlock limit");
      limit.rlim_cur = 0;
      Require(setrlimit(RLIMIT_MEMLOCK, &limit) == 0, "restrict memlock");
    }
    constexpr std::size_t count = (16ULL << 20) / sizeof(std::uint32_t);
    constexpr auto bytes = count * sizeof(std::uint32_t);
    std::vector<std::uint32_t> input(count), expected(count), actual(count);
    for (std::size_t i = 0; i < count; ++i) {
      input[i] = static_cast<std::uint32_t>(i * 2654435761ULL);
      expected[i] = input[i] * 1664525U + 1013904223U;
    }
    for (int kind = 0; kind < 5; ++kind) {
      gufo::hip::HostAllocations host;
      void* memory{};
      switch (kind) {
        case 0:
          Check(gufo::hip::AllocateManaged(&memory, bytes));
          break;
        case 1:
          Check(hipHostMalloc(&memory, bytes));
          break;
        case 2:
          memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
          Require(memory != MAP_FAILED, "cache mapping failed");
          Check(hipHostRegister(memory, bytes, hipHostRegisterDefault));
          break;
        case 3:
          Check(gufo::hip::AllocateDevice(&memory, bytes));
          break;
        default:
          Check(host.Allocate(&memory, bytes, hipHostMallocCoherent));
          Require(host.PinnedBytes() == 0 || host.PinnedBytes() == bytes,
                  "partial host pin");
          if (restricted)
            Require(host.PinnedBytes() == 0, "memlock restriction ignored");
      }
      Check(hipMemcpy(memory, input.data(), bytes, hipMemcpyHostToDevice));
      Transform<<<(count + 255) / 256, 256>>>(
          static_cast<std::uint32_t*>(memory), count);
      Check(hipDeviceSynchronize());
      Check(hipMemcpy(actual.data(), memory, bytes, hipMemcpyDeviceToHost));
      Require(actual == expected, "GPU result differs from integer oracle");
      if (kind != 3) {
        auto* cpu = static_cast<std::uint32_t*>(memory);
        for (std::size_t i = 0; i < count; ++i) {
          Require(cpu[i] == expected[i], "GPU writes not visible to CPU");
          cpu[i] = input[i];
        }
        Transform<<<(count + 255) / 256, 256>>>(cpu, count);
        Check(hipDeviceSynchronize());
        for (std::size_t i = 0; i < count; ++i)
          Require(cpu[i] == expected[i], "CPU writes not visible to GPU");
      }
      if (kind == 0 || kind == 3)
        Check(hipFree(memory));
      else if (kind == 1)
        Check(hipHostFree(memory));
      else if (kind == 2) {
        Check(hipHostUnregister(memory));
        Require(munmap(memory, bytes) == 0, "cache unmap failed");
      } else {
        host.Reset();
        Require(host.PinnedBytes() == 0, "reset retained a host pin");
        Check(host.Allocate(&memory, bytes));
        Check(hipMemcpy(memory, input.data(), bytes, hipMemcpyHostToDevice));
        Check(hipMemcpy(actual.data(), memory, bytes, hipMemcpyDeviceToHost));
        Require(actual == input, "host owner cannot be reused after reset");
      }
    }
    std::cout
        << "PASS: managed, pinned, registered, device and owned host memory\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

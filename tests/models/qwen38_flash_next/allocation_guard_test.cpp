#include <hip/hip_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    std::cerr << operation << ": " << hipGetErrorString(error) << '\n';
    std::exit(1);
  }
}

/// True when the child process aborts on a copy touching [pointer, +bytes).
bool CopyAborts(const void* pointer, std::size_t bytes) {
  std::cout.flush();
  const pid_t child = fork();
  if (child == 0) {
    q::CheckDeviceRange("allocation_guard_test", "src", pointer, bytes);
    _exit(0);
  }
  int status = 0;
  waitpid(child, &status, 0);
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

}  // namespace

int main() {
  constexpr std::size_t kBytes = std::size_t{32} << 20;

  // A freed pinned host buffer leaves no tombstone: ordinary host allocations
  // reuse those address ranges and must stay copyable.
  std::uint8_t* pinned = nullptr;
  CheckHip(q::GuardedHostMalloc("test.stage", &pinned, kBytes), "pinned alloc");
  q::ForgetAllocation(pinned, "test.stage free");
  CheckHip(hipHostFree(pinned), "pinned free");
  Expect(!CopyAborts(pinned, 122880),
         "a freed pinned host range is not a use-after-free tombstone");

  // A freed device buffer still is one.
  void* device = nullptr;
  CheckHip(hipMalloc(&device, kBytes), "device alloc");
  q::RecordAllocation("test.device", device, kBytes);
  q::ForgetAllocation(device, "test.device free");
  CheckHip(hipFree(device), "device free");
  Expect(CopyAborts(device, 4096), "a freed device range still aborts");

  std::cout << "allocation guard ok\n";
  return 0;
}

// Time per mem::RemapBarrier::Settle() with the HIP backend installed
// (mem::InstallRemapBarrierDevice: its drain and translation flush), one
// short kernel launched before each call so the drain has work to wait for.
// Built from the repository root: src/mem/remap_barrier.cpp and
// src/core/hip/remap_barrier_device.cpp (with -D__HIP_PLATFORM_AMD__)
// compiled as C++, this file with clang++ -x hip --offload-arch=gfx1151,
// linked with -lamdhip64. settle.txt runs it against remap_barrier.cpp of
// main 86bb63aa (base: idle time before the flush) and without the idle time
// (new), alternating.
//
//   settle <calls> <label>
//   prints p50, p99 and mean in us, the barrier's own micros per call
//   (stats), and how many calls found no quiet device or failed to flush.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/core/hip/remap_barrier_device.hpp"
#include "src/mem/remap_barrier.hpp"

__global__ void Touch(unsigned* p) {
  p[threadIdx.x] += 1;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 1000;
  const char* label = argc > 2 ? argv[2] : "?";
  unsigned* buf = nullptr;
  if (hipMalloc(&buf, 4096) != hipSuccess)
    return 1;
  gufo::mem::InstallRemapBarrierDevice();
  for (int i = 0; i < 50; ++i) {
    Touch<<<1, 64>>>(buf);
    if (!gufo::mem::RemapBarrier::Settle())
      return 2;
  }
  const auto before = gufo::mem::RemapBarrier::Stats();
  std::vector<double> us;
  us.reserve(n);
  for (int i = 0; i < n; ++i) {
    Touch<<<1, 64>>>(buf);
    const auto t0 = std::chrono::steady_clock::now();
    if (!gufo::mem::RemapBarrier::Settle())
      return 3;
    us.push_back(std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - t0)
                     .count());
  }
  const auto after = gufo::mem::RemapBarrier::Stats();
  std::vector<double> s = us;
  std::sort(s.begin(), s.end());
  double sum = 0;
  for (double v : us)
    sum += v;
  std::printf(
      "%s n=%d p50=%.1f p99=%.1f mean=%.1f stats=%.1f unquiet=%llu "
      "unflushed=%llu\n",
      label, n, s[n / 2], s[n * 99 / 100], sum / n,
      double(after.micros - before.micros) /
          double(after.settles - before.settles),
      (unsigned long long)(after.unquiet - before.unquiet),
      (unsigned long long)(after.unflushed - before.unflushed));
  (void)hipFree(buf);
  return 0;
}

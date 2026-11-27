// Keeps the GPU busy from a second process: short kernels back to back for
// argv[1] seconds.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
__global__ void Spin(uint32_t* d, int iters) {
  uint32_t x = d[threadIdx.x];
  for (int i = 0; i < iters; ++i)
    x = x * 1664525u + 1013904223u;
  d[threadIdx.x] = x;
}
int main(int argc, char** argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 30;
  uint32_t* buf;
  if (hipMalloc(&buf, 4096) != hipSuccess)
    return 1;
  hipStream_t s;
  if (hipStreamCreate(&s) != hipSuccess)
    return 1;
  const auto t0 = std::chrono::steady_clock::now();
  long n = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
             .count() < secs) {
    for (int i = 0; i < 64; ++i)
      Spin<<<1, 256, 0, s>>>(buf, 2000);
    if (hipStreamSynchronize(s) != hipSuccess)
      return 1;
    n += 64;
  }
  printf("busy: %ld kernels in %.0f s\n", n, secs);
  return 0;
}

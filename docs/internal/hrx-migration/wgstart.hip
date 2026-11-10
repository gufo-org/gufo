// Scratch probe: per-workgroup startup cost. Tiny bounded kernels with a
// sweep of grid sizes and kernel-argument sizes; time per launch, 20 chained
// launches per event pair.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

struct Big { unsigned v[64]; };  // 256 bytes of kernel arguments

__global__ void Small(int iters, unsigned* sink) {
  unsigned a = threadIdx.x ^ blockIdx.x;
  for (int i = 0; i < iters; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = a;
}

__global__ void Wide(Big args, int iters, unsigned* sink) {
  unsigned a = threadIdx.x ^ blockIdx.x ^ args.v[threadIdx.x & 63];
  for (int i = 0; i < iters; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = a;
}

template <typename F>
static float PerLaunchUs(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  for (int i = 0; i < 5; ++i) launch();
  std::vector<float> v;
  for (int r = 0; r < 11; ++r) {
    OK(hipDeviceSynchronize());
    OK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 20; ++i) launch();
    OK(hipEventRecord(b, nullptr));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t * 1000.0f / 20.0f);
  }
  std::sort(v.begin(), v.end());
  return v[5];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  Big big{}; for (int i = 0; i < 64; ++i) big.v[i] = i;
  const int grids[] = {20, 80, 640, 6400, 64000};
  for (int threads : {64, 256}) {
    for (int g : grids) {
      const float s = PerLaunchUs([&] { Small<<<g, threads>>>(16, sink); });
      const float w = PerLaunchUs([&] { Wide<<<g, threads>>>(big, 16, sink); });
      std::printf("threads %3d grid %6d  small-args %8.2f us  256B-args %8.2f us  per-WG %.3f us\n",
                  threads, g, s, w, s / g);
    }
  }
  return 0;
}

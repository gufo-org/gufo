// Scratch probe: address-translation and write behavior of hipMalloc memory.
// random4k: each thread reads 16 B from a pseudo-random 4 KiB page of a 1 GiB
// buffer (TLB bound when pages are small); stride: rows of 20 KiB read 64 B per
// row (the HC mixer's stream layout); write: streaming 16 B stores.
// All loops are bounded.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__global__ void Random4k(const u32x4* __restrict__ src, size_t pages, int per_thread,
                         unsigned* sink) {
  unsigned acc = 0;
  unsigned x = (blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u + 1u;
  for (int i = 0; i < per_thread; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    const size_t page = x % pages;
    u32x4 v = src[page * 256 + (threadIdx.x & 255)];
    acc += v.x ^ v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void Stride(const u32x4* __restrict__ src, size_t rows, size_t row_vec,
                       unsigned* sink) {
  unsigned acc = 0;
  const size_t lane = threadIdx.x & 3;  // 4 lanes x 16 B = 64 B per row
  for (size_t r = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 4; r < rows;
       r += (size_t)gridDim.x * blockDim.x / 4) {
    u32x4 v = src[r * row_vec + lane];
    acc += v.y ^ v.z;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void Write(u32x4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = u32x4{(unsigned)i, 1u, 2u, 3u};
}

template <typename F>
static float Median(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  launch(); OK(hipDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < 9; ++r) {
    OK(hipEventRecord(a, nullptr)); launch(); OK(hipEventRecord(b, nullptr));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
  }
  std::sort(v.begin(), v.end());
  return v[4];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  const size_t bytes = 1ull << 30;
  u32x4* buf; OK(hipMalloc(&buf, bytes)); OK(hipMemset(buf, 1, bytes));
  const int grid = 20 * 64, block = 256, per_thread = 64;
  const float tr = Median([&] { Random4k<<<grid, block>>>(buf, bytes / 4096, per_thread, sink); });
  const double accesses = (double)grid * block * per_thread;
  std::printf("random4k  %.3f ms  %.1f M accesses/s\n", tr, accesses / tr / 1e3);
  const size_t row_vec = 20480 / 16, rows = bytes / 20480;
  const float ts = Median([&] { Stride<<<grid, block>>>(buf, rows, row_vec, sink); });
  std::printf("stride20k %.3f ms  %.1f M rows/s\n", ts, rows / ts / 1e3);
  const size_t wbytes = 256u << 20;
  const float tw = Median([&] { Write<<<grid, block>>>(buf, wbytes / 16); });
  std::printf("write     %.3f ms  %.1f GB/s\n", tw, wbytes / tw / 1e6);
  return 0;
}

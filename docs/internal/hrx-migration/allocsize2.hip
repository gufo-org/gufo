// Scratch probe: read bandwidth of separate hipMalloc allocations by size.
// Each size gets its own allocation (as model buffers do); sets that fit in
// L2/MALL are re-read many times, larger ones stream. Bounded loops only.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__global__ void ReadLoop(const u32x4* __restrict__ src, size_t n, int passes, unsigned* sink) {
  unsigned acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; ++p)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
      u32x4 v = src[i];
      acc ^= v.x + v.y + v.z + v.w + p;
    }
  if (acc == 0x12345678u) sink[0] = acc;
}

int main(int argc, char** argv) {
  const int mode = argc > 1 ? std::atoi(argv[1]) : 0;  // 0 memcpy pageable, 1 memset, 2 memcpy pinned, 3 memcpy then kernel rewrite
  unsigned* sink; OK(hipMalloc(&sink, 4));
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  const size_t kib[] = {256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144};
  for (size_t k : kib) {
    const size_t bytes = k * 1024;
    u32x4* buf; OK(hipMalloc(&buf, bytes));
    if (mode == 1) {
      OK(hipMemset(buf, 1, bytes));
    } else if (mode == 2) {
      void* pinned; OK(hipHostMalloc(&pinned, bytes, hipHostMallocDefault));
      std::memset(pinned, 1, bytes);
      OK(hipMemcpy(buf, pinned, bytes, hipMemcpyHostToDevice));
      OK(hipHostFree(pinned));
    } else {
      std::vector<unsigned char> host(bytes, 1);
      OK(hipMemcpy(buf, host.data(), bytes, hipMemcpyHostToDevice));  // as Upload() does
      if (mode == 3) OK(hipMemset(buf, 1, bytes));
    }
    OK(hipDeviceSynchronize());
    const int passes = bytes <= (32u << 20) ? (int)std::max<size_t>(4, (256u << 20) / bytes) : 2;
    ReadLoop<<<640, 256>>>(buf, bytes / 16, 1, sink); OK(hipDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < 7; ++r) {
      OK(hipEventRecord(a, nullptr));
      ReadLoop<<<640, 256>>>(buf, bytes / 16, passes, sink);
      OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
      float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
    }
    std::sort(v.begin(), v.end());
    std::printf("%7zu KiB  %8.1f GB/s\n", k, (double)bytes * passes / v[3] / 1e6);
    OK(hipFree(buf));
  }
  return 0;
}

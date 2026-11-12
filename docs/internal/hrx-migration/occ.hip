// Scratch probe: workgroup residency versus dynamic LDS size. Each workgroup
// does a fixed, bounded amount of ALU work, so time per grid falls as more
// workgroups fit on the GPU at once. A runtime that over-allocates LDS shows a
// step at a smaller LDS size.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

// Dependent loads through a random permutation: latency bound, so throughput
// scales with resident waves.
__global__ void Work(const unsigned* __restrict__ next, int iters, unsigned* sink) {
  extern __shared__ unsigned lds[];
  lds[threadIdx.x] = threadIdx.x;
  __syncthreads();
  unsigned p = (blockIdx.x * blockDim.x + threadIdx.x) * 7919u & ((1u << 22) - 1);
  for (int i = 0; i < iters; ++i) p = next[p];
  if (p == 0xffffffffu) sink[0] = p + lds[threadIdx.x & 255];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  const unsigned n = 1u << 22;  // 16 MiB permutation, MALL resident: latency bound
  std::vector<unsigned> perm(n);
  for (unsigned i = 0; i < n; ++i) perm[i] = i;
  unsigned long long x = 88172645463325252ull;
  for (unsigned i = n - 1; i > 0; --i) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    std::swap(perm[i], perm[x % (i + 1)]);
  }
  unsigned* next; OK(hipMalloc(&next, n * 4ull));
  unsigned* staged; OK(hipMalloc(&staged, n * 4ull));
  OK(hipMemcpy(staged, perm.data(), n * 4ull, hipMemcpyHostToDevice));
  // Rewrite on the GPU so no host-upload cache state remains.
  OK(hipMemcpy(next, staged, n * 4ull, hipMemcpyDeviceToDevice));
  OK(hipDeviceSynchronize()); OK(hipFree(staged));
  OK(hipFuncSetAttribute((const void*)Work, hipFuncAttributeMaxDynamicSharedMemorySize, 65536));
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  const int grid = 2000, block = 256, iters = 256;
  for (int kib : {1, 8, 16, 24, 32, 40, 44, 48, 56, 64}) {
    const size_t lds = (size_t)kib * 1024;
    Work<<<grid, block, lds>>>(next, iters, sink); OK(hipDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < 7; ++r) {
      OK(hipEventRecord(a, nullptr));
      Work<<<grid, block, lds>>>(next, iters, sink);
      OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
      float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
    }
    std::sort(v.begin(), v.end());
    std::printf("lds %2d KiB  %7.3f ms\n", kib, v[3]);
  }
  return 0;
}

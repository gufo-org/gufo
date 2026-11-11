// Scratch probe: read bandwidth by allocation kind and working-set size.
// L2-resident sets (<= 2 MiB) expose cache policy; large sets expose DRAM.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// Each block repeatedly reads the whole working set (wraps), so sets that fit
// in L2 are served from L2 after the first pass.
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

static float Time(const u32x4* buf, size_t bytes, int passes, unsigned* sink) {
  size_t n = bytes / 16;
  int blocks = 20 * 32;
  ReadLoop<<<blocks, 256>>>(buf, n, 1, sink);
  OK(hipDeviceSynchronize());
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  std::vector<float> ms;
  for (int r = 0; r < 7; ++r) {
    OK(hipEventRecord(a, nullptr));
    ReadLoop<<<blocks, 256>>>(buf, n, passes, sink);
    OK(hipEventRecord(b, nullptr));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  return ms[3];
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  unsigned* sink; OK(hipMalloc(&sink, 4));
  const size_t sizes[] = {1u << 20, 2u << 20, 8u << 20, 256u << 20};
  const size_t max_bytes = 256u << 20;
  // Kinds: device malloc, host malloc (pinned), host-registered anonymous memory.
  void* dev; OK(hipMalloc(&dev, max_bytes));
  void* hostm; OK(hipHostMalloc(&hostm, max_bytes, hipHostMallocDefault));
  void* reg = mmap(nullptr, max_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  std::memset(reg, 1, max_bytes);
  OK(hipHostRegister(reg, max_bytes, hipHostRegisterMapped));
  void* reg_dev; OK(hipHostGetDevicePointer(&reg_dev, reg, 0));
  OK(hipMemset(dev, 1, max_bytes));
  std::memset(hostm, 1, max_bytes);
  struct { const char* name; void* p; } kinds[] = {{"hipMalloc", dev}, {"hipHostMalloc", hostm}, {"hipHostRegister", reg_dev}};
  for (auto& k : kinds) {
    if (only && std::strcmp(only, k.name) != 0) continue;
    for (size_t s : sizes) {
      int passes = s <= (8u << 20) ? 200 : 4;
      float ms = Time((const u32x4*)k.p, s, passes, sink);
      std::printf("%-16s %6zu KiB  %9.1f GB/s\n", k.name, s >> 10, (double)s * passes / ms / 1e6);
    }
  }
  return 0;
}

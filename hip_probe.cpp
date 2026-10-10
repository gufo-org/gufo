#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int n = 0;
  printf("hipGetDeviceCount: %s (n=%d)\n", hipGetErrorString(hipGetDeviceCount(&n)), n);
  size_t freeB = 0, totalB = 0;
  printf("hipMemGetInfo: %s free=%zu MiB total=%zu MiB\n",
         hipGetErrorString(hipMemGetInfo(&freeB, &totalB)), freeB >> 20, totalB >> 20);
  hipStream_t s{};
  printf("hipStreamCreate: %s\n", hipGetErrorString(hipStreamCreate(&s)));
  void* p = nullptr;
  printf("hipMalloc 256MiB: %s\n", hipGetErrorString(hipMalloc(&p, 256ull << 20)));
  return 0;
}

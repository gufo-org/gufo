#pragma once
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define C(x)                                                \
  do {                                                      \
    hipError_t e_ = (x);                                    \
    if (e_ != hipSuccess) {                                 \
      printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, \
             hipGetErrorString(e_));                        \
      std::exit(1);                                         \
    }                                                       \
  } while (0)
inline double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
struct Vmm {
  hipMemAllocationProp prop{};
  hipMemAccessDesc acc{};
  Vmm() {
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    acc.location = prop.location;
    acc.flags = hipMemAccessFlagsProtReadWrite;
  }
  // Reserve `total` VA and back it with `chunk`-sized handles.
  void* alloc(size_t total, size_t chunk,
              std::vector<hipMemGenericAllocationHandle_t>* hs) {
    hipDeviceptr_t base{};
    C(hipMemAddressReserve(&base, total, 0, nullptr, 0));
    for (size_t off = 0; off < total; off += chunk) {
      hipMemGenericAllocationHandle_t h;
      C(hipMemCreate(&h, chunk, &prop, 0));
      C(hipMemMap((char*)base + off, chunk, 0, h, 0));
      hs->push_back(h);
    }
    C(hipMemSetAccess(base, total, &acc, 1));
    return base;
  }
  void free(void* base, size_t total, size_t chunk,
            std::vector<hipMemGenericAllocationHandle_t>& hs) {
    for (size_t i = 0; i < hs.size(); ++i) {
      C(hipMemUnmap((char*)base + i * chunk, chunk));
      C(hipMemRelease(hs[i]));
    }
    hs.clear();
    C(hipMemAddressFree(base, total));
  }
};

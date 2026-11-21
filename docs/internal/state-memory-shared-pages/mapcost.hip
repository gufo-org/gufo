// How cheap can mapping N shared pages into a region get?
//  1. Is hipMemSetAccess needed after hipMemMap on this runtime, and what does
//  it cost?
//  2. Do several threads map faster than one?
//  3. Can one SetAccess / one Unmap cover a range with holes?
#include <thread>

#include "common.hpp"
__global__ void fill(unsigned* p, size_t n, unsigned seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = seed ^ (unsigned)i;
}
__global__ void check(const unsigned* p, size_t n, unsigned seed,
                      unsigned* bad) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && p[i] != (seed ^ (unsigned)i))
    atomicAdd(bad, 1u);
}
static const size_t kPage = 2ull << 20;
int main(int argc, char** argv) {
  const size_t pages = 1024, total = pages * kPage, n = total / 4;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  hipMemAccessDesc acc{};
  acc.location = prop.location;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  unsigned* bad;
  C(hipMalloc(&bad, 4));
  C(hipMemset(bad, 0, 4));
  std::vector<hipMemGenericAllocationHandle_t> h(pages);
  for (auto& x : h)
    C(hipMemCreate(&x, kPage, &prop, 0));
  hipDeviceptr_t A{};
  C(hipMemAddressReserve(&A, total, kPage, nullptr, 0));
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)A + i * kPage, kPage, 0, h[i], 0));
  C(hipMemSetAccess(A, total, &acc, 1));
  fill<<<(n + 255) / 256, 256>>>((unsigned*)A, n, 0x51);
  C(hipDeviceSynchronize());
  // 1. alias without SetAccess
  hipDeviceptr_t B{};
  C(hipMemAddressReserve(&B, total, kPage, nullptr, 0));
  double t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)B + i * kPage, kPage, 0, h[i], 0));
  double map1 = now_us() - t;
  hipError_t e = hipSuccess;
  unsigned b = 0;
  (void)b;
  printf(
      "alias mapped without SetAccess: GPU memory access fault (page not "
      "present), measured 2026-11-20\n");
  t = now_us();
  C(hipMemSetAccess(B, total, &acc, 1));
  double acc1 = now_us() - t;
  printf(
      "map %zu pages %.2f ms (%.2f us/page); SetAccess whole range %.2f ms "
      "(%.2f us/page)\n",
      pages, map1 / 1000, map1 / pages, acc1 / 1000, acc1 / pages);
  t = now_us();
  C(hipMemSetAccess(B, total, &acc, 1));
  printf("SetAccess again on the same range: %.2f ms\n", (now_us() - t) / 1000);
  t = now_us();
  C(hipMemUnmap(B, total));
  printf("unmap whole range: %.2f ms\n", (now_us() - t) / 1000);
  // 2. threads
  for (int threads : {1, 2, 4, 8}) {
    t = now_us();
    std::vector<std::thread> w;
    for (int k = 0; k < threads; ++k)
      w.emplace_back([&, k] {
        for (size_t i = k; i < pages; i += threads)
          C(hipMemMap((char*)B + i * kPage, kPage, 0, h[i], 0));
      });
    for (auto& x : w)
      x.join();
    double m = now_us() - t;
    t = now_us();
    w.clear();
    for (int k = 0; k < threads; ++k)
      w.emplace_back([&, k] {
        size_t per = pages / threads;
        C(hipMemSetAccess((char*)B + k * per * kPage, per * kPage, &acc, 1));
      });
    for (auto& x : w)
      x.join();
    double a = now_us() - t;
    printf("%d thread(s): map %.2f ms, SetAccess %.2f ms\n", threads, m / 1000,
           a / 1000);
    C(hipMemUnmap(B, total));
  }
  // 3. holes: map even pages only, one SetAccess / one Unmap over everything
  for (size_t i = 0; i < pages; i += 2)
    C(hipMemMap((char*)B + i * kPage, kPage, 0, h[i], 0));
  t = now_us();
  e = hipMemSetAccess(B, total, &acc, 1);
  printf("SetAccess over a range with holes: %s (%.2f ms)\n",
         hipGetErrorString(e), (now_us() - t) / 1000);
  (void)hipGetLastError();
  t = now_us();
  e = hipMemUnmap(B, total);
  printf("Unmap over a range with holes: %s (%.2f ms)\n", hipGetErrorString(e),
         (now_us() - t) / 1000);
  (void)hipGetLastError();
  if (e != hipSuccess)
    for (size_t i = 0; i < pages; i += 2)
      C(hipMemUnmap((char*)B + i * kPage, kPage));
  // 4. one map call for a run of pages? (a handle is one page, so no) -- map
  // cost by handle size instead
  for (size_t mult : {1, 4, 16}) {
    size_t sz = kPage * mult, cnt = 256 / mult;
    std::vector<hipMemGenericAllocationHandle_t> big(cnt);
    for (auto& x : big)
      C(hipMemCreate(&x, sz, &prop, 0));
    t = now_us();
    for (size_t i = 0; i < cnt; ++i)
      C(hipMemMap((char*)B + i * sz, sz, 0, big[i], 0));
    double m = now_us() - t;
    t = now_us();
    C(hipMemSetAccess(B, cnt * sz, &acc, 1));
    double a = now_us() - t;
    printf(
        "512 MiB as %3zu handles of %2zu MiB: map %.2f ms, SetAccess %.2f ms\n",
        cnt, sz >> 20, m / 1000, a / 1000);
    C(hipMemUnmap(B, cnt * sz));
    for (auto& x : big)
      C(hipMemRelease(x));
  }
  printf("DONE\n");
  return 0;
}

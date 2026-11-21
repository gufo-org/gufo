// Remap-based restores and shared prefix pages (RFC items 7 + 8, #1240).
// One physical 2 MiB page mapped in several address ranges at once:
//   A     the session that wrote it
//   park  the cache's stable view of it
//   B     a second session that restores it by mapping
// Checks: coherence, unmapping one alias while the others are in use,
// replacing a page under a captured graph (copy on write), and the cost of
// every step idle and beside running work.
#include <atomic>
#include <cstring>
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
// A decode-like reader: sums a strided gather over the whole range.
__global__ void gather(const unsigned* p, size_t n, unsigned* out, int rounds) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (int r = 0; r < rounds; ++r)
    acc += p[(i * 2654435761ull + r * 40503ull) % n];
  if (i < 1024)
    out[i] = acc;
}
__global__ void spin(float* p, size_t n, int iters) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    float v = p[i];
    for (int k = 0; k < iters; ++k)
      v = v * 1.0001f + 0.5f;
    p[i] = v;
  }
}

static const size_t kPage = 2ull << 20;
static unsigned* g_bad;
static unsigned Bad(const void* p, size_t pages, unsigned seed,
                    hipStream_t s = nullptr) {
  C(hipMemsetAsync(g_bad, 0, 4, s));
  size_t n = pages * kPage / 4;
  check<<<(n + 255) / 256, 256, 0, s>>>((const unsigned*)p, n, seed, g_bad);
  unsigned b;
  C(hipMemcpyAsync(&b, g_bad, 4, hipMemcpyDeviceToHost, s));
  C(hipStreamSynchronize(s));
  return b;
}

int main(int argc, char** argv) {
  const size_t pages =
      argc > 1 ? atoi(argv[1]) : 1400;  // 100K Flash-Next tokens
  const size_t total = pages * kPage;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  prop.requestedHandleType =
      hipMemHandleTypePosixFileDescriptor;  // as the pool
  hipMemAccessDesc acc{};
  acc.location = prop.location;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  C(hipMalloc(&g_bad, 4));
  hipStream_t s;
  C(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  hipDeviceptr_t A{}, P{}, B{};
  double t = now_us();
  C(hipMemAddressReserve(&A, total, kPage, nullptr, 0));
  C(hipMemAddressReserve(&P, total, kPage, nullptr, 0));
  C(hipMemAddressReserve(&B, total, kPage, nullptr, 0));
  printf("reserve 3 x %.1f GiB: %.1f us each\n", total / 1073741824.0,
         (now_us() - t) / 3);
  // One huge reservation, as a process-wide park range would be.
  {
    hipDeviceptr_t big{};
    t = now_us();
    hipError_t e = hipMemAddressReserve(&big, 256ull << 30, kPage, nullptr, 0);
    printf("reserve 256 GiB: %s, %.1f us\n", hipGetErrorString(e),
           now_us() - t);
    (void)hipGetLastError();
    if (e == hipSuccess)
      C(hipMemAddressFree(big, 256ull << 30));
  }

  std::vector<hipMemGenericAllocationHandle_t> h(pages), fresh(pages);
  for (auto& x : h)
    C(hipMemCreate(&x, kPage, &prop, 0));
  for (auto& x : fresh)
    C(hipMemCreate(&x, kPage, &prop, 0));
  t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)A + i * kPage, kPage, 0, h[i], 0));
  double map_a = (now_us() - t) / pages;
  C(hipMemSetAccess(A, total, &acc, 1));
  size_t n = total / 4;
  fill<<<(n + 255) / 256, 256, 0, s>>>((unsigned*)A, n, 0xA11CE);
  C(hipStreamSynchronize(s));

  // 1. Alias the pages at the park range and in session B.
  t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)P + i * kPage, kPage, 0, h[i], 0));
  double map_p = (now_us() - t) / pages;
  t = now_us();
  C(hipMemSetAccess(P, total, &acc, 1));
  double acc_p = now_us() - t;
  t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)B + i * kPage, kPage, 0, h[i], 0));
  double map_b = (now_us() - t) / pages;
  t = now_us();
  C(hipMemSetAccess(B, total, &acc, 1));
  double acc_b = now_us() - t;
  printf(
      "map first %.2f us/page; alias 2 %.2f us/page (+access %.0f us); alias 3 "
      "%.2f us/page (+access %.0f us); %zu pages = %.2f ms per alias\n",
      map_a, map_p, acc_p, map_b, acc_b, pages, (map_b * pages + acc_b) / 1000);
  // Access set per page instead of once per run.
  {
    hipDeviceptr_t X{};
    C(hipMemAddressReserve(&X, total, kPage, nullptr, 0));
    t = now_us();
    for (size_t i = 0; i < pages; ++i) {
      C(hipMemMap((char*)X + i * kPage, kPage, 0, h[i], 0));
      C(hipMemSetAccess((char*)X + i * kPage, kPage, &acc, 1));
    }
    printf("map + access per page: %.2f us/page\n", (now_us() - t) / pages);
    t = now_us();
    for (size_t i = 0; i < pages; ++i)
      C(hipMemUnmap((char*)X + i * kPage, kPage));
    printf("unmap alias idle, per page calls: %.2f us/page\n",
           (now_us() - t) / pages);
    for (size_t i = 0; i < pages; ++i)
      C(hipMemMap((char*)X + i * kPage, kPage, 0, h[i], 0));
    t = now_us();
    C(hipMemUnmap(X, total));
    printf("unmap alias idle, one call for %zu pages: %.2f us/page\n", pages,
           (now_us() - t) / pages);
    C(hipMemAddressFree(X, total));
  }
  printf("coherence: park sees A's rows: bad=%u; B sees them: bad=%u\n",
         Bad(P, pages, 0xA11CE, s), Bad(B, pages, 0xA11CE, s));
  // A writes through its mapping (appending in a shared page); others see it.
  fill<<<(n + 255) / 256, 256, 0, s>>>((unsigned*)A, n, 0xB0B);
  C(hipStreamSynchronize(s));
  printf("write through A: park bad=%u, B bad=%u\n", Bad(P, pages, 0xB0B, s),
         Bad(B, pages, 0xB0B, s));

  // 2. Detach: A unmaps its alias and maps fresh pages; park and B keep the old
  // rows.
  t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemUnmap((char*)A + i * kPage, kPage));
  double unmap_idle = (now_us() - t) / pages;
  t = now_us();
  for (size_t i = 0; i < pages; ++i)
    C(hipMemMap((char*)A + i * kPage, kPage, 0, fresh[i], 0));
  C(hipMemSetAccess(A, total, &acc, 1));
  double remap_idle = (now_us() - t) / pages;
  fill<<<(n + 255) / 256, 256, 0, s>>>((unsigned*)A, n, 0xF00D);
  C(hipStreamSynchronize(s));
  printf(
      "detach idle: unmap %.2f us/page, map fresh %.2f us/page => %.2f ms for "
      "%zu pages\n",
      unmap_idle, remap_idle, (unmap_idle + remap_idle) * pages / 1000, pages);
  printf(
      "after detach: park bad=%u, B bad=%u (old rows), A bad=%u (new rows)\n",
      Bad(P, pages, 0xB0B, s), Bad(B, pages, 0xB0B, s),
      Bad(A, pages, 0xF00D, s));

  // 3. A captured graph that reads A, replayed after its pages were replaced.
  {
    unsigned* out;
    C(hipMalloc(&out, 4096));
    hipGraph_t g;
    hipGraphExec_t ge;
    C(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    C(hipMemsetAsync(g_bad, 0, 4, s));
    check<<<(n + 255) / 256, 256, 0, s>>>((const unsigned*)A, n, 0xC0FFEE,
                                          g_bad);
    C(hipStreamEndCapture(s, &g));
    C(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    // Swap half the pages back to the old handles... no: swap A's pages among
    // themselves.
    for (size_t i = 0; i + 1 < pages; i += 2) {
      C(hipMemUnmap((char*)A + i * kPage, 2 * kPage));
      C(hipMemMap((char*)A + i * kPage, kPage, 0, fresh[i + 1], 0));
      C(hipMemMap((char*)A + (i + 1) * kPage, kPage, 0, fresh[i], 0));
    }
    C(hipMemSetAccess(A, total, &acc, 1));
    fill<<<(n + 255) / 256, 256, 0, s>>>((unsigned*)A, n, 0xC0FFEE);
    C(hipStreamSynchronize(s));
    C(hipGraphLaunch(ge, s));
    unsigned b;
    C(hipMemcpyAsync(&b, g_bad, 4, hipMemcpyDeviceToHost, s));
    C(hipStreamSynchronize(s));
    printf("graph captured before page swap, replayed after: bad=%u\n", b);
    C(hipGraphExecDestroy(ge));
    C(hipGraphDestroy(g));
    C(hipFree(out));
  }
  printf("park still holds the old rows after the swap: bad=%u\n",
         Bad(P, pages, 0xB0B, s));

  // 4. Read bandwidth through an alias against the first mapping (same kernel).
  {
    unsigned* out;
    C(hipMalloc(&out, 4096));
    for (const char* name : {"A (own pages)", "B (alias)", "park (alias)"}) {
      const void* p = name[0] == 'A' ? A : name[0] == 'B' ? B : P;
      double best = 1e30;
      for (int r = 0; r < 5; ++r) {
        t = now_us();
        gather<<<1 << 12, 256, 0, s>>>((const unsigned*)p, n, out, 64);
        C(hipStreamSynchronize(s));
        best = std::min(best, now_us() - t);
      }
      printf("gather through %-14s %.2f ms\n", name, best / 1000);
    }
    C(hipFree(out));
  }

  // 5. Beside running work. A decoder thread replays a kernel that reads B
  // (the shared pages) back to back; this thread unmaps and maps aliases.
  {
    float* buf;
    size_t bn = 4ull << 20;
    C(hipMalloc(&buf, bn * 4));
    C(hipMemset(buf, 0, bn * 4));
    for (int mode = 0; mode < 3; ++mode) {
      // 0: reader of the shared pages (via B); 1: unrelated 30 ms kernels; 2: 1
      // MiB copies (persistence-like)
      std::atomic<bool> stop{false};
      std::atomic<long> rounds{0};
      std::atomic<double> worst{0};
      std::thread decoder([&] {
        hipStream_t ds;
        C(hipStreamCreateWithFlags(&ds, hipStreamNonBlocking));
        unsigned* out;
        C(hipMalloc(&out, 4096));
        void* stage;
        C(hipMalloc(&stage, 1 << 20));
        while (!stop) {
          double a = now_us();
          if (mode == 0)
            gather<<<1 << 10, 256, 0, ds>>>((const unsigned*)B, n, out, 16);
          else if (mode == 1)
            spin<<<bn / 256, 256, 0, ds>>>(buf, bn, 4000);
          else
            C(hipMemcpyAsync(stage, (char*)P + (rounds % pages) * kPage,
                             1 << 20, hipMemcpyDeviceToDevice, ds));
          C(hipStreamSynchronize(ds));
          double d = now_us() - a;
          if (d > worst)
            worst = d;
          ++rounds;
        }
        C(hipFree(out));
        C(hipFree(stage));
        C(hipStreamDestroy(ds));
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      long r0 = rounds;
      double w0 = worst;
      worst = 0;
      const size_t k = std::min<size_t>(pages, 16);
      hipDeviceptr_t X{};
      C(hipMemAddressReserve(&X, k * kPage, kPage, nullptr, 0));
      t = now_us();
      for (size_t i = 0; i < k; ++i)
        C(hipMemMap((char*)X + i * kPage, kPage, 0, h[i], 0));
      C(hipMemSetAccess(X, k * kPage, &acc, 1));
      double m = (now_us() - t) / k;
      double wm = worst;
      worst = 0;
      double worst_unmap = 0;
      t = now_us();
      for (size_t i = 0; i < k; ++i) {
        double a = now_us();
        C(hipMemUnmap((char*)X + i * kPage, kPage));
        worst_unmap = std::max(worst_unmap, now_us() - a);
      }
      double u = (now_us() - t) / k;
      double wu = worst;
      // Unmap A's own (fresh) pages, as a detach beside this work would.
      t = now_us();
      C(hipMemUnmap(A, k * kPage));
      double u1 = now_us() - t;
      for (size_t i = 0; i < k; ++i)
        C(hipMemMap((char*)A + i * kPage, kPage, 0, fresh[i], 0));
      C(hipMemSetAccess(A, k * kPage, &acc, 1));
      stop = true;
      decoder.join();
      C(hipMemAddressFree(X, k * kPage));
      const char* names[] = {"reader of the shared pages",
                             "unrelated ~30 ms kernels",
                             "1 MiB copies from park"};
      printf(
          "beside %-26s (%ld rounds warm, worst %.1f ms): map alias %.2f "
          "us/page (worst round %.1f ms); unmap alias %.1f us/page, worst call "
          "%.1f ms (worst round %.1f ms); one unmap of %zu pages %.1f ms\n",
          names[mode], r0, w0 / 1000, m, wm / 1000, u, worst_unmap / 1000,
          wu / 1000, k, u1 / 1000);
    }
    C(hipFree(buf));
  }

  // 6. Copy on write of one boundary page: fresh page takes rows [0, r) from
  // the parked one.
  {
    t = now_us();
    const int reps = 64;
    for (int i = 0; i < reps; ++i)
      C(hipMemcpyAsync((char*)A + i * kPage, (char*)P + i * kPage, kPage / 2,
                       hipMemcpyDeviceToDevice, s));
    C(hipStreamSynchronize(s));
    printf("boundary copies: %d x 1 MiB in %.2f ms\n", reps,
           (now_us() - t) / 1000);
  }

  // 7. Release a page that is still mapped elsewhere: refused, or deferred?
  {
    hipMemGenericAllocationHandle_t x;
    C(hipMemCreate(&x, kPage, &prop, 0));
    hipDeviceptr_t X{};
    C(hipMemAddressReserve(&X, kPage, kPage, nullptr, 0));
    C(hipMemMap(X, kPage, 0, x, 0));
    C(hipMemSetAccess(X, kPage, &acc, 1));
    hipError_t e = hipMemRelease(x);
    printf("release while mapped: %s\n", hipGetErrorString(e));
    (void)hipGetLastError();
    fill<<<kPage / 4 / 256, 256, 0, s>>>((unsigned*)X, kPage / 4, 7);
    C(hipStreamSynchronize(s));
    printf("still usable after release-while-mapped: bad=%u\n",
           Bad(X, 1, 7, s));
    C(hipMemUnmap(X, kPage));
  }
  printf("DONE\n");
  return 0;
}

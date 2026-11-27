// Does a KFD TLB flush settle a remapped address, also while another process
// keeps the GPU busy? Derived from upstream's idle2.hip (v0.36.0): K addresses
// + 1 canary cycle through M handle sets with distinct constants; the handles
// stay alive, and the probe only reads after the remap, so a stale
// translation reads a live page and nothing is written through it.
//
//   flush <reps> <method>...
//   method 0: read at once (no settle)
//          1: spin 400 us on the host first (RemapBarrier-like, process-local
//             idleness)
//          2: hipMalloc(4 MiB) + hipFree first (KFD map/unmap ioctls flush
//             the TLB of this process's PASID)
//          3: as 2 with 64 KiB
//          4: hipDeviceSynchronize + spin 400 us (= the C++ barrier's drain)
//          5: hipFree of a 4 MiB block allocated before the remap (free only)
//          6: hipMalloc(4 MiB) only; freed after the reads (map only)
// Witnesses: one kernel reading word 0 of each page, then hipMemcpy D2H of
// the same words (copy path). A batch counts as new when every page reads
// its new handle's constant.
#include <algorithm>
#include <cstdint>

#include "common.hpp"
static const size_t P = 2u << 20, W = P / 4;
__global__ void Fill(uint32_t* d, size_t words, uint32_t v) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    d[i] = v;
}
__global__ void ReadK(const uint32_t* d, size_t stride_words, int k,
                      uint32_t* out) {
  if (threadIdx.x < (unsigned)k)
    out[threadIdx.x] = d[threadIdx.x * stride_words];
}
static Vmm v;
static uint32_t* dout;
static void* reserve(size_t n = 1) {
  hipDeviceptr_t p{};
  C(hipMemAddressReserve(&p, n * P, P, nullptr, 0));
  return p;
}
static void spin_us(double us) {
  const double t = now_us();
  while (now_us() - t < us) {
  }
}
int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 100;
  std::vector<int> methods;
  for (int i = 2; i < argc; ++i)
    methods.push_back(atoi(argv[i]));
  if (methods.empty())
    methods = {0, 1, 2, 3, 4};
  const int K = 8, M = 5;
  C(hipMalloc(&dout, 4096));
  hipStream_t s;
  C(hipStreamCreate(&s));
  hipMemGenericAllocationHandle_t h[M][K + 1];
  for (int j = 0; j < M; ++j)
    for (int k = 0; k <= K; ++k) {
      C(hipMemCreate(&h[j][k], P, &v.prop, 0));
      void* home = reserve();
      C(hipMemMap(home, P, 0, h[j][k], 0));
      C(hipMemSetAccess(home, P, &v.acc, 1));
      Fill<<<64, 256, 0, s>>>((uint32_t*)home, W, 0x1000u * (j + 1) + k);
      C(hipStreamSynchronize(s));
      C(hipMemUnmap(home, P));
    }
  void* X = reserve(K + 1);
  auto read_kernel = [&](uint32_t* o) {
    ReadK<<<1, 64, 0, s>>>((const uint32_t*)X, W, K + 1, dout);
    C(hipStreamSynchronize(s));
    C(hipMemcpy(o, dout, 4 * (K + 1), hipMemcpyDeviceToHost));
  };
  auto read_copy = [&](uint32_t* o) {
    for (int k = 0; k <= K; ++k)
      C(hipMemcpy(&o[k], (char*)X + k * P, 4, hipMemcpyDeviceToHost));
  };
  auto count_new = [&](const uint32_t* o, int n) {
    int x = 0;
    for (int k = 0; k <= K; ++k)
      x += o[k] == 0x1000u * (n + 1) + k;
    return x;
  };
  int j = 0;
  for (int k = 0; k <= K; ++k)
    C(hipMemMap((char*)X + k * P, P, 0, h[0][k], 0));
  C(hipMemSetAccess(X, (K + 1) * P, &v.acc, 1));
  for (int method : methods) {
    int kernel_new = 0, copy_new = 0, never = 0;
    double settle_sum = 0, settle_max = 0, wait_max = 0;
    for (int r = 0; r < reps; ++r) {
      uint32_t o[K + 1];
      read_kernel(o);  // the old translations are in use
      read_copy(o);
      void* held = nullptr;
      if (method == 5)
        C(hipMalloc(&held, 4u << 20));
      const int n = (j + 1) % M;
      C(hipMemUnmap(X, (K + 1) * P));
      for (int k = 0; k <= K; ++k)
        C(hipMemMap((char*)X + k * P, P, 0, h[n][k], 0));
      C(hipMemSetAccess(X, (K + 1) * P, &v.acc, 1));
      const double t = now_us();
      if (method == 1) {
        spin_us(400);
      } else if (method == 2 || method == 3) {
        void* f = nullptr;
        C(hipMalloc(&f, method == 2 ? (4u << 20) : (64u << 10)));
        C(hipFree(f));
      } else if (method == 4) {
        C(hipDeviceSynchronize());
        spin_us(400);
      } else if (method == 5) {
        C(hipFree(held));
        held = nullptr;
      } else if (method == 6) {
        C(hipMalloc(&held, 4u << 20));
      }
      const double cost = now_us() - t;
      settle_sum += cost;
      settle_max = std::max(settle_max, cost);
      read_kernel(o);
      const bool kn = count_new(o, n) == K + 1;
      read_copy(o);
      const bool cn = count_new(o, n) == K + 1;
      kernel_new += kn;
      copy_new += cn;
      // Poll (kernel reads) for up to 200 ms when not yet all new.
      bool done = kn;
      const double t0 = now_us();
      while (!done && now_us() - t0 < 200000) {
        spin_us(100);
        read_kernel(o);
        done = count_new(o, n) == K + 1;
      }
      if (held)
        C(hipFree(held));
      if (!done)
        ++never;
      else if (!kn)
        wait_max = std::max(wait_max, now_us() - t0);
      j = n;
    }
    printf(
        "method %d: kernel all new %3d/%d, copy all new %3d/%d; settle cost "
        "mean %.0f us max %.0f us; not new after 200 ms of polling %d (max "
        "wait of the others %.0f us)\n",
        method, kernel_new, reps, copy_new, reps, settle_sum / reps,
        settle_max, never, wait_max);
    fflush(stdout);
  }
  return 0;
}

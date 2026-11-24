// As idle.hip, without its alternation artefact: every address cycles through M
// handles with distinct constants, so an old translation never looks new.
// Counts, per idle time after a remap batch: batches whose addresses all reach
// the new handle, and whether a canary remapped last in the batch is a sound
// witness (canary new => every address new). Modes: 0 quiet; 2 another thread
// runs short kernels with 60 us gaps; 4 quiet, with 3,000 other pages touched
// after the remap (TLB pressure); 5 as 0 but the idle time is spent BEFORE the
// unmaps (after the touches).
#include <atomic>
#include <cstdint>
#include <thread>

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
__global__ void Touch(const uint32_t* d, size_t stride_words, uint32_t* out) {
  out[0] += d[(size_t)blockIdx.x * stride_words];
}
__global__ void Spin(uint32_t* d, int iters) {
  uint32_t x = d[threadIdx.x];
  for (int i = 0; i < iters; ++i)
    x = x * 1664525u + 1013904223u;
  d[threadIdx.x] = x;
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
  const int mode = argc > 1 ? atoi(argv[1]) : 0;
  const int reps = argc > 2 ? atoi(argv[2]) : 200;
  const int K = 8, M = 5;  // K addresses + 1 canary, M handle sets
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
  uint32_t* press = nullptr;
  const size_t press_pages = 3000;
  if (mode == 4) {
    C(hipMalloc(&press, press_pages * P));
  }
  void* X = reserve(K + 1);  // canary = page K
  std::atomic<bool> stop{false};
  std::thread bg;
  if (mode == 2)
    bg = std::thread([&] {
      hipStream_t b;
      C(hipStreamCreate(&b));
      uint32_t* buf;
      C(hipMalloc(&buf, 4096));
      while (!stop) {
        Spin<<<1, 256, 0, b>>>(buf, 2000);
        C(hipStreamSynchronize(b));
        spin_us(60);
      }
    });
  auto read_all = [&](uint32_t* o) {
    ReadK<<<1, 64, 0, s>>>((const uint32_t*)X, W, K + 1, dout);
    C(hipStreamSynchronize(s));
    C(hipMemcpy(o, dout, 4 * (K + 1), hipMemcpyDeviceToHost));
  };
  int j = 0;
  for (int k = 0; k <= K; ++k) {
    C(hipMemMap((char*)X + k * P, P, 0, h[0][k], 0));
  }
  C(hipMemSetAccess(X, (K + 1) * P, &v.acc, 1));
  std::vector<double> idles = {0.0, 20.0, 40.0, 60.0, 100.0, 200.0, 400.0};
  if (argc > 3) {
    idles.clear();
    for (int i = 3; i < argc; ++i)
      idles.push_back(atof(argv[i]));
  }
  for (double idle : idles) {
    int all_new = 0, canary_new = 0, unsound = 0, any_old_with_canary_old = 0;
    double max_wait = 0;
    int never = 0;
    for (int r = 0; r < reps; ++r) {
      uint32_t o[K + 1];
      read_all(o);  // touch everything; one kernel, canary included
      const int n = (j + 1) % M;
      if (mode == 5)
        spin_us(idle);
      C(hipMemUnmap(X, (K + 1) * P));
      for (int k = 0; k <= K; ++k)
        C(hipMemMap((char*)X + k * P, P, 0, h[n][k], 0));
      C(hipMemSetAccess(X, (K + 1) * P, &v.acc, 1));
      if (mode == 4) {
        Touch<<<press_pages, 1, 0, s>>>(press, W, dout + 512);
        C(hipStreamSynchronize(s));
      }
      if (mode != 5)
        spin_us(idle);
      read_all(o);
      int x_new = 0;
      for (int k = 0; k < K; ++k)
        x_new += o[k] == 0x1000u * (n + 1) + k;
      const bool c_new = o[K] == 0x1000u * (n + 1) + K;
      all_new += c_new && x_new == K;
      canary_new += c_new;
      unsound += c_new && x_new != K;
      any_old_with_canary_old += !c_new;
      const double t0 = now_us();
      bool done = c_new && x_new == K;
      while (!done && now_us() - t0 < 500000) {
        spin_us(100);
        read_all(o);
        x_new = 0;
        for (int k = 0; k < K; ++k)
          x_new += o[k] == 0x1000u * (n + 1) + k;
        const bool c2 = o[K] == 0x1000u * (n + 1) + K;
        if (c2 && x_new != K)
          ++unsound;
        done = c2 && x_new == K;
      }
      if (!done)
        ++never;
      else
        max_wait = std::max(max_wait, now_us() - t0);
      j = n;
    }
    printf(
        "mode %d idle %4.0f us: all new %4d/%d; canary new %4d; canary new "
        "while an address was old %d; settled by polling within %.0f us max, "
        "never %d\n",
        mode, idle, all_new, reps, canary_new, unsound, max_wait, never);
    fflush(stdout);
  }
  stop = true;
  if (bg.joinable())
    bg.join();
  return 0;
}

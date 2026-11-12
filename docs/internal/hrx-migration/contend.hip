// Scratch probe: two full-grid kernels on two streams, as in Flash-Next where
// the routed MoE (memory bound) overlaps the shared-expert GEMM (compute bound).
// Reports serial vs parallel time for mem+mem, alu+alu and mem+alu pairs.
// All kernels are bounded (fixed iteration counts).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__global__ void Mem(const u32x4* __restrict__ src, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    u32x4 v = src[i];
    acc ^= v.x + v.y + v.z + v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void Alu(long long iters, unsigned* sink) {
  unsigned a = threadIdx.x, b = blockIdx.x;
  for (long long i = 0; i < iters; ++i) { a = a * 1664525u + b; b = b * 22695477u + a; }
  if ((a ^ b) == 0x9e3779b9u) sink[0] = a;
}

template <typename F>
static float Median(F body) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  std::vector<float> v;
  for (int r = 0; r < 11; ++r) {
    OK(hipDeviceSynchronize());
    OK(hipEventRecord(a, nullptr));
    body();
    OK(hipEventRecord(b, nullptr));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
  }
  std::sort(v.begin(), v.end());
  return v[5];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  const size_t bytes = 512u << 20;
  u32x4 *m1, *m2; OK(hipMalloc(&m1, bytes)); OK(hipMalloc(&m2, bytes));
  OK(hipMemset(m1, 1, bytes)); OK(hipMemset(m2, 2, bytes));
  hipStream_t s1, s2;
  OK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  OK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  hipEvent_t fork, j1, j2;
  OK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
  OK(hipEventCreateWithFlags(&j1, hipEventDisableTiming));
  OK(hipEventCreateWithFlags(&j2, hipEventDisableTiming));
  const int grid = 20 * 32;  // fills all WGPs
  const long long iters = 40000;
  auto mem = [&](hipStream_t s, u32x4* p) { Mem<<<grid, 256, 0, s>>>(p, bytes / 16, sink); };
  auto alu = [&](hipStream_t s) { Alu<<<grid, 256, 0, s>>>(iters, sink); };
  auto run = [&](auto k1, auto k2, bool parallel) {
    return Median([&] {
      OK(hipEventRecord(fork, nullptr));
      OK(hipStreamWaitEvent(s1, fork, 0)); OK(hipStreamWaitEvent(s2, fork, 0));
      k1(s1); k2(parallel ? s2 : s1);
      OK(hipEventRecord(j1, s1)); OK(hipEventRecord(j2, s2));
      OK(hipStreamWaitEvent(nullptr, j1, 0)); OK(hipStreamWaitEvent(nullptr, j2, 0));
    });
  };
  auto M1 = [&](hipStream_t s) { mem(s, m1); };
  auto M2 = [&](hipStream_t s) { mem(s, m2); };
  auto A = [&](hipStream_t s) { alu(s); };
  run(M1, A, true);  // warm up
  std::fprintf(stderr, "warm ok\n");
  const float mm_s = run(M1, M2, false), mm_p = run(M1, M2, true);
  const float aa_s = run(A, A, false), aa_p = run(A, A, true);
  const float ma_s = run(M1, A, false), ma_p = run(M1, A, true);
  std::printf("mem+mem serial %.3f parallel %.3f ms\n", mm_s, mm_p);
  std::printf("alu+alu serial %.3f parallel %.3f ms\n", aa_s, aa_p);
  std::printf("mem+alu serial %.3f parallel %.3f ms  (gain %.1f%%)\n", ma_s, ma_p,
              100.0 * (ma_s - ma_p) / ma_s);
  return 0;
}

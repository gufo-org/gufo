// Scratch probe: do kernels on two streams overlap, and what does a
// cross-stream event join cost? Each kernel spins a fixed cycle count on a
// small grid so two of them fit on the GPU together.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

__global__ void Spin(long long iters, unsigned* sink) {
  unsigned acc = threadIdx.x;
  for (long long i = 0; i < iters; ++i) acc = acc * 1664525u + (unsigned)i;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

template <typename F>
static float Median(F body) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  std::vector<float> v;
  for (int r = 0; r < 15; ++r) {
    OK(hipDeviceSynchronize());
    OK(hipEventRecord(a, nullptr));
    body();
    OK(hipEventRecord(b, nullptr));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
  }
  std::sort(v.begin(), v.end());
  return v[7];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  hipStream_t s1, s2;
  OK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  OK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  hipEvent_t fork, join; OK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
  OK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
  const long long cyc = 400000;  // fixed ALU iterations, about a millisecond
  const int blocks = 10;          // half the 20 WGPs
  // Legacy null stream records the events; streams are non-blocking, so
  // fork/join through events like gufo does.
  auto one = [&] {
    OK(hipEventRecord(fork, nullptr)); OK(hipStreamWaitEvent(s1, fork, 0));
    Spin<<<blocks, 256, 0, s1>>>(cyc, sink);
    OK(hipEventRecord(join, s1)); OK(hipStreamWaitEvent(nullptr, join, 0));
  };
  auto serial = [&] {
    OK(hipEventRecord(fork, nullptr)); OK(hipStreamWaitEvent(s1, fork, 0));
    Spin<<<blocks, 256, 0, s1>>>(cyc, sink); Spin<<<blocks, 256, 0, s1>>>(cyc, sink);
    OK(hipEventRecord(join, s1)); OK(hipStreamWaitEvent(nullptr, join, 0));
  };
  hipEvent_t j2; OK(hipEventCreateWithFlags(&j2, hipEventDisableTiming));
  auto parallel = [&] {
    OK(hipEventRecord(fork, nullptr));
    OK(hipStreamWaitEvent(s1, fork, 0)); OK(hipStreamWaitEvent(s2, fork, 0));
    Spin<<<blocks, 256, 0, s1>>>(cyc, sink); Spin<<<blocks, 256, 0, s2>>>(cyc, sink);
    OK(hipEventRecord(join, s1)); OK(hipEventRecord(j2, s2));
    OK(hipStreamWaitEvent(nullptr, join, 0)); OK(hipStreamWaitEvent(nullptr, j2, 0));
  };
  // Fork/join ping-pong: 100 tiny kernels alternating streams with event joins.
  auto pingpong = [&] {
    for (int i = 0; i < 100; ++i) {
      hipStream_t a = (i & 1) ? s1 : s2, b = (i & 1) ? s2 : s1;
      Spin<<<1, 64, 0, a>>>(1000, sink);
      OK(hipEventRecord(join, a)); OK(hipStreamWaitEvent(b, join, 0));
    }
    OK(hipEventRecord(join, s1)); OK(hipStreamWaitEvent(nullptr, join, 0));
    OK(hipEventRecord(j2, s2)); OK(hipStreamWaitEvent(nullptr, j2, 0));
  };
  // Same-stream chain of 100 tiny kernels for comparison.
  auto chain = [&] {
    OK(hipEventRecord(fork, nullptr)); OK(hipStreamWaitEvent(s1, fork, 0));
    for (int i = 0; i < 100; ++i) Spin<<<1, 64, 0, s1>>>(1000, sink);
    OK(hipEventRecord(join, s1)); OK(hipStreamWaitEvent(nullptr, join, 0));
  };
  std::fprintf(stderr, "start\n"); Spin<<<1, 64>>>(10, sink); OK(hipDeviceSynchronize()); std::fprintf(stderr, "tiny ok\n");
  one(); OK(hipDeviceSynchronize()); std::fprintf(stderr, "one ok\n");
  for (int w = 0; w < 3; ++w) parallel();
  std::fprintf(stderr, "warm ok\n");
  const float t1 = Median(one), ts = Median(serial), tp = Median(parallel);
  const float tpp = Median(pingpong), tc = Median(chain);
  std::printf("one %.3f ms  serial %.3f ms  parallel %.3f ms  overlap %.0f%%\n", t1, ts, tp,
              100.0 * (ts - tp) / (ts - t1));
  std::printf("tiny chain %.1f us/kernel  cross-stream ping-pong %.1f us/hop\n", tc * 10.0, tpp * 10.0);
  return 0;
}

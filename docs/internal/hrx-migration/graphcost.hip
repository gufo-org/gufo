// Scratch probe: per-kernel cost of eager launches versus HIP graph replay.
// A chain of N short bounded kernels on one stream, eager or captured once and
// replayed; small (16 B) and 256-byte kernel arguments.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

struct Big { unsigned v[62]; };  // with the two other arguments, 256 bytes

__global__ void Small(int iters, unsigned* sink) {
  unsigned a = threadIdx.x ^ blockIdx.x;
  for (int i = 0; i < iters; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = a;
}

__global__ void Wide(Big args, int iters, unsigned* sink) {
  unsigned a = threadIdx.x ^ blockIdx.x ^ args.v[threadIdx.x & 31];
  for (int i = 0; i < iters; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = a;
}

template <typename F>
static float Median(hipStream_t s, F body) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  body(); OK(hipStreamSynchronize(s));
  std::vector<float> v;
  for (int r = 0; r < 11; ++r) {
    OK(hipEventRecord(a, s)); body(); OK(hipEventRecord(b, s));
    OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
  }
  std::sort(v.begin(), v.end());
  return v[5];
}

int main() {
  unsigned* sink; OK(hipMalloc(&sink, 4));
  hipStream_t s; OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  Big big{}; for (int i = 0; i < 62; ++i) big.v[i] = i;
  const int n = 200;
  for (int grid : {20, 640}) {
    for (int wide = 0; wide < 2; ++wide) {
      auto chain = [&] {
        for (int i = 0; i < n; ++i) {
          if (wide) Wide<<<grid, 64, 0, s>>>(big, 16, sink);
          else Small<<<grid, 64, 0, s>>>(16, sink);
        }
      };
      const float eager = Median(s, chain);
      hipGraph_t graph; hipGraphExec_t exec;
      OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
      chain();
      OK(hipStreamEndCapture(s, &graph));
      OK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      const float replay = Median(s, [&] { OK(hipGraphLaunch(exec, s)); });
      std::printf("grid %4d %s args: eager %6.2f us/kernel  graph %6.2f us/kernel\n",
                  grid, wide ? "256 B" : " 16 B", eager * 1000 / n, replay * 1000 / n);
      OK(hipGraphExecDestroy(exec)); OK(hipGraphDestroy(graph));
    }
  }
  // Capture + instantiate + one launch for a prefill-sized chain, host wall time.
  for (int count : {1200, 2400}) {
    std::vector<double> build, update;
    hipGraphExec_t keep = nullptr;
    for (int r = 0; r < 7; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      hipGraph_t graph; hipGraphExec_t exec;
      OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
      for (int i = 0; i < count; ++i) Wide<<<640, 64, 0, s>>>(big, 16, sink);
      OK(hipStreamEndCapture(s, &graph));
      OK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      auto t1 = std::chrono::steady_clock::now();
      build.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      if (keep) {
        // Re-capture and update an existing executable instead of instantiating.
        hipGraph_t g2;
        auto u0 = std::chrono::steady_clock::now();
        OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < count; ++i) Wide<<<640, 64, 0, s>>>(big, 17, sink);
        OK(hipStreamEndCapture(s, &g2));
        hipGraphExecUpdateResult result; hipGraphNode_t bad;
        OK(hipGraphExecUpdate(keep, g2, &bad, &result));
        auto u1 = std::chrono::steady_clock::now();
        update.push_back(std::chrono::duration<double, std::micro>(u1 - u0).count());
        OK(hipGraphDestroy(g2));
        OK(hipGraphExecDestroy(exec));
      } else {
        keep = exec;
      }
      OK(hipGraphDestroy(graph));
    }
    std::sort(build.begin(), build.end()); std::sort(update.begin(), update.end());
    std::printf("%d kernels: capture+instantiate %.0f us (%.2f us/kernel), capture+update %.0f us\n",
                count, build[3], build[3] / count, update[update.size() / 2]);
    OK(hipGraphExecDestroy(keep));
  }
  return 0;
}

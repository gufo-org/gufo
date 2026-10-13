// Does committing pinned host memory stall concurrent GPU work on gfx1151?
// A "decode" thread launches short kernels and synchronizes in a loop while a
// second thread commits segments as CommittedBackingPool grows (hipHostMalloc
// coherent + memset), then frees them. Reports commit cost and step times.
//
//   nix develop -c tools/bench/build.sh host_commit_stall && /tmp/host_commit_stall
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(x)                                                          \
  do {                                                                    \
    hipError_t e = (x);                                                   \
    if (e != hipSuccess) {                                                \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,              \
                   hipGetErrorString(e));                                 \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

using Clock = std::chrono::steady_clock;
static double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

__global__ void Spin(float* out, int iterations) {
  float x = threadIdx.x;
  for (int i = 0; i < iterations; ++i)
    x = x * 1.000001f + 0.5f;
  out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

struct Sample {
  Clock::time_point start, end;
};

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 200000;
  float* out{};
  CHECK(hipMalloc(&out, 256 * 256 * sizeof(float)));
  hipStream_t stream;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  // Warm up and report the kernel time.
  for (int i = 0; i < 20; ++i)
    Spin<<<256, 256, 0, stream>>>(out, iterations);
  CHECK(hipStreamSynchronize(stream));

  for (const std::size_t segment :
       {std::size_t{64} << 20, std::size_t{256} << 20, std::size_t{1} << 30}) {
    std::atomic<bool> stop{false};
    std::vector<Sample> samples;
    samples.reserve(200000);
    std::thread decode([&] {
      while (!stop.load()) {
        const auto start = Clock::now();
        // A decode step is many launches; four per sample keeps the lock
        // contention visible.
        for (int k = 0; k < 4; ++k)
          Spin<<<256, 256, 0, stream>>>(out, iterations / 4);
        CHECK(hipStreamSynchronize(stream));
        samples.push_back({start, Clock::now()});
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    struct Event {
      Clock::time_point start, allocated, touched, freed_start, freed;
    };
    std::vector<Event> events;
    for (int round = 0; round < 8; ++round) {
      Event e{};
      void* data{};
      e.start = Clock::now();
      CHECK(hipHostMalloc(&data, segment, hipHostMallocCoherent));
      e.allocated = Clock::now();
      std::memset(data, 0, segment);
      e.touched = Clock::now();
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      e.freed_start = Clock::now();
      CHECK(hipHostFree(data));
      e.freed = Clock::now();
      events.push_back(e);
      std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    stop = true;
    decode.join();
    // Baseline: samples that overlap no allocation/free window.
    auto overlaps = [&](const Sample& s, Clock::time_point a,
                        Clock::time_point b) {
      return s.start < b && s.end > a;
    };
    std::vector<double> quiet, during_alloc, during_touch, during_free;
    for (const auto& s : samples) {
      const double d = Ms(s.start, s.end);
      bool any = false;
      for (const auto& e : events) {
        if (overlaps(s, e.start, e.allocated)) {
          during_alloc.push_back(d);
          any = true;
        } else if (overlaps(s, e.allocated, e.touched)) {
          during_touch.push_back(d);
          any = true;
        } else if (overlaps(s, e.freed_start, e.freed)) {
          during_free.push_back(d);
          any = true;
        }
      }
      if (!any)
        quiet.push_back(d);
    }
    auto stats = [](std::vector<double> v) {
      if (v.empty())
        return std::string("n=0");
      std::sort(v.begin(), v.end());
      char text[128];
      std::snprintf(text, sizeof(text), "n=%zu p50=%.2f p99=%.2f max=%.2f",
                    v.size(), v[v.size() / 2], v[v.size() * 99 / 100],
                    v.back());
      return std::string(text);
    };
    double alloc = 0, touch = 0, release = 0, alloc_max = 0, touch_max = 0;
    for (const auto& e : events) {
      alloc += Ms(e.start, e.allocated);
      touch += Ms(e.allocated, e.touched);
      release += Ms(e.freed_start, e.freed);
      alloc_max = std::max(alloc_max, Ms(e.start, e.allocated));
      touch_max = std::max(touch_max, Ms(e.allocated, e.touched));
    }
    const double n = events.size();
    std::printf(
        "segment=%zu MiB  hipHostMalloc mean=%.2f max=%.2f ms  memset "
        "mean=%.2f max=%.2f ms  hipHostFree mean=%.2f ms\n",
        segment >> 20, alloc / n, alloc_max, touch / n, touch_max,
        release / n);
    std::printf("  decode step quiet:        %s\n", stats(quiet).c_str());
    std::printf("  decode step during alloc: %s\n",
                stats(during_alloc).c_str());
    std::printf("  decode step during memset:%s\n",
                stats(during_touch).c_str());
    std::printf("  decode step during free:  %s\n", stats(during_free).c_str());
  }
  return 0;
}

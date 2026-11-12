// Scratch probe: cost of reading per-workgroup job data from a large by-value
// kernel argument (like LFM3's Q8_0MultiGEMV Projections) against reading the
// same struct from device memory through a pointer. Each workgroup picks its
// job by scanning first_block, then streams a slice of a buffer.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

struct Job { const float* src; float* dst; std::size_t rows; };
struct Jobs { Job job[4]; int first_block[4]; int count; };  // about 120 bytes

__device__ __forceinline__ void Body(const Jobs& jobs, std::size_t per_block) {
  int j = 0;
  for (int k = 1; k < jobs.count; ++k)
    if (int(blockIdx.x) >= jobs.first_block[k]) j = k;
  const Job& job = jobs.job[j];
  const std::size_t base = (blockIdx.x - jobs.first_block[j]) * per_block;
  float acc = 0;
  for (std::size_t i = threadIdx.x; i < per_block; i += blockDim.x)
    acc += job.src[(base + i) % job.rows];
  if (acc == 1234.5f) job.dst[0] = acc;
}

__global__ void ByValue(Jobs jobs, std::size_t per_block) { Body(jobs, per_block); }
// Per-wave (divergent) job index into the by-value table, as Q8_0MultiGEMV's
// SwiGLU path does with jobs.job[source]: forces vector loads of kernel args.
__global__ void ByValueWave(Jobs jobs, std::size_t per_block) {
  const int j = (threadIdx.x >> 5) & 3;
  const Job& job = jobs.job[j];
  const std::size_t base = blockIdx.x * per_block;
  float acc = 0;
  for (std::size_t i = threadIdx.x; i < per_block; i += blockDim.x)
    acc += job.src[(base + i) % job.rows];
  if (acc == 1234.5f) job.dst[0] = acc;
}
__global__ void ByPointerWave(const Jobs* __restrict__ jobs, std::size_t per_block) {
  const int j = (threadIdx.x >> 5) & 3;
  const Job& job = jobs->job[j];
  const std::size_t base = blockIdx.x * per_block;
  float acc = 0;
  for (std::size_t i = threadIdx.x; i < per_block; i += blockDim.x)
    acc += job.src[(base + i) % job.rows];
  if (acc == 1234.5f) job.dst[0] = acc;
}
__global__ void ByPointer(const Jobs* __restrict__ jobs, std::size_t per_block) {
  Body(*jobs, per_block);
}

template <typename F>
static float Median(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();
  OK(hipDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < 11; ++r) {
    OK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 20; ++i) launch();
    OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t * 1000 / 20);
  }
  std::sort(v.begin(), v.end());
  return v[5];
}

int main() {
  const std::size_t rows = 1u << 22;
  float *src, *dst; OK(hipMalloc(&src, rows * 4)); OK(hipMalloc(&dst, 64));
  OK(hipMemset(src, 0, rows * 4));
  for (int blocks : {512, 2048, 8192}) {
    for (std::size_t per_block : {std::size_t{256}, std::size_t{2048}}) {
      Jobs jobs{};
      jobs.count = 4;
      for (int k = 0; k < 4; ++k) {
        jobs.job[k] = {src, dst, rows};
        jobs.first_block[k] = k * blocks / 4;
      }
      Jobs* djobs; OK(hipMalloc(&djobs, sizeof(Jobs)));
      OK(hipMemcpy(djobs, &jobs, sizeof(Jobs), hipMemcpyHostToDevice));
      const float tv = Median([&] { ByValue<<<blocks, 64>>>(jobs, per_block); });
      const float tp = Median([&] { ByPointer<<<blocks, 64>>>(djobs, per_block); });
      const float twv = Median([&] { ByValueWave<<<blocks, 128>>>(jobs, per_block); });
      const float twp = Median([&] { ByPointerWave<<<blocks, 128>>>(djobs, per_block); });
      std::printf("blocks %5d per_block %5zu: uniform value %7.2f ptr %7.2f | per-wave value %7.2f ptr %7.2f us (%+.0f%%)\n",
                  blocks, per_block, tv, tp, twv, twp, 100.0 * (twv - twp) / twp);
      OK(hipFree(djobs));
    }
  }
  return 0;
}

// Runs Gufo's SwigluHalfKernel (copied verbatim) and a Loom-compiled HSACO of
// the same contract, launched through the HIP module API; compares outputs
// and times both. Usage: swiglu_host <swiglu.hsaco>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

__device__ __forceinline__ float SiluF(float x) { return x / (1.0f + __expf(-x)); }

__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 4;
  if (i + 4 > count) {
    for (std::size_t j = i; j < count; ++j) out[j] = __float2half(SiluF(gate[j]) * up[j]);
    return;
  }
  const float4 g = *reinterpret_cast<const float4*>(gate + i);
  const float4 u = *reinterpret_cast<const float4*>(up + i);
  const __half2 lo = __floats2half2_rn(SiluF(g.x) * u.x, SiluF(g.y) * u.y);
  const __half2 hi = __floats2half2_rn(SiluF(g.z) * u.z, SiluF(g.w) * u.w);
  *reinterpret_cast<__half2*>(out + i) = lo;
  *reinterpret_cast<__half2*>(out + i + 2) = hi;
}

struct LoomArgs {
  std::uint32_t count;
  std::uint32_t pad;
  const float* gate;
  const float* up;
  __half* out;
};

template <typename F>
static float Median(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();
  OK(hipDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < 15; ++r) {
    OK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 20; ++i) launch();
    OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t / 20.0f);
  }
  std::sort(v.begin(), v.end());
  return v[7];
}

int main(int argc, char** argv) {
  hipModule_t module; OK(hipModuleLoad(&module, argv[1]));
  hipFunction_t loom; OK(hipModuleGetFunction(&loom, module, "swiglu_f16"));
  for (std::size_t count : {std::size_t{512} * 1536, std::size_t{2048} * 1536,
                            std::size_t{2048} * 10240}) {
    std::vector<float> hg(count), hu(count);
    for (std::size_t i = 0; i < count; ++i) {
      hg[i] = std::sin(0.001f * i) * 6.0f;
      hu[i] = std::cos(0.0007f * i) * 2.0f;
    }
    float *g, *u; __half *o1, *o2;
    OK(hipMalloc(&g, count * 4)); OK(hipMalloc(&u, count * 4));
    OK(hipMalloc(&o1, count * 2)); OK(hipMalloc(&o2, count * 2));
    // Device-to-device copies avoid HRX's host-upload cache state (#2984).
    float *sg, *su; OK(hipMalloc(&sg, count * 4)); OK(hipMalloc(&su, count * 4));
    OK(hipMemcpy(sg, hg.data(), count * 4, hipMemcpyHostToDevice));
    OK(hipMemcpy(su, hu.data(), count * 4, hipMemcpyHostToDevice));
    OK(hipMemcpy(g, sg, count * 4, hipMemcpyDeviceToDevice));
    OK(hipMemcpy(u, su, count * 4, hipMemcpyDeviceToDevice));
    const unsigned blocks = static_cast<unsigned>((count / 4 + 255) / 256);
    auto hip_launch = [&] { SwigluHalfKernel<<<blocks, 256>>>(g, u, o1, count); };
    LoomArgs args{static_cast<std::uint32_t>(count), 0, g, u, o2};
    std::size_t size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    auto loom_launch = [&] {
      OK(hipModuleLaunchKernel(loom, blocks, 1, 1, 256, 1, 1, 0, nullptr, nullptr, config));
    };
    hip_launch(); loom_launch(); OK(hipDeviceSynchronize());
    std::vector<__half> r1(count), r2(count);
    OK(hipMemcpy(r1.data(), o1, count * 2, hipMemcpyDeviceToHost));
    OK(hipMemcpy(r2.data(), o2, count * 2, hipMemcpyDeviceToHost));
    float max_diff = 0; std::size_t differ = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const float d = std::fabs(__half2float(r1[i]) - __half2float(r2[i]));
      max_diff = std::max(max_diff, d); differ += d != 0.0f;
    }
    const float th = Median(hip_launch), tl = Median(loom_launch);
    const double bytes = count * 10.0;
    std::printf("count %9zu  HIP %7.1f us (%5.0f GB/s)  Loom %7.1f us (%5.0f GB/s)  "
                "max |diff| %.3g, %zu of %zu differ\n", count, th * 1000, bytes / th / 1e6,
                tl * 1000, bytes / tl / 1e6, max_diff, differ, count);
    OK(hipFree(g)); OK(hipFree(u)); OK(hipFree(o1)); OK(hipFree(o2));
    OK(hipFree(sg)); OK(hipFree(su));
  }
  return 0;
}

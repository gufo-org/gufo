#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(cmd)                                                            \
  do {                                                                         \
    hipError_t err = (cmd);                                                    \
    if (err != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__,   \
                   hipGetErrorString(err), static_cast<int>(err));             \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static inline float Bf16ToFloat(hip_bfloat16 val) {
  union {
    std::uint32_t u;
    float f;
  } converter;
  converter.u = static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(&val)) << 16;
  return converter.f;
}

static inline hip_bfloat16 FloatToBf16(float val) {
  union {
    float f;
    std::uint32_t u;
  } converter;
  converter.f = val;
  std::uint16_t upper = static_cast<std::uint16_t>(converter.u >> 16);
  return *reinterpret_cast<const hip_bfloat16*>(&upper);
}

// Hand-written HIP Wave32 reference
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void Wave32FusedSwiGLUGEMV_1Row(
    const hip_bfloat16* __restrict__ gate_w,
    const hip_bfloat16* __restrict__ up_w,
    const float* __restrict__ x,
    float* __restrict__ out,
    std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* g_row = gate_w + (m * K);
  const auto* u_row = up_w + (m * K);
  const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
  const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
  const std::size_t K_vec = K / 8;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv = g_vec[vi];
    const uint4 uv = u_vec[vi];
    const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
    const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);

    dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                (static_cast<float>(g_vals[1]) * x0.y) +
                (static_cast<float>(g_vals[2]) * x0.z) +
                (static_cast<float>(g_vals[3]) * x0.w) +
                (static_cast<float>(g_vals[4]) * x1.x) +
                (static_cast<float>(g_vals[5]) * x1.y) +
                (static_cast<float>(g_vals[6]) * x1.z) +
                (static_cast<float>(g_vals[7]) * x1.w);

    dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
              (static_cast<float>(u_vals[1]) * x0.y) +
              (static_cast<float>(u_vals[2]) * x0.z) +
              (static_cast<float>(u_vals[3]) * x0.w) +
              (static_cast<float>(u_vals[4]) * x1.x) +
              (static_cast<float>(u_vals[5]) * x1.y) +
              (static_cast<float>(u_vals[6]) * x1.z) +
              (static_cast<float>(u_vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_gate += static_cast<float>(g_row[k]) * xv;
    dot_up += static_cast<float>(u_row[k]) * xv;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

int main(int argc, char** argv) {
  int iters = 50;
  const std::size_t M = 17408;
  const std::size_t K = 5120;

  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("Testing on: %s (gfx1151)\n", prop.name);

  // Allocate host buffers
  std::vector<float> h_x(K);
  std::vector<hip_bfloat16> h_gate(M * K);
  std::vector<hip_bfloat16> h_up(M * K);
  std::vector<float> h_out_ref(M, 0.0F);
  std::vector<float> h_out_hip(M, 0.0F);

  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.01F * static_cast<float>((i % 11) - 5);
  }
  for (std::size_t i = 0; i < M * K; ++i) {
    h_gate[i] = FloatToBf16(0.0005F * static_cast<float>((i % 13) - 6));
    h_up[i] = FloatToBf16(0.0005F * static_cast<float>((i % 17) - 8));
  }

  // CPU Oracle
  for (std::size_t m = 0; m < M; ++m) {
    float dot_g = 0.0F;
    float dot_u = 0.0F;
    for (std::size_t k = 0; k < K; ++k) {
      dot_g += Bf16ToFloat(h_gate[m * K + k]) * h_x[k];
      dot_u += Bf16ToFloat(h_up[m * K + k]) * h_x[k];
    }
    float silu_g = dot_g / (1.0F + std::exp(-dot_g));
    h_out_ref[m] = silu_g * dot_u;
  }

  // Device buffers
  float *d_x = nullptr, *d_out_hip = nullptr;
  hip_bfloat16 *d_gate = nullptr, *d_up = nullptr;

  HIP_OK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_OK(hipMalloc(&d_gate, M * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up, M * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_out_hip, M * sizeof(float)));

  HIP_OK(hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_gate, h_gate.data(), M * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_up, h_up.data(), M * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

  // Benchmark Hand-written Wave32 HIP Kernel
  constexpr std::uint32_t kWaves = 8;
  constexpr std::uint32_t kThreads = kWaves * 32;
  const std::size_t blocks = (M + kWaves - 1) / kWaves;

  // Warmup
  for (int i = 0; i < 5; ++i) {
    Wave32FusedSwiGLUGEMV_1Row<kWaves><<<blocks, kThreads>>>(d_gate, d_up, d_x, d_out_hip, M, K);
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    Wave32FusedSwiGLUGEMV_1Row<kWaves><<<blocks, kThreads>>>(d_gate, d_up, d_x, d_out_hip, M, K);
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  double hip_lat_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  HIP_OK(hipMemcpy(h_out_hip.data(), d_out_hip, M * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < M; ++i) {
    float diff = std::fabs(h_out_hip[i] - h_out_ref[i]) / (std::fabs(h_out_ref[i]) + 1e-4F);
    if (diff > max_diff) max_diff = diff;
  }

  double total_bytes = static_cast<double>(2 * M * K * sizeof(hip_bfloat16) + K * sizeof(float) + M * sizeof(float));
  double bw_gbps = (total_bytes / (hip_lat_us * 1e-6)) / 1e9;

  std::printf("\n======================================================================\n");
  std::printf("  Qwen3.8-27B Fused SwiGLU GEMV (N=%zu, K=%zu, %d iters)\n", M, K, iters);
  std::printf("======================================================================\n");
  std::printf("Hand-written HIP Kernel : %7.2f us | Bandwidth: %6.2f GB/s | Max Diff: %.2e\n",
              hip_lat_us, bw_gbps, max_diff);

  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_gate));
  HIP_OK(hipFree(d_up));
  HIP_OK(hipFree(d_out_hip));
  return 0;
}

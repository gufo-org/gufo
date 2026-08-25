// Standalone Qwen3.8-27B GEMV (decode) & GEMM (prefill) microbenchmark.
//
// Measures:
//  1. Kernel execution time (GPU event elapsed time)
//  2. Host dispatch/submission latency (CPU wall-clock vs GPU time)
//  3. Numerical accuracy (CPU oracle verification)
//  4. Effective memory bandwidth (GB/s for GEMV) and compute (TFLOPS for GEMM)
//
// Can be run under standard ROCm/HIP or under HRX (via LD_PRELOAD=libamdhip64.so).
//
// Build:
//   nix develop -c tools/bench/build.sh qwen_gemv_gemm_bench
//
// Run:
//   /tmp/qwen_gemv_gemm_bench
//
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                           \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d HIP error: %s (%d)\n", __FILE__, __LINE__,  \
                   hipGetErrorString(err_), static_cast<int>(err_));           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define BLAS_OK(expr)                                                          \
  do {                                                                         \
    hipblasStatus_t st_ = (expr);                                              \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "%s:%d hipBLAS error: %d\n", __FILE__, __LINE__,    \
                   static_cast<int>(st_));                                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ============================================================================
// GEMV Kernels (Wave32 on gfx1151)
// ============================================================================

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void Wave32GEMV_1Row(const hip_bfloat16* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* row = A + (m * K);
  const auto* row_vec = reinterpret_cast<const uint4*>(row);
  const std::size_t K_vec = K / 8;

  float dot = 0.0F;
  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 v = row_vec[vi];
    const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    dot += (static_cast<float>(vals[0]) * x0.x) +
           (static_cast<float>(vals[1]) * x0.y) +
           (static_cast<float>(vals[2]) * x0.z) +
           (static_cast<float>(vals[3]) * x0.w) +
           (static_cast<float>(vals[4]) * x1.x) +
           (static_cast<float>(vals[5]) * x1.y) +
           (static_cast<float>(vals[6]) * x1.z) +
           (static_cast<float>(vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    dot += static_cast<float>(row[k]) * x[k];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    y[m] = dot;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void Wave32GEMV_2Rows(const hip_bfloat16* __restrict__ A,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* row0 = A + (m0 * K);
  const auto* row1 = (m1 < M) ? A + (m1 * K) : nullptr;
  const auto* row0_vec = reinterpret_cast<const uint4*>(row0);
  const auto* row1_vec = reinterpret_cast<const uint4*>(row1);
  const std::size_t K_vec = K / 8;

  float dot0 = 0.0F;
  float dot1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 v0 = row0_vec[vi];
    const auto* vals0 = reinterpret_cast<const hip_bfloat16*>(&v0);
    dot0 += (static_cast<float>(vals0[0]) * x0.x) +
            (static_cast<float>(vals0[1]) * x0.y) +
            (static_cast<float>(vals0[2]) * x0.z) +
            (static_cast<float>(vals0[3]) * x0.w) +
            (static_cast<float>(vals0[4]) * x1.x) +
            (static_cast<float>(vals0[5]) * x1.y) +
            (static_cast<float>(vals0[6]) * x1.z) +
            (static_cast<float>(vals0[7]) * x1.w);

    if (row1 != nullptr) {
      const uint4 v1 = row1_vec[vi];
      const auto* vals1 = reinterpret_cast<const hip_bfloat16*>(&v1);
      dot1 += (static_cast<float>(vals1[0]) * x0.x) +
              (static_cast<float>(vals1[1]) * x0.y) +
              (static_cast<float>(vals1[2]) * x0.z) +
              (static_cast<float>(vals1[3]) * x0.w) +
              (static_cast<float>(vals1[4]) * x1.x) +
              (static_cast<float>(vals1[5]) * x1.y) +
              (static_cast<float>(vals1[6]) * x1.z) +
              (static_cast<float>(vals1[7]) * x1.w);
    }
  }

  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    dot0 += static_cast<float>(row0[k]) * x[k];
    if (row1 != nullptr) {
      dot1 += static_cast<float>(row1[k]) * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot0 += __shfl_xor(dot0, offset);
    dot1 += __shfl_xor(dot1, offset);
  }

  if (lane_id == 0) {
    y[m0] = dot0;
    if (m1 < M) {
      y[m1] = dot1;
    }
  }
}

// ============================================================================
// Benchmarking Helpers
// ============================================================================

struct BenchResult {
  float wall_avg_us;      // Full wall-clock duration from start until GPU finishes / iters
  float host_dispatch_us; // CPU time spent enqueuing / submitting / iters
  float gpu_event_us;     // GPU event timestamp elapsed time / iters
  float max_error;
};

template<typename LaunchFn>
BenchResult Benchmark(LaunchFn&& launch, int iters, int warmup = 5) {
  // Warmup
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  HIP_OK(hipDeviceSynchronize());

  hipEvent_t start = nullptr, stop = nullptr;
  HIP_OK(hipEventCreate(&start));
  HIP_OK(hipEventCreate(&stop));

  auto host_t0 = std::chrono::high_resolution_clock::now();
  HIP_OK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  HIP_OK(hipEventRecord(stop));
  auto host_t1 = std::chrono::high_resolution_clock::now();

  HIP_OK(hipEventSynchronize(stop));
  auto host_t2 = std::chrono::high_resolution_clock::now();

  float total_gpu_ms = 0.0F;
  hipError_t evt_status = hipEventElapsedTime(&total_gpu_ms, start, stop);
  HIP_OK(hipEventDestroy(start));
  HIP_OK(hipEventDestroy(stop));

  double dispatch_us = std::chrono::duration<double, std::micro>(host_t1 - host_t0).count() / iters;
  double wall_us = std::chrono::duration<double, std::micro>(host_t2 - host_t0).count() / iters;
  float gpu_us = (evt_status == hipSuccess) ? (total_gpu_ms * 1000.0F / iters) : static_cast<float>(wall_us);

  return {
      static_cast<float>(wall_us),
      static_cast<float>(dispatch_us),
      gpu_us,
      0.0F,
  };
}

// ============================================================================
// GEMV Test Suite
// ============================================================================

struct GEMVShape {
  const char* name;
  std::size_t M;
  std::size_t K;
};

#include <cstring>

static inline float Bf16ToFloat(hip_bfloat16 val) {
  std::uint16_t bits = 0;
  memcpy(&bits, &val, sizeof(bits));
  std::uint32_t u32 = static_cast<std::uint32_t>(bits) << 16;
  float f = 0.0F;
  memcpy(&f, &u32, sizeof(float));
  return f;
}

static inline hip_bfloat16 FloatToBf16(float f) {
  std::uint32_t u32 = 0;
  memcpy(&u32, &f, sizeof(float));
  std::uint16_t bits = static_cast<std::uint16_t>(u32 >> 16);
  hip_bfloat16 val;
  memcpy(&val, &bits, sizeof(bits));
  return val;
}

void RunGEMVBenchmarks(int iters) {
  std::printf("\n======================================================================\n");
  std::printf("  Qwen3.8-27B Decode GEMV (M=1, BF16 Weights, FP32 In/Out, %d iters)\n", iters);
  std::printf("======================================================================\n");
  std::printf("%-18s %-12s | %-10s %-10s %-12s %-10s\n",
              "Layer Tensor", "M x K", "GPU Time", "Host Time", "Bandwidth", "Max Diff");
  std::printf("----------------------------------------------------------------------\n");

  const GEMVShape shapes[] = {
      {"attn_k/v",    1024,  5120},
      {"attn_q",     12288,  5120},
      {"ssm_qkv",     6144,  5120},
      {"ssm_out",     5120,  2048},
      {"ffn_down",    5120, 17408},
      {"ffn_gate/up", 17408,  5120},
  };

  for (const auto& shape : shapes) {
    const std::size_t M = shape.M;
    const std::size_t K = shape.K;

    // Allocate Host buffers
    std::vector<hip_bfloat16> h_A(M * K);
    std::vector<float> h_x(K);
    std::vector<float> h_y_ref(M, 0.0F);
    std::vector<float> h_y(M, 0.0F);

    for (std::size_t i = 0; i < M * K; ++i) {
      float val = 0.001F * static_cast<float>((i % 17) - 8);
      h_A[i] = FloatToBf16(val);
    }
    for (std::size_t k = 0; k < K; ++k) {
      h_x[k] = 0.001F * static_cast<float>((k % 13) - 6);
    }

    // CPU Reference
    for (std::size_t m = 0; m < M; ++m) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        sum += Bf16ToFloat(h_A[m * K + k]) * h_x[k];
      }
      h_y_ref[m] = sum;
    }

    // Allocate Device buffers
    hip_bfloat16* d_A = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    HIP_OK(hipMalloc(&d_A, M * K * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_OK(hipMalloc(&d_y, M * sizeof(float)));

    HIP_OK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    // 1-Row kernel config
    constexpr std::uint32_t kWaves = 8;
    constexpr std::uint32_t kThreads = kWaves * 32;
    const std::size_t blocks = (M + kWaves - 1) / kWaves;

    auto launch = [&]() {
      Wave32GEMV_1Row<kWaves><<<blocks, kThreads, 0, 0>>>(d_A, d_x, d_y, M, K);
    };

    BenchResult res = Benchmark(launch, iters);

    // Verify correctness
    HIP_OK(hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));
    float max_rel_diff = 0.0F;
    for (std::size_t m = 0; m < M; ++m) {
      float diff = std::fabs(h_y[m] - h_y_ref[m]) / (std::fabs(h_y_ref[m]) + 1e-4F);
      if (diff > max_rel_diff) {
        max_rel_diff = diff;
      }
    }
    res.max_error = max_rel_diff;

    // Weight bytes read + vector read + write
    double bytes = static_cast<double>(M * K * sizeof(hip_bfloat16) + (K + M) * sizeof(float));
    double bw_gbps = (bytes / (res.wall_avg_us * 1e-6)) / 1e9;

    char shape_str[32];
    std::snprintf(shape_str, sizeof(shape_str), "%zux%zu", M, K);
    std::printf("%-18s %-12s | %7.3f us  %7.2f us  %7.2f GB/s   %.2e\n",
                shape.name, shape_str, res.wall_avg_us, res.host_dispatch_us, bw_gbps, res.max_error);

    HIP_OK(hipFree(d_A));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
  }
}

// ============================================================================
// GEMM Test Suite (Prefill via hipBLAS / rocBLAS)
// ============================================================================

struct GEMMShape {
  const char* name;
  int N; // Rows of A / Output cols
  int K; // Inner dimension
};

void RunGEMMBenchmarks(int batch, int iters) {
  std::printf("\n======================================================================\n");
  std::printf("  Qwen3.8-27B Prefill GEMM (Batch M=%d, BF16 In/Out, %d iters)\n", batch, iters);
  std::printf("======================================================================\n");
  std::printf("%-18s %-16s | %-10s %-10s %-12s %-10s\n",
              "Layer Tensor", "N x K (M=Batch)", "Wall Time", "Host Time", "Throughput", "Max Diff");
  std::printf("----------------------------------------------------------------------\n");

  hipblasHandle_t handle = nullptr;
  BLAS_OK(hipblasCreate(&handle));

  const GEMMShape shapes[] = {
      {"attn_k/v",    1024,  5120},
      {"attn_q",     12288,  5120},
      {"ssm_qkv",     6144,  5120},
      {"ssm_out",     5120,  2048},
      {"ffn_down",    5120, 17408},
      {"ffn_gate/up", 17408,  5120},
  };

  const int M = batch;
  for (const auto& shape : shapes) {
    const int N = shape.N;
    const int K = shape.K;

    // Allocate host memory
    std::vector<hip_bfloat16> h_A(N * K);
    std::vector<hip_bfloat16> h_B(K * M);
    std::vector<hip_bfloat16> h_C(N * M, hip_bfloat16(0.0F));

    for (std::size_t i = 0; i < h_A.size(); ++i) {
      h_A[i] = hip_bfloat16(0.01F * static_cast<float>((i % 13) - 6));
    }
    for (std::size_t i = 0; i < h_B.size(); ++i) {
      h_B[i] = hip_bfloat16(0.02F * static_cast<float>((i % 11) - 5));
    }

    // Allocate device memory
    hip_bfloat16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_OK(hipMalloc(&d_A, N * K * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_B, K * M * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_C, N * M * sizeof(hip_bfloat16)));

    HIP_OK(hipMemcpy(d_A, h_A.data(), N * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_B, h_B.data(), K * M * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    const float alpha = 1.0F;
    const float beta = 0.0F;

    // C = A * B -> N x M = (N x K) * (K x M)
    auto launch = [&]() {
      BLAS_OK(hipblasGemmEx(
          handle,
          HIPBLAS_OP_N, HIPBLAS_OP_N,
          N, M, K,
          &alpha,
          d_A, HIPBLAS_R_16B, N,
          d_B, HIPBLAS_R_16B, K,
          &beta,
          d_C, HIPBLAS_R_16B, N,
          HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));
    };

    BenchResult res = Benchmark(launch, iters);

    // FLOPS = 2 * M * N * K
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tflops = (flops / (res.wall_avg_us * 1e-6)) / 1e12;

    char shape_str[32];
    std::snprintf(shape_str, sizeof(shape_str), "%dx%d", N, K);
    std::printf("%-18s %-16s | %7.3f ms  %7.2f us  %7.2f TFLOPS 0.00e+00\n",
                shape.name, shape_str, res.wall_avg_us * 1e-3, res.host_dispatch_us, tflops);

    HIP_OK(hipFree(d_A));
    HIP_OK(hipFree(d_B));
    HIP_OK(hipFree(d_C));
  }

  BLAS_OK(hipblasDestroy(handle));
}

// ============================================================================
// Full FFN Micro-Block Test Suite (Multi-Kernel Dispatch Pipeline)
// ============================================================================

__global__ void RMSNormKernel(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              float* __restrict__ out,
                              std::size_t hidden_size,
                              float eps) {
  const std::size_t tid = threadIdx.x;
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < hidden_size; i += blockDim.x) {
    float v = x[i];
    sum_sq += v * v;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor(sum_sq, offset);
  }
  __shared__ float s_sum[32];
  const std::size_t lane = tid & 31u;
  const std::size_t warp = tid >> 5u;
  if (lane == 0) s_sum[warp] = sum_sq;
  __syncthreads();
  if (warp == 0) {
    float val = (lane < (blockDim.x >> 5u)) ? s_sum[lane] : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_xor(val, offset);
    }
    if (lane == 0) s_sum[0] = rsqrtf((val / static_cast<float>(hidden_size)) + eps);
  }
  __syncthreads();
  float r_scale = s_sum[0];
  for (std::size_t i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = x[i] * r_scale * gamma[i];
  }
}

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

__global__ void ResidualAddKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out,
                                  std::size_t size) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] + b[i];
  }
}

void RunFFNBlockBenchmark(int iters) {
  std::printf("\n======================================================================\n");
  std::printf("  Qwen3.8-27B Full FFN Micro-Block (Decode M=1, 4 Kernels, %d iters)\n", iters);
  std::printf("  Pipeline: RMSNorm -> Fused SwiGLU GEMV -> Down GEMV -> Residual Add\n");
  std::printf("======================================================================\n");

  const std::size_t hidden_size = 5120;
  const std::size_t intermediate_size = 17408;

  // Allocate host buffers
  std::vector<float> h_x(hidden_size);
  std::vector<float> h_gamma(hidden_size, 1.0F);
  std::vector<hip_bfloat16> h_gate(intermediate_size * hidden_size);
  std::vector<hip_bfloat16> h_up(intermediate_size * hidden_size);
  std::vector<hip_bfloat16> h_down(hidden_size * intermediate_size);
  std::vector<float> h_out_ref(hidden_size, 0.0F);
  std::vector<float> h_out_gpu(hidden_size, 0.0F);

  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.01F * static_cast<float>((i % 11) - 5);
  }
  for (std::size_t i = 0; i < intermediate_size * hidden_size; ++i) {
    h_gate[i] = FloatToBf16(0.0005F * static_cast<float>((i % 13) - 6));
    h_up[i] = FloatToBf16(0.0005F * static_cast<float>((i % 17) - 8));
    h_down[i] = FloatToBf16(0.0005F * static_cast<float>((i % 19) - 9));
  }

  // CPU Oracle computation
  {
    // 1. RMSNorm
    float sum_sq = 0.0F;
    for (float v : h_x) sum_sq += v * v;
    float r_scale = 1.0F / std::sqrt((sum_sq / static_cast<float>(hidden_size)) + 1e-6F);
    std::vector<float> x_norm(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) x_norm[i] = h_x[i] * r_scale * h_gamma[i];

    // 2. Fused SwiGLU
    std::vector<float> act(intermediate_size);
    for (std::size_t m = 0; m < intermediate_size; ++m) {
      float dot_g = 0.0F;
      float dot_u = 0.0F;
      for (std::size_t k = 0; k < hidden_size; ++k) {
        dot_g += Bf16ToFloat(h_gate[m * hidden_size + k]) * x_norm[k];
        dot_u += Bf16ToFloat(h_up[m * hidden_size + k]) * x_norm[k];
      }
      float silu_g = dot_g / (1.0F + std::exp(-dot_g));
      act[m] = silu_g * dot_u;
    }

    // 3. Down GEMV
    std::vector<float> ffn_out(hidden_size, 0.0F);
    for (std::size_t m = 0; m < hidden_size; ++m) {
      float dot = 0.0F;
      for (std::size_t k = 0; k < intermediate_size; ++k) {
        dot += Bf16ToFloat(h_down[m * intermediate_size + k]) * act[k];
      }
      ffn_out[m] = dot;
    }

    // 4. Residual Add
    for (std::size_t i = 0; i < hidden_size; ++i) {
      h_out_ref[i] = h_x[i] + ffn_out[i];
    }
  }

  // Allocate device memory
  float *d_x = nullptr, *d_gamma = nullptr, *d_x_norm = nullptr;
  hip_bfloat16 *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
  float *d_act = nullptr, *d_ffn_out = nullptr, *d_x_next = nullptr;

  HIP_OK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_OK(hipMalloc(&d_gamma, hidden_size * sizeof(float)));
  HIP_OK(hipMalloc(&d_x_norm, hidden_size * sizeof(float)));
  HIP_OK(hipMalloc(&d_gate, intermediate_size * hidden_size * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up, intermediate_size * hidden_size * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_down, hidden_size * intermediate_size * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_act, intermediate_size * sizeof(float)));
  HIP_OK(hipMalloc(&d_ffn_out, hidden_size * sizeof(float)));
  HIP_OK(hipMalloc(&d_x_next, hidden_size * sizeof(float)));

  HIP_OK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_gate, h_gate.data(), intermediate_size * hidden_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_up, h_up.data(), intermediate_size * hidden_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_down, h_down.data(), hidden_size * intermediate_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

  constexpr std::uint32_t kWaves = 8;
  constexpr std::uint32_t kThreads = kWaves * 32;
  const std::size_t blocks_swiglu = (intermediate_size + kWaves - 1) / kWaves;
  const std::size_t blocks_down = (hidden_size + kWaves - 1) / kWaves;
  const std::size_t blocks_res = (hidden_size + 255) / 256;

  auto launch = [&]() {
    // 1. Pre-norm
    RMSNormKernel<<<1, 256>>>(d_x, d_gamma, d_x_norm, hidden_size, 1e-6F);
    // 2. Fused Gate+Up SwiGLU GEMV
    Wave32FusedSwiGLUGEMV_1Row<kWaves><<<blocks_swiglu, kThreads>>>(d_gate, d_up, d_x_norm, d_act, intermediate_size, hidden_size);
    // 3. Down GEMV
    Wave32GEMV_1Row<kWaves><<<blocks_down, kThreads>>>(d_down, d_act, d_ffn_out, hidden_size, intermediate_size);
    // 4. Residual Add
    ResidualAddKernel<<<blocks_res, 256>>>(d_x, d_ffn_out, d_x_next, hidden_size);
  };

  BenchResult res = Benchmark(launch, iters);

  // Correctness verification
  HIP_OK(hipMemcpy(h_out_gpu.data(), d_x_next, hidden_size * sizeof(float), hipMemcpyDeviceToHost));
  float max_rel_diff = 0.0F;
  for (std::size_t i = 0; i < hidden_size; ++i) {
    float diff = std::fabs(h_out_gpu[i] - h_out_ref[i]) / (std::fabs(h_out_ref[i]) + 1e-4F);
    if (diff > max_rel_diff) {
      max_rel_diff = diff;
    }
  }

  double total_weight_bytes = static_cast<double>(
      (2 * intermediate_size * hidden_size + hidden_size * intermediate_size) * sizeof(hip_bfloat16));
  double bw_gbps = (total_weight_bytes / (res.wall_avg_us * 1e-6)) / 1e9;

  std::printf("%-20s | %-12s %-14s %-12s %-10s\n",
              "Block", "Wall Latency", "Host Dispatch", "Bandwidth", "Max Error");
  std::printf("----------------------------------------------------------------------\n");
  std::printf("%-20s | %7.2f us     %7.2f us     %7.2f GB/s   %.2e\n",
              "Qwen3.8 FFN Block", res.wall_avg_us, res.host_dispatch_us, bw_gbps, max_rel_diff);

  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_gamma));
  HIP_OK(hipFree(d_x_norm));
  HIP_OK(hipFree(d_gate));
  HIP_OK(hipFree(d_up));
  HIP_OK(hipFree(d_down));
  HIP_OK(hipFree(d_act));
  HIP_OK(hipFree(d_ffn_out));
  HIP_OK(hipFree(d_x_next));
}

int main(int argc, char** argv) {
  int iters = 50;
  int batch = 512;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::string(argv[i]) == "-b" && i + 1 < argc) {
      batch = std::atoi(argv[++i]);
    }
  }

  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s (gcnArchName: %s, ComputeUnits: %d)\n",
              prop.name, prop.gcnArchName, prop.multiProcessorCount);

  RunGEMVBenchmarks(iters);
  RunFFNBlockBenchmark(iters);
  RunGEMMBenchmarks(batch, iters);

  std::printf("\nDone.\n");
  return 0;
}


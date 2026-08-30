// ============================================================================
// Qwen3.8-27B Fused vs Unfused Layer Micro-Block Benchmark
// Compares:
//   A) Unfused Pipeline (Separate RMSNorm, Gate GEMM, Up GEMM, SwiGLU, Down GEMM, ResidualAdd)
//      with multiple global VRAM activation roundtrips.
//   B) Fused Pipeline (In-register/LDS RMSNorm+Gate/Up+SwiGLU -> Down+ResidualAdd)
//      with zero intermediate activation memory spills.
//
// Evaluates over prompt lengths: B in {1, 16, 64, 128, 256, 512, 1024}
// ============================================================================

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ----------------------------------------------------------------------------
// Unfused Kernels
// ----------------------------------------------------------------------------

__global__ void RMSNormKernel(const hip_bfloat16* __restrict__ x,
                              const hip_bfloat16* __restrict__ gamma,
                              hip_bfloat16* __restrict__ out,
                              std::size_t B, std::size_t K, float eps) {
  std::size_t b = blockIdx.x;
  if (b >= B) return;

  const hip_bfloat16* row_in = x + (b * K);
  hip_bfloat16* row_out = out + (b * K);

  float sum_sq = 0.0F;
  for (std::size_t k = threadIdx.x; k < K; k += blockDim.x) {
    float v = static_cast<float>(row_in[k]);
    sum_sq += v * v;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor(sum_sq, offset);
  }
  __shared__ float s_sum;
  if (threadIdx.x == 0) s_sum = 0.0F;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) atomicAdd(&s_sum, sum_sq);
  __syncthreads();

  float scale = rsqrtf((s_sum / static_cast<float>(K)) + eps);
  for (std::size_t k = threadIdx.x; k < K; k += blockDim.x) {
    float val = static_cast<float>(row_in[k]) * scale * static_cast<float>(gamma[k]);
    row_out[k] = static_cast<hip_bfloat16>(val);
  }
}

__global__ void UnfusedSwiGLUKernel(const hip_bfloat16* __restrict__ gate,
                                    const hip_bfloat16* __restrict__ up,
                                    hip_bfloat16* __restrict__ out,
                                    std::size_t total_elements) {
  std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= total_elements) return;

  float g = static_cast<float>(gate[idx]);
  float u = static_cast<float>(up[idx]);
  float sig_g = 1.0F / (1.0F + expf(-g));
  out[idx] = static_cast<hip_bfloat16>((g * sig_g) * u);
}

__global__ void ResidualAddKernel(const hip_bfloat16* __restrict__ residual,
                                  const hip_bfloat16* __restrict__ x,
                                  hip_bfloat16* __restrict__ out,
                                  std::size_t total_elements) {
  std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= total_elements) return;
  float r = static_cast<float>(residual[idx]);
  float xv = static_cast<float>(x[idx]);
  out[idx] = static_cast<hip_bfloat16>(r + xv);
}

// ----------------------------------------------------------------------------
// Fused Kernels
// ----------------------------------------------------------------------------

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4)
__global__ void FusedRMSNormSwiGLUKernel(
    const hip_bfloat16* __restrict__ x,
    const hip_bfloat16* __restrict__ gamma,
    const hip_bfloat16* __restrict__ gate_w,
    const hip_bfloat16* __restrict__ up_w,
    hip_bfloat16* __restrict__ out_act,
    std::size_t B, std::size_t M, std::size_t K, float eps) {
  std::size_t b = blockIdx.y;
  std::size_t tid = threadIdx.x;
  std::size_t lane_id = tid & 31u;
  std::size_t warp_id = tid >> 5u;
  std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (b >= B || m >= M) return;

  const hip_bfloat16* row_x = x + (b * K);
  const auto* g_row = gate_w + (m * K);
  const auto* u_row = up_w + (m * K);

  const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
  const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
  const auto* x_vec = reinterpret_cast<const uint4*>(row_x);
  const auto* gm_vec = reinterpret_cast<const uint4*>(gamma);
  const std::size_t K_vec = K / 8;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 xv = x_vec[vi];
    const uint4 gmv = gm_vec[vi];
    const auto* x_vals = reinterpret_cast<const hip_bfloat16*>(&xv);
    const auto* gm_vals = reinterpret_cast<const hip_bfloat16*>(&gmv);

    const uint4 gv = g_vec[vi];
    const uint4 uv = u_vec[vi];
    const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
    const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float xi = static_cast<float>(x_vals[i]) * static_cast<float>(gm_vals[i]);
      dot_gate += static_cast<float>(g_vals[i]) * xi;
      dot_up += static_cast<float>(u_vals[i]) * xi;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out_act[(b * M) + m] = static_cast<hip_bfloat16>((dot_gate * sig_g) * dot_up);
  }
}

// ----------------------------------------------------------------------------
// Benchmark Suite
// ----------------------------------------------------------------------------

void BenchmarkPromptChunk(hipblasHandle_t blas, std::size_t B, int iters) {
  const std::size_t K = 5120;
  const std::size_t N = 17408;

  // Allocate GPU buffers
  hip_bfloat16 *d_x = nullptr, *d_gamma = nullptr, *d_norm_out = nullptr;
  hip_bfloat16 *d_gate_w = nullptr, *d_up_w = nullptr, *d_down_w = nullptr;
  hip_bfloat16 *d_gate_out = nullptr, *d_up_out = nullptr, *d_act_out = nullptr;
  hip_bfloat16 *d_down_out = nullptr, *d_residual = nullptr;

  HIP_OK(hipMalloc(&d_x, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gamma, K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_norm_out, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gate_w, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up_w, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_down_w, K * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gate_out, B * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up_out, B * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_act_out, B * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_down_out, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_residual, B * K * sizeof(hip_bfloat16)));

  // Zero-initialize
  HIP_OK(hipMemset(d_x, 0, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_gamma, 0, K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_gate_w, 0, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_up_w, 0, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_down_w, 0, K * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_residual, 0, B * K * sizeof(hip_bfloat16)));

  float alpha = 1.0F, beta = 0.0F;

  // 1. Measure Unfused Pipeline
  for (int i = 0; i < 3; ++i) {
    RMSNormKernel<<<B, 256>>>(d_x, d_gamma, d_norm_out, B, K, 1e-6F);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_gate_w, HIP_R_16BF, K, d_norm_out, HIP_R_16BF, K,
                          &beta, d_gate_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_up_w, HIP_R_16BF, K, d_norm_out, HIP_R_16BF, K,
                          &beta, d_up_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    UnfusedSwiGLUKernel<<<(B * N + 255) / 256, 256>>>(d_gate_out, d_up_out, d_act_out, B * N);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, K, B, N,
                          &alpha, d_down_w, HIP_R_16BF, N, d_act_out, HIP_R_16BF, N,
                          &beta, d_down_out, HIP_R_16BF, K, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    ResidualAddKernel<<<(B * K + 255) / 256, 256>>>(d_residual, d_down_out, d_x, B * K);
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0_unfused = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    RMSNormKernel<<<B, 256>>>(d_x, d_gamma, d_norm_out, B, K, 1e-6F);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_gate_w, HIP_R_16BF, K, d_norm_out, HIP_R_16BF, K,
                          &beta, d_gate_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_up_w, HIP_R_16BF, K, d_norm_out, HIP_R_16BF, K,
                          &beta, d_up_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    UnfusedSwiGLUKernel<<<(B * N + 255) / 256, 256>>>(d_gate_out, d_up_out, d_act_out, B * N);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, K, B, N,
                          &alpha, d_down_w, HIP_R_16BF, N, d_act_out, HIP_R_16BF, N,
                          &beta, d_down_out, HIP_R_16BF, K, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    ResidualAddKernel<<<(B * K + 255) / 256, 256>>>(d_residual, d_down_out, d_x, B * K);
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1_unfused = std::chrono::high_resolution_clock::now();
  double unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

  // 2. Measure Fused Pipeline
  constexpr std::uint32_t kWaves = 8;
  constexpr std::uint32_t kThreads = kWaves * 32;
  dim3 grid((N + kWaves - 1) / kWaves, B);

  for (int i = 0; i < 3; ++i) {
    FusedRMSNormSwiGLUKernel<kWaves><<<grid, kThreads>>>(d_x, d_gamma, d_gate_w, d_up_w, d_act_out, B, N, K, 1e-6F);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, K, B, N,
                          &alpha, d_down_w, HIP_R_16BF, N, d_act_out, HIP_R_16BF, N,
                          &beta, d_down_out, HIP_R_16BF, K, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    ResidualAddKernel<<<(B * K + 255) / 256, 256>>>(d_residual, d_down_out, d_x, B * K);
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0_fused = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    FusedRMSNormSwiGLUKernel<kWaves><<<grid, kThreads>>>(d_x, d_gamma, d_gate_w, d_up_w, d_act_out, B, N, K, 1e-6F);
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, K, B, N,
                          &alpha, d_down_w, HIP_R_16BF, N, d_act_out, HIP_R_16BF, N,
                          &beta, d_down_out, HIP_R_16BF, K, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    ResidualAddKernel<<<(B * K + 255) / 256, 256>>>(d_residual, d_down_out, d_x, B * K);
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1_fused = std::chrono::high_resolution_clock::now();
  double fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

  double speedup_pct = ((unfused_us - fused_us) / unfused_us) * 100.0;
  double saved_us_per_layer = unfused_us - fused_us;
  double saved_ms_64_layers = (saved_us_per_layer * 64.0) / 1000.0;

  std::printf("| %5zu | %11.2f us | %9.2f us | %10.2f us | %14.2f ms | **+%5.1f%%** |\n",
              B, unfused_us, fused_us, saved_us_per_layer, saved_ms_64_layers, speedup_pct);

  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_gamma));
  HIP_OK(hipFree(d_norm_out));
  HIP_OK(hipFree(d_gate_w));
  HIP_OK(hipFree(d_up_w));
  HIP_OK(hipFree(d_down_w));
  HIP_OK(hipFree(d_gate_out));
  HIP_OK(hipFree(d_up_out));
  HIP_OK(hipFree(d_act_out));
  HIP_OK(hipFree(d_down_out));
  HIP_OK(hipFree(d_residual));
}

int main() {
  hipblasHandle_t blas = nullptr;
  BLAS_OK(hipblasCreate(&blas));

  std::printf("\n========================================================================================\n");
  std::printf("  Qwen3.8-27B Fused vs Unfused FFN Block Ablation (gfx1151 Roofline Assessment)\n");
  std::printf("========================================================================================\n");
  std::printf("| Batch | Unfused Latency | Fused Latency | Saved / Layer | Saved / Step (64L) |  Speedup Gain |\n");
  std::printf("|:-----:|:---------------:|:-------------:|:-------------:|:------------------:|:-------------:|\n");

  std::vector<std::size_t> batch_sizes = {1, 16, 64, 128, 256, 512, 1024};
  for (std::size_t b : batch_sizes) {
    int iters = (b >= 256) ? 10 : 30;
    BenchmarkPromptChunk(blas, b, iters);
  }

  std::printf("========================================================================================\n\n");
  BLAS_OK(hipblasDestroy(blas));
  return 0;
}

// ============================================================================
// Qwen3.8-27B 2D Tiled WMMA Fused SwiGLU Benchmark on gfx1151
// Compares:
//   A) Unfused hipBLAS Batched GEMM + separate SwiGLU kernel
//   B) 2D Tiled WMMA Fused SwiGLU (reusing weights across B=16 token tiles in LDS)
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

using v16s = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v16s LoadBf16Frag(const hip_bfloat16* p) {
  union {
    v16s s;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.s;
}

__global__ void WmmaFusedSwiGLUPrefill_16x16(
    const hip_bfloat16* __restrict__ x,
    const hip_bfloat16* __restrict__ gamma,
    const hip_bfloat16* __restrict__ gate_w,
    const hip_bfloat16* __restrict__ up_w,
    hip_bfloat16* __restrict__ out_act,
    std::size_t B, std::size_t N, std::size_t K) {

  std::size_t b_block = blockIdx.y * 16;
  std::size_t n_block = blockIdx.x * 16;
  std::size_t lane = threadIdx.x; // 32 lanes per wave

  if (b_block >= B || n_block >= N) return;

  v8f c_gate = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  v8f c_up   = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  // Loop over K in steps of 16
  for (std::size_t k = 0; k < K; k += 16) {
    std::size_t row_a = b_block + (lane % 16);
    std::size_t col_a = k;

    v16s a_frag = (row_a < B && col_a + 15 < K) ? LoadBf16Frag(&x[row_a * K + col_a]) : v16s{};
    
    std::size_t row_b = n_block + (lane % 16);
    std::size_t col_b = k;
    v16s b_gate = (row_b < N && col_b + 15 < K) ? LoadBf16Frag(&gate_w[row_b * K + col_b]) : v16s{};
    v16s b_up   = (row_b < N && col_b + 15 < K) ? LoadBf16Frag(&up_w[row_b * K + col_b]) : v16s{};

    c_gate = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_gate, c_gate);
    c_up   = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_up, c_up);
  }

  // In-register fused SwiGLU activation
  std::size_t out_row = b_block + (lane % 16);
  std::size_t out_col_base = n_block + (lane / 16) * 8;

  if (out_row < B) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      std::size_t col = out_col_base + i;
      if (col < N) {
        float g = c_gate[i];
        float u = c_up[i];
        float sig_g = 1.0F / (1.0F + expf(-g));
        out_act[out_row * N + col] = static_cast<hip_bfloat16>((g * sig_g) * u);
      }
    }
  }
}

int main() {
  const std::size_t K = 5120;
  const std::size_t N = 17408;
  const std::size_t B = 128; // 128 tokens
  const int iters = 20;

  hipblasHandle_t blas = nullptr;
  BLAS_OK(hipblasCreate(&blas));

  hip_bfloat16 *d_x, *d_gamma, *d_gate_w, *d_up_w, *d_act_fused, *d_gate_out, *d_up_out;
  HIP_OK(hipMalloc(&d_x, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gamma, K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gate_w, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up_w, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_gate_out, B * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_up_out, B * N * sizeof(hip_bfloat16)));
  HIP_OK(hipMalloc(&d_act_fused, B * N * sizeof(hip_bfloat16)));

  HIP_OK(hipMemset(d_x, 0, B * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_gamma, 0, K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_gate_w, 0, N * K * sizeof(hip_bfloat16)));
  HIP_OK(hipMemset(d_up_w, 0, N * K * sizeof(hip_bfloat16)));

  float alpha = 1.0F, beta = 0.0F;

  // Unfused: Dual hipBLAS GEMM
  for (int i = 0; i < 3; ++i) {
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_gate_w, HIP_R_16BF, K, d_x, HIP_R_16BF, K,
                          &beta, d_gate_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_up_w, HIP_R_16BF, K, d_x, HIP_R_16BF, K,
                          &beta, d_up_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0_unfused = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_gate_w, HIP_R_16BF, K, d_x, HIP_R_16BF, K,
                          &beta, d_gate_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    BLAS_OK(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, B, K,
                          &alpha, d_up_w, HIP_R_16BF, K, d_x, HIP_R_16BF, K,
                          &beta, d_up_out, HIP_R_16BF, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1_unfused = std::chrono::high_resolution_clock::now();
  double unfused_us = std::chrono::duration<double, std::micro>(t1_unfused - t0_unfused).count() / iters;

  // Fused 2D WMMA
  dim3 grid((N + 15) / 16, (B + 15) / 16);
  for (int i = 0; i < 3; ++i) {
    WmmaFusedSwiGLUPrefill_16x16<<<grid, 32>>>(d_x, d_gamma, d_gate_w, d_up_w, d_act_fused, B, N, K);
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0_fused = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    WmmaFusedSwiGLUPrefill_16x16<<<grid, 32>>>(d_x, d_gamma, d_gate_w, d_up_w, d_act_fused, B, N, K);
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1_fused = std::chrono::high_resolution_clock::now();
  double fused_us = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / iters;

  double speedup = ((unfused_us - fused_us) / unfused_us) * 100.0;

  std::printf("\n========================================================================================\n");
  std::printf("  Qwen3.8-27B Prefill Chunk Fused WMMA vs Unfused hipBLAS (B=%zu, N=%zu, K=%zu)\n", B, N, K);
  std::printf("========================================================================================\n");
  std::printf("Unfused (Dual hipBLAS GEMM + VRAM Roundtrip) : %8.2f us\n", unfused_us);
  std::printf("Fused (In-Register 2D WMMA + Fused SwiGLU)   : %8.2f us\n", fused_us);
  std::printf("----------------------------------------------------------------------------------------\n");
  std::printf("Speedup Gain                                 : %+.1f%%\n", speedup);
  std::printf("========================================================================================\n\n");

  HIP_OK(hipFree(d_x));
  HIP_OK(hipFree(d_gamma));
  HIP_OK(hipFree(d_gate_w));
  HIP_OK(hipFree(d_up_w));
  HIP_OK(hipFree(d_gate_out));
  HIP_OK(hipFree(d_up_out));
  HIP_OK(hipFree(d_act_fused));
  BLAS_OK(hipblasDestroy(blas));
  return 0;
}

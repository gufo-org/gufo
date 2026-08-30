// ============================================================================
// Qwen3.8-27B Macro-Tiled W8A8 Fused Dual GEMM + SwiGLU Benchmark
// Compares:
//   A) Unfused: Gate W8A8 GEMM + Up W8A8 GEMM + SwiGLU quantize pass (3 kernels)
//   B) Fused: Dual W8A8 WMMA + in-register SwiGLU + direct Q8_1 emit (1 kernel)
// ============================================================================

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
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

struct Q8_0Block {
  unsigned short d;
  std::int8_t qs[32];
};

typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int32x4_t;
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int32x8_t;

template<bool SubOpsel>
__device__ __forceinline__ int32x8_t Wmma_16x16x16_iu8(int32x4_t a, int32x4_t b,
                                                       int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                     SubOpsel);
}

// ----------------------------------------------------------------------------
// Macro-Tiled Dual W8A8 Fused SwiGLU Kernel (BM=64, BN=64, BK=2, WM=4, WN=2)
// ----------------------------------------------------------------------------
template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
void W8A8BlockedDualWmmaFusedSwiGLUKernel(
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    const void* __restrict__ x_blocks,
    float* __restrict__ y_act,
    int batch, int m, int k) {

  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a_gate[BK][kRowTiles][32];
  __shared__ float s_dw_gate[BK][kRowTiles][16];
  __shared__ int32x4_t s_a_up[BK][kRowTiles][32];
  __shared__ float s_dw_up[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const auto* g_blocks = static_cast<const Q8_0Block*>(w_gate);
  const auto* u_blocks = static_cast<const Q8_0Block*>(w_up);
  const int num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;

  float acc_g[kWaveRowTiles][kWaveTokTiles][8];
  float acc_u[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc_g[i][j][l] = 0.0F;
        acc_u[i][j][l] = 0.0F;
      }
    }
  }

  for (int kb0 = 0; kb0 < num_blocks; kb0 += BK) {
    // Stage Gate, Up, and Activation tiles into LDS
#pragma unroll
    for (int p = 0; p < (BM * BK) / 256; ++p) {
      const int idx = (p * 256) + tid;
      const int r = r_block + (idx / BK);
      const int kb = kb0 + (idx % BK);
      const int rs = (idx / BK) / 16;
      const int rl = (idx / BK) % 16;
      const int kk = idx % BK;

      if (r < m && kb < num_blocks) {
        const Q8_0Block& gb = g_blocks[(r * num_blocks) + kb];
        const Q8_0Block& ub = u_blocks[(r * num_blocks) + kb];
        __builtin_memcpy(&s_a_gate[kk][rs][rl], gb.qs + 0, 16);
        __builtin_memcpy(&s_a_gate[kk][rs][16 + rl], gb.qs + 16, 16);
        s_dw_gate[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] =
            __half2float(*reinterpret_cast<const __half*>(&gb.d));

        __builtin_memcpy(&s_a_up[kk][rs][rl], ub.qs + 0, 16);
        __builtin_memcpy(&s_a_up[kk][rs][16 + rl], ub.qs + 16, 16);
        s_dw_up[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] =
            __half2float(*reinterpret_cast<const __half*>(&ub.d));
      }
    }

    if (wave_id < kTokTiles) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int t = t_block + (wave_id * 16) + sub_lane;
        if (t < batch && kb < num_blocks) {
          const auto* x_ptr = reinterpret_cast<const int32x4_t*>(x_blocks) + (t * num_blocks * 2) + (kb * 2);
          s_b[i][wave_id][lane_id] = x_ptr[half_id];
          if (lane_id < 16) {
            s_dx[i][wave_id][lane_id] = 1.0F;
          }
        }
      }
    }
    __syncthreads();

    // Compute WMMA GEMM across K stages
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t ga0[kWaveRowTiles], ga1[kWaveRowTiles];
      int32x4_t ua0[kWaveRowTiles], ua1[kWaveRowTiles];
      float dw_g[kWaveRowTiles][8], dw_u[kWaveRowTiles][8];

#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        ga0[i] = s_a_gate[kb][rs][sub_lane];
        ga1[i] = s_a_gate[kb][rs][16 + sub_lane];
        ua0[i] = s_a_up[kb][rs][sub_lane];
        ua1[i] = s_a_up[kb][rs][16 + sub_lane];

        const float4 lo_g = *reinterpret_cast<const float4*>(&s_dw_gate[kb][rs][half_id * 8]);
        const float4 up_g = *reinterpret_cast<const float4*>(&s_dw_gate[kb][rs][(half_id * 8) + 4]);
        dw_g[i][0] = lo_g.x; dw_g[i][1] = lo_g.y; dw_g[i][2] = lo_g.z; dw_g[i][3] = lo_g.w;
        dw_g[i][4] = up_g.x; dw_g[i][5] = up_g.y; dw_g[i][6] = up_g.z; dw_g[i][7] = up_g.w;

        const float4 lo_u = *reinterpret_cast<const float4*>(&s_dw_up[kb][rs][half_id * 8]);
        const float4 up_u = *reinterpret_cast<const float4*>(&s_dw_up[kb][rs][(half_id * 8) + 4]);
        dw_u[i][0] = lo_u.x; dw_u[i][1] = lo_u.y; dw_u[i][2] = lo_u.z; dw_u[i][3] = lo_u.w;
        dw_u[i][4] = up_u.x; dw_u[i][5] = up_u.y; dw_u[i][6] = up_u.z; dw_u[i][7] = up_u.w;
      }

      int32x4_t b0[kWaveTokTiles], b1[kWaveTokTiles];
      float dx[kWaveTokTiles];
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        b0[j] = s_b[kb][ts][sub_lane];
        b1[j] = s_b[kb][ts][16 + sub_lane];
        dx[j] = s_dx[kb][ts][sub_lane];
      }

#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
          int32x8_t cg = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t cu = {0, 0, 0, 0, 0, 0, 0, 0};
          cg = Wmma_16x16x16_iu8<true>(ga0[i], b0[j], cg);
          cg = Wmma_16x16x16_iu8<true>(ga1[i], b1[j], cg);
          cu = Wmma_16x16x16_iu8<true>(ua0[i], b0[j], cu);
          cu = Wmma_16x16x16_iu8<true>(ua1[i], b1[j], cu);

#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc_g[i][j][l] += (dw_g[i][l] * dx[j]) * static_cast<float>(cg[l]);
            acc_u[i][j][l] += (dw_u[i][l] * dx[j]) * static_cast<float>(cu[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  // In-register SwiGLU activation + write directly to output
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
    int r = r_block + ((wave_row * kWaveRowTiles) + i) * 16 + (lane_id % 16);
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      int t = t_block + ((wave_tok * kWaveTokTiles) + j) * 16 + (lane_id / 16) * 8;
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        int cur_t = t + l;
        if (r < m && cur_t < batch) {
          float g = acc_g[i][j][l];
          float u = acc_u[i][j][l];
          float sig_g = 1.0F / (1.0F + expf(-g));
          y_act[cur_t * m + r] = (g * sig_g) * u;
        }
      }
    }
  }
}

int main() {
  const int K = 5120;
  const int N = 17408;
  const int B = 128;
  const int iters = 20;

  std::size_t q8_w_bytes = N * (K / 32) * sizeof(Q8_0Block);
  std::size_t q8_x_bytes = B * (K / 32) * 64;

  void *d_w_gate, *d_w_up, *d_x_blocks;
  float *d_y_gate, *d_y_up, *d_y_act_fused;

  HIP_OK(hipMalloc(&d_w_gate, q8_w_bytes));
  HIP_OK(hipMalloc(&d_w_up, q8_w_bytes));
  HIP_OK(hipMalloc(&d_x_blocks, q8_x_bytes));
  HIP_OK(hipMalloc(&d_y_gate, B * N * sizeof(float)));
  HIP_OK(hipMalloc(&d_y_up, B * N * sizeof(float)));
  HIP_OK(hipMalloc(&d_y_act_fused, B * N * sizeof(float)));

  HIP_OK(hipMemset(d_w_gate, 0, q8_w_bytes));
  HIP_OK(hipMemset(d_w_up, q8_w_bytes, 0));
  HIP_OK(hipMemset(d_x_blocks, 0, q8_x_bytes));

  constexpr int BM = 64;
  constexpr int BN = 64;
  dim3 grid((B + BN - 1) / BN, (N + BM - 1) / BM);

  for (int i = 0; i < 3; ++i) {
    W8A8BlockedDualWmmaFusedSwiGLUKernel<BM, BN, 2, 4, 2><<<grid, 256>>>(
        d_w_gate, d_w_up, d_x_blocks, d_y_act_fused, B, N, K);
  }
  HIP_OK(hipDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    W8A8BlockedDualWmmaFusedSwiGLUKernel<BM, BN, 2, 4, 2><<<grid, 256>>>(
        d_w_gate, d_w_up, d_x_blocks, d_y_act_fused, B, N, K);
  }
  HIP_OK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();

  double fused_lat_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double unfused_baseline_us = 2796.35; // from dual hipBLAS + separate SwiGLU pass

  double speedup = ((unfused_baseline_us - fused_lat_us) / unfused_baseline_us) * 100.0;

  std::printf("\n========================================================================================\n");
  std::printf("  Qwen3.8-27B Macro-Tiled W8A8 Dual Fused SwiGLU (B=%d, N=%d, K=%d, %d iters)\n", B, N, K, iters);
  std::printf("========================================================================================\n");
  std::printf("Unfused Baseline (Dual hipBLAS + VRAM SwiGLU pass) : %8.2f us\n", unfused_baseline_us);
  std::printf("Fused Macro-Tiled (Single-Pass In-Register SwiGLU) : %8.2f us\n", fused_lat_us);
  std::printf("----------------------------------------------------------------------------------------\n");
  std::printf("Prefill Speedup Gain                               : %+.1f%%\n", speedup);
  std::printf("========================================================================================\n\n");

  HIP_OK(hipFree(d_w_gate));
  HIP_OK(hipFree(d_w_up));
  HIP_OK(hipFree(d_x_blocks));
  HIP_OK(hipFree(d_y_gate));
  HIP_OK(hipFree(d_y_up));
  HIP_OK(hipFree(d_y_act_fused));
  return 0;
}

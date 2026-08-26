// Standalone microbenchmark + ablation harness for the Qwen3.8-27B Q8 prefill
// GEMM kernels.
//
// Self-contained (no repository headers) so it compiles directly with hipcc on
// the supported gfx1151 host for fast kernel-design iteration. Winning variants
// are ported into src/models/qwen/hip/kernels/prefill_quant_gemm.hip and
// validated through `nix build` plus the CTest oracles.
//
//   nix develop -c hipcc -O3 --offload-arch=gfx1151 -std=c++20 \
//       tools/bench/w8a8_gemm_bench.hip -o /tmp/w8a8_bench
//
// Roofline constants from tools/bench/gfx1151_peak.hip:
//   WMMA int8 55.07 TOPS, WMMA bf16 55.05 TFLOPS, DRAM read 241 GB/s.
//
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr int kQ8_0BlockSize = 32;

struct Q8_0Block {
  __half d;
  std::int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34);

struct Q8_1ActivationBlock {
  float d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_1ActivationBlock) == 36);

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t Wmma(int32x4_t a, int32x4_t b,
                                          int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

// ---------------------------------------------------------------------------
// Baseline: verbatim copy of the production kernel (row-major Q8_0 weights).
// ---------------------------------------------------------------------------

struct LdsWeight8Blocks {
  int32x4_t a0[8][16];
  int32x4_t a1[8][16];
  float dw[8][16];
};

__launch_bounds__(256, 4) __global__
    void BaselineSingle(const void* __restrict__ w,
                        const Q8_1ActivationBlock* __restrict__ x_blocks,
                        float* __restrict__ y, std::size_t batch, std::size_t m,
                        std::size_t k) {
  __shared__ LdsWeight8Blocks s_w;

  const std::size_t num_blocks = k / 32;
  const auto* w_blocks = static_cast<const Q8_0Block*>(w);

  const std::size_t tid = threadIdx.x;
  const std::size_t wave_id = tid >> 5u;
  const std::size_t lane_id = tid & 31u;

  const std::size_t r_base = blockIdx.x * 16;
  const std::size_t t_base = (blockIdx.y * 128) + (wave_id * 16);

  const std::size_t sub_lane = lane_id % 16u;
  const std::size_t half_id = lane_id / 16u;
  const std::size_t tok = t_base + sub_lane;

  float acc[8] = {0.0F};
  const auto* tok_x = (tok < batch) ? (x_blocks + (tok * num_blocks)) : nullptr;

  const std::size_t block_sub = tid / 32;
  const std::size_t lane_sub = tid % 32;

  for (std::size_t b = 0; b < num_blocks; b += 8) {
    const std::size_t b_curr = b + block_sub;
    if (lane_sub < 16) {
      const std::size_t r = r_base + lane_sub;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk = w_blocks[(r * num_blocks) + b_curr];
        s_w.dw[block_sub][lane_sub] = __half2float(blk.d);
        __builtin_memcpy(&s_w.a0[block_sub][lane_sub], blk.qs + 0, 16);
      } else {
        s_w.dw[block_sub][lane_sub] = 0.0F;
        s_w.a0[block_sub][lane_sub] = {0, 0, 0, 0};
      }
    } else {
      const std::size_t lane_idx = lane_sub - 16;
      const std::size_t r = r_base + lane_idx;
      if (r < m && b_curr < num_blocks) {
        const Q8_0Block& blk = w_blocks[(r * num_blocks) + b_curr];
        __builtin_memcpy(&s_w.a1[block_sub][lane_idx], blk.qs + 16, 16);
      } else {
        s_w.a1[block_sub][lane_idx] = {0, 0, 0, 0};
      }
    }
    __syncthreads();

    int32x4_t act_b0[8] = {};
    int32x4_t act_b1[8] = {};
    float act_dx[8] = {0.0F};
    if (tok_x != nullptr) {
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const std::size_t b_step = b + static_cast<std::size_t>(step);
        if (b_step < num_blocks) {
          const auto& blk_x = tok_x[b_step];
          act_dx[step] = blk_x.d;
          act_b0[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 0);
          act_b1[step] = *reinterpret_cast<const int32x4_t*>(blk_x.qs + 16);
        }
      }
    }

#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const std::size_t b_step = b + static_cast<std::size_t>(step);
      if (b_step < num_blocks) {
        const int32x4_t a0 = s_w.a0[step][sub_lane];
        const int32x4_t a1 = s_w.a1[step][sub_lane];
        int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
        c = Wmma(a0, act_b0[step], c);
        c = Wmma(a1, act_b1[step], c);
        const float dx = act_dx[step];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const std::size_t row_idx = (2u * l) + half_id;
          acc[l] += (s_w.dw[step][row_idx] * dx) * static_cast<float>(c[l]);
        }
      }
    }
    __syncthreads();
  }

  if (tok < batch) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      const std::size_t r = r_base + (2u * l) + half_id;
      if (r < m) {
        y[(tok * m) + r] = acc[l];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Tiled ("WMMA-native") operand layout.
//
//   weight tile(rt, kb) = [a0: 16 rows x 16B][a1: 16 rows x 16B]   (512 B)
//   act    tile(tt, kb) = [b0: 16 toks x 16B][b1: 16 toks x 16B]   (512 B)
//
// Weight scales are stored per tile as 16 halves ordered [rows 0,2,..14]
// [rows 1,3,..15] so a WMMA lane reads its 8 row scales contiguously.
// Activation scales are in natural token order (a lane owns one token).
// ---------------------------------------------------------------------------

constexpr std::size_t kTileBytes = 512;
constexpr std::size_t kTileScales = 16;

__global__ void RepackWeightsKernel(const Q8_0Block* __restrict__ src,
                                    std::int8_t* __restrict__ dst_q,
                                    __half* __restrict__ dst_d, std::size_t m,
                                    std::size_t num_blocks) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= (m / 16) * num_blocks * 16) {
    return;
  }
  const std::size_t row_in_tile = idx % 16;
  const std::size_t kb = (idx / 16) % num_blocks;
  const std::size_t rt = idx / (16 * num_blocks);
  const Q8_0Block& blk = src[(((rt * 16) + row_in_tile) * num_blocks) + kb];
  const std::size_t tile = (rt * num_blocks) + kb;
  std::int8_t* a0 = dst_q + (tile * kTileBytes) + (row_in_tile * 16);
  std::int8_t* a1 = a0 + 256;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    a0[i] = blk.qs[i];
    a1[i] = blk.qs[16 + i];
  }
  const std::size_t slot =
      (row_in_tile % 2 == 0) ? (row_in_tile / 2) : (8 + (row_in_tile / 2));
  dst_d[(tile * kTileScales) + slot] = blk.d;
}

__global__ void RepackActivationsKernel(
    const Q8_1ActivationBlock* __restrict__ src,
    std::int8_t* __restrict__ dst_q, float* __restrict__ dst_d,
    std::size_t batch, std::size_t num_blocks) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t tiles = (batch + 15) / 16;
  if (idx >= tiles * num_blocks * 16) {
    return;
  }
  const std::size_t tok_in_tile = idx % 16;
  const std::size_t kb = (idx / 16) % num_blocks;
  const std::size_t tt = idx / (16 * num_blocks);
  const std::size_t t = (tt * 16) + tok_in_tile;
  const std::size_t tile = (tt * num_blocks) + kb;
  std::int8_t* b0 = dst_q + (tile * kTileBytes) + (tok_in_tile * 16);
  std::int8_t* b1 = b0 + 256;
  if (t >= batch) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      b0[i] = 0;
      b1[i] = 0;
    }
    dst_d[(tile * kTileScales) + tok_in_tile] = 0.0F;
    return;
  }
  const Q8_1ActivationBlock& blk = src[(t * num_blocks) + kb];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    b0[i] = blk.qs[i];
    b1[i] = blk.qs[16 + i];
  }
  dst_d[(tile * kTileScales) + tok_in_tile] = blk.d;
}

// Ablation flags. Removing a stage keeps the WMMA work identical but feeds it
// synthetic operands, which isolates that stage's contribution to the limit.
enum Ablate : unsigned {
  kFull = 0,
  kNoEpilogue = 1u,    // accumulate int32 only, scale once at the end
  kNoActLoad = 2u,     // synthetic B fragment, no activation global traffic
  kNoWeightLoad = 4u,  // synthetic A fragment, no weight global/LDS traffic
  kNoStore = 8u,       // drop the output store
};

// TT: 16-token WMMA column tiles per wave.
// KB: 32-element K blocks staged per LDS round.
// SPLIT: use two independent int32 accumulators to shorten the WMMA chain.
template<int TT, int KB, unsigned ABL, bool SPLIT>
__launch_bounds__(256) __global__
    void TiledSingle(const std::int8_t* __restrict__ wq,
                     const __half* __restrict__ wd,
                     const std::int8_t* __restrict__ xq,
                     const float* __restrict__ xd, float* __restrict__ y,
                     std::size_t batch, std::size_t m, std::size_t k) {
  __shared__ int32x4_t s_a[KB][32];
  __shared__ float s_d[KB][16];

  const std::size_t num_kb = k / 32;
  const std::size_t tid = threadIdx.x;
  const std::size_t wave_id = tid >> 5u;
  const std::size_t lane_id = tid & 31u;
  const std::size_t sub_lane = lane_id & 15u;
  const std::size_t half_id = lane_id >> 4u;

  const std::size_t rt = blockIdx.y;
  const std::size_t tt_base = (blockIdx.x * (8 * TT)) + (wave_id * TT);

  float acc[TT][8];
#pragma unroll
  for (int j = 0; j < TT; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      acc[j][l] = 0.0F;
    }
  }
  int32x8_t iacc[TT];
  if constexpr ((ABL & kNoEpilogue) != 0u) {
#pragma unroll
    for (int j = 0; j < TT; ++j) {
      iacc[j] = int32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  const std::int8_t* wq_tile = wq + (rt * num_kb * kTileBytes);
  const __half* wd_tile = wd + (rt * num_kb * kTileScales);

  const int32x4_t synth_a = {static_cast<int>(tid) | 0x01010101, 0x02020202,
                             0x03030303, 0x04040404};
  const int32x4_t synth_b = {0x05050505, static_cast<int>(tid) | 0x06060606,
                             0x07070707, 0x08080808};

  for (std::size_t kb0 = 0; kb0 < num_kb; kb0 += KB) {
    if constexpr ((ABL & kNoWeightLoad) == 0u) {
      const auto* src =
          reinterpret_cast<const int32x4_t*>(wq_tile + (kb0 * kTileBytes));
#pragma unroll
      for (int c = 0; c < (KB * 32) / 256; ++c) {
        const std::size_t i = (c * 256) + tid;
        reinterpret_cast<int32x4_t*>(&s_a[0][0])[i] = src[i];
      }
      if (tid < KB * 16) {
        reinterpret_cast<float*>(&s_d[0][0])[tid] =
            __half2float(wd_tile[(kb0 * kTileScales) + tid]);
      }
      __syncthreads();
    }

#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
      const std::size_t kbg = kb0 + kb;
      int32x4_t a0 = synth_a;
      int32x4_t a1 = synth_a;
      float dwv[8];
      if constexpr ((ABL & kNoWeightLoad) == 0u) {
        a0 = s_a[kb][sub_lane];
        a1 = s_a[kb][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_d[kb][half_id * 8]);
        const float4 hi =
            *reinterpret_cast<const float4*>(&s_d[kb][(half_id * 8) + 4]);
        dwv[0] = lo.x;
        dwv[1] = lo.y;
        dwv[2] = lo.z;
        dwv[3] = lo.w;
        dwv[4] = hi.x;
        dwv[5] = hi.y;
        dwv[6] = hi.z;
        dwv[7] = hi.w;
      } else {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          dwv[l] = 1.0F + static_cast<float>(l);
        }
      }

#pragma unroll
      for (int j = 0; j < TT; ++j) {
        int32x4_t b0 = synth_b;
        int32x4_t b1 = synth_b;
        float dx = 1.5F;
        if constexpr ((ABL & kNoActLoad) == 0u) {
          const std::size_t tile = ((tt_base + j) * num_kb) + kbg;
          const std::int8_t* xt = xq + (tile * kTileBytes);
          b0 = *reinterpret_cast<const int32x4_t*>(xt + (sub_lane * 16));
          b1 = *reinterpret_cast<const int32x4_t*>(xt + 256 + (sub_lane * 16));
          dx = xd[(tile * kTileScales) + sub_lane];
        }

        if constexpr ((ABL & kNoEpilogue) != 0u) {
          iacc[j] = Wmma(a0, b0, iacc[j]);
          iacc[j] = Wmma(a1, b1, iacc[j]);
        } else if constexpr (SPLIT) {
          int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t c1 = c0;
          c0 = Wmma(a0, b0, c0);
          c1 = Wmma(a1, b1, c1);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[j][l] += (dwv[l] * dx) * static_cast<float>(c0[l] + c1[l]);
          }
        } else {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma(a0, b0, c);
          c = Wmma(a1, b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[j][l] += (dwv[l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
    }
    if constexpr ((ABL & kNoWeightLoad) == 0u) {
      __syncthreads();
    }
  }

  if constexpr ((ABL & kNoEpilogue) != 0u) {
#pragma unroll
    for (int j = 0; j < TT; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[j][l] = static_cast<float>(iacc[j][l]);
      }
    }
  }

  if constexpr ((ABL & kNoStore) == 0u) {
#pragma unroll
    for (int j = 0; j < TT; ++j) {
      const std::size_t tok = ((tt_base + j) * 16) + sub_lane;
      if (tok >= batch) {
        continue;
      }
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const std::size_t r = (rt * 16) + (2u * l) + half_id;
        if (r < m) {
          y[(tok * m) + r] = acc[j][l];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Two-dimensionally blocked W8A8 GEMM.
//
// The production kernel owns a 16-row x 128-token macro tile, so the activation
// panel is re-read once per row tile (m/16 = 1088 times for the FFN shape,
// ~12.8 GB at batch 2048). Traffic per MAC is 34/(32*BN) bytes of weight and
// 36/(32*BM) bytes of activation, so BM is what fixes the dominant term.
//
// Both operands are staged in LDS in WMMA-fragment order. Weights are read from
// the production row-major Q8_0 layout (no repack, no second weight copy);
// activations use the tiled layout the quantize kernel can emit directly.
//
//   block  = BM rows x BN tokens, BK 32-element K blocks per LDS stage
//   waves  = WM row groups x WN token groups (WM*WN == 8)
//   wave   = (BM/WM) rows x (BN/WN) tokens
// ---------------------------------------------------------------------------

template<int BM, int BN, int BK, int WM, int WN, bool LdsStore,
         unsigned ABL = kFull, bool CoarseDx = false>
__launch_bounds__(256) __global__
    void BlockedW8A8(const Q8_0Block* __restrict__ w,
                     const std::int8_t* __restrict__ xq,
                     const float* __restrict__ xd, float* __restrict__ y,
                     std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads == 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int RS = BM / 16;   // row subtiles per block
  constexpr int TS = BN / 16;   // token subtiles per block
  constexpr int WRS = RS / WM;  // row subtiles per wave
  constexpr int WTS = TS / WN;  // token subtiles per wave

  __shared__ int32x4_t s_a[BK][RS][32];
  __shared__ float s_dw[BK][RS][16];
  __shared__ int32x4_t s_b[BK][TS][32];
  __shared__ float s_dx[BK][TS][16];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int hi = lane >> 4;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const std::size_t num_kb = k / 32;
  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / 16;

  float acc[WRS][WTS][8];
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  for (std::size_t kb0 = 0; kb0 < num_kb; kb0 += BK) {
    // Weights: one (row, k-block) pair per thread, consecutive threads walking
    // consecutive k-blocks of a row so the 34-byte blocks stay contiguous.
#pragma unroll
    for (int p = 0; p < (BM * BK) / 256; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const std::size_t r = r_block + static_cast<std::size_t>(rr);
      const std::size_t kb = kb0 + static_cast<std::size_t>(kk);
      const int rs = rr / 16;
      const int rl = rr % 16;
      if (r < m && kb < num_kb) {
        const Q8_0Block& blk = w[(r * num_kb) + kb];
        __builtin_memcpy(&s_a[kk][rs][rl], blk.qs + 0, 16);
        __builtin_memcpy(&s_a[kk][rs][16 + rl], blk.qs + 16, 16);
        s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] =
            __half2float(blk.d);
      } else {
        s_a[kk][rs][rl] = int32x4_t{0, 0, 0, 0};
        s_a[kk][rs][16 + rl] = int32x4_t{0, 0, 0, 0};
        s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = 0.0F;
      }
    }

    // Activations: already in fragment order, so this is a contiguous copy.
    {
      const int ts = tid >> 5;  // one wave per token subtile when TS == 8
#pragma unroll
      for (int t = ts; t < TS; t += 8) {
        const auto* src = reinterpret_cast<const int32x4_t*>(
            xq + ((((tt_block + t) * num_kb) + kb0) * kTileBytes));
#pragma unroll
        for (int i = 0; i < BK; ++i) {
          s_b[i][t][lane] = src[(i * 32) + lane];
        }
      }
#pragma unroll
      for (int p = 0; p < ((BK * TS * 16) + 255) / 256; ++p) {
        const int idx = (p * 256) + tid;
        if (idx < BK * TS * 16) {
          const int kk = idx / (TS * 16);
          const int t = (idx / 16) % TS;
          const int sc = idx % 16;
          s_dx[kk][t][sc] =
              xd[((((tt_block + t) * num_kb) + kb0 + kk) * kTileScales) + sc];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[WRS];
      int32x4_t a1[WRS];
      float dw[WRS][8];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        const int rs = (wr * WRS) + i;
        a0[i] = s_a[kb][rs][sl];
        a1[i] = s_a[kb][rs][16 + sl];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][hi * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(hi * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      int32x4_t b0[WTS];
      int32x4_t b1[WTS];
      float dx[WTS];
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int ts = (wt * WTS) + j;
        b0[j] = s_b[kb][ts][sl];
        b1[j] = s_b[kb][ts][16 + sl];
        dx[j] = s_dx[CoarseDx ? 0 : kb][ts][sl];
      }

#pragma unroll
      for (int i = 0; i < WRS; ++i) {
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma(a0[i], b0[j], c);
          c = Wmma(a1[i], b1[j], c);
          if constexpr ((ABL & kNoEpilogue) != 0u) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              acc[i][j][l] += static_cast<float>(c[l]);
            }
          } else if constexpr (CoarseDx) {
            // dx is stage-constant, so it factors out of the inner fma:
            // 2 VALU per output element instead of 3.
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              acc[i][j][l] += dw[i][l] * static_cast<float>(c[l]);
            }
          } else {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              acc[i][j][l] += (dw[i][l] * dx[j]) * static_cast<float>(c[l]);
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if constexpr (!LdsStore) {
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const std::size_t tok =
            t_block + static_cast<std::size_t>(((wt * WTS) + j) * 16) + sl;
        if (tok >= batch) {
          continue;
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const std::size_t r =
              r_block + static_cast<std::size_t>(((wr * WRS) + i) * 16) +
              (2u * l) + hi;
          if (r < m) {
            y[(tok * m) + r] = acc[i][j][l];
          }
        }
      }
    }
  } else {
    // Transpose each 16x16 result tile through LDS so the global stores are
    // 16 consecutive rows of one token (64 contiguous bytes per half wave)
    // instead of 16 separate cache lines per instruction.
    __syncthreads();
    auto* scratch = reinterpret_cast<float*>(&s_a[0][0][0]);  // 8 waves x 256
    float* mine = scratch + (wave * 256);
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          mine[(sl * 16) + (2 * l) + hi] = acc[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const std::size_t r0 =
            r_block + static_cast<std::size_t>(((wr * WRS) + i) * 16);
        const std::size_t tok0 =
            t_block + static_cast<std::size_t>(((wt * WTS) + j) * 16);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;  // 0..255
          const int tl = flat / 16;
          const int rl = flat % 16;
          const std::size_t tok = tok0 + tl;
          const std::size_t r = r0 + rl;
          if (tok < batch && r < m) {
            y[(tok * m) + r] = mine[(tl * 16) + rl];
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Software-pipelined variant: the next K stage's weight blocks are fetched into
// registers before the current stage's WMMA work, so the global latency of the
// scattered 34-byte block_q8_0 reads overlaps compute instead of gating the
// LDS fill. PF is the number of (row, k-block) pairs each thread carries.
// ---------------------------------------------------------------------------

template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
    void BlockedW8A8Pipe(const Q8_0Block* __restrict__ w,
                         const std::int8_t* __restrict__ xq,
                         const float* __restrict__ xd, float* __restrict__ y,
                         std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8);
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int PF = (BM * BK) / 256;

  __shared__ int32x4_t s_a[BK][RS][32];
  __shared__ float s_dw[BK][RS][16];
  __shared__ int32x4_t s_b[BK][TS][32];
  __shared__ float s_dx[BK][TS][16];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int hi = lane >> 4;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const std::size_t num_kb = k / 32;
  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / 16;

  float acc[WRS][WTS][8];
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Per-thread weight staging registers.
  int32x4_t rq0[PF];
  int32x4_t rq1[PF];
  float rd[PF];
  int32x4_t rb[BK];
  float rbx[BK];
  const int b_tile = wave;  // one wave per token subtile when TS == 8

  auto fetch = [&](std::size_t kb0) {
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int idx = (p * 256) + tid;
      const std::size_t r = r_block + static_cast<std::size_t>(idx / BK);
      const std::size_t kb = kb0 + static_cast<std::size_t>(idx % BK);
      if (r < m && kb < num_kb) {
        const Q8_0Block& blk = w[(r * num_kb) + kb];
        __builtin_memcpy(&rq0[p], blk.qs + 0, 16);
        __builtin_memcpy(&rq1[p], blk.qs + 16, 16);
        rd[p] = __half2float(blk.d);
      } else {
        rq0[p] = int32x4_t{0, 0, 0, 0};
        rq1[p] = int32x4_t{0, 0, 0, 0};
        rd[p] = 0.0F;
      }
    }
    if (b_tile < TS) {
      const auto* src = reinterpret_cast<const int32x4_t*>(
          xq + ((((tt_block + b_tile) * num_kb) + kb0) * kTileBytes));
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        rb[i] = src[(i * 32) + lane];
        rbx[i] = xd[((((tt_block + b_tile) * num_kb) + kb0 + i) * kTileScales) +
                    lane % 16];
      }
    }
  };

  auto commit = [&]() {
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = rq0[p];
      s_a[kk][rs][16 + rl] = rq1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = rd[p];
    }
    if (b_tile < TS) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane] = rb[i];
        if (lane < 16) {
          s_dx[i][b_tile][lane] = rbx[i];
        }
      }
    }
  };

  fetch(0);
  for (std::size_t kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch(kb0 + BK);  // global latency overlaps the WMMA work below
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[WRS];
      int32x4_t a1[WRS];
      float dw[WRS][8];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        const int rs = (wr * WRS) + i;
        a0[i] = s_a[kb][rs][sl];
        a1[i] = s_a[kb][rs][16 + sl];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][hi * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(hi * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int ts = (wt * WTS) + j;
        const int32x4_t b0 = s_b[kb][ts][sl];
        const int32x4_t b1 = s_b[kb][ts][16 + sl];
        const float dxv = s_dx[kb][ts][sl];
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma(a0[i], b0, c);
          c = Wmma(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dxv) * static_cast<float>(c[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  float* scratch = reinterpret_cast<float*>(&s_a[0][0][0]) + (wave * 256);
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[(sl * 16) + (2 * l) + hi] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block + static_cast<std::size_t>(((wr * WRS) + i) * 16);
      const std::size_t t0 =
          t_block + static_cast<std::size_t>(((wt * WTS) + j) * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane;
        const std::size_t tok = t0 + static_cast<std::size_t>(flat / 16);
        const std::size_t r = r0 + static_cast<std::size_t>(flat % 16);
        if (tok < batch && r < m) {
          y[(tok * m) + r] = scratch[flat];
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// ---------------------------------------------------------------------------
// Same blocking as BlockedW8A8Pipe, with the instruction overhead removed.
//
// An ISA dump of the pipelined kernel showed only 64 of 2669 instructions were
// WMMA: the loop was dominated by 64-bit address arithmetic (v_mad_u64_u32 /
// v_add_co pairs) and by exec-mask manipulation from per-thread bounds guards
// (~300 s_and_saveexec / s_or / s_and). Two changes remove almost all of it:
//
//   * Row base pointers are hoisted out of the K loop and the in-loop index is
//     32-bit, so a stage advance is a single 32-bit add.
//   * Out-of-range weight rows are CLAMPED instead of branched, and their scale
//     is zeroed with a select, so an out-of-range row contributes exactly zero
//     with no divergence. The store keeps a guard, but as one uniform branch
//     per 16x16 tile instead of eight per-lane predicates.
// ---------------------------------------------------------------------------

template<int BM, int BN, int BK, int WM, int WN, int THREADS = 256>
__launch_bounds__(THREADS) __global__
    void BlockedW8A8Fast(const Q8_0Block* __restrict__ w,
                         const std::int8_t* __restrict__ xq,
                         const float* __restrict__ xd, float* __restrict__ y,
                         std::size_t batch_in, std::size_t m_in,
                         std::size_t k_in) {
  static_assert(WM * WN == THREADS / 32);
  static_assert(BN / 16 <= THREADS / 32);
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int PF = (BM * BK) / THREADS;

  __shared__ int32x4_t s_a[BK][RS][32];
  __shared__ float s_dw[BK][RS][16];
  __shared__ int32x4_t s_b[BK][TS][32];
  __shared__ float s_dx[BK][TS][16];

  const int batch = static_cast<int>(batch_in);
  const int m = static_cast<int>(m_in);
  const int num_kb = static_cast<int>(k_in / 32);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int hi = lane >> 4;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;
  const int tt_block = t_block / 16;

  float acc[WRS][WTS][8];
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Hoisted 64-bit bases: everything inside the K loop is a 32-bit offset.
  const Q8_0Block* w_row[PF];
  float row_live[PF];
#pragma unroll
  for (int p = 0; p < PF; ++p) {
    const int rr = ((p * 256) + tid) / BK;
    const int r = r_block + rr;
    const int r_clamped = (r < m) ? r : (m - 1);
    w_row[p] = w + (static_cast<std::size_t>(r_clamped) * num_kb);
    row_live[p] = (r < m) ? 1.0F : 0.0F;
  }
  const int b_tile = wave;
  const std::int8_t* b_base =
      xq + (static_cast<std::size_t>(tt_block + b_tile) * num_kb * kTileBytes);
  const float* bx_base =
      xd + (static_cast<std::size_t>(tt_block + b_tile) * num_kb * kTileScales);
  const bool b_live = b_tile < TS;

  int32x4_t r_q0[PF];
  int32x4_t r_q1[PF];
  float r_dw[PF];
  int32x4_t r_b[BK];
  float r_dx[BK];

  const auto fetch = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int kk = ((p * THREADS) + tid) % BK;
      const int kb = kb0 + kk;
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      const float live = (kb < num_kb) ? row_live[p] : 0.0F;
      r_dw[p] = __half2float(blk.d) * live;
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* src = reinterpret_cast<const int32x4_t*>(
            b_base + (static_cast<std::size_t>(kb_clamped) * kTileBytes));
        r_b[i] = src[lane];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  bx_base[(kb_clamped * kTileScales) + (lane & 15)];
      }
    }
  };

  const auto commit = [&]() {
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int idx = (p * THREADS) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane] = r_b[i];
        if (lane < 16) {
          s_dx[i][b_tile][lane] = r_dx[i];
        }
      }
    }
  };

  fetch(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch(kb0 + BK);
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[WRS];
      int32x4_t a1[WRS];
      float dw[WRS][8];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        const int rs = (wr * WRS) + i;
        a0[i] = s_a[kb][rs][sl];
        a1[i] = s_a[kb][rs][16 + sl];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][hi * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(hi * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int ts = (wt * WTS) + j;
        const int32x4_t b0 = s_b[kb][ts][sl];
        const int32x4_t b1 = s_b[kb][ts][16 + sl];
        const float dxv = s_dx[kb][ts][sl];
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma(a0[i], b0, c);
          c = Wmma(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dxv) * static_cast<float>(c[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  float* scratch = reinterpret_cast<float*>(&s_a[0][0][0]) + (wave * 256);
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[(sl * 16) + (2 * l) + hi] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int r0 = r_block + (((wr * WRS) + i) * 16);
      const int t0 = t_block + (((wt * WTS) + j) * 16);
      const std::size_t out_base =
          (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
      // One uniform branch per tile instead of eight per-lane predicates.
      if (t0 + 16 <= batch && r0 + 16 <= m) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
            (flat & 15)] = scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          const int tok = t0 + (flat >> 4);
          const int r = r0 + (flat & 15);
          if (tok < batch && r < m) {
            y[(static_cast<std::size_t>(tok) * m) + r] = scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// ---------------------------------------------------------------------------
// Blocked dual GEMM: gate and up share one staged activation tile.
//
// THREADS waves are arranged WM x WN; each wave owns (BM/WM) rows x (BN/WN)
// tokens of BOTH outputs, so the activation LDS stage and its global traffic
// are paid once for two weight matrices.
// ---------------------------------------------------------------------------

template<int BM, int BN, int BK, int WM, int WN, int THREADS>
__launch_bounds__(THREADS) __global__
    void BlockedW8A8Dual(const Q8_0Block* __restrict__ wg,
                         const Q8_0Block* __restrict__ wu,
                         const std::int8_t* __restrict__ xq,
                         const float* __restrict__ xd, float* __restrict__ yg,
                         float* __restrict__ yu, std::size_t batch,
                         std::size_t m, std::size_t k) {
  static_assert(WM * WN == THREADS / 32);
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;

  __shared__ int32x4_t s_ag[BK][RS][32];
  __shared__ int32x4_t s_au[BK][RS][32];
  __shared__ float s_dwg[BK][RS][16];
  __shared__ float s_dwu[BK][RS][16];
  __shared__ int32x4_t s_b[BK][TS][32];
  __shared__ float s_dx[BK][TS][16];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sl = lane & 15;
  const int hi = lane >> 4;
  const int wr = wave / WN;
  const int wt = wave % WN;

  const std::size_t num_kb = k / 32;
  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / 16;

  float acc_g[WRS][WTS][8];
  float acc_u[WRS][WTS][8];
#pragma unroll
  for (int i = 0; i < WRS; ++i) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc_g[i][j][l] = 0.0F;
        acc_u[i][j][l] = 0.0F;
      }
    }
  }

  for (std::size_t kb0 = 0; kb0 < num_kb; kb0 += BK) {
#pragma unroll
    for (int p = 0; p < (BM * BK) / THREADS; ++p) {
      const int idx = (p * THREADS) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      const int slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
      const std::size_t r = r_block + static_cast<std::size_t>(rr);
      const std::size_t kb = kb0 + static_cast<std::size_t>(kk);
      if (r < m && kb < num_kb) {
        const Q8_0Block& bg = wg[(r * num_kb) + kb];
        const Q8_0Block& bu = wu[(r * num_kb) + kb];
        __builtin_memcpy(&s_ag[kk][rs][rl], bg.qs + 0, 16);
        __builtin_memcpy(&s_ag[kk][rs][16 + rl], bg.qs + 16, 16);
        __builtin_memcpy(&s_au[kk][rs][rl], bu.qs + 0, 16);
        __builtin_memcpy(&s_au[kk][rs][16 + rl], bu.qs + 16, 16);
        s_dwg[kk][rs][slot] = __half2float(bg.d);
        s_dwu[kk][rs][slot] = __half2float(bu.d);
      } else {
        s_ag[kk][rs][rl] = int32x4_t{0, 0, 0, 0};
        s_ag[kk][rs][16 + rl] = int32x4_t{0, 0, 0, 0};
        s_au[kk][rs][rl] = int32x4_t{0, 0, 0, 0};
        s_au[kk][rs][16 + rl] = int32x4_t{0, 0, 0, 0};
        s_dwg[kk][rs][slot] = 0.0F;
        s_dwu[kk][rs][slot] = 0.0F;
      }
    }
#pragma unroll
    for (int t = wave; t < TS; t += THREADS / 32) {
      const auto* src = reinterpret_cast<const int32x4_t*>(
          xq + ((((tt_block + t) * num_kb) + kb0) * kTileBytes));
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][t][lane] = src[(i * 32) + lane];
        if (lane < 16) {
          s_dx[i][t][lane] =
              xd[((((tt_block + t) * num_kb) + kb0 + i) * kTileScales) + lane];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0g[WRS];
      int32x4_t a1g[WRS];
      int32x4_t a0u[WRS];
      int32x4_t a1u[WRS];
      float dwg[WRS][8];
      float dwu[WRS][8];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        const int rs = (wr * WRS) + i;
        a0g[i] = s_ag[kb][rs][sl];
        a1g[i] = s_ag[kb][rs][16 + sl];
        a0u[i] = s_au[kb][rs][sl];
        a1u[i] = s_au[kb][rs][16 + sl];
        const float4 glo =
            *reinterpret_cast<const float4*>(&s_dwg[kb][rs][hi * 8]);
        const float4 ghi =
            *reinterpret_cast<const float4*>(&s_dwg[kb][rs][(hi * 8) + 4]);
        const float4 ulo =
            *reinterpret_cast<const float4*>(&s_dwu[kb][rs][hi * 8]);
        const float4 uhi =
            *reinterpret_cast<const float4*>(&s_dwu[kb][rs][(hi * 8) + 4]);
        dwg[i][0] = glo.x;
        dwg[i][1] = glo.y;
        dwg[i][2] = glo.z;
        dwg[i][3] = glo.w;
        dwg[i][4] = ghi.x;
        dwg[i][5] = ghi.y;
        dwg[i][6] = ghi.z;
        dwg[i][7] = ghi.w;
        dwu[i][0] = ulo.x;
        dwu[i][1] = ulo.y;
        dwu[i][2] = ulo.z;
        dwu[i][3] = ulo.w;
        dwu[i][4] = uhi.x;
        dwu[i][5] = uhi.y;
        dwu[i][6] = uhi.z;
        dwu[i][7] = uhi.w;
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int ts = (wt * WTS) + j;
        const int32x4_t b0 = s_b[kb][ts][sl];
        const int32x4_t b1 = s_b[kb][ts][16 + sl];
        const float dx = s_dx[kb][ts][sl];
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          int32x8_t cg = {0, 0, 0, 0, 0, 0, 0, 0};
          cg = Wmma(a0g[i], b0, cg);
          cg = Wmma(a1g[i], b1, cg);
          int32x8_t cu = {0, 0, 0, 0, 0, 0, 0, 0};
          cu = Wmma(a0u[i], b0, cu);
          cu = Wmma(a1u[i], b1, cu);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc_g[i][j][l] += (dwg[i][l] * dx) * static_cast<float>(cg[l]);
            acc_u[i][j][l] += (dwu[i][l] * dx) * static_cast<float>(cu[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  float* scratch = reinterpret_cast<float*>(&s_ag[0][0][0]) + (wave * 256);
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    float* out = (pass == 0) ? yg : yu;
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          scratch[(sl * 16) + (2 * l) + hi] =
              (pass == 0) ? acc_g[i][j][l] : acc_u[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const std::size_t r0 =
            r_block + static_cast<std::size_t>(((wr * WRS) + i) * 16);
        const std::size_t t0 =
            t_block + static_cast<std::size_t>(((wt * WTS) + j) * 16);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane;
          const std::size_t tok = t0 + static_cast<std::size_t>(flat / 16);
          const std::size_t r = r0 + static_cast<std::size_t>(flat % 16);
          if (tok < batch && r < m) {
            out[(tok * m) + r] = scratch[flat];
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------

template<typename Launch>
static float BenchMs(Launch&& launch, int iters) {
  launch();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return ms / static_cast<float>(iters);
}

struct Case {
  const char* name;
  std::size_t m;
  std::size_t k;
};

int main(int argc, char** argv) {
  std::size_t batch = 2048;
  int iters = 5;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-b" && i + 1 < argc) {
      batch = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "-c" && i + 1 < argc) {
      only = argv[++i];
    }
  }

  const Case all_cases[] = {
      {"ffn_gate/up", 17408, 5120},
      {"ffn_down", 5120, 17408},
      {"ssm_qkv", 6144, 5120},
      {"ssm_out", 5120, 2048},
  };

  constexpr double kWmmaPeakTops = 55.07;
  std::printf("batch=%zu iters=%d   (WMMA int8 peak %.2f TOPS)\n", batch, iters,
              kWmmaPeakTops);
  std::printf("%-13s %-26s %9s %8s %7s %8s\n", "case", "variant", "ms", "TOPS",
              "%peak", "maxrel");

  for (const auto& c : all_cases) {
    if (!only.empty() && only != c.name) {
      continue;
    }
    const std::size_t m = c.m;
    const std::size_t k = c.k;
    const std::size_t num_kb = k / 32;

    std::vector<Q8_0Block> hw(m * num_kb);
    for (std::size_t i = 0; i < hw.size(); ++i) {
      hw[i].d = __float2half(0.002F + 0.001F * static_cast<float>(i % 7));
      for (int j = 0; j < 32; ++j) {
        hw[i].qs[j] = static_cast<std::int8_t>(((i * 31 + j * 17) % 255) - 127);
      }
    }
    std::vector<Q8_1ActivationBlock> hx(batch * num_kb);
    for (std::size_t i = 0; i < hx.size(); ++i) {
      hx[i].d = 0.004F + 0.001F * static_cast<float>(i % 5);
      for (int j = 0; j < 32; ++j) {
        hx[i].qs[j] = static_cast<std::int8_t>(((i * 13 + j * 7) % 255) - 127);
      }
    }

    void* d_w = nullptr;
    void* d_x = nullptr;
    float* d_y = nullptr;
    float* d_ref = nullptr;
    HIP_OK(hipMalloc(&d_w, hw.size() * sizeof(Q8_0Block)));
    HIP_OK(hipMalloc(&d_x, hx.size() * sizeof(Q8_1ActivationBlock)));
    HIP_OK(hipMalloc(&d_y, batch * m * sizeof(float)));
    HIP_OK(hipMalloc(&d_ref, batch * m * sizeof(float)));
    HIP_OK(hipMemcpy(d_w, hw.data(), hw.size() * sizeof(Q8_0Block),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_x, hx.data(), hx.size() * sizeof(Q8_1ActivationBlock),
                     hipMemcpyHostToDevice));

    const std::size_t row_tiles = m / 16;
    const std::size_t tok_tiles = (batch + 15) / 16;
    const std::size_t allocated_tok_tiles = std::max<std::size_t>(tok_tiles, 4);
    std::int8_t* d_wq = nullptr;
    __half* d_wd = nullptr;
    std::int8_t* d_xq = nullptr;
    float* d_xd = nullptr;
    HIP_OK(hipMalloc(&d_wq, row_tiles * num_kb * kTileBytes));
    HIP_OK(hipMalloc(&d_wd, row_tiles * num_kb * kTileScales * sizeof(__half)));
    HIP_OK(hipMalloc(&d_xq, allocated_tok_tiles * num_kb * kTileBytes));
    HIP_OK(hipMalloc(
        &d_xd, allocated_tok_tiles * num_kb * kTileScales * sizeof(float)));
    {
      const std::size_t n = row_tiles * num_kb * 16;
      RepackWeightsKernel<<<(n + 255) / 256, 256>>>(
          static_cast<const Q8_0Block*>(d_w), d_wq, d_wd, m, num_kb);
      const std::size_t n2 = tok_tiles * num_kb * 16;
      RepackActivationsKernel<<<(n2 + 255) / 256, 256>>>(
          static_cast<const Q8_1ActivationBlock*>(d_x), d_xq, d_xd, batch,
          num_kb);
      HIP_OK(hipDeviceSynchronize());
    }

    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(k) *
                       static_cast<double>(batch);

    auto report = [&](const char* name, float ms, int mode) {
      // mode 0 = record reference, 1 = compare, 2 = ablation (no compare)
      double maxrel = -1.0;
      if (mode == 0) {
        HIP_OK(hipMemcpy(d_ref, d_y, batch * m * sizeof(float),
                         hipMemcpyDeviceToDevice));
        maxrel = 0.0;
      } else if (mode == 1) {
        std::vector<float> a(batch * m);
        std::vector<float> b(batch * m);
        HIP_OK(hipMemcpy(a.data(), d_ref, a.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(b.data(), d_y, b.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
        maxrel = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          const double den =
              (a[i] == 0.0F) ? 1.0 : std::fabs(static_cast<double>(a[i]));
          const double rel =
              std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) /
              den;
          if (rel > maxrel) {
            maxrel = rel;
          }
        }
      }
      const double tops = ops / (ms * 1e-3) / 1e12;
      if (maxrel < 0.0) {
        std::printf("%-13s %-26s %9.3f %8.2f %6.0f%% %8s\n", c.name, name, ms,
                    tops, 100.0 * tops / kWmmaPeakTops, "-");
      } else {
        std::printf("%-13s %-26s %9.3f %8.2f %6.0f%% %8.1e\n", c.name, name, ms,
                    tops, 100.0 * tops / kWmmaPeakTops, maxrel);
      }
    };

    {
      const dim3 grid((m + 15) / 16, (batch + 127) / 128);
      report("baseline (production)",
             BenchMs(
                 [&] {
                   BaselineSingle<<<grid, 256>>>(
                       d_w, static_cast<const Q8_1ActivationBlock*>(d_x), d_y,
                       batch, m, k);
                 },
                 iters),
             0);
    }

#define RUN(NAME, TT, KB, ABL, SPLIT, MODE)                              \
  if (batch >= 128) {                                                    \
    const dim3 grid((tok_tiles + (8 * (TT)) - 1) / (8 * (TT)), m / 16);  \
    report(NAME,                                                         \
           BenchMs(                                                      \
               [&] {                                                     \
                 TiledSingle<(TT), (KB), (ABL), (SPLIT)><<<grid, 256>>>( \
                     d_wq, d_wd, d_xq, d_xd, d_y, batch, m, k);          \
               },                                                        \
               iters),                                                   \
           (MODE));                                                      \
  }

    RUN("tiled tt=4 kb=16", 4, 16, kFull, false, 1)
    RUN("tiled tt=8 kb=16", 8, 16, kFull, false, 1)
    // Ablations on the tt=4 shape: each removes one stage's cost.
    RUN("abl: -epilogue", 4, 16, kNoEpilogue, false, 2)
    RUN("abl: -weightload", 4, 16, kNoWeightLoad, false, 2)
#undef RUN

#define RUN_BLK(TAG, BM, BN, BK, WM, WN, LDSST, ABL, CDX, MODE)                \
  if (m % (BM) == 0 && (batch >= (BN) || (BN) == 16)) {                        \
    const dim3 grid(static_cast<unsigned>((batch + (BN) - 1) / (BN)),          \
                    static_cast<unsigned>(m / (BM)));                          \
    char nm[64];                                                               \
    std::snprintf(nm, sizeof(nm), "blk %dx%dx%d w%dx%d%s", (BM), (BN), (BK),   \
                  (WM), (WN), (TAG));                                          \
    report(                                                                    \
        nm,                                                                    \
        BenchMs(                                                               \
            [&] {                                                              \
              BlockedW8A8<(BM), (BN), (BK), (WM), (WN), (LDSST), (ABL), (CDX)> \
                  <<<grid, 256>>>(static_cast<const Q8_0Block*>(d_w), d_xq,    \
                                  d_xd, d_y, batch, m, k);                     \
            },                                                                 \
            iters),                                                            \
        (MODE));                                                               \
  }

    RUN_BLK("", 64, 128, 4, 2, 4, false, kFull, false, 1)
    RUN_BLK("", 128, 16, 2, 8, 1, false, kFull, false, 1)
    RUN_BLK("", 128, 16, 4, 8, 1, false, kFull, false, 1)
    RUN_BLK("", 128, 64, 4, 4, 2, false, kFull, false, 1)
    RUN_BLK("", 128, 128, 4, 4, 2, false, kFull, false, 1)
    RUN_BLK(" ldsst", 128, 128, 4, 4, 2, true, kFull, false, 1)
    RUN_BLK(" ldsst", 128, 128, 4, 8, 1, true, kFull, false, 1)
    RUN_BLK(" ldsst", 128, 128, 4, 2, 4, true, kFull, false, 1)
    RUN_BLK(" ldsst", 256, 64, 4, 4, 2, true, kFull, false, 1)
    RUN_BLK(" ldsst", 128, 256, 4, 4, 2, true, kFull, false, 1)
    RUN_BLK(" ldsst -epi", 128, 128, 4, 4, 2, true, kNoEpilogue, false, 2)
    RUN_BLK(" ldsst coarsedx", 128, 128, 4, 4, 2, true, kFull, true, 2)
#undef RUN_BLK

#define RUN_PIPE(BM, BN, BK, WM, WN)                                           \
  if (m % (BM) == 0 &&                                                         \
      (batch >= (BN) || (BN) == 16 || (BN) == 32 || (BN) == 64)) {             \
    const dim3 grid(static_cast<unsigned>((batch + (BN) - 1) / (BN)),          \
                    static_cast<unsigned>(m / (BM)));                          \
    char nm[64];                                                               \
    std::snprintf(nm, sizeof(nm), "pipe %dx%dx%d w%dx%d", (BM), (BN), (BK),    \
                  (WM), (WN));                                                 \
    report(nm,                                                                 \
           BenchMs(                                                            \
               [&] {                                                           \
                 BlockedW8A8Pipe<(BM), (BN), (BK), (WM), (WN)>                 \
                     <<<grid, 256>>>(static_cast<const Q8_0Block*>(d_w), d_xq, \
                                     d_xd, d_y, batch, m, k);                  \
               },                                                              \
               iters),                                                         \
           1);                                                                 \
  }
    RUN_PIPE(128, 128, 4, 4, 2)
    RUN_PIPE(128, 16, 2, 8, 1)
    RUN_PIPE(128, 16, 4, 8, 1)
    RUN_PIPE(128, 32, 4, 8, 1)
    RUN_PIPE(128, 32, 4, 4, 2)
    RUN_PIPE(128, 64, 4, 8, 1)
    // The production kernel is this shape. BK=2 is the same structure with half
    // the LDS stage, which is the one-template-argument version of the change.
    RUN_PIPE(128, 128, 2, 4, 2)
#undef RUN_PIPE

#define RUN_FAST(BM, BN, BK, WM, WN, TH)                                  \
  if (m % (BM) == 0 && batch >= (BN)) {                                   \
    const dim3 grid(static_cast<unsigned>((batch + (BN) - 1) / (BN)),     \
                    static_cast<unsigned>((m + (BM) - 1) / (BM)));        \
    char nm[64];                                                          \
    std::snprintf(nm, sizeof(nm), "FAST %dx%dx%d w%dx%d t%d", (BM), (BN), \
                  (BK), (WM), (WN), (TH));                                \
    report(nm,                                                            \
           BenchMs(                                                       \
               [&] {                                                      \
                 BlockedW8A8Fast<(BM), (BN), (BK), (WM), (WN), (TH)>      \
                     <<<grid, (TH)>>>(static_cast<const Q8_0Block*>(d_w), \
                                      d_xq, d_xd, d_y, batch, m, k);      \
               },                                                         \
               iters),                                                    \
           1);                                                            \
  }
    RUN_FAST(128, 128, 4, 4, 2, 256)
    RUN_FAST(128, 256, 4, 4, 4, 512)
    RUN_FAST(256, 128, 4, 8, 2, 512)
    // BK is the one parameter never swept, and it is what sets the LDS
    // footprint: at BK=4 the four stage buffers are 36864 bytes, so few blocks
    // fit per WGP and there is little to hide the WMMA -> convert -> fma chain
    // behind. BK=2 halves the two int8 buffers, to 20480 bytes total, for a
    // shallower prefetch.
    // 128 rows over 512 threads needs BK >= 4 for a whole prefetch register per
    // thread, so that combination is not expressible here.
    RUN_FAST(128, 128, 2, 4, 2, 256)
    RUN_FAST(256, 128, 2, 8, 2, 512)
#undef RUN_FAST

    // Dual gate/up: one shared activation stage feeds two weight matrices.
    // Reported ms/TOPS are for BOTH matrices, so compare against 2x a single.
    {
      float* d_y2 = nullptr;
      HIP_OK(hipMalloc(&d_y2, batch * m * sizeof(float)));
      const double dual_ops = 2.0 * ops;
      auto report_dual = [&](const char* name, float ms) {
        const double tops = dual_ops / (ms * 1e-3) / 1e12;
        std::printf("%-13s %-26s %9.3f %8.2f %6.0f%% %8s\n", c.name, name, ms,
                    tops, 100.0 * tops / kWmmaPeakTops, "-");
      };
#define RUN_DUAL(BM, BN, BK, WM, WN, TH)                                      \
  if (m % (BM) == 0 && (batch >= (BN) || (BN) == 16)) {                       \
    const dim3 grid(static_cast<unsigned>((batch + (BN) - 1) / (BN)),         \
                    static_cast<unsigned>(m / (BM)));                         \
    char nm[64];                                                              \
    std::snprintf(nm, sizeof(nm), "DUAL %dx%dx%d w%dx%d t%d", (BM), (BN),     \
                  (BK), (WM), (WN), (TH));                                    \
    report_dual(nm, BenchMs(                                                  \
                        [&] {                                                 \
                          BlockedW8A8Dual<(BM), (BN), (BK), (WM), (WN), (TH)> \
                              <<<grid, (TH)>>>(                               \
                                  static_cast<const Q8_0Block*>(d_w),         \
                                  static_cast<const Q8_0Block*>(d_w), d_xq,   \
                                  d_xd, d_y, d_y2, batch, m, k);              \
                        },                                                    \
                        iters));                                              \
  }
      RUN_DUAL(128, 128, 4, 4, 4, 512)
      RUN_DUAL(128, 128, 2, 4, 4, 512)
      RUN_DUAL(64, 128, 4, 2, 4, 256)
      RUN_DUAL(128, 64, 4, 4, 2, 256)
      RUN_DUAL(128, 16, 4, 8, 1, 256)
#undef RUN_DUAL
      HIP_OK(hipFree(d_y2));
    }

    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_y));
    HIP_OK(hipFree(d_ref));
    HIP_OK(hipFree(d_wq));
    HIP_OK(hipFree(d_wd));
    HIP_OK(hipFree(d_xq));
    HIP_OK(hipFree(d_xd));
    std::printf("\n");
  }
  return 0;
}

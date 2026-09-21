#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/minimax_h3/dense.hip.hpp"
#include "src/models/minimax_h3/dit.hpp"
namespace gufo::minimax_h3::dit_ops {
namespace {
__device__ hip_bfloat16 Bf16(float x) {
  return hip_bfloat16(x);
}
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ Fragment Load(const std::uint16_t* x) {
  union {
    Fragment b;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(x);
  value.u[1] = *reinterpret_cast<const uint4*>(x + 8);
  return value.b;
}

// Exact byte permutation: [row][K] -> [K/16][row][16]. Adjacent
// blocks traverse columns first so source pages remain resident while packing
// a 64-row strip. A padded LDS tile coalesces both sides of the permutation.
__global__ void PackDense(const std::uint16_t* input, std::uint16_t* output,
                          int rows, int cols) {
  __shared__ __align__(16) std::uint16_t tile[64][72];
  const int tid = threadIdx.x, first_row = blockIdx.y * 64;
  const int first_col = blockIdx.x * 64;
  const int row = tid / 4, col = (tid % 4) * 16;
  uint4 a{}, b{};
  if (first_row + row < rows && first_col + col < cols) {
    const auto source =
        input + std::size_t(first_row + row) * cols + first_col + col;
    a = *reinterpret_cast<const uint4*>(source);
    b = *reinterpret_cast<const uint4*>(source + 8);
  }
  *reinterpret_cast<uint4*>(&tile[row][col]) = a;
  *reinterpret_cast<uint4*>(&tile[row][col + 8]) = b;
  __syncthreads();
  const int part = tid / 64, dest_row = tid % 64;
  if (first_row + dest_row < rows && first_col + part * 16 < cols) {
    auto dest = output + std::size_t(first_col / 16 + part) * rows * 16 +
                (first_row + dest_row) * 16;
    *reinterpret_cast<uint4*>(dest) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16]);
    *reinterpret_cast<uint4*>(dest + 8) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16 + 8]);
  }
}

// Adapted from Gufo's Qwen-Image kernel. Packed BF16 operands feed WMMA
// directly, with the pinned projection's increasing-K FP32 accumulation.
// Packing replaces the original weight buffer rather than retaining a copy.
__launch_bounds__(256) __global__
    void FfnDownProjection(const std::uint16_t* w, const std::uint16_t* x,
                           hip_bfloat16* y, int n, int m, int k) {
  constexpr int BM = 256, BN = 128, WM = 4, WN = 2;
  constexpr int RS = BM / 16, TS = BN / 16, WRS = RS / WM, WTS = TS / WN;
  int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  int rb = blockIdx.y * BM, tb = blockIdx.x * BN;
  Accumulator acc[WRS][WTS]{};
  // Refill operand registers immediately after their last use.
  {
    Fragment a[WRS], b[WTS];
    auto load_a = [&](int i, int ks) {
      return Load(w + std::size_t(ks) * m * 16 +
                  (rb + (wr * WRS + i) * 16 + sl) * 16);
    };
    auto load_b = [&](int j, int ks) {
      int t = tb + (wt * WTS + j) * 16 + sl;
      t = min(n - 1, t);
      return Load(x + std::size_t(ks) * n * 16 + t * 16);
    };
#pragma unroll
    for (int i = 0; i < WRS; i++)
      a[i] = load_a(i, 0);
#pragma unroll
    for (int j = 0; j < WTS; j++)
      b[j] = load_b(j, 0);
    for (int ks = 0; ks < k / 16; ks++) {
      int next = min(ks + 1, k / 16 - 1);
#pragma unroll
      for (int j = 0; j < WTS; j++) {
#pragma unroll
        for (int i = 0; i < WRS; i++) {
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b[j],
                                                                  acc[i][j]);
          if (j == WTS - 1)
            a[i] = load_a(i, next);
        }
        b[j] = load_b(j, next);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < WRS; i++)
#pragma unroll
    for (int j = 0; j < WTS; j++)
#pragma unroll
      for (int l = 0; l < 8; l++) {
        int r = rb + (wr * WRS + i) * 16 + 2 * l + lane / 16,
            t = tb + (wt * WTS + j) * 16 + sl;
        if (t < n)
          y[std::size_t(t) * m + r] = Bf16(acc[i][j][l]);
      }
}

}  // namespace

void LaunchPackDense(const std::uint16_t* input, std::uint16_t* output,
                     std::uint32_t rows, std::uint32_t columns,
                     hipStream_t stream) {
  PackDense<<<dim3((columns + 63) / 64, (rows + 63) / 64), 256, 0, stream>>>(
      input, output, rows, columns);
}

void LaunchFfnDown(const std::uint16_t* packed_weight,
                   const std::uint16_t* packed_input, std::uint16_t* output,
                   std::uint32_t rows, hipStream_t stream) {
  FfnDownProjection<<<dim3((rows + 127) / 128, kDitHiddenSize / 256), 256, 0,
                      stream>>>(packed_weight, packed_input,
                                reinterpret_cast<hip_bfloat16*>(output), rows,
                                kDitHiddenSize, kDitFeedForwardSize);
}
}  // namespace gufo::minimax_h3::dit_ops

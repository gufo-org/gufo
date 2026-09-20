#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {
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
  if (first_row + row < rows) {
    const auto source =
        input + std::size_t(first_row + row) * cols + first_col + col;
    a = *reinterpret_cast<const uint4*>(source);
    b = *reinterpret_cast<const uint4*>(source + 8);
  }
  *reinterpret_cast<uint4*>(&tile[row][col]) = a;
  *reinterpret_cast<uint4*>(&tile[row][col + 8]) = b;
  __syncthreads();
  const int part = tid / 64, dest_row = tid % 64;
  if (first_row + dest_row < rows) {
    auto dest = output + std::size_t(first_col / 16 + part) * rows * 16 +
                (first_row + dest_row) * 16;
    *reinterpret_cast<uint4*>(dest) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16]);
    *reinterpret_cast<uint4*>(dest + 8) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16 + 8]);
  }
}

// Packed operands feed WMMA directly. FP32 accumulation follows increasing K;
// no LDS staging, extra weight copy, probability narrowing or tensor
// conversion.
template<int BM, int BN, int WM, int WN, bool Ragged>
__launch_bounds__(WM * WN * 32) __global__
    void DenseProjection(const std::uint16_t* w, const std::uint16_t* x,
                         hip_bfloat16* y, int n, int m, int k) {
  constexpr int RS = BM / 16, TS = BN / 16, WRS = RS / WM, WTS = TS / WN;
  int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  int rb = blockIdx.y * BM, tb = blockIdx.x * BN;
  Accumulator acc[WRS][WTS]{};
  for (int ks = 0; ks < k / 16; ks++) {
    Fragment a[WRS];
#pragma unroll
    for (int i = 0; i < WRS; i++) {
      int r = rb + (wr * WRS + i) * 16 + sl;
      a[i] = Load(w + std::size_t(ks) * m * 16 + r * 16);
    }
#pragma unroll
    for (int j = 0; j < WTS; j++) {
      int t = tb + (wt * WTS + j) * 16 + sl;
      if constexpr (Ragged)
        t = min(n - 1, t);
      Fragment b = Load(x + std::size_t(ks) * n * 16 + t * 16);
#pragma unroll
      for (int i = 0; i < WRS; i++)
        acc[i][j] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b, acc[i][j]);
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
        if (!Ragged || t < n)
          y[std::size_t(t) * m + r] = hip_bfloat16(acc[i][j][l]);
      }
}

}  // namespace

bool DenseShape(int channels, int inner) {
  return (channels == 4096 && (inner == 4096 || inner == 12288)) ||
         (channels == 12288 && inner == 4096);
}

void PackDense16(const std::uint16_t* input, std::uint16_t* output, int rows,
                 int cols, hipStream_t stream) {
  PackDense<<<dim3(cols / 64, (rows + 63) / 64), 256, 0, stream>>>(
      input, output, rows, cols);
}

bool DenseBf16(const std::uint16_t* weights, const std::uint16_t* input,
               std::uint16_t* output, int rows, int channels, int inner,
               hipStream_t stream) {
  if (rows < 1 || !DenseShape(channels, inner))
    return false;
  if (rows <= 64 || (rows <= 256 && channels == 4096)) {
    const dim3 grid((rows + 31) / 32, channels / 128);
    if (rows % 32)
      DenseProjection<128, 32, 2, 2, true><<<grid, 128, 0, stream>>>(
          weights, input, reinterpret_cast<hip_bfloat16*>(output), rows,
          channels, inner);
    else
      DenseProjection<128, 32, 2, 2, false><<<grid, 128, 0, stream>>>(
          weights, input, reinterpret_cast<hip_bfloat16*>(output), rows,
          channels, inner);
  } else {
    const dim3 grid((rows + 127) / 128, channels / 256);
    if (rows % 128)
      DenseProjection<256, 128, 4, 2, true><<<grid, 256, 0, stream>>>(
          weights, input, reinterpret_cast<hip_bfloat16*>(output), rows,
          channels, inner);
    else
      DenseProjection<256, 128, 4, 2, false><<<grid, 256, 0, stream>>>(
          weights, input, reinterpret_cast<hip_bfloat16*>(output), rows,
          channels, inner);
  }
  return true;
}
}  // namespace gufo::models::qwen_image_21::hip

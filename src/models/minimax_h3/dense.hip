#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/minimax_h3/dense.hip.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/dit_ops.hip.hpp"
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
template<bool Pack>
__global__ void PackDense(const std::uint16_t* input, std::uint16_t* output,
                          int rows, int cols) {
  __shared__ __align__(16) std::uint16_t tile[64][72];
  const int tid = threadIdx.x, first_row = blockIdx.y * 64;
  const int first_col = blockIdx.x * 64;
  const int row = tid / 4, col = (tid % 4) * 16;
  uint4 a{}, b{};
  if (first_row + row < rows && first_col + col < cols) {
    const auto source =
        input + (Pack ? std::size_t(first_row + row) * cols + first_col + col
                      : std::size_t((first_col + col) / 16) * rows * 16 +
                            (first_row + row) * 16);
    a = *reinterpret_cast<const uint4*>(source);
    b = *reinterpret_cast<const uint4*>(source + 8);
  }
  *reinterpret_cast<uint4*>(&tile[row][col]) = a;
  *reinterpret_cast<uint4*>(&tile[row][col + 8]) = b;
  __syncthreads();
  const int part = tid / 64, dest_row = tid % 64;
  if (first_row + dest_row < rows && first_col + part * 16 < cols) {
    auto dest =
        output + (Pack ? std::size_t(first_col / 16 + part) * rows * 16 +
                             (first_row + dest_row) * 16
                       : std::size_t(first_row + dest_row) * cols + first_col +
                             part * 16);
    *reinterpret_cast<uint4*>(dest) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16]);
    *reinterpret_cast<uint4*>(dest + 8) =
        *reinterpret_cast<const uint4*>(&tile[dest_row][part * 16 + 8]);
  }
}

// Preserve AdaLN arithmetic while writing its rounded result directly in
// the projection layout. LDS coalesces the permutation across 64 token rows.
template<int Shift, int Scale>
__global__ void PackedAdaLn(const std::uint16_t* input,
                            const std::uint16_t* weight,
                            const std::uint16_t* modulation,
                            const std::uint32_t* row_map, const float* inverse,
                            std::uint16_t* output, int rows) {
  constexpr int cols = kDitHiddenSize;
  __shared__ __align__(16) std::uint16_t tile[64][72];
  const int tid = threadIdx.x, first_row = blockIdx.y * 64;
  const int first_col = blockIdx.x * 64, row = tid / 4, col = (tid % 4) * 16;
  if (first_row + row < rows) {
    const float row_inverse = inverse[first_row + row];
    const std::size_t row_base = std::size_t(first_row + row) * cols;
    const std::size_t modulation_base =
        std::size_t(row_map[first_row + row]) * kDitModulationSlots * cols;
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int column = first_col + col + z;
      const float normalized = Bf16ToFloat(input[row_base + column]) *
                               row_inverse * Bf16ToFloat(weight[column]);
      const float scale =
          Bf16ToFloat(modulation[modulation_base + Scale * cols + column]);
      const float shift =
          Bf16ToFloat(modulation[modulation_base + Shift * cols + column]);
      tile[row][col + z] = FloatToBf16(normalized * (1.0F + scale) + shift);
    }
  }
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

// Adapted from Gufo's Qwen-Image kernel. Packed BF16 operands feed WMMA
// directly, with the pinned projection's increasing-K FP32 accumulation.
// Packing replaces the original weight buffer rather than retaining a copy.
template<bool Tail>
__launch_bounds__(256) __global__
    void HiddenProjection(const std::uint16_t* w, const std::uint16_t* x,
                          hip_bfloat16* y, int n, int m, int k,
                          int tile_offset) {
  constexpr int BM = 256, BN = 128, WM = 4, WN = 2;
  constexpr int RS = BM / 16, TS = BN / 16, WRS = RS / WM, WTS = TS / WN;
  int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  int rb = blockIdx.y * BM, tb = (blockIdx.x + tile_offset) * BN;
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
      if constexpr (Tail)
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
  for (int i = 0; i < WRS; i++) {
#pragma unroll
    for (int j = 0; j < WTS; j++) {
      const int r = rb + (wr * WRS + i) * 16;
      const int t = tb + (wt * WTS + j) * 16 + sl;
      union {
        unsigned words[8];
        uint4 vectors[2];
      } store;
#pragma unroll
      for (int z = 0; z < 8; z++) {
        const unsigned own = __float_as_uint(float(Bf16(acc[i][j][z]))) >> 16;
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        store.words[z] = own | (peer << 16);
      }
      // Pair even/odd channel lanes so each row uses contiguous vector stores.
      if (lane < 16 && (!Tail || t < n)) {
        auto dest = reinterpret_cast<uint4*>(y + std::size_t(t) * m + r);
        dest[0] = store.vectors[0];
        dest[1] = store.vectors[1];
      }
    }
  }
}

// Project the checkpoint's interleaved [head][Q,K,V][128] rows, then
// consume each BF16-rounded head in LDS. This retains the four-FMA norm
// sum and direct paired RoPE expressions, without a global QKV tensor.
__launch_bounds__(256) __global__
    void QkvProjection(const std::uint16_t* w, const std::uint16_t* x,
                       std::uint16_t* q, std::uint16_t* key,
                       std::uint16_t* value, const std::uint16_t* qw,
                       const std::uint16_t* kw, const std::uint16_t* cosine,
                       const std::uint16_t* sine, int n, int m, int k) {
  constexpr int BM = 256, BN = 128, WM = 4, WN = 2;
  constexpr int RS = BM / 16, TS = BN / 16, WRS = RS / WM, WTS = TS / WN;
  int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  constexpr int group_m = 8;
  int group = pid / (group_m * gridDim.x), first = group * group_m;
  int size = min(int(gridDim.y) - first, group_m),
      within = pid % (group_m * gridDim.x);
  int rb = (first + within % size) * BM, tb = (within / size) * BN;
  __shared__ std::uint16_t tile[BN][136];
  Accumulator acc[WRS][WTS]{};
  // Ragged strides benefit from refilling operand registers after their last
  // use. Aligned rows are faster with the simpler direct-load loop.
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

  for (int part = 0; part < BM / 128; ++part) {
    int kind = (rb / 128 + part) % 3, head = (rb / 128 + part) / 3;
    if (kind == 2) {
#pragma unroll
      for (int i = 0; i < WRS; ++i)
#pragma unroll
        for (int j = 0; j < WTS; ++j)
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            int d = (wr * WRS + i) * 16 + 2 * l + lane / 16 - part * 128;
            int t = tb + (wt * WTS + j) * 16 + sl;
            if (d >= 0 && d < 128 && t < ((n + 31) & ~31))
              value[(std::size_t(head) * 128 + d) * ((n + 31) & ~31) + t] =
                  t < n ? FloatToBf16(acc[i][j][l]) : 0;
          }
    } else {
#pragma unroll
      for (int i = 0; i < WRS; ++i)
#pragma unroll
        for (int j = 0; j < WTS; ++j)
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            int d = (wr * WRS + i) * 16 + 2 * l + lane / 16 - part * 128;
            if (d >= 0 && d < 128)
              tile[(wt * WTS + j) * 16 + sl][d] = FloatToBf16(acc[i][j][l]);
          }
      __syncthreads();
      for (int tr = wave; tr < BN; tr += WM * WN) {
        int row = tb + tr;
        if (row < n) {
          auto* nw = kind == 0 ? qw : kw;
          auto* out = kind == 0 ? q : key;
          float sum = 0;
#pragma unroll
          for (unsigned d = lane; d < 128; d += 32) {
            float v = Bf16ToFloat(tile[tr][d]);
            sum = fmaf(v, v, sum);
          }
#pragma unroll
          for (unsigned off = 16; off; off /= 2)
            sum += __shfl_down(sum, off);
          float inverse = rsqrtf(__shfl(sum, 0) / 128.f + 1e-5f);
#pragma unroll
          for (unsigned d = lane; d < 128; d += 32) {
            auto output = (std::size_t(head) * n + row) * 128 + d;
            if (d < 48) {
              float v = Bf16ToFloat(tile[tr][d]) * inverse * Bf16ToFloat(nw[d]);
              float pair = Bf16ToFloat(tile[tr][d + 48]) * inverse *
                           Bf16ToFloat(nw[d + 48]);
              float c = Bf16ToFloat(cosine[std::size_t(row) * 48 + d]),
                    s = Bf16ToFloat(sine[std::size_t(row) * 48 + d]);
              out[output] = FloatToBf16(v * c - pair * s);
              out[output + 48] = FloatToBf16(pair * c + v * s);
            } else if (d >= 96)
              out[output] = FloatToBf16(Bf16ToFloat(tile[tr][d]) * inverse *
                                        Bf16ToFloat(nw[d]));
          }
        }
      }
      __syncthreads();
    }
  }
}
// Fuse both BF16 projections and SwiGLU into the down projection layout.
template<bool Tail>
__launch_bounds__(256) __global__
    void FfnUpProjection(const std::uint16_t* g, const std::uint16_t* u,
                         const std::uint16_t* x, std::uint16_t* y, int n, int m,
                         int k, const float* lookup, int tile_offset) {
  constexpr int M = 128, N = 128, WM = 4, WN = 2;
  constexpr int WR = M / (WM * 16), WT = N / (WN * 16);
  int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane % 16,
      wr = wave / WN, wt = wave % WN;
  const int pid = blockIdx.y * gridDim.x + blockIdx.x, group_m = 8;
  const int group = pid / (group_m * gridDim.x), first = group * group_m;
  const int size = min(int(gridDim.y) - first, group_m),
            within = pid % (group_m * gridDim.x);
  const int rb = (first + within % size) * M,
            tb = (within / size + tile_offset) * N;
  Accumulator ag[WR][WT]{}, au[WR][WT]{};
  Fragment a[2][WR], b[WT];
  auto la = [&](const std::uint16_t* w, int i, int ks) {
    return Load(w + size_t(ks) * m * 16 + (rb + (wr * WR + i) * 16 + sl) * 16);
  };
  auto lb = [&](int j, int ks) {
    return Load(x + size_t(ks) * n * 16 +
                (Tail ? min(n - 1, tb + (wt * WT + j) * 16 + sl)
                      : tb + (wt * WT + j) * 16 + sl) *
                    16);
  };
#pragma unroll
  for (int i = 0; i < WR; i++) {
    a[0][i] = la(g, i, 0);
    a[1][i] = la(u, i, 0);
  }
#pragma unroll
  for (int j = 0; j < WT; j++)
    b[j] = lb(j, 0);
  for (int ks = 0; ks < k / 16; ks++) {
#pragma unroll
    for (int j = 0; j < WT; j++) {
#pragma unroll
      for (int i = 0; i < WR; i++) {
        ag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[0][i], b[j],
                                                               ag[i][j]);
        au[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[1][i], b[j],
                                                               au[i][j]);
        if (j == WT - 1) {
          a[0][i] = la(g, i, min(ks + 1, k / 16 - 1));
          a[1][i] = la(u, i, min(ks + 1, k / 16 - 1));
        }
      }
      b[j] = lb(j, min(ks + 1, k / 16 - 1));
    }
  }
#pragma unroll
  for (int i = 0; i < WR; i++) {
#pragma unroll
    for (int j = 0; j < WT; j++) {
      const int r = rb + (wr * WR + i) * 16;
      const int t = tb + (wt * WT + j) * 16 + sl;
      union {
        unsigned words[8];
        uint4 vectors[2];
      } store;
#pragma unroll
      for (int z = 0; z < 8; z++) {
        const auto gate = Bf16(ag[i][j][z]), up = Bf16(au[i][j][z]);
        const unsigned own =
            ApplySwiGluLookup(__float_as_uint(float(gate)) >> 16,
                              __float_as_uint(float(up)) >> 16, lookup);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        store.words[z] = own | (peer << 16);
      }
      // Pair the even/odd channel lanes, then issue two contiguous stores
      // per token instead of sixteen individual BF16 stores.
      if (lane < 16 && (!Tail || t < n)) {
        auto dest =
            reinterpret_cast<uint4*>(y + size_t(r / 16) * n * 16 + t * 16);
        dest[0] = store.vectors[0];
        dest[1] = store.vectors[1];
      }
    }
  }
}

void LaunchHiddenProjection(const std::uint16_t* packed_weight,
                            const std::uint16_t* packed_input,
                            std::uint16_t* output, std::uint32_t rows,
                            int width, hipStream_t stream) {
  if (rows >= 128)
    HiddenProjection<false>
        <<<dim3(rows / 128, kDitHiddenSize / 256), 256, 0, stream>>>(
            packed_weight, packed_input,
            reinterpret_cast<hip_bfloat16*>(output), rows, kDitHiddenSize,
            width, 0);
  if (rows % 128 != 0)
    HiddenProjection<true><<<dim3(1, kDitHiddenSize / 256), 256, 0, stream>>>(
        packed_weight, packed_input, reinterpret_cast<hip_bfloat16*>(output),
        rows, kDitHiddenSize, width, rows / 128);
}

}  // namespace

void LaunchPackDense(const std::uint16_t* input, std::uint16_t* output,
                     std::uint32_t rows, std::uint32_t columns,
                     hipStream_t stream) {
  PackDense<true>
      <<<dim3((columns + 63) / 64, (rows + 63) / 64), 256, 0, stream>>>(
          input, output, rows, columns);
}

void LaunchUnpackDense(const std::uint16_t* input, std::uint16_t* output,
                       std::uint32_t rows, std::uint32_t columns,
                       hipStream_t stream) {
  PackDense<false>
      <<<dim3((columns + 63) / 64, (rows + 63) / 64), 256, 0, stream>>>(
          input, output, rows, columns);
}

void LaunchPackFfnUpWeights(const std::uint16_t* input, std::uint16_t* output,
                            hipStream_t stream) {
  constexpr std::size_t half =
      std::size_t(kDitFeedForwardSize) * kDitHiddenSize;
  LaunchPackDense(input, output, kDitFeedForwardSize, kDitHiddenSize, stream);
  LaunchPackDense(input + half, output + half, kDitFeedForwardSize,
                  kDitHiddenSize, stream);
}

void LaunchPackedMlpAdaLn(const std::uint16_t* input,
                          const std::uint16_t* weight,
                          const std::uint16_t* modulation,
                          const std::uint32_t* row_map, const float* inverse,
                          std::uint16_t* output, std::uint32_t rows,
                          hipStream_t stream) {
  PackedAdaLn<3, 4>
      <<<dim3(kDitHiddenSize / 64, (rows + 63) / 64), 256, 0, stream>>>(
          input, weight, modulation, row_map, inverse, output, rows);
}

void LaunchPackedAttentionAdaLn(const std::uint16_t* input,
                                const std::uint16_t* weight,
                                const std::uint16_t* modulation,
                                const std::uint32_t* row_map,
                                const float* inverse, std::uint16_t* output,
                                std::uint32_t rows, hipStream_t stream) {
  PackedAdaLn<0, 1>
      <<<dim3(kDitHiddenSize / 64, (rows + 63) / 64), 256, 0, stream>>>(
          input, weight, modulation, row_map, inverse, output, rows);
}

void LaunchQkv(const std::uint16_t* packed_weight,
               const std::uint16_t* packed_input,
               const std::uint16_t* query_weight,
               const std::uint16_t* key_weight, const std::uint16_t* rope_cos,
               const std::uint16_t* rope_sin, std::uint16_t* query,
               std::uint16_t* key, std::uint16_t* packed_value,
               std::uint32_t rows, hipStream_t stream) {
  QkvProjection<<<dim3((rows + 127) / 128, kDitAttentionWidth * 3 / 256), 256,
                  0, stream>>>(packed_weight, packed_input, query, key,
                               packed_value, query_weight, key_weight, rope_cos,
                               rope_sin, rows, kDitAttentionWidth * 3,
                               kDitHiddenSize);
}

void LaunchFfnUp(const std::uint16_t* packed_weight,
                 const std::uint16_t* packed_input, std::uint16_t* output,
                 const float* silu_lookup, std::uint32_t rows,
                 hipStream_t stream) {
  constexpr std::size_t half =
      std::size_t(kDitFeedForwardSize) * kDitHiddenSize;
  // The interior has no bounds work inside its projection loop. A separate
  // final tile retains exact ragged loads/stores without padding the state.
  if (rows >= 128)
    FfnUpProjection<false>
        <<<dim3(rows / 128, kDitFeedForwardSize / 128), 256, 0, stream>>>(
            packed_weight, packed_weight + half, packed_input, output, rows,
            kDitFeedForwardSize, kDitHiddenSize, silu_lookup, 0);
  if (rows % 128 != 0)
    FfnUpProjection<true>
        <<<dim3(1, kDitFeedForwardSize / 128), 256, 0, stream>>>(
            packed_weight, packed_weight + half, packed_input, output, rows,
            kDitFeedForwardSize, kDitHiddenSize, silu_lookup, rows / 128);
}

void LaunchFfnDown(const std::uint16_t* packed_weight,
                   const std::uint16_t* packed_input, std::uint16_t* output,
                   std::uint32_t rows, hipStream_t stream) {
  LaunchHiddenProjection(packed_weight, packed_input, output, rows,
                         kDitFeedForwardSize, stream);
}

void LaunchAttentionOutput(const std::uint16_t* packed_weight,
                           const std::uint16_t* packed_input,
                           std::uint16_t* output, std::uint32_t rows,
                           hipStream_t stream) {
  LaunchHiddenProjection(packed_weight, packed_input, output, rows,
                         kDitAttentionWidth, stream);
}
}  // namespace gufo::minimax_h3::dit_ops

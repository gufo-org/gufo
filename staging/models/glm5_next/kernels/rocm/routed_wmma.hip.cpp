#include "staging/models/glm5_next/kernels/rocm/routed_wmma.hpp"

#include <cassert>

#include "staging/models/glm5_next/kernels/rocm/iq_tables.hip.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kRowsPerBlock = 128;
constexpr int kStageK = 256;  // one IQ super-block
constexpr int kTileTokens = 32;
constexpr int kActStride = kStageK + 8;

// Staging geometry per codec: `kSuper` code bytes per 256-element block,
// `kStride` staging bytes per staged row. The stride is the ceil(kSuper/4)
// dwords rounded to an odd dword count, so the 16 rows a fragment read
// touches sit on distinct bank groups.
constexpr int kIq1MBytes = 56;
constexpr int kIq1MStride = 60;
constexpr int kIq2XxsBytes = 66;
constexpr int kIq2XxsStride = 68;
constexpr int kIq3XxsBytes = 98;
constexpr int kIq3XxsStride = 100;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ std::uint32_t U32At(const std::uint8_t* p) {
  std::uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

/// f16 at a byte offset: the block scale `d` of the IQ2/IQ3 super-blocks.
__device__ __forceinline__ float F16At(const std::uint8_t* p) {
  __half_raw raw;
  raw.x = static_cast<std::uint16_t>(
      p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
  return __half2float(*reinterpret_cast<const __half*>(&raw));
}

// Byte j (little-endian) of a packed grid entry.
__device__ __forceinline__ int GridByte(std::uint64_t grid, int j) {
  return static_cast<int>((grid >> (8 * j)) & 0xFFU);
}

__device__ __forceinline__ float IqSign(std::uint8_t signs, int j) {
  return (signs & static_cast<std::uint8_t>(1U << j)) != 0 ? -1.0F : 1.0F;
}

/// The last staged word of the last block: the ceil-bytes read spills at
/// most 2 bytes past the block, which is inside the stack only for earlier
/// blocks, so the tail word of the final block assembles from `avail` (>= 1)
/// in-stack bytes instead of running past the tensor.
__device__ __forceinline__ std::uint32_t TailU32(const std::uint8_t* p,
                                                 std::size_t avail) {
  std::uint32_t v = p[0];
  if (avail > 1U) {
    v |= static_cast<std::uint32_t>(p[1]) << 8;
  }
  if (avail > 2U) {
    v |= static_cast<std::uint32_t>(p[2]) << 16;
  }
  return v;
}

/// The 4 grid nibbles of one iq1s entry as 8 halves: nibble j sits at bit
/// 4 * (2 * (j & 3) + (j >> 2)) of the GPU-packed word.
__device__ __forceinline__ v16h GridHalves(std::uint32_t gv, float dl,
                                           float delta) {
  v16h out;
  auto* h = reinterpret_cast<__half*>(&out);
  const int db = delta > 0.0F ? 1 : -1;  // * 0.125 folded below
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int nib = static_cast<int>(
        (gv >> (4 * (2 * (j & 3) + (j >> 2)))) & 0xFU);
    h[j] = __float2half(dl * (static_cast<float>(nib - 1) + 0.125F * db));
  }
  return out;
}

/// The 16 elements [k0, k0 + 16) (= two 8-element groups) of the staged
/// super-block row `blk` as one f16 wmma fragment operand. Selected by the
/// codec's super-block size (the staging row holds exactly one block).
template <int kSuper>
__device__ __forceinline__ v16h DecodeFragmentHalf(const std::uint8_t* blk,
                                                   int k0);

template <>
__device__ __forceinline__ v16h DecodeFragmentHalf<kIq1MBytes>(
    const std::uint8_t* __restrict__ blk, int k0) {
  const std::uint32_t scw[4] = {U32At(blk + 48) & 0xFFFFU,
                                U32At(blk + 50) & 0xFFFFU,
                                U32At(blk + 52) & 0xFFFFU,
                                U32At(blk + 54) & 0xFFFFU};
  const std::uint16_t packed = static_cast<std::uint16_t>(
      (scw[0] >> 12) | ((scw[1] >> 8) & 0x00F0U) | ((scw[2] >> 4) & 0x0F00U) |
      (scw[3] & 0xF000U));
  __half_raw d_raw;
  d_raw.x = packed;
  const float d = __half2float(*reinterpret_cast<const __half*>(&d_raw));
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int sel = 0; sel < 2; ++sel) {
    const int g = k0 / 8 + sel;  // group index 0..31
    const int ib = g >> 2;
    const int l = g & 3;
    const std::uint8_t qs = blk[g];
    const std::uint8_t qh = blk[32 + (g >> 1)];
    const float dl = d * static_cast<float>(
                             2 *
                                 ((scw[ib / 2] >> (6 * (ib % 2) + 3 * (l >> 1))) &
                                  7) +
                             1);
    const std::uint32_t idx =
        static_cast<std::uint32_t>(qs) |
        ((static_cast<std::uint32_t>(qh) << ((l & 1U) ? 4U : 8U)) & 0x700U);
    const float delta =
        ((qh >> ((l & 1U) ? 7U : 3U)) & 1U) != 0 ? -0.125F : 0.125F;
    const v16h part = GridHalves(kIq1sGridGpu[idx], dl, delta);
    const __half* pp = reinterpret_cast<const __half*>(&part);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      halves[8 * sel + j] = pp[j];
    }
  }
  return frag;
}

// IQ2_XXS: group g of the block reads its grid byte at 2 + 8*ib32 + l and
// the aux word at 6 + 8*ib32 (4-bit scale extension, 4 x 7 sign bits; core
// DequantizeIQ2_XXS).
template <>
__device__ __forceinline__ v16h DecodeFragmentHalf<kIq2XxsBytes>(
    const std::uint8_t* __restrict__ blk, int k0) {
  const float d = F16At(blk);
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int sel = 0; sel < 2; ++sel) {
    const int g = k0 / 8 + sel;  // group index 0..31
    const int ib32 = g >> 2;
    const int l = g & 3;
    const std::uint8_t* q = blk + 2 + 8 * ib32;
    const std::uint32_t aux = U32At(q + 4);
    const float db = d * (0.5F + static_cast<float>(aux >> 28)) * 0.25F;
    const std::uint64_t grid = kIq2xxsGrid[q[l]];
    const std::uint8_t signs = kKsignsIq2xs[(aux >> (7 * l)) & 127U];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      halves[8 * sel + j] = __float2half(db *
                                         static_cast<float>(GridByte(grid, j)) *
                                         IqSign(signs, j));
    }
  }
  return frag;
}

// IQ3_XXS: group g reads two 4-value grid words at 2 + 8*ib32 + 2l and the
// aux word at 66 + 4*ib32 (core DequantizeIQ3_XXS).
template <>
__device__ __forceinline__ v16h DecodeFragmentHalf<kIq3XxsBytes>(
    const std::uint8_t* __restrict__ blk, int k0) {
  const float d = F16At(blk);
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int sel = 0; sel < 2; ++sel) {
    const int g = k0 / 8 + sel;  // group index 0..31
    const int ib32 = g >> 2;
    const int l = g & 3;
    const std::uint8_t* qs = blk + 2 + 8 * ib32;
    const std::uint32_t aux = U32At(blk + 66 + 4 * ib32);
    const float db = d * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
    const std::uint8_t signs = kKsignsIq2xs[(aux >> (7 * l)) & 127U];
    const std::uint32_t g1 = kIq3xxsGrid[qs[2 * l]];
    const std::uint32_t g2 = kIq3xxsGrid[qs[2 * l + 1]];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      halves[8 * sel + j] =
          __float2half(db * static_cast<float>((g1 >> (8 * j)) & 0xFFU) *
                       IqSign(signs, j));
      halves[8 * sel + 4 + j] =
          __float2half(db * static_cast<float>((g2 >> (8 * j)) & 0xFFU) *
                       IqSign(signs, j + 4));
    }
  }
  return frag;
}

/// Fused routed GEMM over one 32-slot token tile per weight block. `kStacks`
/// is 2 for the gate/up pair (shared activations, swiglu-clamp epilogue into
/// out[slot * rows + r]) and 1 for the down projection (out[slot * rows + r]
/// = raw dot, the slot being the activation row, slots_per_row = 1).
/// Accumulation runs in f16 fragments over wmma 16x16x16 for every codec.
template <int kSuper, int kStride, int kStacks>
__global__ void __launch_bounds__(kThreads) RoutedWmmaKernel(
    const std::uint8_t* __restrict__ stack0,
    const std::uint8_t* __restrict__ stack1, std::size_t expert_stride,
    std::size_t stack_bytes, const __half* __restrict__ x,
    std::uint32_t x_stride, const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_slot, std::uint32_t slots_per_row,
    std::uint32_t rows, std::uint32_t cols, float clamp,
    float* __restrict__ out) {
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kChunksPerTok = kStageK / 8;
  constexpr int kActChunks = kTileTokens * kChunksPerTok;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  constexpr int kWords = (kSuper + 3) / 4;
  constexpr int kWords0 = (kWords + 1) / 2;
  constexpr int kWords1 = kWords - kWords0;
  constexpr bool kSpill = 4 * kWords > kSuper;
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_w[kStacks][kRowsPerBlock * kStride];
  __shared__ __attribute__((aligned(16))) __half
      s_act[kTileTokens * kActStride];

  const std::int32_t tile = tiles[blockIdx.y];
  if ((tile >> 16) == 0x7FFF) {
    return;
  }
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * kTileTokens;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{cols} / 256 * kSuper;

  // Weight fetch: two threads per staged row move the ceil(kSuper/4) words
  // of the stage's super-block (p0 the first kWords0, p1 the rest, each
  // starting at its own byte offset; word loads are misaligned by design,
  // blocks sit kSuper bytes apart).
  const int f_row = tid >> 1;
  const int f_part = tid & 1;
  const std::uint32_t f_global = blockIdx.x * kRowsPerBlock + f_row;
  const bool f_live = f_global < rows;
  const std::size_t f_off =
      static_cast<std::size_t>(expert) * expert_stride +
      (f_live ? f_global : rows - 1) * row_bytes;
  const std::uint8_t* f_w[kStacks];
  f_w[0] = stack0;
  if constexpr (kStacks > 1) {
    f_w[1] = stack1;
  } else {
    static_cast<void>(stack1);
  }
  std::uint32_t f_word[kStacks][kWords0];
  const auto fetch_weights = [&](int stage) {
    const std::size_t tail_off =
        f_off + static_cast<std::size_t>(stage) * kSuper +
        4U * (kWords - 1);
    const std::size_t tail_avail =
        stack_bytes > tail_off ? stack_bytes - tail_off : 0;
    const std::size_t base = f_part == 0 ? 0U : 4U * kWords0;
#pragma unroll
    for (int s = 0; s < kStacks; ++s) {
      const std::uint8_t* block = f_w[s] + f_off + stage * kSuper + base;
#pragma unroll
      for (int i = 0; i < kWords0; ++i) {
        if (i < kWords1 || f_part == 0) {
          const std::uint8_t* p = block + i * 4;
          if constexpr (kSpill) {
            const bool tail = f_part == 1 && i == kWords1 - 1;
            f_word[s][i] =
                tail && tail_avail < 4U ? TailU32(p, tail_avail) : U32At(p);
          } else {
            f_word[s][i] = U32At(p);
          }
        }
      }
    }
  };
  const auto commit_weights = [&] {
    const std::uint32_t d0 = f_part == 0 ? 0U : kWords0;
#pragma unroll
    for (int s = 0; s < kStacks; ++s) {
      auto* dst = reinterpret_cast<std::uint32_t*>(s_w[s] + f_row * kStride);
#pragma unroll
      for (int i = 0; i < kWords1 || (i < kWords0 && f_part == 0); ++i) {
        dst[d0 + i] = f_word[s][i];
      }
    }
  };

  // Activation fetch (flint chunking: chunk c = token c / 32, eight values
  // at K offset 8 * (c % 32)); rows_slot maps a bucket row to the slot and
  // the slot / slots_per_row back to the token row of x.
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / kChunksPerTok;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t slot = rows_slot[bucket_begin + t_local + t];
      if (slot >= 0) {
        a_src[i] = static_cast<std::int32_t>(slot / slots_per_row) *
                       static_cast<std::int32_t>(x_stride) +
                   (chunk % kChunksPerTok) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] = a_src[i] >= 0
                      ? *reinterpret_cast<const uint4*>(x + a_src[i] +
                                                        stage * kStageK)
                      : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / kChunksPerTok) * kActStride +
                                  (chunk % kChunksPerTok) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kStacks][kTokTiles];
#pragma unroll
  for (int s = 0; s < kStacks; ++s) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[s][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }
  const int stages = static_cast<int>(cols / kStageK);
  fetch_weights(0);
  fetch_act(0);
  for (int stage = 0; stage < stages; ++stage) {
    commit_weights();
    commit_act();
    __syncthreads();
    if (stage + 1 < stages) {
      fetch_weights(stage + 1);
      fetch_act(stage + 1);
    }
#pragma unroll
    for (int s = 0; s < kStageK / 32; ++s) {
      v16h w_lo[kStacks];
      v16h w_hi[kStacks];
#pragma unroll
      for (int w = 0; w < kStacks; ++w) {
        const std::uint8_t* row = s_w[w] + (wave * 16 + sub) * kStride;
        w_lo[w] = DecodeFragmentHalf<kSuper>(row, 32 * s);
        w_hi[w] = DecodeFragmentHalf<kSuper>(row, 32 * s + 16);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
#pragma unroll
          for (int w = 0; w < kStacks; ++w) {
            acc[w][j] = Wmma(w_lo[w], b_lo, acc[w][j]);
            acc[w][j] = Wmma(w_hi[w], b_hi, acc[w][j]);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    const int t = t_local + j * 16 + sub;
    if (t >= bucket_rows) {
      continue;
    }
    const std::int32_t slot = rows_slot[bucket_begin + t];
    if (slot < 0) {
      continue;
    }
    const std::size_t base = static_cast<std::size_t>(slot) * rows;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t r =
          blockIdx.x * kRowsPerBlock + wave * 16 + 2 * i + half;
      if (r < rows) {
        if constexpr (kStacks == 2) {
          const float g = fminf(acc[0][j][i], clamp);
          const float u = fminf(fmaxf(acc[1][j][i], -clamp), clamp);
          out[base + r] = g * (1.0F / (1.0F + expf(-g))) * u;
        } else {
          out[base + r] = acc[0][j][i];
        }
      }
    }
  }
}

}  // namespace

void RoutedGateUpWmma(GemvType type, const void* gate, const void* up,
                      std::size_t expert_stride, std::size_t stack_bytes,
                      const __half* x, std::uint32_t x_stride,
                      const std::int32_t* tiles, std::uint32_t capacity,
                      const std::int32_t* pad_bounds,
                      const std::int32_t* rows_slot,
                      std::uint32_t slots_per_row, std::uint32_t rows,
                      std::uint32_t cols, float clamp, float* out,
                      hipStream_t stream) {
  const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, capacity);
  const std::uint8_t* g = static_cast<const std::uint8_t*>(gate);
  const std::uint8_t* u = static_cast<const std::uint8_t*>(up);
  switch (type) {
    case GemvType::kIQ1_M:
      RoutedWmmaKernel<kIq1MBytes, kIq1MStride, 2><<<grid, kThreads, 0,
                                                     stream>>>(
          g, u, expert_stride, stack_bytes, x, x_stride, tiles, pad_bounds,
          rows_slot, slots_per_row, rows, cols, clamp, out);
      return;
    case GemvType::kIQ2_XXS:
      RoutedWmmaKernel<kIq2XxsBytes, kIq2XxsStride, 2><<<grid, kThreads, 0,
                                                         stream>>>(
          g, u, expert_stride, stack_bytes, x, x_stride, tiles, pad_bounds,
          rows_slot, slots_per_row, rows, cols, clamp, out);
      return;
    default:
      break;
  }
  assert(false && "RoutedGateUpWmma: unsupported expert quantization");
}

void RoutedDownWmma(GemvType type, const void* down, std::size_t expert_stride,
                    std::size_t stack_bytes, const __half* x,
                    std::uint32_t x_stride, const std::int32_t* tiles,
                    std::uint32_t capacity, const std::int32_t* pad_bounds,
                    const std::int32_t* rows_slot, std::uint32_t rows,
                    std::uint32_t cols, float* out, hipStream_t stream) {
  const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, capacity);
  const std::uint8_t* d = static_cast<const std::uint8_t*>(down);
  if (type == GemvType::kIQ3_XXS) {
    RoutedWmmaKernel<kIq3XxsBytes, kIq3XxsStride, 1><<<grid, kThreads, 0,
                                                       stream>>>(
        d, nullptr, expert_stride, stack_bytes, x, x_stride, tiles, pad_bounds,
        rows_slot, 1U, rows, cols, 0.0F, out);
    return;
  }
  assert(false && "RoutedDownWmma: unsupported expert quantization");
}

}  // namespace gufo::models::glm5_next::rocm

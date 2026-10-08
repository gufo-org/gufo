// Routed expert GEMM for prefill over binary16 activations for the K-quant
// expert formats (Q4_K, Q5_K, Q6_K), and for the blockwise ones (Q5_1, Q8_0,
// binary16). It reads the routing layout of Flash-Next's routed GEMM:
// RoutedCompact's 16-padded expert buckets and a tile map of
// expert | tile << 16 entries.
//
// A block computes 128 output rows (eight waves of 16) for one tile of
// kTileTokens bucket rows, skipping the tile's 16-row parts past the bucket.
// A stage is half a super-block (128 K values, four 32-value K blocks): two
// threads per row fetch its code bytes and scale header one stage ahead in
// registers, neighbouring threads on neighbouring bytes, and commit them with
// the tile's activations to LDS, the scales already as binary16. Each lane
// then decodes its own row's K blocks straight into binary16 WMMA fragments.
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and B
// column L%16, each with 16 K values in both half-waves; C lane L holds
// column L%16, rows 2 i + L/16.
//
// Q4_K and Q5_K weights are q * (d sc) - dmin m, evaluated as one binary16
// FMA per pair with the two factors rounded to binary16, as in Flash-Next's
// routed GEMM, so the two produce the same products; Q6_K weights are
// (q - 32) * (d sc).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kRowsPerBlock = 128;
/// K values per stage: half a super-block.
constexpr int kStageK = 128;

/// Bytes per staged row: 64 low-bit bytes, 32 high-bit bytes (Q5_K, Q6_K)
/// and 16 bytes of binary16 scales. 80 and 112 bytes (20 and 28 dwords) put
/// the 16 rows a fragment read touches on distinct bank groups.
template<ExpertFormat F>
constexpr int kRowStride = F == ExpertFormat::kQ4_K ? 80 : 112;

template<ExpertFormat F>
constexpr int kBlockBytes = F == ExpertFormat::kQ4_K   ? 144
                            : F == ExpertFormat::kQ5_K ? 176
                                                       : 210;
/// Halves per staged token row: 128 values plus padding (68 dwords) against
/// bank conflicts on the fragment reads.
constexpr int kActStride = kStageK + 8;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

/// Four byte codes q (< 1024) as two half2 of (q - bias) * scale: the codes
/// become 1024 + q through the exponent byte 0x64, exactly.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 offset, __half2 scale,
                                              __half2* out) {
  constexpr std::uint32_t kMagic = 0x64646464U;
  const std::uint32_t p0 = __builtin_amdgcn_perm(codes, kMagic, 0x01050004U);
  const std::uint32_t p1 = __builtin_amdgcn_perm(codes, kMagic, 0x03070206U);
  out[0] = __hmul2(__hsub2(__builtin_bit_cast(__half2, p0), offset), scale);
  out[1] = __hmul2(__hsub2(__builtin_bit_cast(__half2, p1), offset), scale);
}

/// Four byte codes c (< 1024) as two half2 of (c - offset) * scale + bias,
/// offset 0 or 128 (a signed byte carried as q + 128).
__device__ __forceinline__ void CodesToHalvesAffine(std::uint32_t codes,
                                                    __half2 scale, __half2 bias,
                                                    __half2* out,
                                                    float offset = 0.0F) {
  constexpr std::uint32_t kMagic = 0x64646464U;
  const __half2 magic = __float2half2_rn(-1024.0F - offset);
  const std::uint32_t p0 = __builtin_amdgcn_perm(codes, kMagic, 0x01050004U);
  const std::uint32_t p1 = __builtin_amdgcn_perm(codes, kMagic, 0x03070206U);
  out[0] =
      __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale, bias);
  out[1] =
      __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale, bias);
}

/// Q4_K / Q5_K K block s (< 4) of a staged half super-block n as two
/// fragments: low (s even) or high nibbles of qs[32 (s / 2) + l], plus bit
/// 4 n + s of qh[l] as 16 (Q5_K), under sub-block s's (scale, bias).
template<ExpertFormat F>
__device__ __forceinline__ void DecodeQ45K(const uint4 (&qs)[4],
                                           const uint4 (&qh)[2],
                                           const __half2 (&scale)[4], int n,
                                           int s, v16h* lo, v16h* hi) {
  const int shift = 4 * (s & 1);
  const int bit = 4 * n + s;
  const __half2 sc = __low2half2(scale[s]);
  const __half2 bias = __high2half2(scale[s]);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const uint4 q4 = qs[2 * (s >> 1) + part];
    const uint4 h4 = qh[part];
    const std::uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
    const std::uint32_t hw[4] = {h4.x, h4.y, h4.z, h4.w};
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      std::uint32_t codes = (qw[i] >> shift) & 0x0F0F0F0FU;
      if constexpr (F == ExpertFormat::kQ5_K) {
        codes |= ((hw[i] >> bit) & 0x01010101U) << 4U;
      }
      CodesToHalvesAffine(codes, sc, bias, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q4_K / Q5_K scale and min of sub-block sb (< 8) from the 12 packed
/// 6-bit bytes that follow d and dmin in the header.
__device__ __forceinline__ void ScaleMinK4(const uint4& header, int sb,
                                           std::uint32_t* sc,
                                           std::uint32_t* mn) {
  const std::uint32_t words[4] = {header.x, header.y, header.z, header.w};
  const auto byte = [&](int i) {
    return (words[i / 4] >> (8 * (i % 4))) & 0xFFU;
  };
  // scales[i] is header byte 4 + i.
  if (sb < 4) {
    *sc = byte(4 + sb) & 0x3FU;
    *mn = byte(8 + sb) & 0x3FU;
  } else {
    *sc = (byte(8 + sb) & 0x0FU) | ((byte(sb) >> 6U) << 4U);
    *mn = (byte(8 + sb) >> 4U) | ((byte(4 + sb) >> 6U) << 4U);
  }
}

/// Q6_K K block s (< 4) of a staged half super-block as two fragments: low
/// (s < 2) or high nibbles of ql[32 (s % 2) + l], bits 2 s of qh[l], scales
/// 2 s (l < 16) and 2 s + 1, offset 32.
__device__ __forceinline__ void DecodeQ6K(const uint4 (&ql)[4],
                                          const uint4 (&qh)[2],
                                          const __half2 (&scale)[4], int s,
                                          v16h* lo, v16h* hi) {
  const int nibble = (s >> 1) * 4;
  const int high = 2 * s;
  const __half2 offset = __float2half2_rn(1056.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const uint4 l4 = ql[2 * (s & 1) + part];
    const uint4 h4 = qh[part];
    const std::uint32_t lw[4] = {l4.x, l4.y, l4.z, l4.w};
    const std::uint32_t hw[4] = {h4.x, h4.y, h4.z, h4.w};
    const __half2 sc =
        part == 0 ? __low2half2(scale[s]) : __high2half2(scale[s]);
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t codes = ((lw[i] >> nibble) & 0x0F0F0F0FU) |
                                  (((hw[i] >> high) & 0x03030303U) << 4U);
      CodesToHalves(codes, offset, sc, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// gelu_tanh(x) * u, the expression (and fast-math build) of the Gemma
/// elementwise kernels' GeGluValue.
__device__ __attribute__((noinline)) float GeGluValue(float x, float u) {
  constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876F;
  constexpr float kCoefA = 0.044715F;
  const float g =
      0.5F * x * (1.0F + tanhf(kSqrt2OverPi * x * (1.0F + kCoefA * x * x)));
  return g * u;
}

__device__ __forceinline__ __half SaturatedHalf(float v) {
  return __float2half(fminf(fmaxf(v, -65504.0F), 65504.0F));
}

/// kGeGlu: W holds fused [gate | up] rows; a block's first four waves take
/// 64 gate rows and the last four the matching up rows (m / 2 further), and
/// the block writes GeGLU of the binary16-rounded pair, as the separate
/// GeGluPackedHalf pass would, into `out_half` rows of width m / 2.
template<ExpertFormat F, int kTileTokens, bool kGeGlu = false,
         bool kDense = false>
__global__ void __launch_bounds__(kThreads)
    RoutedHalfKQuantKernel(const std::uint8_t* __restrict__ w,
                           const __half* __restrict__ x,
                           const std::int32_t* __restrict__ tiles,
                           const std::int32_t* __restrict__ pad_bounds,
                           const std::int32_t* __restrict__ rows_in,
                           const std::int32_t* __restrict__ rows_out,
                           float* __restrict__ out,
                           __half* __restrict__ out_half, std::uint32_t m,
                           std::uint32_t k, std::uint32_t dense_rows = 0,
                           std::uint32_t dense_group = 0) {
  constexpr bool kQ6 = F == ExpertFormat::kQ6_K;
  constexpr bool kQ4 = F == ExpertFormat::kQ4_K;
  constexpr int kStride = kRowStride<F>;
  constexpr int kScaleChunk = kStride / 16 - 1;
  constexpr int kTokTiles = kTileTokens / 16;
  // uint4 activation chunks per stage (16 per token) and per thread.
  constexpr int kActChunks = kTileTokens * kStageK / 8;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  __shared__
      __attribute__((aligned(16))) std::uint8_t s_rows[kRowsPerBlock * kStride];
  __shared__
      __attribute__((aligned(16))) __half s_act[kTileTokens * kActStride];
  int expert = 0;
  int t_local = 0;
  int bucket_begin = 0;
  int bucket_rows = 0;
  unsigned row_block = blockIdx.x;
  if constexpr (kDense) {
    // One matrix over rows 0..dense_rows of x and out, on a 1-D grid that
    // takes `dense_group` token tiles for one row block, then the next row
    // block: the blocks running together share a weight tile and a few
    // activation tiles.
    const unsigned token_tiles = (dense_rows + kTileTokens - 1) / kTileTokens;
    const unsigned span = dense_group * (gridDim.x / token_tiles);
    const unsigned first = blockIdx.x / span * dense_group;
    const unsigned width = min(dense_group, token_tiles - first);
    const unsigned within = blockIdx.x % span;
    t_local = static_cast<int>(first + within % width) * kTileTokens;
    row_block = within / width;
    bucket_rows = static_cast<int>(dense_rows);
  } else {
    const std::int32_t tile = tiles[blockIdx.y];
    expert = tile & 0xFFFF;
    t_local = (tile >> 16) * kTileTokens;
    bucket_begin = pad_bounds[expert];
    bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  }
  // Bucket row t's activation and output rows (the identity when dense).
  const auto source_row = [&](int t) {
    return kDense ? t : rows_in[bucket_begin + t];
  };
  const auto dest_row = [&](int t) {
    return kDense ? t : rows_out[bucket_begin + t];
  };
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{k} / 256 * kBlockBytes<F>;

  // Weight fetch: thread tid moves row tid / 2 of the block, half its code
  // chunks each (part 0 first), and part 1 commits the scales as binary16.
  // Q6_K: ql 0-47 | ql 48-63, qh 0-31; Q5_K: qs 0-47 | qs 48-63, qh 0-31;
  // Q4_K: qs 0-31 | qs 32-63.
  const int f_row = tid >> 1;
  const int f_part = tid & 1;
  const std::uint32_t half_m = m / 2;
  // Block row l's matrix row: kGeGlu, gate rows then their up rows.
  const auto matrix_row = [&](int l) -> std::uint32_t {
    if constexpr (kGeGlu) {
      return (l >= kRowsPerBlock / 2 ? half_m : 0U) +
             row_block * (kRowsPerBlock / 2) + l % (kRowsPerBlock / 2);
    } else {
      return row_block * kRowsPerBlock + l;
    }
  };
  const std::uint32_t f_global = matrix_row(f_row);
  const bool f_live =
      kGeGlu ? row_block * (kRowsPerBlock / 2) + f_row % (kRowsPerBlock / 2) <
                   half_m
             : f_global < m;
  const std::uint8_t* f_ptr =
      w + (std::size_t{static_cast<std::uint32_t>(expert)} * m +
           (f_live ? f_global : m - 1)) *
              row_bytes;
  uint4 f_code0;
  uint4 f_code1;
  uint4 f_code2;
  uint4 f_header;  // Q4_K / Q5_K: d, dmin and the packed scales
  uint2 f_sc;      // Q6_K: the half's eight scales
  float f_d = 0.0F;
  int f_half = 0;
  const auto fetch_weights = [&](int stage) {
    const std::uint8_t* block = f_ptr + (stage / 2) * kBlockBytes<F>;
    const int n = stage % 2;
    f_half = n;
    if constexpr (kQ6) {
      // The qh bytes at 128 + 32 n follow ql 48-63 after 64 - 32 n bytes.
      const std::uint8_t* first = block + 64 * n + 48 * f_part;
      const int gap = f_part * (64 - 32 * n);
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(first + 16 + gap);
      f_code2 = *reinterpret_cast<const uint4*>(first + 32 + gap);
      f_sc = *reinterpret_cast<const uint2*>(block + 192 + 8 * n);
      f_d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    } else if constexpr (kQ4) {
      const std::uint8_t* first = block + 16 + 64 * n + 32 * f_part;
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(first + 16);
      f_header = *reinterpret_cast<const uint4*>(block);
    } else {
      // Q5_K: header, qh[32], qs[128].
      const std::uint8_t* first = block + 48 + 64 * n + 48 * f_part;
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(f_part != 0 ? block + 16
                                                            : first + 16);
      f_code2 = *reinterpret_cast<const uint4*>(f_part != 0 ? block + 32
                                                            : first + 32);
      f_header = *reinterpret_cast<const uint4*>(block);
    }
  };
  const auto commit_weights = [&] {
    auto* dst = reinterpret_cast<uint4*>(s_rows + f_row * kStride);
    if constexpr (kQ4) {
      dst[2 * f_part] = f_code0;
      dst[2 * f_part + 1] = f_code1;
    } else {
      dst[3 * f_part] = f_code0;
      dst[3 * f_part + 1] = f_code1;
      dst[3 * f_part + 2] = f_code2;
    }
    if (f_part == 1) {
      std::uint32_t packed[4];
      if constexpr (kQ6) {
        const float d = f_live ? f_d : 0.0F;
        const std::uint32_t sw[2] = {f_sc.x, f_sc.y};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const auto lo = static_cast<std::int8_t>(sw[i / 2] >> (16 * (i % 2)));
          const auto hi =
              static_cast<std::int8_t>(sw[i / 2] >> (16 * (i % 2) + 8));
          // d * sc in FP32, then binary16.
          packed[i] = __builtin_bit_cast(
              std::uint32_t,
              __halves2half2(__float2half_rn(d * static_cast<float>(lo)),
                             __float2half_rn(d * static_cast<float>(hi))));
        }
      } else {
        const __half2 dm = __builtin_bit_cast(__half2, f_header.x);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          std::uint32_t sc = 0;
          std::uint32_t mn = 0;
          ScaleMinK4(f_header, 4 * f_half + i, &sc, &mn);
          const float scale =
              f_live ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
          const float offset =
              f_live ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
          packed[i] = __builtin_bit_cast(std::uint32_t,
                                         __floats2half2_rn(scale, -offset));
        }
      }
      dst[kScaleChunk] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
  };

  // Activation fetch: chunk c = tid + 256 i is token c / 16, eight values at
  // K offset 8 (c % 16) of the stage; a_src is that chunk's offset in x at
  // stage 0, or -1 for a padding row.
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / 16;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = source_row(t_local + t);
      if (src >= 0) {
        a_src[i] = src * static_cast<std::int32_t>(k) + (chunk % 16) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] =
          a_src[i] >= 0
              ? *reinterpret_cast<const uint4*>(x + a_src[i] + stage * kStageK)
              : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / 16) * kActStride +
                                  (chunk % 16) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const int stages = static_cast<int>(k / kStageK);
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
    const auto* row =
        reinterpret_cast<const uint4*>(s_rows + (wave * 16 + sub) * kStride);
    const uint4 q[4] = {row[0], row[1], row[2], row[3]};
    uint4 qh[2] = {};
    if constexpr (!kQ4) {
      qh[0] = row[4];
      qh[1] = row[5];
    }
    const uint4 sc4 = row[kScaleChunk];
    const __half2 scale[4] = {
        __builtin_bit_cast(__half2, sc4.x), __builtin_bit_cast(__half2, sc4.y),
        __builtin_bit_cast(__half2, sc4.z), __builtin_bit_cast(__half2, sc4.w)};
    const int n = stage % 2;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      v16h a_lo;
      v16h a_hi;
      if constexpr (kQ6) {
        DecodeQ6K(q, qh, scale, s, &a_lo, &a_hi);
      } else {
        DecodeQ45K<F>(q, qh, scale, n, s, &a_lo, &a_hi);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(a_lo, b_lo, acc[j]);
          acc[j] = Wmma(a_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
  }
  if constexpr (kGeGlu) {
    // Up waves hand their binary16 rows to the gate waves through LDS (the
    // stages are free now); lane layouts match wave for wave.
    auto* s_up = reinterpret_cast<__half*>(s_act);
    static_assert(4 * kTokTiles * 8 * kWave * 2 <=
                  static_cast<int>(sizeof(s_act)));
    if (wave >= 4) {
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_up[(((wave - 4) * kTokTiles + j) * 8 + i) * kWave + lane] =
              SaturatedHalf(acc[j][i]);
        }
      }
    }
    __syncthreads();
    if (wave >= 4) {
      return;
    }
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      const int t = t_local + j * 16 + sub;
      if (t >= bucket_rows) {
        continue;
      }
      const std::int32_t dst = dest_row(t);
      if (dst < 0) {
        continue;
      }
      const std::size_t base =
          std::size_t{static_cast<std::uint32_t>(dst)} * half_m;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t r =
            row_block * (kRowsPerBlock / 2) +
            static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
        if (r < half_m) {
          const __half up =
              s_up[((wave * kTokTiles + j) * 8 + i) * kWave + lane];
          const float v = GeGluValue(__half2float(SaturatedHalf(acc[j][i])),
                                     __half2float(up));
          out_half[base + r] = SaturatedHalf(v);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    const int t = t_local + j * 16 + sub;
    if (t >= bucket_rows) {
      continue;
    }
    const std::int32_t dst = dest_row(t);
    if (dst < 0) {
      continue;
    }
    const std::size_t base = std::size_t{static_cast<std::uint32_t>(dst)} * m;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t r =
          row_block * kRowsPerBlock +
          static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
      if (r < m) {
        if (out_half != nullptr) {
          out_half[base + r] =
              __float2half(fminf(fmaxf(acc[j][i], -65504.0F), 65504.0F));
        } else {
          out[base + r] = acc[j][i];
        }
      }
    }
  }
}

/// K values per stage of the down projection kernel: two 32-value blocks.
constexpr int kDownStageK = 64;
/// Halves per staged token row of the down kernel: 64 values plus padding
/// (36 dwords, conflict-free fragment reads).
constexpr int kDownActStride = kDownStageK + 8;
/// Bytes per staged Q5_1 row: the stage's two raw 24-byte blocks (d, m, qh,
/// 16 code bytes each). 48 bytes (12 dwords) spread the 16 rows a fragment
/// read touches over distinct banks.
constexpr int kQ51Stride = 48;
/// Bytes per staged Q8_0 row: the stage's two raw 34-byte blocks as 17
/// words (d0, codes 0-31, d1, codes 0-31; the first block's codes start two
/// bytes into a word), padded to 80 bytes (20 dwords, conflict-free).
constexpr int kQ80Stride = 80;
constexpr int kQ80Words = 17;
/// Bytes per staged binary16 row: the stage's 64 values, padded to 144 bytes
/// (36 dwords, conflict-free fragment reads).
constexpr int kF16Stride = 144;
/// Bytes per staged Q4_0 row: the stage's two raw 18-byte blocks as nine
/// words (d0, codes 0-15, d1, codes 0-15; the first block's codes start two
/// bytes into a word, the second's on a word), padded like Q5_1 to 48 bytes.
constexpr int kQ40Stride = 48;
constexpr int kQ40Words = 9;

template<ExpertFormat F>
constexpr int kDownStride = F == ExpertFormat::kQ5_1   ? kQ51Stride
                            : F == ExpertFormat::kQ8_0 ? kQ80Stride
                            : F == ExpertFormat::kQ4_0 ? kQ40Stride
                                                       : kF16Stride;
template<ExpertFormat F>
constexpr int kDownBlockBytes = F == ExpertFormat::kQ5_1   ? 24
                                : F == ExpertFormat::kQ8_0 ? 34
                                : F == ExpertFormat::kQ4_0 ? 18
                                                           : 64;
/// Words a word-staged format (Q8_0, Q4_0) keeps per row and stage.
template<ExpertFormat F>
constexpr int kDownWords = F == ExpertFormat::kQ8_0 ? kQ80Words : kQ40Words;
/// Floats per token row of the down kernel's output transpose (128 rows plus
/// padding: 132 = 4 mod 64 puts a tile's writes on distinct banks).
constexpr int kOutStride = kRowsPerBlock + 4;

/// Q5_1 K block (32 values) from its staged raw bytes as two fragments:
/// element j < 16 is the low nibble of qs[j] plus bit j of qh as 16, j >= 16
/// the high nibble of qs[j - 16] plus bit j; weight q * d + m, one binary16
/// FMA as in Flash-Next's routed GEMM.
__device__ __forceinline__ void DecodeQ51(const std::uint8_t* block, bool live,
                                          v16h* lo, v16h* hi) {
  const uint2 head = *reinterpret_cast<const uint2*>(block);
  const uint2 q0 = *reinterpret_cast<const uint2*>(block + 8);
  const uint2 q1 = *reinterpret_cast<const uint2*>(block + 16);
  const std::uint32_t qs[4] = {q0.x, q0.y, q1.x, q1.y};
  const std::uint32_t qh = head.y;
  const __half2 dm = __builtin_bit_cast(__half2, live ? head.x : 0U);
  const __half2 scale = __low2half2(dm);
  const __half2 bias = __high2half2(dm);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t bits = qh >> (16 * part + 4 * i);
      // Bit b of `bits` (b < 4) to bit 4 of byte b.
      const std::uint32_t high = ((bits & 1U) | ((bits & 2U) << 7U) |
                                  ((bits & 4U) << 14U) | ((bits & 8U) << 21U))
                                 << 4U;
      const std::uint32_t codes = ((qs[i] >> (4 * part)) & 0x0F0F0F0FU) | high;
      CodesToHalvesAffine(codes, scale, bias, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q8_0 K block s (< 2) of a staged row's 17 words as two fragments: codes
/// flipped to q + 128 and taken back out by the magic, weight q * d, the
/// binary16 FMA of Flash-Next's routed GEMM.
__device__ __forceinline__ void DecodeQ80(const std::uint32_t (&w)[kQ80Words],
                                          int s, bool live, v16h* lo,
                                          v16h* hi) {
  std::uint32_t codes[8];
  std::uint32_t d_bits = 0;
  if (s == 0) {
    d_bits = w[0] & 0xFFFFU;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      codes[i] = __builtin_amdgcn_alignbit(w[i + 1], w[i], 16);
    }
  } else {
    d_bits = w[8] >> 16U;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      codes[i] = w[9 + i];
    }
  }
  const __half2 dm = __builtin_bit_cast(__half2, live ? d_bits : 0U);
  const __half2 scale = __low2half2(dm);
  const __half2 bias = __high2half2(dm);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      CodesToHalvesAffine(codes[4 * part + i] ^ 0x80808080U, scale, bias,
                          &h[2 * i], 128.0F);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q4_0 K block s (< 2) of a staged row's nine words as two fragments: the
/// low nibbles of its 16 code bytes (values 0-15), then the high nibbles
/// (16-31); weight (q - 8) * d, exact before the binary16 rounding.
__device__ __forceinline__ void DecodeQ40(const std::uint32_t (&w)[kQ40Words],
                                          int s, bool live, v16h* lo,
                                          v16h* hi) {
  std::uint32_t codes[4];
  std::uint32_t d_bits = 0;
  if (s == 0) {
    d_bits = w[0] & 0xFFFFU;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = __builtin_amdgcn_alignbit(w[i + 1], w[i], 16);
    }
  } else {
    d_bits = w[4] >> 16U;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      codes[i] = w[5 + i];
    }
  }
  const __half2 scale = __half2half2(__builtin_bit_cast(
      __half, static_cast<std::uint16_t>(live ? d_bits : 0U)));
  const __half2 offset = __float2half2_rn(1032.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      CodesToHalves((codes[i] >> (4 * part)) & 0x0F0F0F0FU, offset, scale,
                    &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Routed Q5_1 / Q8_0 / binary16 down projection with FP32 outputs: the
/// layout and tiling of RoutedHalfKQuantKernel over 64-value stages (three
/// 16-byte pieces per Q5_1 row, consecutive threads on consecutive pieces).
/// Each token tile's results are transposed through LDS so a token's 128
/// outputs leave as one contiguous 512-byte store. kGeGlu (Q8_0, binary16):
/// the gate/up projection with RoutedHalfKQuantKernel's row split and GeGLU
/// epilogue into `out_half`.
template<ExpertFormat F, int kTileTokens, bool kGeGlu = false,
         bool kDense = false>
__global__ void __launch_bounds__(kThreads)
    RoutedHalfDownKernel(const std::uint8_t* __restrict__ w,
                         const __half* __restrict__ x,
                         const std::int32_t* __restrict__ tiles,
                         const std::int32_t* __restrict__ pad_bounds,
                         const std::int32_t* __restrict__ rows_in,
                         const std::int32_t* __restrict__ rows_out,
                         float* __restrict__ out, __half* __restrict__ out_half,
                         std::uint32_t m, std::uint32_t k,
                         std::uint32_t dense_rows = 0,
                         std::uint32_t dense_group = 0) {
  constexpr bool kQ8 = F == ExpertFormat::kQ8_0;
  // Q8_0 and Q4_0 rows are only 2-byte aligned: staged as words.
  constexpr bool kWordStaged = kQ8 || F == ExpertFormat::kQ4_0;
  constexpr int kRowWords = kDownWords<F>;
  constexpr bool kF16 = F == ExpertFormat::kF16;
  constexpr int kStride = kDownStride<F>;
  constexpr int kStageBytes = 2 * kDownBlockBytes<F>;
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kPieces = kRowsPerBlock * kQ51Stride / 16;  // per stage
  constexpr int kActChunks = kTileTokens * kDownStageK / 8;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  constexpr int kActBytes = kTileTokens * kDownActStride * 2;
  constexpr int kOutBytes = 16 * kOutStride * 4;
  __shared__
      __attribute__((aligned(16))) std::uint8_t s_rows[kRowsPerBlock * kStride];
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_act_bytes[kActBytes > kOutBytes ? kActBytes : kOutBytes];
  auto* s_act = reinterpret_cast<__half*>(s_act_bytes);
  int expert = 0;
  int t_local = 0;
  int bucket_begin = 0;
  int bucket_rows = 0;
  unsigned row_block = blockIdx.x;
  if constexpr (kDense) {
    // RoutedHalfKQuantKernel's dense raster.
    const unsigned token_tiles = (dense_rows + kTileTokens - 1) / kTileTokens;
    const unsigned span = dense_group * (gridDim.x / token_tiles);
    const unsigned first = blockIdx.x / span * dense_group;
    const unsigned width = min(dense_group, token_tiles - first);
    const unsigned within = blockIdx.x % span;
    t_local = static_cast<int>(first + within % width) * kTileTokens;
    row_block = within / width;
    bucket_rows = static_cast<int>(dense_rows);
  } else {
    const std::int32_t tile = tiles[blockIdx.y];
    expert = tile & 0xFFFF;
    t_local = (tile >> 16) * kTileTokens;
    bucket_begin = pad_bounds[expert];
    bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  }
  const auto source_row = [&](int t) {
    return kDense ? t : rows_in[bucket_begin + t];
  };
  const auto dest_row = [&](int t) {
    return kDense ? t : rows_out[bucket_begin + t];
  };
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::uint32_t row0 = row_block * kRowsPerBlock;
  const std::uint32_t half_m = m / 2;
  const std::size_t row_bytes = std::size_t{k} / 32 * kDownBlockBytes<F>;
  const std::uint8_t* w_expert =
      w + std::size_t{static_cast<std::uint32_t>(expert)} * m * row_bytes;
  // Block row l's matrix row: kGeGlu, gate rows then their up rows.
  const auto matrix_row = [&](int l) -> std::uint32_t {
    if constexpr (kGeGlu) {
      return (l >= kRowsPerBlock / 2 ? half_m : 0U) +
             row_block * (kRowsPerBlock / 2) +
             static_cast<std::uint32_t>(l % (kRowsPerBlock / 2));
    } else {
      return row0 + static_cast<std::uint32_t>(l);
    }
  };
  const auto row_live = [&](int l) {
    if constexpr (kGeGlu) {
      return row_block * (kRowsPerBlock / 2) +
                 static_cast<std::uint32_t>(l % (kRowsPerBlock / 2)) <
             half_m;
    } else {
      return matrix_row(l) < m;
    }
  };

  // Weight fetch. Q5_1: piece c (tid, and tid + 256 for tid < 128) is bytes
  // 16 (c % 3) of row c / 3's stage. Q8_0 / Q4_0: word c = tid + 256 i is
  // word c % 17 (c % 9) of row c / 17 (c / 9)'s stage.
  // Binary16: piece c = tid + 256 i is bytes 16 (c % 8) of row c / 8's stage.
  static_assert(kPieces > kThreads && kPieces <= 2 * kThreads);
  constexpr int kWords = kRowsPerBlock * kRowWords;
  constexpr int kWordsPer =
      kWordStaged ? (kWords + kThreads - 1) / kThreads : 1;
  constexpr int kHalfPieces = kRowsPerBlock * 8;
  constexpr int kHalfPer = kF16 ? kHalfPieces / kThreads : 1;
  const auto row_src = [&](int r_local) {
    return w_expert +
           std::size_t{row_live(r_local) ? matrix_row(r_local) : m - 1} *
               row_bytes;
  };
  const bool f_second = tid + kThreads < kPieces;
  const std::uint8_t* f_src0 = row_src(tid / 3) + (tid % 3) * 16;
  const int c1 = f_second ? tid + kThreads : tid;
  const std::uint8_t* f_src1 = row_src(c1 / 3) + (c1 % 3) * 16;
  uint4 f_data0;
  uint4 f_data1;
  std::uint32_t f_words[kWordsPer];
  // Binary16 stages as plain words: a uint4 array here stayed in scratch.
  std::uint32_t f_pieces[4 * kHalfPer];
  const auto word_of = [&](int i) {
    const int c = tid + i * kThreads;
    return c < kWords ? c : kWords - 1;
  };
  const auto fetch_weights = [&](int stage) {
    if constexpr (kF16) {
#pragma unroll
      for (int i = 0; i < kHalfPer; ++i) {
        const int c = tid + i * kThreads;
        const uint4 piece = *reinterpret_cast<const uint4*>(
            row_src(c / 8) + stage * kStageBytes + (c % 8) * 16);
        f_pieces[4 * i] = piece.x;
        f_pieces[4 * i + 1] = piece.y;
        f_pieces[4 * i + 2] = piece.z;
        f_pieces[4 * i + 3] = piece.w;
      }
    } else if constexpr (kWordStaged) {
#pragma unroll
      for (int i = 0; i < kWordsPer; ++i) {
        const int c = word_of(i);
        f_words[i] = *reinterpret_cast<const std::uint32_t*>(
            row_src(c / kRowWords) + stage * kStageBytes + (c % kRowWords) * 4);
      }
    } else {
      f_data0 = *reinterpret_cast<const uint4*>(f_src0 + stage * kStageBytes);
      f_data1 = *reinterpret_cast<const uint4*>(f_src1 + stage * kStageBytes);
    }
  };
  const auto commit_weights = [&] {
    if constexpr (kF16) {
#pragma unroll
      for (int i = 0; i < kHalfPer; ++i) {
        const int c = tid + i * kThreads;
        *reinterpret_cast<uint4*>(s_rows + (c / 8) * kStride + (c % 8) * 16) =
            make_uint4(f_pieces[4 * i], f_pieces[4 * i + 1],
                       f_pieces[4 * i + 2], f_pieces[4 * i + 3]);
      }
    } else if constexpr (kWordStaged) {
#pragma unroll
      for (int i = 0; i < kWordsPer; ++i) {
        const int c = tid + i * kThreads;
        if (c < kWords) {
          *reinterpret_cast<std::uint32_t*>(s_rows + (c / kRowWords) * kStride +
                                            (c % kRowWords) * 4) = f_words[i];
        }
      }
    } else {
      *reinterpret_cast<uint4*>(s_rows + (tid / 3) * kStride + (tid % 3) * 16) =
          f_data0;
      if (f_second) {
        *reinterpret_cast<uint4*>(s_rows + (c1 / 3) * kStride + (c1 % 3) * 16) =
            f_data1;
      }
    }
  };

  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / 8;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = source_row(t_local + t);
      if (src >= 0) {
        a_src[i] = src * static_cast<std::int32_t>(k) + (chunk % 8) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] = a_src[i] >= 0 ? *reinterpret_cast<const uint4*>(
                                      x + a_src[i] + stage * kDownStageK)
                                : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / 8) * kDownActStride +
                                  (chunk % 8) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const bool live = row_live(wave * 16 + sub);
  const int stages = static_cast<int>(k / kDownStageK);
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
    const std::uint8_t* row = s_rows + (wave * 16 + sub) * kStride;
    std::uint32_t words[kQ80Words];
    if constexpr (kQ8) {
      const auto* r4 = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = r4[i];
        words[4 * i] = v.x;
        words[4 * i + 1] = v.y;
        words[4 * i + 2] = v.z;
        words[4 * i + 3] = v.w;
      }
      words[16] = reinterpret_cast<const std::uint32_t*>(row)[16];
    }
    std::uint32_t q40_words[kQ40Words];
    if constexpr (F == ExpertFormat::kQ4_0) {
      const auto* r4 = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 v = r4[i];
        q40_words[4 * i] = v.x;
        q40_words[4 * i + 1] = v.y;
        q40_words[4 * i + 2] = v.z;
        q40_words[4 * i + 3] = v.w;
      }
      q40_words[8] = reinterpret_cast<const std::uint32_t*>(row)[8];
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      v16h a_lo;
      v16h a_hi;
      if constexpr (kF16) {
        // Padding rows (fetched from the last row) contribute zeros.
        using v8u =
            std::uint32_t __attribute__((ext_vector_type(8), aligned(16)));
        const auto* v = reinterpret_cast<const v8u*>(row + 64 * s);
        const v8u zero = {};
        a_lo = __builtin_bit_cast(v16h, live ? v[0] : zero);
        a_hi = __builtin_bit_cast(v16h, live ? v[1] : zero);
      } else if constexpr (kQ8) {
        DecodeQ80(words, s, live, &a_lo, &a_hi);
      } else if constexpr (F == ExpertFormat::kQ4_0) {
        DecodeQ40(q40_words, s, live, &a_lo, &a_hi);
      } else {
        DecodeQ51(row + 24 * s, live, &a_lo, &a_hi);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kDownActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(a_lo, b_lo, acc[j]);
          acc[j] = Wmma(a_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
  }
  if constexpr (kGeGlu) {
    // Up waves hand their binary16 rows to the gate waves through LDS, as in
    // RoutedHalfKQuantKernel.
    static_assert(4 * kTokTiles * 8 * kWave * 2 <=
                  static_cast<int>(sizeof(s_act_bytes)));
    if (wave >= 4) {
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_act[(((wave - 4) * kTokTiles + j) * 8 + i) * kWave + lane] =
              SaturatedHalf(acc[j][i]);
        }
      }
    }
    __syncthreads();
    if (wave >= 4) {
      return;
    }
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      const int t = t_local + j * 16 + sub;
      if (t >= bucket_rows) {
        continue;
      }
      const std::int32_t dst = dest_row(t);
      if (dst < 0) {
        continue;
      }
      const std::size_t base =
          std::size_t{static_cast<std::uint32_t>(dst)} * half_m;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t r =
            row_block * (kRowsPerBlock / 2) +
            static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
        if (r < half_m) {
          const __half up =
              s_act[((wave * kTokTiles + j) * 8 + i) * kWave + lane];
          const float v = GeGluValue(__half2float(SaturatedHalf(acc[j][i])),
                                     __half2float(up));
          out_half[base + r] = SaturatedHalf(v);
        }
      }
    }
    return;
  }
  // Epilogue per token tile: [token][row] through LDS, then 16 threads per
  // token store its 128 rows contiguously.
  auto* s_out = reinterpret_cast<float*>(s_act_bytes);
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tiles) {
      break;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_out[sub * kOutStride + wave * 16 + 2 * i + half] = acc[j][i];
    }
    __syncthreads();
    const int tt = tid / 16;
    const int t = t_local + j * 16 + tt;
    const std::int32_t dst = t < bucket_rows ? dest_row(t) : -1;
    if (dst >= 0) {
      const int r = (tid % 16) * 8;
      const float* src = s_out + tt * kOutStride + r;
      float* o = out + std::size_t{static_cast<std::uint32_t>(dst)} * m + row0 +
                 static_cast<std::uint32_t>(r);
      if (row0 + static_cast<std::uint32_t>(r) + 8 <= m) {
        reinterpret_cast<float4*>(o)[0] = *reinterpret_cast<const float4*>(src);
        reinterpret_cast<float4*>(o)[1] =
            *reinterpret_cast<const float4*>(src + 4);
      } else {
        for (int q = 0; q < 8; ++q) {
          if (row0 + static_cast<std::uint32_t>(r + q) < m) {
            o[q] = src[q];
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

bool LaunchDenseHalfGemm(ExpertFormat format, const void* w, const void* x,
                         float* out, void* out_half, std::uint32_t rows,
                         std::uint32_t m, std::uint32_t k, hipStream_t stream,
                         bool geglu) {
  const bool q40 = format == ExpertFormat::kQ4_0;
  if (rows == 0 || k % (q40 ? kDownStageK : 256) != 0 ||
      (out == nullptr) == (out_half == nullptr) ||
      (geglu && (out_half == nullptr || m % 2 != 0)) ||
      (q40 && out_half != nullptr && !geglu)) {
    return false;
  }
  constexpr int kTile = 96;
  // Token tiles per row-block sweep (cold 2048-row Gemma 31B shapes: 2 or 4
  // beat both one and every tile).
  constexpr std::uint32_t kGroup = 2;
  const std::uint32_t token_tiles = (rows + kTile - 1) / kTile;
  const std::uint32_t row_blocks =
      geglu ? (m / 2 + kRowsPerBlock / 2 - 1) / (kRowsPerBlock / 2)
            : (m + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 grid(token_tiles * row_blocks);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x);
  auto* oh = static_cast<__half*>(out_half);
  const auto launch = [&]<ExpertFormat F>() {
    if (geglu) {
      RoutedHalfKQuantKernel<F, kTile, true, true>
          <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                          nullptr, out, oh, m, k, rows, kGroup);
    } else {
      RoutedHalfKQuantKernel<F, kTile, false, true>
          <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                          nullptr, out, oh, m, k, rows, kGroup);
    }
  };
  switch (format) {
    case ExpertFormat::kQ4_K:
      launch.template operator()<ExpertFormat::kQ4_K>();
      return true;
    case ExpertFormat::kQ5_K:
      launch.template operator()<ExpertFormat::kQ5_K>();
      return true;
    case ExpertFormat::kQ6_K:
      launch.template operator()<ExpertFormat::kQ6_K>();
      return true;
    case ExpertFormat::kQ4_0:
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kQ4_0, kTile, true, true>
            <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, oh, m, k, rows,
                                            kGroup);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kQ4_0, kTile, false, true>
            <<<grid, kThreads, 0, stream>>>(wb, xh, nullptr, nullptr, nullptr,
                                            nullptr, out, nullptr, m, k, rows,
                                            kGroup);
      }
      return true;
    default:
      return false;
  }
}

bool LaunchRoutedHalfGemm(ExpertFormat format, const void* w, const void* x,
                          const std::int32_t* tiles, std::uint32_t n_tiles,
                          std::uint32_t tile_rows,
                          const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, float* out,
                          void* out_half, std::uint32_t m, std::uint32_t k,
                          hipStream_t stream, bool geglu) {
  if (tile_rows != 96 || n_tiles == 0 ||
      (out == nullptr) == (out_half == nullptr) ||
      (format != ExpertFormat::kQ5_1 && format != ExpertFormat::kQ8_0 &&
       format != ExpertFormat::kF16 && k % 256 != 0)) {
    return false;
  }
  if (geglu &&
      (out_half == nullptr || m % 2 != 0 || format == ExpertFormat::kQ5_1)) {
    return false;
  }
  const dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock, n_tiles);
  const dim3 geglu_grid((m / 2 + kRowsPerBlock / 2 - 1) / (kRowsPerBlock / 2),
                        n_tiles);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x);
  auto* oh = static_cast<__half*>(out_half);
  switch (format) {
    case ExpertFormat::kQ4_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ4_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ4_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ5_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ5_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ5_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ6_K:
      if (geglu) {
        RoutedHalfKQuantKernel<ExpertFormat::kQ6_K, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(
                wb, xh, tiles, pad_bounds, rows_in, rows_out, out, oh, m, k);
      } else {
        RoutedHalfKQuantKernel<ExpertFormat::kQ6_K, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, oh, m, k);
      }
      return true;
    case ExpertFormat::kQ5_1:
      if (out == nullptr || k % kDownStageK != 0) {
        return false;
      }
      RoutedHalfDownKernel<ExpertFormat::kQ5_1, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, nullptr, m, k);
      return true;
    case ExpertFormat::kQ8_0:
      if (k % kDownStageK != 0 || (out == nullptr) != geglu) {
        return false;
      }
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kQ8_0, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds,
                                                  rows_in, rows_out, nullptr,
                                                  oh, m, k);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kQ8_0, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, nullptr, m, k);
      }
      return true;
    case ExpertFormat::kF16:
      if (k % kDownStageK != 0 || (out == nullptr) != geglu) {
        return false;
      }
      if (geglu) {
        RoutedHalfDownKernel<ExpertFormat::kF16, 96, true>
            <<<geglu_grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds,
                                                  rows_in, rows_out, nullptr,
                                                  oh, m, k);
      } else {
        RoutedHalfDownKernel<ExpertFormat::kF16, 96>
            <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                            rows_out, out, nullptr, m, k);
      }
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::models::gemma4::rocm

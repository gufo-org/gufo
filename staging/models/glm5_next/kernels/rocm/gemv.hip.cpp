#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"

#include <cstddef>
#include <cstdint>

#include "staging/models/glm5_next/kernels/rocm/iq_tables.hip.hpp"

// Decode semantics are transcribed element for element from
// src/core/quant/ggml_dequant.cpp (the fork CPU contract for each GGML
// block): IQ2_XXS (66 B), IQ3_XXS (98 B), IQ1_M (56 B), IQ4_XS (136 B, 256
// elements), Q2_K (84 B), Q3_K (110 B) and the Q*_K super-blocks shared with
// the scalar oracle. The IQ block structs stay raw-byte: every field sits at
// the GGUF offset the core decoder uses, and the helpers below load
// little-endian so misaligned reads never depend on host packing.
namespace gufo::models::glm5_next::rocm {
namespace {

// IQ format super-block sizes (bytes per 256 elements).
constexpr std::size_t kIq2XxsBytes = 66;
constexpr std::size_t kIq3XxsBytes = 98;
constexpr std::size_t kIq1MBytes = 56;
constexpr std::size_t kIq4XsBytes = 136;
constexpr std::size_t kQ2KBytes = 84;
constexpr std::size_t kQ3KBytes = 110;

// Q5_K / Q6_K / Q8_0 blocks and the Q4_K helpers are byte-compatible with the
// scalar oracle (src/core/quant/ggml_dequant.hpp block_iq*_*/block_q*_*).
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  __half d;
};
static_assert(sizeof(Q6KBlock) == 210, "block_q6_K must be 210 bytes");

// Non-linear 4-bit codebook shared by IQ4_XS (kValuesIq4Nl in
// src/core/quant/ggml_dequant.hpp).
__device__ __constant__ std::int8_t kIq4Vals[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ std::uint16_t U16At(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) | static_cast<std::uint16_t>(p[1])
                                                << 8;
}

__device__ __forceinline__ std::uint32_t U32At(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         static_cast<std::uint32_t>(p[1]) << 8 |
         static_cast<std::uint32_t>(p[2]) << 16 |
         static_cast<std::uint32_t>(p[3]) << 24;
}

__device__ __forceinline__ float F16At(const std::uint8_t* p) {
  __half_raw raw;
  raw.x = U16At(p);
  return __half2float(*reinterpret_cast<const __half*>(&raw));
}

// Byte j (little-endian) of a packed grid entry.
__device__ __forceinline__ int GridByte(std::uint64_t grid, int j) {
  return static_cast<int>((grid >> (8 * j)) & 0xFFU);
}

__device__ __forceinline__ float IqSign(std::uint8_t signs, int j) {
  return (signs & static_cast<std::uint8_t>(1U << j)) != 0 ? -1.0F : 1.0F;
}

/// Element j of iq1s_grid entry `index`, from the GPU packing (value + 1);
/// the core formula Iq1sGrid in ggml_dequant.cpp.
__device__ __forceinline__ float Iq1sValue(std::uint32_t index, int j) {
  const int nibble = 2 * (j & 3) + (j >> 2);
  return static_cast<float>(
      static_cast<int>((kIq1sGridGpu[index] >> (4 * nibble)) & 0x0FU) - 1);
}

// ---------------------------------------------------------------------------
// IQ element decoders. Each lane of a 32-lane wave decodes its own aligned
// chunk of one 256-element super-block and returns the eight F32 values, so
// the same helper serves the dot, multi-row, grouped and embed tiers.
// ---------------------------------------------------------------------------

// IQ2_XXS (66 B): lane L covers elements [8L, 8L+8) = the l-th 8-element grid
// group of the ib32 = L/4, l = L%32th 32-element group. d is the fp16 word at
// byte 0; each 32-element group carries four grid bytes and an aux word whose
// top four bits extend the per-group scale (db = d*(0.5+aux>>28)/4) and
// whose seven-bit fields index the sign table (core DequantizeIQ2_XXS).
__device__ __forceinline__ void DecodeIq2Xxs8(const std::uint8_t* blk,
                                              std::uint32_t lane, float* v) {
  const std::uint32_t ib32 = lane >> 2;
  const std::uint32_t l = lane & 3U;
  const std::uint8_t* q = blk + 2 + 8 * ib32;
  const float d = F16At(blk);
  const std::uint32_t aux1 = U32At(q + 4);
  const float db = d * (0.5F + static_cast<float>(aux1 >> 28)) * 0.25F;
  const std::uint64_t grid = kIq2xxsGrid[q[l]];
  const std::uint8_t signs = kKsignsIq2xs[(aux1 >> (7 * l)) & 127U];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = db * static_cast<float>(GridByte(grid, j)) * IqSign(signs, j);
  }
}

// IQ3_XXS (98 B): lane L covers elements [8L, 8L+8) as two 4-element grid
// lookups (bytes 0..3 of kIq3xxsGrid[qs[2l]] and [qs[2l+1]]); the aux word at
// byte 66 + 4*ib32 carries the four 7-bit sign groups and the 4-bit scale
// (db = d*(0.5+aux>>28)/2; core DequantizeIQ3_XXS).
__device__ __forceinline__ void DecodeIq3Xxs8(const std::uint8_t* blk,
                                              std::uint32_t lane, float* v) {
  const std::uint32_t ib32 = lane >> 2;
  const std::uint32_t l = lane & 3U;
  const std::uint8_t* qs = blk + 2 + 8 * ib32;
  const float d = F16At(blk);
  const std::uint32_t aux = U32At(blk + 66 + 4 * ib32);
  const float db = d * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
  const std::uint8_t signs = kKsignsIq2xs[(aux >> (7 * l)) & 127U];
  const std::uint32_t g1 = kIq3xxsGrid[qs[2 * l]];
  const std::uint32_t g2 = kIq3xxsGrid[qs[2 * l + 1]];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = db * static_cast<float>(g1 >> (8 * j) & 0xFFU) * IqSign(signs, j);
    v[j + 4] =
        db * static_cast<float>(g2 >> (8 * j) & 0xFFU) * IqSign(signs, j + 4);
  }
}

// IQ1_M (56 B): lane L covers elements [8L, 8L+8) = group l of super-block
// ib = L/4. qs[l] is the low grid byte; the two qh BYTES at byte 32 + 2*ib
// add bits 8..10 and the delta sign per group (b0 -> l=0/1, b1 -> l=2/3;
// grid bits [2:0]/[6:4], sign bit 3/7). The four 16-bit scale words at byte
// 48 hold twelve 3-bit per-group scales plus the fp16 d in their top nibbles
// (core DequantizeIQ1_M).
__device__ __forceinline__ void DecodeIq1M8(const std::uint8_t* blk,
                                            std::uint32_t lane, float* v) {
  const std::uint32_t ib = lane >> 2;
  const std::uint32_t l = lane & 3U;
  const std::uint8_t qs = blk[4U * ib + l];
  const std::uint8_t qh = blk[32U + 2U * ib + (l >> 1)];
  const std::uint16_t sc[4] = {U16At(blk + 48), U16At(blk + 50),
                               U16At(blk + 52), U16At(blk + 54)};
  const std::uint16_t packed =
      static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0U) |
                                 ((sc[2] >> 4) & 0x0F00U) | (sc[3] & 0xF000U));
  __half_raw d_raw;
  d_raw.x = packed;
  const float d = __half2float(*reinterpret_cast<const __half*>(&d_raw));
  const float dl1 =
      d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
  const float dl2 =
      d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
  const std::uint32_t idx =
      static_cast<std::uint32_t>(qs) |
      ((static_cast<std::uint32_t>(qh) << ((l & 1U) ? 4U : 8U)) & 0x700U);
  const float delta =
      ((qh >> ((l & 1U) ? 7U : 3U)) & 1U) != 0 ? -0.125F : 0.125F;
  const float dl = l < 2 ? dl1 : dl2;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = dl * (Iq1sValue(idx, j) + delta);
  }
}

// IQ4_XS (136 bytes per 256 elements): lane L covers elements [8L, 8L+8),
// inside sub-block ib = L/4 (four 32-element pairs across the 128 qs bytes).
// The 4-bit code indexes the shared iq4 codebook and scales by (ls-32) with
// the sub-block scale split between scales_l nibbles and scales_h bits
// (core Iq4XsValue with d at byte 0, scales_h at 2, scales_l at 4).
__device__ __forceinline__ void DecodeIq4Xs8(const std::uint8_t* blk,
                                             std::uint32_t lane, float* v) {
  const std::uint32_t ib = lane >> 2;
  const std::uint32_t within = (8 * lane) & 31U;
  const std::uint32_t lane16 = within & 15U;
  const bool high = within >= 16;
  const float d = F16At(blk);
  const int ls = static_cast<int>((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0x0FU) |
                 static_cast<int>(((U16At(blk + 2) >> (2 * ib)) & 0x03U) << 4U);
  const float ds = d * static_cast<float>(ls - 32);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    // Consecutive elements take consecutive qs bytes (the two nibbles of a
    // byte are elements 16 apart inside the 32-element sub-block).
    const std::uint8_t packed =
        blk[8U + ib * 16U + lane16 + static_cast<std::uint32_t>(j)];
    const int code = high ? (packed >> 4) : (packed & 0x0F);
    v[j] = ds * static_cast<float>(kIq4Vals[code]);
  }
}

// Q2_K (84 B): d at 80, min at 82, scales[16] at 0, 2-bit codes in 64 bytes
// at 16. Lane L covers elements [8L, 8L+8) of the j = L/4, half = (L/2)%2
// 16-element sub-block of the (L/16)-th 128-element half (core DequantizeQ2_K
// iterates j-major, half-minor within each half).
__device__ __forceinline__ void DecodeQ2K8(const std::uint8_t* blk,
                                           std::uint32_t lane, float* v) {
  const std::uint32_t n = lane >> 4;  // 128-element half (q += 32)
  const std::uint32_t j = (lane >> 2) & 3U;
  const std::uint32_t half = (lane >> 1) & 1U;
  const std::uint32_t l0 = (lane & 1U) * 8U;
  const float d = F16At(blk + 80);
  const float mn = F16At(blk + 82);
  const std::uint8_t sc = blk[(n * 8) + j * 2 + half];
  const float dl = d * static_cast<float>(sc & 0x0FU);
  const float ml = mn * static_cast<float>(sc >> 4);
  const std::uint8_t* q = blk + 16 + n * 32 + 16 * half + l0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = dl * static_cast<float>((q[i] >> (2 * j)) & 3U) - ml;
  }
}

// Q3_K (110 B): hmask[32] at 0, qs[128 at 2 bits] -> qs[64] at 32, scales[12]
// at 96, fp16 d at 108 (core block_q3_K). Lane L covers elements [8L, 8L+8):
// half = L>>4 sub-half ... see core Q3Value: index -> half = index/128,
// within = index%128, scale_pair = within/32, lane = within%32; the 4-bit
// quant is the 2-bit code shifted by 2*scale_pair minus 4 unless the hmask
// bit ((half*4)+scale_pair) for that lane is set (set = -0 offset, clear = -4;
// core adds 4 when the bit is NOT set: quant = low - (bit ? 0 : 4)).
__device__ __forceinline__ int Q3Scale(const std::uint8_t* packed,
                                       std::uint32_t index) {
  const std::uint32_t low = (index < 8U) ? (packed[index] & 0x0FU)
                                         : ((packed[index - 8] >> 4U) & 0x0FU);
  const std::uint32_t high =
      (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
  return static_cast<int>(low | (high << 4U)) - 32;
}

__device__ __forceinline__ void DecodeQ3K8(const std::uint8_t* blk,
                                           std::uint32_t lane, float* v) {
  const std::uint32_t base = 8 * lane;
  const std::uint32_t half = base >> 7;
  const std::uint32_t within = base & 127U;
  const std::uint32_t scale_pair = within >> 5;
  const std::uint32_t lane32 = within & 31U;
  const std::uint32_t shift = 2 * scale_pair;
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>(1U << ((half * 4U) + scale_pair));
  const float d = F16At(blk + 108);
  const int scale =
      Q3Scale(blk + 96, (half * 8U) + (scale_pair * 2U) + (lane32 >> 4));
  const float ds = d * static_cast<float>(scale);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Each element carries its own hmask/qs byte at lane32 + i (the 2-bit
    // codes of a byte belong to elements 32 apart, not neighbours).
    const std::uint8_t hb = blk[lane32 + static_cast<std::uint32_t>(i)];
    const std::uint8_t qb =
        blk[32U + half * 32U + lane32 + static_cast<std::uint32_t>(i)];
    const int low = static_cast<int>((qb >> shift) & 3U);
    const int quant = low - (((hb & high_mask) != 0) ? 0 : 4);
    v[i] = ds * static_cast<float>(quant);
  }
}

// One block of four waves handles four consecutive output rows: wave w owns
// row blockIdx.x*4+w, walks the row once and reduces within the wave.

__global__ void GemvQ8_0(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const Q8_0Block& blk = row[i >> 5];
    acc += __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]) * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Q4_K walk: lane L takes the element at offset L of each of the eight
// 32-element groups (e = L + 32 j), so the nibble reads coalesce across the
// warp. value = d*scale_j*q - dmin*min_j with the nibble low for even j and
// high for odd j (core Q4Value).
__device__ __forceinline__ void Q4ScaleMinFromWords(std::uint32_t lo,
                                                    std::uint32_t mid,
                                                    std::uint32_t hi, int index,
                                                    int& scale, int& minimum) {
  if (index < 4) {
    scale = static_cast<int>((lo >> (8 * index)) & 0x3FU);
    minimum = static_cast<int>((mid >> (8 * index)) & 0x3FU);
  } else {
    const std::uint32_t pj4 = (hi >> (8 * (index - 4))) & 0xFFU;
    scale = static_cast<int>(pj4 & 0xFU) |
            (static_cast<int>(((lo >> (8 * (index - 4))) & 0xFFU) >> 6) << 4);
    minimum =
        static_cast<int>(pj4 >> 4) |
        (static_cast<int>(((mid >> (8 * (index - 4))) & 0xFFU) >> 6) << 4);
  }
}

__global__ void GemvQ4K(const Q4KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q4KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const float* xrow = x + b * 256 + lane;
    const std::uint32_t* sp =
        reinterpret_cast<const std::uint32_t*>(blk.scales);
    int scv[8];
    float dmn[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int s0 = 0;
      int m0 = 0;
      Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, s0, m0);
      scv[j] = s0;
      dmn[j] = dmin * static_cast<float>(m0);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
      const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
      acc += ((d * static_cast<float>(scv[j] * quant)) - dmn[j]) * xrow[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Q5_K walk: a 5-bit code is the Q4_K nibble plus a fifth bit from qh[lane]
// at position j; the sub-block scale index is j (core Q5Value).
__global__ void GemvQ5K(const Q5KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q5KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q5KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint8_t qhb = blk.qh[lane];
    const float* xrow = x + b * 256 + lane;
    const std::uint32_t* sp =
        reinterpret_cast<const std::uint32_t*>(blk.scales);
    int scv[8];
    float dmn[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int s0 = 0;
      int m0 = 0;
      Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, s0, m0);
      scv[j] = s0;
      dmn[j] = dmin * static_cast<float>(m0);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t qb = blk.qs[(j >> 1) * 32 + lane];
      const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
      const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
      acc += ((d * static_cast<float>(scv[j] * quant)) - dmn[j]) * xrow[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Q6_K walk: lane L covers the eight (half, segment) groups at element
// half*128 + segment*32 + L; the 6-bit code is the ql nibble plus two top
// bits from qh[half*32+lane], minus 32, times the int8 sub-block scale
// (core Q6Value).
__global__ void GemvQ6K(const Q6KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q6KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q6KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float* xrow = x + b * 256 + lane;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::uint8_t qhb = blk.qh[half * 32 + lane];
      const std::uint8_t q0 = blk.ql[half * 64 + lane];
      const std::uint8_t q1 = blk.ql[half * 64 + 32 + lane];
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        const std::uint8_t byte = (seg & 1) ? q1 : q0;
        const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
        const int high = (qhb >> (2 * seg)) & 0x3;
        const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
        const int scale = blk.scales[half * 8 + (lane >> 4) + seg * 2];
        acc += d * static_cast<float>(scale) * static_cast<float>(quant) *
               xrow[128 * half + 32 * seg];
      }
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvF32(const float* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const float* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += row[i] * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvF16(const __half* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const __half* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += static_cast<float>(row[i]) * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// 8-element-per-lane IQ/low-bit walker: lane L owns elements [8L, 8L+8) of
// each 256-element super-block, decodes them once and dots against x. Used
// unchanged by Q2_K, Q3_K and the four IQ types; DecodeFn is one of the
// DecodeXxx8 helpers above.
template<void (*DecodeFn)(const std::uint8_t*, std::uint32_t, float*)>
__global__ void GemvSuper256(const std::uint8_t* __restrict__ w,
                             std::size_t block_bytes,
                             const float* __restrict__ x,
                             float* __restrict__ out, std::uint32_t rows,
                             std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const std::uint8_t* __restrict__ row =
      w + static_cast<std::size_t>(r) * nblocks * block_bytes;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    float v[8];
    DecodeFn(row + static_cast<std::size_t>(b) * block_bytes, lane, v);
    const float* xrow = x + static_cast<std::size_t>(b) * 256 + 8 * lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc += v[j] * xrow[j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Dequantize a single row of a quantized matrix (the token-embedding
// lookup). One block per 256-element super-block for the block formats.
__global__ void EmbedRowQ8_0(const Q8_0Block* __restrict__ w,
                             float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  out[b * 32 + lane] = d * static_cast<float>(w[b].qs[lane]);
}

__global__ void EmbedRowQ4K(const Q4KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  const float dmin = __half2float(w[b].dmin);
  const std::uint32_t* sp = reinterpret_cast<const std::uint32_t*>(w[b].scales);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint8_t packed = w[b].qs[(j >> 1) * 32 + lane];
    const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
    int sc = 0;
    int mn = 0;
    Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, sc, mn);
    out[b * 256 + j * 32 + lane] =
        d * static_cast<float>(sc) * static_cast<float>(quant) -
        dmin * static_cast<float>(mn);
  }
}

__global__ void EmbedRowQ5K(const Q5KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  const float dmin = __half2float(w[b].dmin);
  const std::uint8_t qhb = w[b].qh[lane];
  const std::uint32_t* sp = reinterpret_cast<const std::uint32_t*>(w[b].scales);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint8_t qb = w[b].qs[(j >> 1) * 32 + lane];
    const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
    const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
    int sc = 0;
    int mn = 0;
    Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, sc, mn);
    out[b * 256 + j * 32 + lane] =
        d * static_cast<float>(sc) * static_cast<float>(quant) -
        dmin * static_cast<float>(mn);
  }
}

__global__ void EmbedRowQ6K(const Q6KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t qhb = w[b].qh[half * 32 + lane];
    const std::uint8_t q0 = w[b].ql[half * 64 + lane];
    const std::uint8_t q1 = w[b].ql[half * 64 + 32 + lane];
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      const std::uint8_t byte = (seg & 1) ? q1 : q0;
      const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
      const int high = (qhb >> (2 * seg)) & 0x3;
      const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
      const int scale = w[b].scales[half * 8 + (lane >> 4) + seg * 2];
      out[b * 256 + half * 128 + seg * 32 + lane] =
          d * static_cast<float>(scale) * static_cast<float>(quant);
    }
  }
}

__global__ void EmbedRowF32(const float* __restrict__ w,
                            float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = w[idx];
  }
}

__global__ void EmbedRowF16(const __half* __restrict__ w,
                            float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = static_cast<float>(w[idx]);
  }
}

// Embed tier for the 8-per-lane super-block formats: write the decoded chunk.
template<void (*DecodeFn)(const std::uint8_t*, std::uint32_t, float*)>
__global__ void EmbedRowSuper256(const std::uint8_t* __restrict__ w,
                                 std::size_t block_bytes,
                                 float* __restrict__ out) {
  const std::uint32_t lane = threadIdx.x;
  float v[8];
  DecodeFn(reinterpret_cast<const std::uint8_t*>(w) +
               static_cast<std::size_t>(blockIdx.x) * block_bytes,
           lane, v);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[static_cast<std::size_t>(blockIdx.x) * 256 + 8 * lane + j] = v[j];
  }
}

// F32 -> binary16 retype with overflow counting (router upload).
__global__ void F32ToHalfKernel(const float* __restrict__ src,
                                __half* __restrict__ dst, std::size_t count,
                                std::uint32_t* __restrict__ overflow) {
  std::uint32_t outside = 0;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const __half h = __float2half_rn(src[i]);
    outside += __hisinf(h) != 0 || __hisnan(h) ? 1U : 0U;
    dst[i] = h;
  }
  if (outside != 0) {
    atomicAdd(overflow, outside);
  }
}

// ---------------------------------------------------------------------------
// Multi-row (speculative verify) tier. One wave owns an output row, walks the
// encoded row once, decodes each block into v[] and dots it against all
// `Rows` activation rows before the next decode. Each per-token accumulator
// runs the identical j-sequence as Gemv, so results are bit-identical.
// ---------------------------------------------------------------------------

// Grouped heads: when group_rows is non-zero the rows are `heads` contiguous
// stacked slices (attention heads); wave r uses activation base x plus
// (r / group_rows) * group_advance floats. Every per-row dot is the same
// address sequence as the matching per-head call, so results are bit-
// identical to the host loop.

template<std::uint32_t Rows>
__global__ void GemvRowsQ8_0(const Q8_0Block* __restrict__ w,
                             const float* __restrict__ x, std::uint32_t xs,
                             float* __restrict__ out, std::uint32_t os,
                             std::uint32_t rows, std::uint32_t cols,
                             std::uint32_t group_rows,
                             std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const Q8_0Block& blk = row[i >> 5];
    const float b = __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += b * x[static_cast<std::size_t>(o) * xs + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

// Tiled Q8_0 tier: one wave accumulates four weight rows (kRpt) and grid.y
// enumerates independent 16-token slices, turning the whole chunk into one
// launch. The activation load stream x[o*xs+i] is identical for every row,
// so each load feeds kRpt FMAs while every per-row accumulation order and
// shuffle tree stay exactly as in GemvRowsQ8_0; results are bit-identical.
// A wave's four rows never straddle a group boundary because group_rows is
// a multiple of four whenever grouped.
template<std::uint32_t Rows>
__global__ void GemvRowsQ8_0Tiled(const Q8_0Block* __restrict__ w,
                                  const float* __restrict__ x,
                                  std::uint32_t xs, float* __restrict__ out,
                                  std::uint32_t os, std::uint32_t rows,
                                  std::uint32_t cols, std::uint32_t group_rows,
                                  std::uint32_t group_advance) {
  constexpr std::uint32_t kRpt = 4U;
  x += static_cast<std::size_t>(blockIdx.y) * Rows * xs;
  out += static_cast<std::size_t>(blockIdx.y) * Rows * os;
  const std::uint32_t r0 = (blockIdx.x * 4U + (threadIdx.x >> 5)) * kRpt;
  if (r0 >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r0 / group_rows) * group_advance;
  }
  const std::uint32_t stride32 = cols / 32U;
  const Q8_0Block* __restrict__ row[kRpt];
#pragma unroll
  for (std::uint32_t u = 0; u < kRpt; ++u) {
    row[u] = w + static_cast<std::size_t>(r0 + u) * stride32;
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[kRpt][Rows] = {};
#pragma unroll 2
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    const std::uint32_t blk_idx = i >> 5;
    float wv[kRpt];
#pragma unroll
    for (std::uint32_t u = 0; u < kRpt; ++u) {
      const Q8_0Block& blk = row[u][blk_idx];
      wv[u] = __half2float(blk.d) * static_cast<float>(blk.qs[i & 31U]);
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const float xv = x[static_cast<std::size_t>(o) * xs + i];
#pragma unroll
      for (std::uint32_t u = 0; u < kRpt; ++u) {
        acc[u][o] += wv[u] * xv;
      }
    }
  }
#pragma unroll
  for (std::uint32_t u = 0; u < kRpt; ++u) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[u][o] = WarpReduceSum(acc[u][o]);
    }
    if (lane == 0) {
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        out[static_cast<std::size_t>(o) * os + r0 + u] = acc[u][o];
      }
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvRowsQ4K(const Q4KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                            std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t group_rows,
                            std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q4KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint32_t* sp =
        reinterpret_cast<const std::uint32_t*>(blk.scales);
    float wj[8];
    float dmn[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int s0 = 0;
      int m0 = 0;
      Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, s0, m0);
      wj[j] = d * static_cast<float>(s0);
      dmn[j] = dmin * static_cast<float>(m0);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
      const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
      const float wv = (wj[j] * static_cast<float>(quant)) - dmn[j];
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        acc[o] +=
            wv * x[static_cast<std::size_t>(o) * xs + b * 256 + j * 32 + lane];
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvRowsQ5K(const Q5KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                            std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t group_rows,
                            std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q5KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q5KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint8_t qhb = blk.qh[lane];
    const std::uint32_t* sp =
        reinterpret_cast<const std::uint32_t*>(blk.scales);
    float wj[8];
    float dmn[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int s0 = 0;
      int m0 = 0;
      Q4ScaleMinFromWords(sp[0], sp[1], sp[2], j, s0, m0);
      wj[j] = d * static_cast<float>(s0);
      dmn[j] = dmin * static_cast<float>(m0);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t qb = blk.qs[(j >> 1) * 32 + lane];
      const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
      const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
      const float wv = (wj[j] * static_cast<float>(quant)) - dmn[j];
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        acc[o] +=
            wv * x[static_cast<std::size_t>(o) * xs + b * 256 + j * 32 + lane];
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvRowsQ6K(const Q6KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                            std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t group_rows,
                            std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q6KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q6KBlock& blk = row[b];
    const float d = __half2float(blk.d);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::uint8_t qhb = blk.qh[half * 32 + lane];
      const std::uint8_t q0 = blk.ql[half * 64 + lane];
      const std::uint8_t q1 = blk.ql[half * 64 + 32 + lane];
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        const std::uint8_t byte = (seg & 1) ? q1 : q0;
        const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
        const int high = (qhb >> (2 * seg)) & 0x3;
        const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
        const int scale = blk.scales[half * 8 + (lane >> 4) + seg * 2];
        const float wv =
            d * static_cast<float>(scale) * static_cast<float>(quant);
#pragma unroll
        for (std::uint32_t o = 0; o < Rows; ++o) {
          acc[o] += wv * x[static_cast<std::size_t>(o) * xs + b * 256 +
                           128 * half + 32 * seg + lane];
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvRowsF32(const float* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                            std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t group_rows,
                            std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const float* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    const float wv = row[i];
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += wv * x[static_cast<std::size_t>(o) * xs + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvRowsF16(const __half* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t xs,
                            float* __restrict__ out, std::uint32_t os,
                            std::uint32_t rows, std::uint32_t cols,
                            std::uint32_t group_rows,
                            std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const __half* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    const float wv = static_cast<float>(row[i]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += wv * x[static_cast<std::size_t>(o) * xs + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

template<std::uint32_t Rows,
         void (*DecodeFn)(const std::uint8_t*, std::uint32_t, float*)>
__global__ void GemvRowsSuper256(const std::uint8_t* __restrict__ w,
                                 std::size_t block_bytes,
                                 const float* __restrict__ x, std::uint32_t xs,
                                 float* __restrict__ out, std::uint32_t os,
                                 std::uint32_t rows, std::uint32_t cols,
                                 std::uint32_t group_rows,
                                 std::uint32_t group_advance) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  if (group_rows != 0U) {
    x += static_cast<std::size_t>(r / group_rows) * group_advance;
  }
  const std::uint32_t nblocks = cols / 256;
  const std::uint8_t* __restrict__ row =
      w + static_cast<std::size_t>(r) * nblocks * block_bytes;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    float v[8];
    DecodeFn(row + static_cast<std::size_t>(b) * block_bytes, lane, v);
    const float* xrow = x + static_cast<std::size_t>(b) * 256 + 8 * lane;
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[o] += v[j] * xrow[static_cast<std::size_t>(o) * xs + j];
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * os + r] = acc[o];
    }
  }
}

// ---------------------------------------------------------------------------
// Grouped decode tier: one wave per (slot, row), expert ids read on device.
// ---------------------------------------------------------------------------

template<void (*DecodeFn)(const std::uint8_t*, std::uint32_t, float*)>
__global__ void GemvGroupedSuper256(
    const std::uint8_t* __restrict__ w, std::size_t block_bytes,
    std::size_t expert_stride, const std::int32_t* __restrict__ ids,
    std::uint32_t used, std::uint32_t slots_per_row, std::uint32_t rows,
    std::uint32_t cols, const float* __restrict__ x, std::uint32_t xs,
    float* __restrict__ out) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t pairs = used * rows;
  for (std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5); pair < pairs;
       pair += gridDim.x * 4U) {
    const std::uint32_t s = pair / rows;
    const std::uint32_t r = pair - s * rows;
    const std::uint32_t nblocks = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(ids[s]) * expert_stride +
        static_cast<std::size_t>(r) * nblocks * block_bytes;
    const float* xr = x + static_cast<std::size_t>(s / slots_per_row) * xs;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nblocks; ++b) {
      float v[8];
      DecodeFn(row + static_cast<std::size_t>(b) * block_bytes, lane, v);
      const float* xrow = xr + static_cast<std::size_t>(b) * 256 + 8 * lane;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc += v[j] * xrow[j];
      }
    }
    acc = WarpReduceSum(acc);
    if (lane == 0) {
      out[static_cast<std::size_t>(s) * rows + r] = acc;
    }
  }
}

__global__ void GemvGroupedF32(const float* __restrict__ w,
                               std::size_t expert_stride,
                               const std::int32_t* __restrict__ ids,
                               std::uint32_t used, std::uint32_t slots_per_row,
                               std::uint32_t rows, std::uint32_t cols,
                               const float* __restrict__ x, std::uint32_t xs,
                               float* __restrict__ out) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t pairs = used * rows;
  const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  for (std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5); pair < pairs;
       pair += gridDim.x * 4U) {
    const std::uint32_t s = pair / rows;
    const std::uint32_t r = pair - s * rows;
    const float* row = reinterpret_cast<const float*>(
        reinterpret_cast<const std::uint8_t*>(w) +
        static_cast<std::size_t>(ids[s]) * expert_stride +
        static_cast<std::size_t>(r) * row_bytes);
    const float* xr = x + static_cast<std::size_t>(s / slots_per_row) * xs;
    float acc = 0.0f;
#pragma unroll 4
    for (std::uint32_t i = lane; i < cols; i += 32) {
      acc += row[i] * xr[i];
    }
    acc = WarpReduceSum(acc);
    if (lane == 0) {
      out[static_cast<std::size_t>(s) * rows + r] = acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Bucketed tile tier: one wave per (16-row expert bucket, row). The wave
// decodes its weight row once and dots it against every slot of the
// bucket, so a shared expert weight is read once for up to 16 slots
// instead of once per slot. Each per-slot dot runs the exact grouped
// sequence (b ascending, j ascending, one warp reduce at the end), so the
// bucketed MoE output is bit-identical to GemvGrouped.
// ---------------------------------------------------------------------------

template<void (*DecodeFn)(const std::uint8_t*, std::uint32_t, float*)>
__global__ void GemvRoutedSuper256(
    const std::uint8_t* __restrict__ w, std::size_t block_bytes,
    std::size_t expert_stride, const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ pad_bounds,
    const std::int32_t* __restrict__ rows_slot, std::uint32_t slots_per_row,
    std::uint32_t rows, std::uint32_t cols, const float* __restrict__ x,
    std::uint32_t xs, float* __restrict__ out) {
  const std::int32_t entry = tiles[blockIdx.y];
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if ((entry >> 16) == 0x7FFF || r >= rows) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t e = static_cast<std::uint32_t>(entry) & 0xFFFFU;
  const std::int32_t start =
      pad_bounds[e] + static_cast<std::int32_t>(entry >> 16) * 16;
  std::int32_t slot[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    slot[k] = rows_slot[start + k];
  }
  const std::uint32_t nblocks = cols / 256;
  const std::uint8_t* row = w + static_cast<std::size_t>(e) * expert_stride +
                            static_cast<std::size_t>(r) * nblocks * block_bytes;
  float acc[16] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    float v[8];
    DecodeFn(row + static_cast<std::size_t>(b) * block_bytes, lane, v);
    const float* xb = x + static_cast<std::size_t>(b) * 256 + 8 * lane;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      if (slot[k] < 0) {
        continue;
      }
      const float* xrow =
          xb + static_cast<std::size_t>(slot[k] / slots_per_row) * xs;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[k] += v[j] * xrow[j];
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float total = WarpReduceSum(acc[k]);
    if (lane == 0 && slot[k] >= 0) {
      out[static_cast<std::size_t>(slot[k]) * rows + r] = total;
    }
  }
}

// ---------------------------------------------------------------------------
// Host dispatch.
// ---------------------------------------------------------------------------

std::uint32_t RowGrid(std::uint32_t rows) {
  return (rows + 3U) / 4U;
}

std::uint32_t GroupedGrid(std::uint32_t used, std::uint32_t rows) {
  const std::uint32_t need = (used * rows + 3U) / 4U;
  return need < 1024U ? need : 1024U;
}

template<std::uint32_t Rows>
void LaunchRows(const void* base, GemvType type, std::uint32_t rows,
                std::uint32_t cols, std::size_t row_bytes, const float* x,
                std::uint32_t xs, float* out, std::uint32_t os,
                std::uint32_t group_rows, std::uint32_t group_advance,
                hipStream_t stream) {
  const std::uint32_t grid = RowGrid(rows);
  switch (type) {
    case GemvType::kQ8_0:
      GemvRowsQ8_0<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const Q8_0Block*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
    case GemvType::kQ4_K:
      GemvRowsQ4K<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const Q4KBlock*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
    case GemvType::kQ5_K:
      GemvRowsQ5K<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const Q5KBlock*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
    case GemvType::kQ6_K:
      GemvRowsQ6K<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const Q6KBlock*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
    case GemvType::kQ2_K:
      GemvRowsSuper256<Rows, DecodeQ2K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), row_bytes / (cols / 256), x,
          xs, out, os, rows, cols, group_rows, group_advance);
      break;
    case GemvType::kQ3_K:
      GemvRowsSuper256<Rows, DecodeQ3K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), row_bytes / (cols / 256), x,
          xs, out, os, rows, cols, group_rows, group_advance);
      break;
    case GemvType::kIQ2_XXS:
      GemvRowsSuper256<Rows, DecodeIq2Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq2XxsBytes, x, xs, out, os,
          rows, cols, group_rows, group_advance);
      break;
    case GemvType::kIQ1_M:
      GemvRowsSuper256<Rows, DecodeIq1M8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq1MBytes, x, xs, out, os,
          rows, cols, group_rows, group_advance);
      break;
    case GemvType::kIQ3_XXS:
      GemvRowsSuper256<Rows, DecodeIq3Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq3XxsBytes, x, xs, out, os,
          rows, cols, group_rows, group_advance);
      break;
    case GemvType::kIQ4_XS:
      GemvRowsSuper256<Rows, DecodeIq4Xs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq4XsBytes, x, xs, out, os,
          rows, cols, group_rows, group_advance);
      break;
    case GemvType::kF32:
      GemvRowsF32<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const float*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
    case GemvType::kF16:
      GemvRowsF16<Rows><<<grid, 128, 0, stream>>>(
          static_cast<const __half*>(base), x, xs, out, os, rows, cols,
          group_rows, group_advance);
      break;
  }
}

std::size_t RowBytesFor(GemvType type, std::uint32_t cols) {
  switch (type) {
    case GemvType::kQ8_0:
      return static_cast<std::size_t>(cols / 32U) * sizeof(Q8_0Block);
    case GemvType::kQ4_K:
      return static_cast<std::size_t>(cols / 256U) * sizeof(Q4KBlock);
    case GemvType::kQ5_K:
      return static_cast<std::size_t>(cols / 256U) * sizeof(Q5KBlock);
    case GemvType::kQ6_K:
      return static_cast<std::size_t>(cols / 256U) * sizeof(Q6KBlock);
    case GemvType::kQ2_K:
      return static_cast<std::size_t>(cols / 256U) * kQ2KBytes;
    case GemvType::kQ3_K:
      return static_cast<std::size_t>(cols / 256U) * kQ3KBytes;
    case GemvType::kIQ2_XXS:
      return static_cast<std::size_t>(cols / 256U) * kIq2XxsBytes;
    case GemvType::kIQ1_M:
      return static_cast<std::size_t>(cols / 256U) * kIq1MBytes;
    case GemvType::kIQ3_XXS:
      return static_cast<std::size_t>(cols / 256U) * kIq3XxsBytes;
    case GemvType::kIQ4_XS:
      return static_cast<std::size_t>(cols / 256U) * kIq4XsBytes;
    case GemvType::kF32:
      return static_cast<std::size_t>(cols) * sizeof(float);
    case GemvType::kF16:
      return static_cast<std::size_t>(cols) * sizeof(__half);
  }
  return 0;
}

}  // namespace

void ConvertF32ToHalf(const float* src, __half* dst, std::size_t count,
                      std::uint32_t* overflow, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t blocks = (count + 255U) / 256U;
  const std::size_t launched = blocks < 4096U ? blocks : 4096U;
  F32ToHalfKernel<<<static_cast<unsigned>(launched), 256, 0, stream>>>(
      src, dst, count, overflow);
}

void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  const std::uint32_t grid = RowGrid(rows);
  switch (type) {
    case GemvType::kQ8_0:
      GemvQ8_0<<<grid, 128, 0, stream>>>(static_cast<const Q8_0Block*>(base), x,
                                         out, rows, cols);
      break;
    case GemvType::kQ4_K:
      GemvQ4K<<<grid, 128, 0, stream>>>(static_cast<const Q4KBlock*>(base), x,
                                        out, rows, cols);
      break;
    case GemvType::kQ5_K:
      GemvQ5K<<<grid, 128, 0, stream>>>(static_cast<const Q5KBlock*>(base), x,
                                        out, rows, cols);
      break;
    case GemvType::kQ6_K:
      GemvQ6K<<<grid, 128, 0, stream>>>(static_cast<const Q6KBlock*>(base), x,
                                        out, rows, cols);
      break;
    case GemvType::kQ2_K:
      GemvSuper256<DecodeQ2K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), row_bytes / (cols / 256), x,
          out, rows, cols);
      break;
    case GemvType::kQ3_K:
      GemvSuper256<DecodeQ3K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), row_bytes / (cols / 256), x,
          out, rows, cols);
      break;
    case GemvType::kIQ2_XXS:
      GemvSuper256<DecodeIq2Xxs8>
          <<<grid, 128, 0, stream>>>(static_cast<const std::uint8_t*>(base),
                                     kIq2XxsBytes, x, out, rows, cols);
      break;
    case GemvType::kIQ1_M:
      GemvSuper256<DecodeIq1M8>
          <<<grid, 128, 0, stream>>>(static_cast<const std::uint8_t*>(base),
                                     kIq1MBytes, x, out, rows, cols);
      break;
    case GemvType::kIQ3_XXS:
      GemvSuper256<DecodeIq3Xxs8>
          <<<grid, 128, 0, stream>>>(static_cast<const std::uint8_t*>(base),
                                     kIq3XxsBytes, x, out, rows, cols);
      break;
    case GemvType::kIQ4_XS:
      GemvSuper256<DecodeIq4Xs8>
          <<<grid, 128, 0, stream>>>(static_cast<const std::uint8_t*>(base),
                                     kIq4XsBytes, x, out, rows, cols);
      break;
    case GemvType::kF32:
      GemvF32<<<grid, 128, 0, stream>>>(static_cast<const float*>(base), x, out,
                                        rows, cols);
      break;
    case GemvType::kF16:
      GemvF16<<<grid, 128, 0, stream>>>(static_cast<const __half*>(base), x,
                                        out, rows, cols);
      break;
  }
}

namespace {

// Widening token slices past 16 was measured and rejected: 32 spills
// nothing on the Q/K quants (VGPR<=180) yet runs 2.3x slower at pp2048
// (18.6 vs 42.5 t/s, matched A/B) on activation traffic; 64 additionally
// spills (scratch up to 596 B/lane) and runs 3.2x slower.
constexpr std::uint32_t kMaxRows = 16U;

void LaunchRowsSlices(const void* base, GemvType type, std::uint32_t tokens,
                      std::uint32_t rows, std::uint32_t cols,
                      std::size_t row_bytes, const float* x,
                      std::uint32_t x_stride, float* out,
                      std::uint32_t out_stride, std::uint32_t group_rows,
                      std::uint32_t group_advance, hipStream_t stream) {
  // Any width decomposes into descending power-of-two slices; each slice is
  // an independent contiguous row block, so every per-row dot keeps its
  // exact sequence and only the launch factorization changes.
  if (type == GemvType::kQ8_0 && tokens >= kMaxRows && rows % 16U == 0U) {
    // Full 16-token tiles with four tiled rows per wave go out as one
    // grid.y batched launch; anything shorter falls through to the ladder.
    const std::uint32_t full = tokens / kMaxRows;
    GemvRowsQ8_0Tiled<kMaxRows><<<dim3(rows / 16U, full), 128, 0, stream>>>(
        static_cast<const Q8_0Block*>(base), x, x_stride, out, out_stride,
        rows, cols, group_rows, group_advance);
    const std::uint32_t tiled = full * kMaxRows;
    tokens -= tiled;
    x += static_cast<std::size_t>(tiled) * x_stride;
    out += static_cast<std::size_t>(tiled) * out_stride;
  }
  std::uint32_t done = 0;
  while (done < tokens) {
    const std::uint32_t rem = tokens - done;
    std::uint32_t slice = 1U;
    while (slice * 2U <= rem && slice < kMaxRows) {
      slice *= 2U;
    }
    const float* xs = x + static_cast<std::size_t>(done) * x_stride;
    float* os = out + static_cast<std::size_t>(done) * out_stride;
    switch (slice) {
      case 1:
        LaunchRows<1>(base, type, rows, cols, row_bytes, xs, x_stride, os,
                      out_stride, group_rows, group_advance, stream);
        break;
      case 2:
        LaunchRows<2>(base, type, rows, cols, row_bytes, xs, x_stride, os,
                      out_stride, group_rows, group_advance, stream);
        break;
      case 4:
        LaunchRows<4>(base, type, rows, cols, row_bytes, xs, x_stride, os,
                      out_stride, group_rows, group_advance, stream);
        break;
      case 8:
        LaunchRows<8>(base, type, rows, cols, row_bytes, xs, x_stride, os,
                      out_stride, group_rows, group_advance, stream);
        break;
      case 16:
        LaunchRows<16>(base, type, rows, cols, row_bytes, xs, x_stride, os,
                       out_stride, group_rows, group_advance, stream);
        break;
    }
    done += slice;
  }
}

}  // namespace

void GemvRows(const void* base, GemvType type, std::uint32_t tokens,
              std::uint32_t rows, std::uint32_t cols, std::size_t row_bytes,
              const float* x, std::uint32_t x_stride, float* out,
              std::uint32_t out_stride, hipStream_t stream) {
  LaunchRowsSlices(base, type, tokens, rows, cols, row_bytes, x, x_stride, out,
                   out_stride, 0U, 0U, stream);
}

void GemvRowsGrouped(const void* base, GemvType type, std::uint32_t tokens,
                     std::uint32_t groups, std::uint32_t group_rows,
                     std::uint32_t cols, std::size_t row_bytes, const float* x,
                     std::uint32_t x_stride, std::uint32_t group_advance,
                     float* out, std::uint32_t out_stride, hipStream_t stream) {
  LaunchRowsSlices(base, type, tokens, groups * group_rows, cols, row_bytes, x,
                   x_stride, out, out_stride, group_rows, group_advance,
                   stream);
}

void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t slots_per_row, std::uint32_t rows,
                 std::uint32_t cols, const float* x, std::uint32_t x_stride,
                 float* out, hipStream_t stream) {
  const std::uint32_t grid = GroupedGrid(used, rows);
  switch (type) {
    case GemvType::kIQ2_XXS:
      GemvGroupedSuper256<DecodeIq2Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq2XxsBytes, expert_stride,
          ids, used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kIQ1_M:
      GemvGroupedSuper256<DecodeIq1M8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq1MBytes, expert_stride,
          ids, used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kIQ3_XXS:
      GemvGroupedSuper256<DecodeIq3Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq3XxsBytes, expert_stride,
          ids, used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kIQ4_XS:
      GemvGroupedSuper256<DecodeIq4Xs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq4XsBytes, expert_stride,
          ids, used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kQ2_K:
      GemvGroupedSuper256<DecodeQ2K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base),
          RowBytesFor(GemvType::kQ2_K, cols) / (cols / 256), expert_stride, ids,
          used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kQ3_K:
      GemvGroupedSuper256<DecodeQ3K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base),
          RowBytesFor(GemvType::kQ3_K, cols) / (cols / 256), expert_stride, ids,
          used, slots_per_row, rows, cols, x, x_stride, out);
      break;
    case GemvType::kF32:
      GemvGroupedF32<<<grid, 128, 0, stream>>>(
          static_cast<const float*>(base), expert_stride, ids, used,
          slots_per_row, rows, cols, x, x_stride, out);
      break;
    default:
      break;
  }
}

void GemvRoutedTiles(const void* base, GemvType type, std::size_t expert_stride,
                     const std::int32_t* tiles, std::uint32_t capacity,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_slot, std::uint32_t slots_per_row,
                     std::uint32_t rows, std::uint32_t cols, const float* x,
                     std::uint32_t x_stride, float* out, hipStream_t stream) {
  const dim3 grid((rows + 3U) / 4U, capacity);
  switch (type) {
    case GemvType::kIQ2_XXS:
      GemvRoutedSuper256<DecodeIq2Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq2XxsBytes, expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    case GemvType::kIQ1_M:
      GemvRoutedSuper256<DecodeIq1M8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq1MBytes, expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    case GemvType::kIQ3_XXS:
      GemvRoutedSuper256<DecodeIq3Xxs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq3XxsBytes, expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    case GemvType::kIQ4_XS:
      GemvRoutedSuper256<DecodeIq4Xs8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), kIq4XsBytes, expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    case GemvType::kQ2_K:
      GemvRoutedSuper256<DecodeQ2K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base),
          RowBytesFor(GemvType::kQ2_K, cols) / (cols / 256), expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    case GemvType::kQ3_K:
      GemvRoutedSuper256<DecodeQ3K8><<<grid, 128, 0, stream>>>(
          static_cast<const std::uint8_t*>(base),
          RowBytesFor(GemvType::kQ3_K, cols) / (cols / 256), expert_stride,
          tiles, pad_bounds, rows_slot, slots_per_row, rows, cols, x, x_stride,
          out);
      break;
    default:
      break;
  }
}

void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream) {
  const std::uint8_t* p =
      static_cast<const std::uint8_t*>(base) +
      static_cast<std::size_t>(row) * RowBytesFor(type, cols);
  switch (type) {
    case GemvType::kQ8_0:
      EmbedRowQ8_0<<<cols / 32U, 32, 0, stream>>>(
          reinterpret_cast<const Q8_0Block*>(p), out);
      break;
    case GemvType::kQ4_K:
      EmbedRowQ4K<<<cols / 256U, 32, 0, stream>>>(
          reinterpret_cast<const Q4KBlock*>(p), out);
      break;
    case GemvType::kQ5_K:
      EmbedRowQ5K<<<cols / 256U, 32, 0, stream>>>(
          reinterpret_cast<const Q5KBlock*>(p), out);
      break;
    case GemvType::kQ6_K:
      EmbedRowQ6K<<<cols / 256U, 32, 0, stream>>>(
          reinterpret_cast<const Q6KBlock*>(p), out);
      break;
    case GemvType::kQ2_K:
      EmbedRowSuper256<DecodeQ2K8>
          <<<cols / 256U, 32, 0, stream>>>(p, kQ2KBytes, out);
      break;
    case GemvType::kQ3_K:
      EmbedRowSuper256<DecodeQ3K8>
          <<<cols / 256U, 32, 0, stream>>>(p, kQ3KBytes, out);
      break;
    case GemvType::kIQ2_XXS:
      EmbedRowSuper256<DecodeIq2Xxs8>
          <<<cols / 256U, 32, 0, stream>>>(p, kIq2XxsBytes, out);
      break;
    case GemvType::kIQ1_M:
      EmbedRowSuper256<DecodeIq1M8>
          <<<cols / 256U, 32, 0, stream>>>(p, kIq1MBytes, out);
      break;
    case GemvType::kIQ3_XXS:
      EmbedRowSuper256<DecodeIq3Xxs8>
          <<<cols / 256U, 32, 0, stream>>>(p, kIq3XxsBytes, out);
      break;
    case GemvType::kIQ4_XS:
      EmbedRowSuper256<DecodeIq4Xs8>
          <<<cols / 256U, 32, 0, stream>>>(p, kIq4XsBytes, out);
      break;
    case GemvType::kF32:
      EmbedRowF32<<<(cols + 255U) / 256U, 256, 0, stream>>>(
          reinterpret_cast<const float*>(p), out, cols);
      break;
    case GemvType::kF16:
      EmbedRowF16<<<(cols + 255U) / 256U, 256, 0, stream>>>(
          reinterpret_cast<const __half*>(p), out, cols);
      break;
  }
}

}  // namespace gufo::models::glm5_next::rocm

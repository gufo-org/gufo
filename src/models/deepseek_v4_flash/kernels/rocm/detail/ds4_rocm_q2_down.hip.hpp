#pragma once

#include <rocwmma/rocwmma.hpp>
#include <vector>

#include "ds4_rocm_iq2_gate.hip.hpp"

// Pack 128-row groups first, then a 64-row tail for each odd bucket. Keeping
// small tails at 64 rows avoids wasting matrix work on padding. The return
// value separates the two launch spans; each expert output belongs to one tile.
static uint32_t ds4_rocm_q2_down_tile_map(std::vector<uint32_t>& map,
                                          const uint32_t* counts,
                                          const uint32_t* hot_experts,
                                          uint32_t hot_count,
                                          uint32_t row_tiles, bool pair_tiles) {
  map.clear();
  const uint32_t rows = row_tiles * 16u;
  for (uint32_t h = 0; h < hot_count; ++h) {
    const uint32_t groups = (counts[hot_experts[h]] + rows - 1u) / rows;
    const uint32_t full = pair_tiles ? groups / 2u : groups;
    for (uint32_t g = 0; g < full; ++g)
      map.push_back((h << 16u) | g);
  }
  const auto wide_count = pair_tiles ? static_cast<uint32_t>(map.size()) : 0u;
  if (pair_tiles) {
    for (uint32_t h = 0; h < hot_count; ++h) {
      const uint32_t groups = (counts[hot_experts[h]] + rows - 1u) / rows;
      if (groups & 1u)
        map.push_back((h << 16u) | (groups - 1u));
    }
  }
  return wide_count;
}

/* Dynamic LDS for the wide-N variant.
 *
 * The A and B staging halves, then the raw Q2_K slab window that the staged
 * dequantizer reads. The epilogue's single-fragment C page aliases A and B, so
 * only the larger of those two counts, but the raw window has to sit beyond
 * both because the K loop still needs it. */
static size_t ds4_rocm_q2_down_wide_shmem(uint32_t mtiles, uint32_t bm,
                                          uint32_t bn, uint32_t bk,
                                          uint32_t nfrag) {
  /* Two mid-tile buffers: the kernel prefetches the next K step's tile while
   * the current one still feeds the matrix ops. */
  const size_t ab = (2u * (size_t)mtiles * bm * bk + (size_t)nfrag * bk * bn) *
                    sizeof(__half);
  const size_t c = ((size_t)mtiles * bm * bn) * sizeof(float);
  const size_t raw =
      (size_t)nfrag * bn * (84u / sizeof(uint32_t)) * sizeof(uint32_t);
  return (ab > c ? ab : c) + raw;
}

/* NFRAG-wide dequantizer over Q2_K blocks already staged in shared memory.
 *
 * The global-memory twin below re-reads the same 84-byte block on every one of
 * the sixteen BK steps inside a 256-value K slab, and each read is six scalar
 * sub-word loads from a two-byte-aligned block. Staging the slab once as
 * uint32 words and dequantizing from there leaves the arithmetic and the
 * emitted values identical. */
template<int BN, int BK, int NFRAG>
__device__ __forceinline__ static void
q2_K_dequant_wide_tile_half_rowwise_staged(__half* shB,
                                           const uint32_t* raw_rows,
                                           uint32_t k0, uint32_t tid) {
  const uint32_t g = (k0 & 255u) >> 4u;
  const uint32_t within = g & 7u;
  const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
  const uint32_t shift = (within >> 1u) * 2u;
  constexpr uint32_t KG = 4u;
  constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
  constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
  for (uint32_t j = tid; j < (uint32_t)NFRAG * UNITS_PER_TILE;
       j += blockDim.x) {
    const uint32_t tile = j / UNITS_PER_TILE;
    const uint32_t rem = j - tile * UNITS_PER_TILE;
    const uint32_t nn = rem / (uint32_t)(BK / KG);
    const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
    const uint32_t row_local = tile * (uint32_t)BN + nn;
    const unsigned char* blk = reinterpret_cast<const unsigned char*>(
        raw_rows + row_local * RAW_DWORDS);
    const uint32_t dm_bits = *reinterpret_cast<const uint32_t*>(blk + 80u);
    const float d = dev_f16_to_f32((uint16_t)dm_bits);
    const float dm = dev_f16_to_f32((uint16_t)(dm_bits >> 16u));
    const float s = (float)(blk[g] & 0x0fu);
    const float m = (float)(blk[g] >> 4u);
    const uint32_t qbits =
        *reinterpret_cast<const uint32_t*>(blk + 16u + qbase + kk0);
    const uint32_t q0 = (qbits >> shift) & 3u;
    const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
    const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
    const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
    const float ds = d * s;
    const float dmm = dm * m;
    /* One 8-byte LDS store rather than two adjacent 4-byte ones. Split, the
     * compiler pairs them into a two-address form whose halves land in the
     * same bank; the same change in the IQ2 tile loader took its conflict
     * rate from 19.7% to 11.7%. `kk0` is a multiple of KG = 4, so `dst` is
     * always 8-byte aligned. Same bytes in the same order. */
    __half* dst = shB + tile * (uint32_t)(BK * BN) + nn * (uint32_t)BK + kk0;
    uint2 packed;
    packed.x = dev_pack_half2_bits(ds * (float)q0 - dmm, ds * (float)q1 - dmm);
    packed.y = dev_pack_half2_bits(ds * (float)q2 - dmm, ds * (float)q3 - dmm);
    *reinterpret_cast<uint2*>(dst) = packed;
  }
}

/* Wide-N routed Q2_K down.
 *
 * The n2 kernel above is activation-traffic bound rather than compute or
 * dequantization bound: its 64-row mid tile is 92% of the bytes it touches, and
 * with 32 output columns per workgroup that tile is re-staged 128 times per
 * layer, once per column block.  Arithmetic intensity is about 30 FLOP/byte
 * where roughly 230 would be needed to saturate the matrix cores.
 *
 * This variant stages the same mid tile once and runs NFRAG column fragments
 * against it, halving (NFRAG=4) or quartering (NFRAG=8) the activation
 * re-reads.  The accumulators live in registers, and the epilogue writes them
 * back two fragments at a time through the same 8 KiB staging window the n2
 * kernel uses, so dynamic LDS and resident workgroups per CU are unchanged.
 * The K loop order, fragment shapes, and accumulation order per output element
 * are identical to the n2 kernel, so results stay bit-exact. */
template<int MTILES = 4, int BM = 16, int BN = 16, int BK = 16, int NFRAG = 4,
         bool MID_F16 = false, bool OUT_F16 = false, bool SLOT_MAJOR = false>
__global__ static void moe_down_q2K_hotlist_wmma_wide_kernel(
    float* down_out, __half* down_out_h, const char* down_base,
    const float* mid, const __half* mid_h, const uint32_t* counts,
    const uint32_t* offsets, const uint32_t* pairs, const uint32_t* hot_experts,
    uint32_t hot_count, uint32_t expert_mid_dim, uint32_t out_dim,
    uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t n_tokens = 0u,
    /* When non-null, blockIdx.y indexes a compacted list of the
     * (hot expert, row group) pairs that actually have rows, packed as
     * (hot_index << 16) | row_group, and gridDim.z is 1. */
    const uint32_t* tile_map = nullptr) {
  extern __shared__ unsigned char raw_sh[];
  /* Two mid-tile buffers. Counters put this kernel at 9 to 18%
   * instruction-issue utilisation with 96 VGPRs and no scratch, so it is
   * waiting rather than working, and the mid tile is the only global read
   * left inside the K step once the weights are staged per slab. Fetching the
   * next step's tile into registers before this step's dequantize, barrier
   * and matrix ops lets that load retire underneath them. */
  constexpr uint32_t A_TILE = (uint32_t)(MTILES * BM * BK);
  __half* shA = reinterpret_cast<__half*>(raw_sh);
  __half* shB = shA + 2u * A_TILE;
  /* Raw Q2_K blocks for this workgroup's NFRAG * BN output rows, one 256-value
   * K slab at a time. shC still aliases the A and B staging area, which the K
   * loop has finished with by the time the epilogue runs, so the raw window
   * sits beyond both and is never overwritten. */
  constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
  constexpr uint32_t RAW_ROWS = (uint32_t)(NFRAG * BN);
  uint32_t* shW =
      reinterpret_cast<uint32_t*>(shB + (uint32_t)(NFRAG * BK * BN));
  /* MTILES*BM*(BK/2) uint32 over 32*MTILES threads is BM*(BK/2)/32 each,
   * independent of MTILES. */
  constexpr uint32_t MID_PRE = (uint32_t)(BM * (BK / 2) / 32);
  float* shC = reinterpret_cast<float*>(raw_sh);
  uint32_t hot_idx;
  uint32_t m_group0;
  if (tile_map) {
    const uint32_t packed = tile_map[blockIdx.y];
    hot_idx = packed >> 16u;
    m_group0 = (packed & 0xffffu) * MTILES * BM;
  } else {
    hot_idx = (uint32_t)blockIdx.z;
    m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
  }
  if (hot_idx >= hot_count)
    return;
  const uint32_t expert = hot_experts[hot_idx];
  const uint32_t count = counts[expert];
  if (m_group0 >= count)
    return;
  const uint32_t n0 = (uint32_t)blockIdx.x * (NFRAG * BN);
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid >> 5u;
  const uint32_t first = offsets[expert];
  __shared__ uint32_t shPair[MTILES * BM];
  for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
    const uint32_t bucket_row = m_group0 + j;
    shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
  }
  __syncthreads();

  using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half,
                                   rocwmma::row_major>;
  using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half,
                                   rocwmma::col_major>;
  using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
  frag_a a;
  frag_b b[NFRAG];
  frag_c acc[NFRAG];
  if (wave < MTILES) {
#pragma unroll
    for (int f = 0; f < NFRAG; f++)
      rocwmma::fill_fragment(acc[f], 0.0f);
  }

  const unsigned char* dew =
      (const unsigned char*)down_base + (uint64_t)expert * down_expert_bytes;

  uint32_t pre[MID_PRE];
  const auto fetch_mid = [&](uint32_t k0, uint32_t* dst) {
#pragma unroll
    for (uint32_t u = 0; u < MID_PRE; u++) {
      const uint32_t j = tid + u * blockDim.x;
      const uint32_t pair_row = j / (BK / 2);
      const uint32_t kk2 = j - pair_row * (BK / 2);
      const uint32_t pair = shPair[pair_row];
      uint32_t v = 0u;
      if (pair != UINT32_MAX) {
        const uint64_t moff = (uint64_t)pair * expert_mid_dim + k0 + kk2 * 2u;
        v = *reinterpret_cast<const uint32_t*>(mid_h + moff);
      }
      dst[u] = v;
    }
  };
  if (MID_F16)
    fetch_mid(0u, pre);
  uint32_t abuf = 0u;
  for (uint32_t kb = 0; kb < expert_mid_dim; kb += 256u) {
    for (uint32_t j = tid; j < RAW_ROWS * RAW_DWORDS; j += blockDim.x) {
      const uint32_t row_local = j / RAW_DWORDS;
      const uint32_t word = j - row_local * RAW_DWORDS;
      const uint32_t row = n0 + row_local;
      uint32_t v = 0u;
      if (row < out_dim) {
        const unsigned char* blk =
            dew + (uint64_t)row * down_row_bytes + (uint64_t)(kb >> 8u) * 84u;
        v = *reinterpret_cast<const uint32_t*>(blk + word * sizeof(uint32_t));
      }
      shW[j] = v;
    }
    __syncthreads();

    for (uint32_t krel = 0; krel < 256u && kb + krel < expert_mid_dim;
         krel += BK) {
      const uint32_t k0 = kb + krel;
      uint32_t next[MID_PRE];
      if (MID_F16) {
#pragma unroll
        for (uint32_t u = 0; u < MID_PRE; u++) {
          const uint32_t j = tid + u * blockDim.x;
          const uint32_t pair_row = j / (BK / 2);
          const uint32_t kk2 = j - pair_row * (BK / 2);
          *reinterpret_cast<uint32_t*>(shA + abuf * A_TILE + pair_row * BK +
                                       kk2 * 2u) = pre[u];
        }
        const uint32_t k_next = k0 + BK;
        if (k_next < expert_mid_dim)
          fetch_mid(k_next, next);
      } else {
        for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
          const uint32_t mt = j / (BM * BK);
          const uint32_t rem = j - mt * BM * BK;
          const uint32_t mm = rem / BK;
          const uint32_t kk = rem - mm * BK;
          const uint32_t pair = shPair[mt * BM + mm];
          if (pair != UINT32_MAX) {
            shA[abuf * A_TILE + j] =
                __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
          } else {
            shA[abuf * A_TILE + j] = __float2half(0.0f);
          }
        }
      }
      q2_K_dequant_wide_tile_half_rowwise_staged<BN, BK, NFRAG>(shB, shW, krel,
                                                                tid);
      __syncthreads();
      if (wave < MTILES) {
        rocwmma::load_matrix_sync(a, shA + abuf * A_TILE + wave * BM * BK, BK);
#pragma unroll
        for (int f = 0; f < NFRAG; f++) {
          rocwmma::load_matrix_sync(b[f], shB + f * (BK * BN), BN);
          rocwmma::mma_sync(acc[f], a, b[f], acc[f]);
        }
      }
      if (MID_F16) {
#pragma unroll
        for (uint32_t u = 0; u < MID_PRE; u++)
          pre[u] = next[u];
      }
      abuf ^= 1u;
      __syncthreads();
    }
  }

  /* Page the accumulators out one fragment at a time.
   *
   * Two at a time needed `2 * MTILES * BM * BN` floats, which made the C
   * window rather than the A/B staging the binding term and capped MTILES at
   * four for eight resident workgroups per CU. One fragment halves that
   * window, so MTILES can double without costing residency, and doubling the
   * row tile halves how often each weight column block is dequantized. The
   * epilogue writes the same values in the same order either way. */
#pragma unroll
  for (int f = 0; f < NFRAG; f++) {
    __syncthreads();
    if (wave < MTILES) {
      rocwmma::store_matrix_sync(shC + wave * BM * BN, acc[f], BN,
                                 rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
      const uint32_t mt = j / (BM * BN);
      const uint32_t rem = j - mt * BM * BN;
      const uint32_t mm = rem / BN;
      const uint32_t nn = rem - mm * BN;
      const uint32_t pair = shPair[mt * BM + mm];
      if (pair == UINT32_MAX)
        continue;
      const uint32_t tok = pair / 6u;
      const uint32_t slot = pair - tok * 6u;
      const uint32_t row = n0 + (uint32_t)f * BN + nn;
      if (row >= out_dim)
        continue;
      const float v = shC[j];
      uint64_t dst = (uint64_t)pair * out_dim + row;
      if (SLOT_MAJOR)
        dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
      if (OUT_F16)
        down_out_h[dst] = __float2half(v);
      else
        down_out[dst] = v;
    }
  }
}

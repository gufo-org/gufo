#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/trellis_device.hpp"

// Vector (decode and verification) kernels for Gyro TQ2_T routed experts.
// Every block copies the 8 KB codebook into LDS once and serves many rows
// from it; each lane decodes 16 weights per block-row step and takes exact
// F16 codebook products against F16 activations (v_dot2_f32_f16, F32
// accumulation). Rows routed to the same expert share one decode of its
// weights: the block of the expert's first slot serves all of them, with
// their rotated input rows built in its LDS.

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

using trellis::Half2;

constexpr int kWaves = 8;
constexpr int kBlockThreads = kWaves * 32;

/// One row's code bytes for this lane: the F16 scale of block `b` and bytes
/// 4l..4l+3 of its codes. Blocks are 34 bytes, so the code word is only
/// 2-byte aligned.
__device__ __forceinline__ void LoadLane(const std::uint8_t* row, int b, int l,
                                         float& d, std::uint32_t& w) {
  const auto* block = reinterpret_cast<const std::uint16_t*>(
      row + static_cast<std::size_t>(b) * kTrellisBlockBytes);
  d = __half2float(__ushort_as_half(block[0]));
  w = static_cast<std::uint32_t>(block[1 + 2 * l]) |
      (static_cast<std::uint32_t>(block[2 + 2 * l]) << 16);
}

__device__ __forceinline__ float SumLanes(float v, int width) {
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, 32);
  }
  return v;
}

/// Rotates 128-element block `b` of a k-wide F32 row into F16 LDS (one wave).
__device__ __forceinline__ void RotateBlockToLds(const float* row,
                                                 const float* signs, int b,
                                                 __half* lds) {
  const int lane = static_cast<int>(threadIdx.x & 31U);
  const float4 x4 =
      reinterpret_cast<const float4*>(row + b * kTrellisBlock)[lane];
  float v[4] = {x4.x, x4.y, x4.z, x4.w};
  if (signs != nullptr) {
    const float4 s4 =
        reinterpret_cast<const float4*>(signs + b * kTrellisBlock)[lane];
    v[0] *= s4.x;
    v[1] *= s4.y;
    v[2] *= s4.z;
    v[3] *= s4.w;
    trellis::Fwht128(v);
  }
  auto* dst = reinterpret_cast<__half2*>(lds + b * kTrellisBlock + 4 * lane);
  dst[0] = __floats2half2_rn(v[0], v[1]);
  dst[1] = __floats2half2_rn(v[2], v[3]);
}

/// Sixteen F16 activations from LDS as eight pairs.
__device__ __forceinline__ void LoadPairs(const __half* lds, Half2 out[8]) {
  const uint4* src = reinterpret_cast<const uint4*>(lds);
  const uint4 a = src[0];
  const uint4 b = src[1];
  const std::uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __builtin_bit_cast(Half2, words[i]);
  }
}

/// The 16 weights of one lane (four steps, code bytes `w`) as codebook F16
/// pairs in weight order, unscaled.
__device__ __forceinline__ void DecodeLane(std::uint32_t w,
                                           const std::uint32_t* lut,
                                           Half2 out[8]) {
  std::uint32_t s[4];
  trellis::LaneStates(w, trellis::PrevCodeByte(w), s);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    std::uint32_t a;
    std::uint32_t b;
    trellis::StepIndices(s[t], a, b);
    out[2 * t] = __builtin_bit_cast(Half2, lut[a]);
    out[(2 * t) + 1] = __builtin_bit_cast(Half2, lut[b]);
  }
}

/// Unscaled dot of decoded weights with sixteen activations, accumulated
/// in F32 in weight order.
__device__ __forceinline__ float Dot16(const Half2 w[8], const Half2 x[8]) {
  float acc = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc = __builtin_amdgcn_fdot2(w[i], x[i], acc, false);
  }
  return acc;
}

/// Routing entries a grouped block copies to LDS: tokens * used for the
/// gate/up projection, the slots for the down projection.
constexpr std::uint32_t kMaxGroupSlots = 512;

/// LDS bookkeeping of a block's expert group: the batch's routing (grouped
/// kernels only) and the current members, up to G token (or slot) indices
/// in ascending order with, per member, the mask of its `used` slots routed
/// to the expert.
template<int G>
struct Group {
  std::int32_t ids[G > 1 ? kMaxGroupSlots : 1];
  int member[G];
  std::uint32_t mask[G];
  int count;
  int next;  ///< where the scan for further members resumes
};

/// Wave 0: the next members of `expert`'s group from row `start` on.
template<int G>
__device__ __forceinline__ void NextMembers(Group<G>& group,
                                            std::int32_t expert, int start,
                                            std::uint32_t rows,
                                            std::uint32_t used) {
  const int lane = static_cast<int>(threadIdx.x & 31U);
  int found = 0;
  int next = start;
  while (found < G && next < static_cast<int>(rows)) {
    const int index = next + lane;
    std::uint32_t bits = 0;
    if (index < static_cast<int>(rows)) {
      for (std::uint32_t j = 0; j < used; ++j) {
        bits |=
            static_cast<std::uint32_t>(group.ids[index * used + j] == expert)
            << j;
      }
    }
    const auto hits = static_cast<std::uint32_t>(__ballot(bits != 0));
    const int take = min(G - found, __popc(hits));
    const int rank = __popc(hits & ((1U << lane) - 1U));
    if (bits != 0 && rank < take) {
      group.member[found + rank] = index;
      group.mask[found + rank] = bits;
    }
    found += take;
    if (take < __popc(hits)) {
      // Stopped inside this window: resume at its first untaken member.
      std::uint32_t rest = hits;
      for (int i = 0; i < take; ++i) {
        rest &= rest - 1U;
      }
      next += __ffs(static_cast<int>(rest)) - 1;
    } else {
      next += 32;
    }
  }
  if (lane == 0) {
    group.count = found;
    group.next = next;
  }
}

/// Wave 0: the first members of the group of `slot`'s expert, starting with
/// slot's own row, if `slot` is the first slot (in token-major order) routed
/// to it; otherwise no members, as another block serves the expert.
template<int G>
__device__ __forceinline__ void FirstMembers(Group<G>& group,
                                             std::int32_t expert,
                                             std::uint32_t slot,
                                             std::uint32_t rows,
                                             std::uint32_t used) {
  bool earlier = false;
  for (std::uint32_t a = threadIdx.x & 31U; a < slot; a += 32) {
    earlier |= group.ids[a] == expert;
  }
  if (__any(earlier)) {
    if ((threadIdx.x & 31U) == 0) {
      group.count = 0;
    }
    return;
  }
  NextMembers<G>(group, expert, static_cast<int>(slot / used), rows, used);
}

/// Gate/up: one wave per row; lane l of 8-lane group g takes code bytes
/// 4l..4l+3 of blocks g, g + 4, ... (kUnits = K / 512 of them). With G > 1
/// the block of the first slot routed to an expert serves every token
/// routed to it, G at a time: each weight is decoded once and dotted with
/// all of their rotated rows; the other blocks of that expert exit. A
/// token's sums keep the single-token order at any G.
template<int K, int kRowsPerWave, int G>
__launch_bounds__(kBlockThreads) __global__
    void TrellisGatedVecKernel(const std::uint8_t* __restrict__ gate,
                               const std::uint8_t* __restrict__ up,
                               const float* __restrict__ x,
                               const float* __restrict__ signs,
                               const std::uint32_t* __restrict__ codebook,
                               const std::int32_t* __restrict__ ids,
                               float* __restrict__ out, std::uint32_t m,
                               std::uint32_t tokens, std::uint32_t used) {
  static_assert(K % 512 == 0 && kRowsPerWave % 2 == 0);
  constexpr int kUnits = K / 512;
  constexpr int kBlocks = K / kTrellisBlock;
  constexpr std::size_t kRowBytes = kBlocks * kTrellisBlockBytes;
  __shared__ __attribute__((aligned(16))) std::uint32_t lut[2048];
  __shared__ __attribute__((aligned(16))) __half xh[G][K];
  __shared__ Group<G> group;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const std::uint32_t slot = blockIdx.y;
  const std::int32_t expert = ids[slot];
  const std::uint32_t row0 =
      (blockIdx.x * kWaves + wave) * static_cast<std::uint32_t>(kRowsPerWave);
  if (expert < 0) {
    // An inactive slot still defines its output rows.
    if (lane < kRowsPerWave) {
      out[static_cast<std::size_t>(slot) * m + row0 + lane] = 0.0F;
    }
    return;
  }
  const std::uint32_t own = slot / used;
  int n = 1;
  bool last = true;
  if constexpr (G > 1) {
    // Only the expert's first slot continues, before any codebook or
    // activation traffic.
    for (std::uint32_t a = tid; a < tokens * used; a += kBlockThreads) {
      group.ids[a] = ids[a];
    }
    __syncthreads();
    if (wave == 0) {
      FirstMembers<G>(group, expert, slot, tokens, used);
    }
    __syncthreads();
    // Read once: wave 0 rewrites both for the next members.
    n = group.count;
    last = group.next >= static_cast<int>(tokens);
    if (n == 0) {
      return;
    }
  }
  trellis::LoadCodebook(lut, codebook, tid, kBlockThreads);
  for (int task = wave; task < n * kBlocks; task += kWaves) {
    const int g = task / kBlocks;
    RotateBlockToLds(
        x + static_cast<std::size_t>(G == 1 ? own : group.member[g]) * K, signs,
        task % kBlocks, xh[g]);
  }
  __syncthreads();

  const int lane_group = lane >> 3;
  const int l = lane & 7;
  const std::size_t expert_offset =
      static_cast<std::size_t>(expert) * m * kRowBytes;
  for (bool first = true;; first = false) {
    if constexpr (G > 1) {
      if (!first) {
        // The previous members' rows are done; xh and the lists are free.
        __syncthreads();
        if (wave == 0) {
          NextMembers<G>(group, expert, group.next, tokens, used);
        }
        __syncthreads();
        n = group.count;
        last = group.next >= static_cast<int>(tokens);
        if (n == 0) {
          return;
        }
        for (int task = wave; task < n * kBlocks; task += kWaves) {
          RotateBlockToLds(
              x + static_cast<std::size_t>(group.member[task / kBlocks]) * K,
              signs, task % kBlocks, xh[task / kBlocks]);
        }
        __syncthreads();
      }
    }
    // A single token keeps its activations in registers.
    Half2 xr[G == 1 ? kUnits : 1][8];
    if constexpr (G == 1) {
#pragma unroll
      for (int j = 0; j < kUnits; ++j) {
        LoadPairs(xh[0] + (lane_group + 4 * j) * kTrellisBlock + 16 * l, xr[j]);
      }
    }
    for (int i = 0; i < kRowsPerWave; i += 2) {
      if constexpr (G == 1) {
        // One row at a time, as its loads, decode and dots chain.
        float sums[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const std::size_t offset =
              expert_offset +
              static_cast<std::size_t>(row0 + i + r) * kRowBytes;
          float d[2][kUnits];
          std::uint32_t w[2][kUnits];
#pragma unroll
          for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int j = 0; j < kUnits; ++j) {
              LoadLane((p == 0 ? gate : up) + offset, lane_group + 4 * j, l,
                       d[p][j], w[p][j]);
            }
          }
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            float acc = 0.0F;
#pragma unroll
            for (int j = 0; j < kUnits; ++j) {
              Half2 wv[8];
              DecodeLane(w[p][j], lut, wv);
              acc = __builtin_fmaf(d[p][j], Dot16(wv, xr[j]), acc);
            }
            sums[r][p] = SumLanes(acc, 32);
          }
        }
        if (lane < 2) {
          const float gv = lane == 0 ? sums[0][0] : sums[1][0];
          const float uv = lane == 0 ? sums[0][1] : sums[1][1];
          const float value = (gv * (1.0F / (1.0F + __expf(-gv)))) * uv;
          out[static_cast<std::size_t>(slot) * m + row0 + i + lane] =
              std::isfinite(value) ? value : 0.0F;
        }
        continue;
      }
      // Code words and scales of rows (row0 + i + r), gate (p = 0) and up.
      float d[2][2][kUnits];
      std::uint32_t w[2][2][kUnits];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const std::size_t offset =
            expert_offset + static_cast<std::size_t>(row0 + i + r) * kRowBytes;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const std::uint8_t* row = (p == 0 ? gate : up) + offset;
#pragma unroll
          for (int j = 0; j < kUnits; ++j) {
            LoadLane(row, lane_group + 4 * j, l, d[r][p][j], w[r][p][j]);
          }
        }
      }
      float acc[G][2][2] = {};
#pragma unroll
      for (int j = 0; j < kUnits; ++j) {
        Half2 wv[2][2][8];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            DecodeLane(w[r][p][j], lut, wv[r][p]);
          }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          if (g < n) {
            Half2 xv[8];
            LoadPairs(xh[g] + (lane_group + 4 * j) * kTrellisBlock + 16 * l,
                      xv);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
              for (int p = 0; p < 2; ++p) {
                acc[g][r][p] = __builtin_fmaf(d[r][p][j], Dot16(wv[r][p], xv),
                                              acc[g][r][p]);
              }
            }
          }
        }
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g < n) {
          float sums[2][2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              sums[r][p] = SumLanes(acc[g][r][p], 32);
            }
          }
          if (lane < 2) {
            const float gv = lane == 0 ? sums[0][0] : sums[1][0];
            const float uv = lane == 0 ? sums[0][1] : sums[1][1];
            const float value = (gv * (1.0F / (1.0F + __expf(-gv)))) * uv;
            const float stored = std::isfinite(value) ? value : 0.0F;
            const auto token = static_cast<std::size_t>(group.member[g]);
            for (std::uint32_t bits = group.mask[g]; bits != 0;
                 bits &= bits - 1U) {
              const std::size_t dst_slot =
                  token * used + static_cast<std::size_t>(__ffs(bits) - 1);
              out[dst_slot * m + row0 + i + lane] = stored;
            }
          }
        }
      }
    }
    if (last) {
      return;
    }
  }
}

/// Down: eight lanes per row, four rows per wave at a time; lane l of
/// group g takes code bytes 4l..4l+3 of every block of row (row + g). With
/// G > 1 the block of the first slot routed to an expert serves every slot
/// routed to it, G at a time, as in the gate/up kernel.
template<int K, int kRowsPerWave, int G>
__launch_bounds__(kBlockThreads) __global__
    void TrellisDownVecKernel(const std::uint8_t* __restrict__ w,
                              const float* __restrict__ x,
                              const float* __restrict__ signs,
                              const std::uint32_t* __restrict__ codebook,
                              const std::int32_t* __restrict__ ids,
                              float* __restrict__ out, std::uint32_t m,
                              std::uint32_t slots) {
  static_assert(K % kTrellisBlock == 0 && kRowsPerWave % 8 == 0);
  constexpr int kBlocks = K / kTrellisBlock;
  constexpr std::size_t kRowBytes = kBlocks * kTrellisBlockBytes;
  __shared__ __attribute__((aligned(16))) std::uint32_t lut[2048];
  __shared__ __attribute__((aligned(16))) __half xh[G][K];
  __shared__ Group<G> group;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const std::uint32_t slot = blockIdx.y;
  const std::int32_t expert = ids[slot];
  const std::uint32_t row0 =
      (blockIdx.x * kWaves + wave) * static_cast<std::uint32_t>(kRowsPerWave);
  if (expert < 0) {
    if (lane < kRowsPerWave) {
      out[static_cast<std::size_t>(slot) * m + row0 + lane] = 0.0F;
    }
    return;
  }
  int n = 1;
  bool last = true;
  if constexpr (G > 1) {
    for (std::uint32_t a = tid; a < slots; a += kBlockThreads) {
      group.ids[a] = ids[a];
    }
    __syncthreads();
    if (wave == 0) {
      FirstMembers<G>(group, expert, slot, slots, 1);
    }
    __syncthreads();
    n = group.count;
    last = group.next >= static_cast<int>(slots);
    if (n == 0) {
      return;
    }
  }
  trellis::LoadCodebook(lut, codebook, tid, kBlockThreads);
  for (int task = wave; task < n * kBlocks; task += kWaves) {
    const int g = task / kBlocks;
    RotateBlockToLds(
        x + static_cast<std::size_t>(G == 1 ? slot : group.member[g]) * K,
        signs, task % kBlocks, xh[g]);
  }
  __syncthreads();

  const int lane_group = lane >> 3;
  const int l = lane & 7;
  const std::uint8_t* base =
      w + static_cast<std::size_t>(expert) * m * kRowBytes;
  for (bool first = true;; first = false) {
    if constexpr (G > 1) {
      if (!first) {
        __syncthreads();
        if (wave == 0) {
          NextMembers<G>(group, expert, group.next, slots, 1);
        }
        __syncthreads();
        n = group.count;
        last = group.next >= static_cast<int>(slots);
        if (n == 0) {
          return;
        }
        for (int task = wave; task < n * kBlocks; task += kWaves) {
          RotateBlockToLds(
              x + static_cast<std::size_t>(group.member[task / kBlocks]) * K,
              signs, task % kBlocks, xh[task / kBlocks]);
        }
        __syncthreads();
      }
    }
    Half2 xr[G == 1 ? kBlocks : 1][8];
    if constexpr (G == 1) {
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) {
        LoadPairs(xh[0] + b * kTrellisBlock + 16 * l, xr[b]);
      }
    }
    for (int i = 0; i < kRowsPerWave; i += 8) {
      if constexpr (G == 1) {
        // One row at a time, as its loads, decode and dots chain.
        float sums[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const std::uint8_t* row =
              base + static_cast<std::size_t>(row0 + i + 4 * r + lane_group) *
                         kRowBytes;
          float d[kBlocks];
          std::uint32_t code[kBlocks];
#pragma unroll
          for (int b = 0; b < kBlocks; ++b) {
            LoadLane(row, b, l, d[b], code[b]);
          }
          float acc = 0.0F;
#pragma unroll
          for (int b = 0; b < kBlocks; ++b) {
            Half2 wv[8];
            DecodeLane(code[b], lut, wv);
            acc = __builtin_fmaf(d[b], Dot16(wv, xr[b]), acc);
          }
          sums[r] = SumLanes(acc, 8);
        }
        if (l == 0) {
          float* dst = out + static_cast<std::size_t>(slot) * m;
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            dst[row0 + i + 4 * r + lane_group] =
                std::isfinite(sums[r]) ? sums[r] : 0.0F;
          }
        }
        continue;
      }
      float d[2][kBlocks];
      std::uint32_t code[2][kBlocks];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const std::uint8_t* row =
            base +
            static_cast<std::size_t>(row0 + i + 4 * r + lane_group) * kRowBytes;
#pragma unroll
        for (int b = 0; b < kBlocks; ++b) {
          LoadLane(row, b, l, d[r][b], code[r][b]);
        }
      }
      float acc[G][2] = {};
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) {
        Half2 wv[2][8];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          DecodeLane(code[r][b], lut, wv[r]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          if (g < n) {
            Half2 xv[8];
            LoadPairs(xh[g] + b * kTrellisBlock + 16 * l, xv);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              acc[g][r] = __builtin_fmaf(d[r][b], Dot16(wv[r], xv), acc[g][r]);
            }
          }
        }
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g < n) {
          float sums[2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            sums[r] = SumLanes(acc[g][r], 8);
          }
          if (l == 0) {
            float* dst = out + static_cast<std::size_t>(group.member[g]) * m;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              dst[row0 + i + 4 * r + lane_group] =
                  std::isfinite(sums[r]) ? sums[r] : 0.0F;
            }
          }
        }
      }
    }
    if (last) {
      return;
    }
  }
}

/// One wave per 128-element block of a row.
template<typename T>
__launch_bounds__(kBlockThreads) __global__
    void TrellisRotateKernel(const T* __restrict__ in, __half* out,
                             const float* __restrict__ signs,
                             std::uint32_t blocks, std::uint32_t k) {
  const std::uint32_t index = blockIdx.x * kWaves + (threadIdx.x >> 5);
  if (index >= blocks) {
    return;
  }
  const std::uint32_t per_row = k / kTrellisBlock;
  const std::size_t base =
      static_cast<std::size_t>(index / per_row) * k +
      static_cast<std::size_t>(index % per_row) * kTrellisBlock;
  const int lane = static_cast<int>(threadIdx.x & 31U);
  float v[4];
  if constexpr (sizeof(T) == sizeof(float)) {
    const float4 x4 = reinterpret_cast<const float4*>(in + base)[lane];
    v[0] = x4.x;
    v[1] = x4.y;
    v[2] = x4.z;
    v[3] = x4.w;
  } else {
    const auto* src = reinterpret_cast<const __half2*>(in + base) + 2 * lane;
    const float2 a = __half22float2(src[0]);
    const float2 b = __half22float2(src[1]);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
  }
  if (signs != nullptr) {
    const float4 s4 = reinterpret_cast<const float4*>(
        signs + (index % per_row) * kTrellisBlock)[lane];
    v[0] *= s4.x;
    v[1] *= s4.y;
    v[2] *= s4.z;
    v[3] *= s4.w;
    trellis::Fwht128(v);
  }
  auto* dst = reinterpret_cast<__half2*>(out + base) + 2 * lane;
  dst[0] = __floats2half2_rn(v[0], v[1]);
  dst[1] = __floats2half2_rn(v[2], v[3]);
}

}  // namespace

namespace {

/// Rows per expert group: the batch rounded up to a power of two, capped at
/// `limit` (larger groups take several passes over the expert's weights).
/// Gate/up groups stop at four: eight rotated 2560-wide rows would take
/// 40 KB of LDS and halve the resident waves.
std::uint32_t GroupRows(std::uint32_t tokens, std::uint32_t limit) {
  std::uint32_t g = 1;
  while (g < tokens && g < limit) {
    g *= 2;
  }
  return g;
}

}  // namespace

bool TrellisGatedVec(const void* gate, const void* up, const float* x,
                     const float* signs, const std::uint32_t* codebook,
                     const std::int32_t* ids, float* out, std::uint32_t m,
                     std::uint32_t k, std::uint32_t tokens, std::uint32_t used,
                     hipStream_t stream) {
  constexpr int kRowsPerWave = 8;
  constexpr std::uint32_t kRows = kWaves * kRowsPerWave;
  if (codebook == nullptr || k != 2560 || m % kRows != 0 || tokens == 0 ||
      used == 0 || used > 32 || tokens * used > kMaxGroupSlots) {
    return false;
  }
  const dim3 grid(m / kRows, tokens * used);
  const auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, grid, dim3(kBlockThreads), 0, stream,
                       static_cast<const std::uint8_t*>(gate),
                       static_cast<const std::uint8_t*>(up), x, signs, codebook,
                       ids, out, m, tokens, used);
  };
  switch (GroupRows(tokens, 4)) {
    case 1:
      launch(TrellisGatedVecKernel<2560, kRowsPerWave, 1>);
      break;
    case 2:
      launch(TrellisGatedVecKernel<2560, kRowsPerWave, 2>);
      break;
    default:
      launch(TrellisGatedVecKernel<2560, kRowsPerWave, 4>);
      break;
  }
  return true;
}

bool TrellisDownVec(const void* w, const float* x, const float* signs,
                    const std::uint32_t* codebook, const std::int32_t* ids,
                    float* out, std::uint32_t m, std::uint32_t k,
                    std::uint32_t tokens, std::uint32_t used,
                    hipStream_t stream) {
  constexpr int kRowsPerWave = 16;
  constexpr std::uint32_t kRows = kWaves * kRowsPerWave;
  if (codebook == nullptr || k != 640 || m % kRows != 0 || tokens == 0 ||
      used == 0 || tokens * used > kMaxGroupSlots) {
    return false;
  }
  const std::uint32_t slots = tokens * used;
  const dim3 grid(m / kRows, slots);
  const auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, grid, dim3(kBlockThreads), 0, stream,
                       static_cast<const std::uint8_t*>(w), x, signs, codebook,
                       ids, out, m, slots);
  };
  switch (GroupRows(tokens, 8)) {
    case 1:
      launch(TrellisDownVecKernel<640, kRowsPerWave, 1>);
      break;
    case 2:
      launch(TrellisDownVecKernel<640, kRowsPerWave, 2>);
      break;
    case 4:
      launch(TrellisDownVecKernel<640, kRowsPerWave, 4>);
      break;
    default:
      launch(TrellisDownVecKernel<640, kRowsPerWave, 8>);
      break;
  }
  return true;
}

void TrellisRotateRows(const float* in_f32, const __half* in_f16, __half* out,
                       const float* signs, std::uint32_t rows, std::uint32_t k,
                       hipStream_t stream) {
  const std::uint32_t blocks = rows * (k / kTrellisBlock);
  if (blocks == 0) {
    return;
  }
  const dim3 grid((blocks + kWaves - 1) / kWaves);
  if (in_f32 != nullptr) {
    hipLaunchKernelGGL(TrellisRotateKernel<float>, grid, dim3(kBlockThreads), 0,
                       stream, in_f32, out, signs, blocks, k);
  } else {
    hipLaunchKernelGGL(TrellisRotateKernel<__half>, grid, dim3(kBlockThreads),
                       0, stream, in_f16, out, signs, blocks, k);
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm

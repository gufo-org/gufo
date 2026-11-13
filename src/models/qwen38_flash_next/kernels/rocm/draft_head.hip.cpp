// Load-time requantization of the MTP draft head (DeviceModel::SetDraftVocab).
// Built without fast math or contraction (extension.cmake): every operation
// rounds as in ggml's single-precision reference quantizer. The pragma below
// keeps contraction off whatever the build flags.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#pragma clang fp contract(off)

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr unsigned kThreads = 256;

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

__device__ __forceinline__ int NearestInt(float value) {
  return static_cast<int>(__builtin_rintf(value));  // ggml: ties to even
}

__device__ __forceinline__ int Q4Code(float iscale, float x, float min) {
  return std::max(0, std::min(15, NearestInt(iscale * (x - min))));
}

/// make_qkx2_quants(32, 15, x, w, L, &the_min, Laux, -1, 0.1, 20, false);
/// the caller recomputes the codes, so only the scale and the negated
/// minimum are kept.
__device__ float Qkx2Scale(const float (&x)[32], const float (&w)[32],
                           float* the_min) {
  float min = x[0];
  float max = x[0];
  float sum_w = w[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < 32; ++i) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    sum_w += w[i];
    sum_x += w[i] * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    *the_min = -min;
    return 0.0F;
  }
  float iscale = 15.0F / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < 32; ++i) {
    const int l = Q4Code(iscale, x[i], min);
    const float diff = scale * static_cast<float>(l) + min - x[i];
    best_error += w[i] * (diff * diff);
  }
  for (int step = 0; step <= 20; ++step) {
    iscale = (-1.0F + 0.1F * static_cast<float>(step) + 15.0F) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Q4Code(iscale, x[i], min));
      sum_l += w[i] * l;
      sum_l2 += w[i] * l * l;
      sum_xl += w[i] * l * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0)
      continue;
    float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
    float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (this_min > 0) {
      this_min = 0;
      this_scale = sum_xl / sum_l2;
    }
    float error = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Q4Code(iscale, x[i], min));
      const float diff = this_scale * l + this_min - x[i];
      error += w[i] * (diff * diff);
    }
    if (error < best_error) {
      best_error = error;
      scale = this_scale;
      min = this_min;
    }
  }
  *the_min = -min;
  return scale;
}

// ggml's quantize_row_q4_K_ref over Q8_0 rows. A thread owns one 32-value
// sub-block (one Q8_0 block); a superblock's eight threads are consecutive.
__global__ void QuantizeQ8_0ToQ4KKernel(const Q8_0Block* __restrict__ q8,
                                        Q4KBlock* __restrict__ q4k,
                                        std::size_t superblocks) {
  constexpr unsigned kPerBlock = kThreads / 8;
  __shared__ float scales_s[kPerBlock][8];
  __shared__ float mins_s[kPerBlock][8];
  __shared__ std::uint8_t codes_s[kPerBlock][256];
  const unsigned local = threadIdx.x / 8;
  const unsigned j = threadIdx.x % 8;
  const std::size_t superblock =
      static_cast<std::size_t>(blockIdx.x) * kPerBlock + local;
  const bool active = superblock < superblocks;
  float x[32];
  if (active) {
    const Q8_0Block& b = q8[superblock * 8 + j];
    const float d = __half2float(b.d);
    float sum_x2 = 0;
    for (int i = 0; i < 32; ++i) {
      x[i] = static_cast<float>(b.qs[i]) * d;
      sum_x2 += x[i] * x[i];
    }
    const float av_x = __builtin_sqrtf(sum_x2 / 32);
    float w[32];
    for (int i = 0; i < 32; ++i)
      w[i] = av_x + __builtin_fabsf(x[i]);
    float the_min = 0;
    scales_s[local][j] = Qkx2Scale(x, w, &the_min);
    mins_s[local][j] = the_min;
  }
  __syncthreads();
  if (active) {
    float max_scale = 0;
    float max_min = 0;
    for (int k = 0; k < 8; ++k) {
      if (scales_s[local][k] > max_scale)
        max_scale = scales_s[local][k];
      if (mins_s[local][k] > max_min)
        max_min = mins_s[local][k];
    }
    const float inv_scale = max_scale > 0 ? 63.0F / max_scale : 0.0F;
    const float inv_min = max_min > 0 ? 63.0F / max_min : 0.0F;
    // Every thread packs all eight 6-bit pairs into its own copy.
    std::uint8_t packed[12] = {};
    for (int k = 0; k < 8; ++k) {
      const auto ls = std::min<std::uint8_t>(
          63, static_cast<std::uint8_t>(
                  NearestInt(inv_scale * scales_s[local][k])));
      const auto lm = std::min<std::uint8_t>(
          63,
          static_cast<std::uint8_t>(NearestInt(inv_min * mins_s[local][k])));
      if (k < 4) {
        packed[k] = ls;
        packed[k + 4] = lm;
      } else {
        packed[k + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        packed[k - 4] |= (ls >> 4) << 6;
        packed[k] |= (lm >> 4) << 6;
      }
    }
    const __half d16 = __float2half_rn(max_scale / 63.0F);
    const __half dmin16 = __float2half_rn(max_min / 63.0F);
    Q4KBlock& out = q4k[superblock];
    if (j == 0) {
      out.d = d16;
      out.dmin = dmin16;
      for (int k = 0; k < 12; ++k)
        out.scales[k] = packed[k];
    }
    // get_scale_min_k4(j, scales)
    const std::uint8_t sc =
        j < 4 ? packed[j] & 63
              : (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4);
    const std::uint8_t m = j < 4
                               ? packed[j + 4] & 63
                               : (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    const float d = __half2float(d16) * static_cast<float>(sc);
    const float dm = __half2float(dmin16) * static_cast<float>(m);
    for (int i = 0; i < 32; ++i) {
      // A zero-scale sub-block dequantizes to its minimum whatever its
      // codes; ggml leaves its search codes there, this writes zeros.
      const int l =
          d != 0 ? std::max(0, std::min(15, NearestInt((x[i] + dm) / d))) : 0;
      codes_s[local][32 * j + i] = static_cast<std::uint8_t>(l);
    }
  }
  __syncthreads();
  if (active) {
    // qs[32 * pair + l] = L[64 * pair + l] | L[64 * pair + 32 + l] << 4
    Q4KBlock& out = q4k[superblock];
    for (int i = 0; i < 16; ++i) {
      const int byte = 16 * static_cast<int>(j) + i;
      const int pair = byte / 32;
      const int l = byte % 32;
      out.qs[byte] =
          static_cast<std::uint8_t>(codes_s[local][64 * pair + l] |
                                    (codes_s[local][64 * pair + 32 + l] << 4));
    }
  }
}

}  // namespace

void QuantizeQ8_0ToQ4K(const void* q8, void* q4k, std::size_t rows,
                       std::uint32_t cols, hipStream_t stream) {
  const std::size_t superblocks = rows * (cols / 256);
  constexpr std::size_t kPerBlock = kThreads / 8;
  hipLaunchKernelGGL(
      QuantizeQ8_0ToQ4KKernel,
      dim3(static_cast<unsigned>((superblocks + kPerBlock - 1) / kPerBlock)),
      dim3(kThreads), 0, stream, static_cast<const Q8_0Block*>(q8),
      static_cast<Q4KBlock*>(q4k), superblocks);
}

}  // namespace gufo::models::qwen38_flash_next::rocm

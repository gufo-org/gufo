#ifndef STRIX_MODELS_QWEN_HIP_QUANT_OPS_HPP_
#define STRIX_MODELS_QWEN_HIP_QUANT_OPS_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace strix::hip {

// Shared quantized block layouts + quant row-dot helpers for the decode and
// prefill GPU paths. Layouts/values are numerically identical to the
// historical per-file copies (decode_ops.hip / prefill_ops.hip) and match the
// CPU oracles in ggml_dequant.cpp.

// block_q8_0 layout: {__half d; int8_t qs[32];}, 34 bytes, QK=32. Dominant
// quant in the Q8_K_L model.
constexpr std::size_t kQ8_0BlockSize = 32;

struct Q8_0Block {
  __half d;
  std::int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

// opt-r4-q5k-q6k: block_q5_K layout ({half d; half dmin; uint8 scales[12];
// uint8 qh[32]; uint8 qs[128]; }, 176 bytes, QK_K=256). Dequant-to-fp path:
// Q5KValue read a fp16 d/dmin and the packed 4-bit qs + qh sign bits, then
// apply the (scale, min) pair from GetQKScaleMin. Requires K % 256 == 0.
constexpr std::size_t kQ5KBlockSize = 256;

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

// opt-r4-q5k-q6k: block_q6_K layout ({uint8 ql[128]; uint8 qh[64]; int8
// scales[16]; half d; }, 210 bytes, QK_K=256). Dequant-to-fp path: Q6KValue
// reads a fp16 d and the packed 4-bit ql + 2-bit qh, then applies the int8
// scale. Requires K % 256 == 0.
constexpr std::size_t kQ6KBlockSize = 256;

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  __half d;
};
static_assert(sizeof(Q6KBlock) == 210, "block_q6_K must be 210 bytes");

// opt-c1xx-q8k-gemv: block_q8_K layout ({ float d; int8_t qs[256];
// int16_t bsums[16]; }, 292 bytes, QK_K=256). The dot runs AT Q8, not by
// casting weights to fp16: each 256-wide fp32 x block is quantized in-register
// to int8 (d_x = max|x|, scale = d_x/127) and accumulated as an int32 integer
// MAC with qs; the single fp scale (d_w * scale_x) is applied at block end.
// Requires K % 256 == 0.
constexpr std::size_t kQ8KBlockSize = 256;

struct Q8KBlock {
  float d;
  std::int8_t qs[kQ8KBlockSize];
  std::int16_t bsums[16];
};
static_assert(sizeof(Q8KBlock) == 292, "block_q8_K must be 292 bytes");

// opt-r2-q8_0: type-aware quant block geometry helpers so RowDot can own the
// per-type row stride (bytes per block and elements per block).
__device__ inline std::size_t QuantBlockBytes(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return sizeof(Q8_0Block);
    case core::GgmlType::kQ8_K:
      return sizeof(Q8KBlock);
    case core::GgmlType::kQ5_K:
      return sizeof(Q5KBlock);
    case core::GgmlType::kQ6_K:
      return sizeof(Q6KBlock);
    default:
      return 0;
  }
}

__device__ inline std::size_t QuantBlockQK(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return kQ8_0BlockSize;
    case core::GgmlType::kQ8_K:
      return kQ8KBlockSize;
    case core::GgmlType::kQ5_K:
      return kQ5KBlockSize;
    case core::GgmlType::kQ6_K:
      return kQ6KBlockSize;
    default:
      return 0;
  }
}

// opt-r4-q5k-q6k: unpack the (scale, minimum) pair for Q5_K from the packed
// 12-byte scales array. index 0..7; mirrors CPU GetQ4ScaleMin in
// ggml_dequant.cpp (Q5_K uses the same scale/min encoding as Q4_K).
__device__ inline void GetQKScaleMin(std::size_t index,
                                     const std::uint8_t* packed,
                                     std::uint8_t& sc,
                                     std::uint8_t& m) noexcept {
  if (index < 4) {
    sc = packed[index] & 0x3FU;
    m = packed[index + 4] & 0x3FU;
    return;
  }
  sc = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                 ((packed[index - 4] >> 6U) << 4U));
  m = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                ((packed[index] >> 6U) << 4U));
}

// opt-r4-q5k-q6k: dequantize one Q5_K element (index i in [0,256)) to fp32.
// Mirrors CPU Q5Value in ggml_dequant.cpp exactly.
__device__ inline float Q5KValue(const Q5KBlock& block,
                                 std::size_t index) noexcept {
  const std::size_t gg = index / 64;  // 0..3
  const std::size_t wv = index % 64;  // 0..63
  const std::size_t lane = wv % 32;   // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((block.qh[lane] >> bit) & 1U) ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);        // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQKScaleMin(sis, block.scales, sc, m);
  return __half2float(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         __half2float(block.dmin) * static_cast<float>(m);
}

// opt-r4-q5k-q6k: dequantize one Q6_K element (index i in [0,256)) to fp32.
// Mirrors CPU Q6Value in ggml_dequant.cpp exactly.
__device__ inline float Q6KValue(const Q6KBlock& block,
                                 std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t segment = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh_byte = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = (qh_byte >> 0U) & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh_byte >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh_byte >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh_byte >> 6U) & 0x03U;
      break;
  }
  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return __half2float(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

// opt-r7-decode-parallel: warp-parallel quant row-dot, templated on the input
// element type TX (float in decode, hip_bfloat16 for the prefill bf16 path).
// ONE warp cooperates: all 32 lanes process a slice of the row's quant blocks
// (b in [b0,b1)), each lane owning 8 elements per 256-wide block
// (8_K/5_K/6_K) or 1 element per 32-wide block (Q8_0), then reduce within each
// block via __shfl_xor. Returns this warp's partial row-dot ONLY IN LANE 0
// (other lanes return 0), so it is a drop-in for both the one-warp-per-row
// kernels (a) and the one-block-per-row kernels (b) whose existing __shfl_xor
// / shared-memory reductions then broadcast/combine it correctly. With
// num_warps == 1 the warp processes the entire row. Mirrors Q8KBlockGEMVKernel
// exactly. The only TX-dependent code is the x reads, wrapped in
// static_cast<float> so TX=float is identity and TX=hip_bfloat16 converts.
template<typename TX>
__device__ inline float QuantWarpBlockDot(
    core::GgmlType type, const void* __restrict__ base, std::size_t row_idx,
    const TX* __restrict__ x, std::size_t K, std::size_t lane_id,
    std::size_t warp_id, std::size_t num_warps) {
  const std::size_t qk = QuantBlockQK(type);
  const std::size_t block_bytes = QuantBlockBytes(type);
  const char* row =
      static_cast<const char*>(base) + (row_idx * (K / qk * block_bytes));
  const std::size_t num_blocks = K / qk;
  const std::size_t chunk = (num_blocks + num_warps - 1) / num_warps;
  const std::size_t b0 = warp_id * chunk;
  const std::size_t b1 = (b0 + chunk < num_blocks) ? (b0 + chunk) : num_blocks;
  float sumf = 0.0F;
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const auto* row_blocks = reinterpret_cast<const Q8_0Block*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q8_0Block& wblk = row_blocks[b];
        const float d_w = __half2float(wblk.d);
        const TX* xb = x + (b * kQ8_0BlockSize);
        float local_max = fabsf(static_cast<float>(xb[lane_id]));
        for (int off = 16; off > 0; off >>= 1)
          local_max = fmaxf(local_max, __shfl_xor(local_max, off));
        const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
        const float xv =
            (scale > 0.0F) ? (static_cast<float>(xb[lane_id]) / scale) : 0.0F;
        int xq = static_cast<int>(roundf(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        int acc = static_cast<int>(wblk.qs[lane_id]) * xq;
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor(acc, off);
        sumf += d_w * scale * static_cast<float>(acc);
      }
      break;
    }
    case core::GgmlType::kQ8_K: {
      const auto* row_blocks = reinterpret_cast<const Q8KBlock*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q8KBlock& wblk = row_blocks[b];
        const float d_w = wblk.d;
        const TX* xb = x + (b * kQ8KBlockSize);
        float local_max = 0.0F;
#pragma unroll
        for (std::size_t k = 0; k < 8; ++k)
          local_max = fmaxf(local_max,
                            fabsf(static_cast<float>(xb[lane_id + (32 * k)])));
        for (int off = 16; off > 0; off >>= 1)
          local_max = fmaxf(local_max, __shfl_xor(local_max, off));
        const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
        int acc = 0;
#pragma unroll
        for (std::size_t k = 0; k < 8; ++k) {
          const std::size_t idx = lane_id + (32 * k);
          const float xv =
              (scale > 0.0F) ? (static_cast<float>(xb[idx]) / scale) : 0.0F;
          int xq = static_cast<int>(roundf(xv));
          xq = (xq > 127) ? 127 : xq;
          xq = (xq < -127) ? -127 : xq;
          acc += static_cast<int>(wblk.qs[idx]) * xq;
        }
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor(acc, off);
        sumf += d_w * scale * static_cast<float>(acc);
      }
      break;
    }
    case core::GgmlType::kQ5_K: {
      const auto* row_blocks = reinterpret_cast<const Q5KBlock*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q5KBlock& wblk = row_blocks[b];
        const TX* xb = x + (b * kQ5KBlockSize);
        float acc = 0.0F;
#pragma unroll
        for (std::size_t k = 0; k < 8; ++k)
          acc += Q5KValue(wblk, lane_id + (32u * k)) *
                 static_cast<float>(xb[lane_id + (32u * k)]);
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor(acc, off);
        sumf += acc;
      }
      break;
    }
    case core::GgmlType::kQ6_K: {
      const auto* row_blocks = reinterpret_cast<const Q6KBlock*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q6KBlock& wblk = row_blocks[b];
        const TX* xb = x + (b * kQ6KBlockSize);
        float acc = 0.0F;
#pragma unroll
        for (std::size_t k = 0; k < 8; ++k)
          acc += Q6KValue(wblk, lane_id + (32u * k)) *
                 static_cast<float>(xb[lane_id + (32u * k)]);
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor(acc, off);
        sumf += acc;
      }
      break;
    }
    default:
      break;
  }
  if (lane_id != 0)
    return 0.0F;
  return sumf;
}

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_QUANT_OPS_HPP_

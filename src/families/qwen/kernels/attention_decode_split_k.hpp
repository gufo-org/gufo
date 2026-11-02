#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_SPLIT_K_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_SPLIT_K_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/families/qwen/hip_common/attention_policy.hpp"

// Per-head steps of the split-K fp16 decode attention partials, shared by the
// wave32 kernels in attention_decode.hip and the wave64 verification kernel.
// Every variant reproduces the 32-lane scalar kernel bit for bit: the same
// four-product dot per lane and vector, the same 32 -> 16 -> 8 -> 4 -> 2 -> 1
// score tree, the same online-softmax recurrence and the same rounded value
// products. Only the lane packaging and the exchange instructions of the
// score tree differ between kernels.
namespace gufo::hip::detail {

// XOR exchange inside a 16-lane DPP row (gfx10 and later) on the ALU path
// instead of ds_bpermute. Both lanes of a pair add the same two values, so
// after the 8, 4, 2, 1 ladder every lane of a row holds the row's sum, and
// that sum is lane zero's descending tree bit for bit.
template<int kMask>
__device__ __forceinline__ float XorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

// Sixteen-lane row sum: the last four stages of the scalar kernel's tree.
__device__ __forceinline__ float RowSumDpp(float x) {
  x = XorAddDpp<8>(x);
  x = XorAddDpp<4>(x);
  x = XorAddDpp<2>(x);
  x = XorAddDpp<1>(x);
  return x;
}

struct SplitKPartition {
  std::size_t begin;
  std::size_t end;
};

// Token range [begin, end) of one split for the sequence ending at `pos`.
__device__ __forceinline__ SplitKPartition SplitKDecodePartition(
    std::uint32_t pos, std::uint32_t split, std::uint32_t split_count) {
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t partition_length =
      (sequence_length + split_count - 1) / split_count;
  const std::size_t begin = static_cast<std::size_t>(split) * partition_length;
  return {begin, min(begin + partition_length, sequence_length)};
}

// Four products of one float4 against four consecutive fp16 values in the
// order the scalar kernel's contraction fixed: the y product is rounded,
// then x, z and w fuse onto it. fp16 -> fp32 is exact, so each fmaf with a
// converted operand is a single mixed-precision FMA. With MixedFma the y
// product also goes through fma(a, b, negative_zero), where the caller's
// `negative_zero` is an opaque -0.0f: the result equals a * b bit for bit,
// sign of zero included, and the opaque addend stops the compiler folding
// it back to a convert and a multiply. The 32-lane kernel keeps the plain
// multiply so its instruction pairing is unchanged.
template<bool MixedFma>
__device__ __forceinline__ float DotHalf4(const float4& query,
                                          const __half* key,
                                          float negative_zero) {
  const auto* pairs = reinterpret_cast<const __half2*>(key);
  const __half2 lower = pairs[0], upper = pairs[1];
  float dot;
  if constexpr (MixedFma) {
    dot = fmaf(query.y, __high2float(lower), negative_zero);
  } else {
    dot = __fmul_rn(query.y, __high2float(lower));
  }
  dot = fmaf(query.x, __low2float(lower), dot);
  dot = fmaf(query.z, __low2float(upper), dot);
  dot = fmaf(query.w, __high2float(upper), dot);
  return dot;
}

// One online-softmax value step for four consecutive fp16 values. The scalar
// kernel rounds the value product for x and y and the output product for z
// and w, then fuses the other product; keep both operand roles.
template<bool MixedFma>
__device__ __forceinline__ void UpdateHalf4(float4& output, const __half* value,
                                            float previous_scale,
                                            float value_scale,
                                            float negative_zero) {
  const auto* pairs = reinterpret_cast<const __half2*>(value);
  const __half2 lower = pairs[0], upper = pairs[1];
  if constexpr (MixedFma) {
    output.x = fmaf(output.x, previous_scale,
                    fmaf(value_scale, __low2float(lower), negative_zero));
    output.y = fmaf(output.y, previous_scale,
                    fmaf(value_scale, __high2float(lower), negative_zero));
  } else {
    output.x = fmaf(output.x, previous_scale,
                    __fmul_rn(__low2float(lower), value_scale));
    output.y = fmaf(output.y, previous_scale,
                    __fmul_rn(__high2float(lower), value_scale));
  }
  output.z = fmaf(__low2float(upper), value_scale,
                  __fmul_rn(output.z, previous_scale));
  output.w = fmaf(__high2float(upper), value_scale,
                  __fmul_rn(output.w, previous_scale));
}

// One query head's state across a split: its query slice, output
// accumulators and softmax statistics. Lanes is 32 (one head per wave32) or
// 16 (two heads per wave32 or four per wave64); a 16-lane head keeps the
// 32-lane element layout, so lane l holds the original lanes l and l + 16.
template<unsigned Lanes, bool MixedFma>
struct SplitKAttentionHead {
  static_assert(Lanes == 16 || Lanes == 32);
  static constexpr std::uint32_t kHeadDim = 256;
  static constexpr std::uint32_t kVectorWidth = kHeadDim / 4;
  static constexpr std::uint32_t kVectorsPerLane = kVectorWidth / Lanes;

  float4 query_values[kVectorsPerLane];
  float4 output_values[kVectorsPerLane] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  float negative_zero = -0.0F;

  // float4 index of this lane's vector in the head's 64 vectors.
  __device__ static constexpr std::uint32_t Vector(std::uint32_t lane,
                                                   std::uint32_t vector) {
    return lane + (vector / 2) * Lanes + (vector % 2) * 32;
  }

  __device__ __forceinline__ void LoadQuery(const float4* query,
                                            std::uint32_t lane) {
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      query_values[vector] = query[Vector(lane, vector)];
    }
    if constexpr (MixedFma) {
      asm volatile("" : "+v"(negative_zero));
    }
  }

  // This lane's partial of q . k, ready for the score tree.
  __device__ __forceinline__ float Partial(const __half* key,
                                           std::uint32_t lane) const {
    if constexpr (Lanes == 32) {
      float score = 0.0F;
#pragma unroll
      for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
        score +=
            DotHalf4<MixedFma>(query_values[vector],
                               key + Vector(lane, vector) * 4, negative_zero);
      }
      return score;
    } else {
      // Keep both original lane partials independent until the original
      // offset-16 reduction stage.
      float partial[2] = {};
#pragma unroll
      for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
        for (unsigned vector = 0; vector < 2; ++vector) {
          partial[half] += DotHalf4<MixedFma>(
              query_values[half * 2 + vector],
              key + Vector(lane, half * 2 + vector) * 4, negative_zero);
        }
      }
      return partial[0] + partial[1];
    }
  }

  // Online-softmax step for one token's scaled score and value row.
  __device__ __forceinline__ void Accumulate(float score, const __half* value,
                                             std::uint32_t lane) {
    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      UpdateHalf4<MixedFma>(output_values[vector],
                            value + Vector(lane, vector) * 4, previous_scale,
                            value_scale, negative_zero);
    }
  }

  // Partial statistics and output in the layout the reduce kernel reads.
  __device__ __forceinline__ void Store(float* scratch, std::uint32_t num_heads,
                                        std::uint32_t head, std::uint32_t split,
                                        std::uint32_t lane) const {
    constexpr std::size_t max_splits = kSplitKDecodeAttentionMaxSplits;
    const std::size_t partition_index =
        (static_cast<std::size_t>(head) * max_splits) + split;
    if (lane == 0) {
      scratch[(partition_index * 2) + 0] = running_max;
      scratch[(partition_index * 2) + 1] = running_sum;
    }
    const std::size_t stats_elements =
        static_cast<std::size_t>(num_heads) * max_splits * 2;
    auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
    auto* partial_output = partial_outputs + (partition_index * kVectorWidth);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      partial_output[Vector(lane, vector)] = output_values[vector];
    }
  }
};

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_SPLIT_K_HPP_

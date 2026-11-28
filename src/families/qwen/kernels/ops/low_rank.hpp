#ifndef GUFO_MODELS_QWEN_HIP_OPS_LOW_RANK_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_LOW_RANK_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Largest rank `LaunchLowRankAdd` accepts.
inline constexpr std::uint32_t kLowRankMaxRank = 64;

/// Adds a LoRA update beside a projection, for `rows` contiguous rows:
/// y[row][o] += multiplier * sum_k b[o][k] * (sum_i a[k][i] * x[row][i]),
/// with `a` as [rank][in] and `b` as [out][rank]. Each row is reduced in a
/// fixed order that does not depend on `rows`, without atomics, so a row's
/// result is bit-identical whatever the batch. Requires `in` to be a multiple
/// of 4, 16-byte aligned `x` rows and `a` rows, and 1 <= rank <= 64.
void LaunchLowRankAdd(const float* x, const float* a, const float* b,
                      float multiplier, float* y, std::uint32_t rows,
                      std::uint32_t in, std::uint32_t out, std::uint32_t rank,
                      hipStream_t stream);

/// `LaunchLowRankAdd` reading IEEE half inputs `x` (rows 8-byte aligned).
void LaunchLowRankAddFp16(const void* x, const float* a, const float* b,
                          float multiplier, float* y, std::uint32_t rows,
                          std::uint32_t in, std::uint32_t out,
                          std::uint32_t rank, hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_LOW_RANK_HPP_

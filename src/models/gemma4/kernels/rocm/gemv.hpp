#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// 32 bits wide: the value crosses from GCC-built host code into clang-built
/// HIP code, and clang assumes a narrower argument arrives zero-extended where
/// GCC leaves the upper bits undefined.
enum class GemvFormat : std::uint32_t { kQ4_0, kQ4_K, kQ5_K, kQ6_K, kQ8_0 };

/// y[m] = W x for K-quant, Q4_0 or Q8_0 W ([m][k] in GGUF blocks,
/// k % 256 == 0) and an FP32
/// activation row: the autoregressive decode projection. Returns false without
/// launching when the shape is unsupported.
[[nodiscard]] bool LaunchKQuantGemv(GemvFormat format, const void* w,
                                    const float* x, float* y, std::uint32_t m,
                                    std::uint32_t k, hipStream_t stream);

/// Rows a binary16 small-batch projection takes in one weight pass.
inline constexpr std::uint32_t kMaxHalfGemvRows = 16;

/// y[r][m] = W x[r] for binary16 W ([m][k], k % 8 == 0) and `rows` (1 to
/// kMaxHalfGemvRows) FP32 activation rows: decode, verification and drafter
/// projections. Each row's FMA order depends only on k, so every batch width
/// rounds a row identically. Returns false without launching otherwise.
[[nodiscard]] bool LaunchHalfGemv(const void* w, const float* x, float* y,
                                  std::uint32_t rows, std::uint32_t m,
                                  std::uint32_t k, hipStream_t stream);

/// Rewrites `count` BF16 values as binary16 in place (round to nearest
/// even; exact for every BF16 magnitude from 2^-17 to 65504) and adds the
/// values that leave the binary16 range (or are not finite) to *overflow.
void ConvertBf16ToHalf(void* data, std::size_t count, std::uint32_t* overflow,
                       hipStream_t stream);

/// Requantizes Q8_0 rows ([rows][cols], cols % 256 == 0) as Q4_K with
/// ggml's reference Q4_K quantizer (weighted scale/min fit per 32 values,
/// 6-bit scales). `dst` holds rows * cols / 256 * 144 bytes.
void RepackQ8_0AsQ4K(const void* src, void* dst, std::uint32_t rows,
                     std::uint32_t cols, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_

#ifndef STRIX_CORE_QUANT_GGML_GEMM_HPP_
#define STRIX_CORE_QUANT_GGML_GEMM_HPP_

#include <cstddef>
#include <span>

#include "src/core/gguf_reader.hpp"

namespace strix::quant {

// Unified dequant/dot dispatch that routes by GgmlType. This is the single
// dispatch point for the quant_gemm module; it lives beside (Parallel Change)
// the older per-file switches in models/qwen/forward.cpp, which are removed in
// a later phase. It additionally handles the types the old CPU GEMV switch
// silently skipped (Q8_K, Q8_0, Q5_K) and fails loudly instead of returning
// 0.0F for unsupported types.

/// True when the unified dispatch can dequant/dot this type.
[[nodiscard]] bool IsSupported(core::GgmlType type) noexcept;

/// Dequantize one logical row of `k` elements from packed `src` into `dst`.
/// F32 is byte-copied; F16/BF16 are converted to float; Q3_K/Q4_K/Q5_K/Q6_K/
/// Q8_K/Q8_0 route to the canonical DequantizeQ* routines. Unsupported types
/// assert (loud fail) instead of returning 0.0F.
void Dequantize(core::GgmlType type, const void* src, float* dst,
                std::size_t k);

/// Dot-product of one row against `x`. F32/F16/BF16 are computed natively;
/// quantized types route to DotProductQ*. Unsupported types assert (loud fail).
float Dot(core::GgmlType type, const void* row, std::span<const float> x,
          std::size_t k);

}  // namespace strix::quant

#endif  // STRIX_CORE_QUANT_GGML_GEMM_HPP_

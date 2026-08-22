#ifndef STRIX_CORE_QUANT_GGML_DEQUANT_HPP_
#define STRIX_CORE_QUANT_GGML_DEQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace strix::quant {

/// Returns the encoded byte count for one logical row, or zero when the type
/// is unsupported or the element count is not block aligned.
[[nodiscard]] std::size_t QuantizedRowBytes(core::GgmlType type,
                                            std::size_t elements) noexcept;

// Standard 16-bit float helper
float Fp16ToFloat(std::uint16_t h) noexcept;

// Dequantize row of Q4_K to float
void DequantizeQ4_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q5_K to float
void DequantizeQ5_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q6_K to float
void DequantizeQ6_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q3_K to float
void DequantizeQ3_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q8_K to float
void DequantizeQ8_K(const void* src, float* dst, std::size_t k);

// Compute dot product of quantized row with FP32 vector
float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
// Compute dot product of Q5_K quantized row with FP32 vector
float DotProductQ5_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ8_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);

// Dequantize row of Q8_0 to float
void DequantizeQ8_0(const void* src, float* dst, std::size_t k);

// Compute dot product of Q8_0 quantized row with FP32 vector
float DotProductQ8_0(const void* row_data, std::span<const float> vec,
                     std::size_t k);

}  // namespace strix::quant

#endif  // STRIX_CORE_QUANT_GGML_DEQUANT_HPP_

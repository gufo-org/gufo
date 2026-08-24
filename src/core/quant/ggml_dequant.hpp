#ifndef STRIX_CORE_QUANT_GGML_DEQUANT_HPP_
#define STRIX_CORE_QUANT_GGML_DEQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace strix::quant {

// Canonical quantized block layouts. These are the authoritative field
// layouts for the dequant/dot paths. Relocated verbatim from the internal
// definitions in ggml_dequant.cpp so the header is the single source of
// truth; field order and byte sizes match the HIP path
// (src/models/qwen/hip/quant_ops.hpp) and the historical per-file copies
// (models/qwen/state.hpp, tests/). Do NOT redefine block_* locally. Sizes are
// static_asserted; this is the layout contract (see the parity test that
// proves field order against the byte-level spec).
#pragma pack(push, 1)
struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct block_q5_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};

struct block_q6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct block_q3_K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};

struct block_q8_K {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};

// Q8_0: fp16 scale + 32 int8 quantized values (QK=32).
struct block_q8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

/// Returns the encoded byte count for one logical quantized row, or zero when
/// the type is not quantized or the element count is not block aligned.
[[nodiscard]] std::size_t QuantizedRowBytes(core::GgmlType type,
                                            std::size_t elements) noexcept;

/// Returns the physical byte count for a tensor with `elements` logical
/// elements, or zero when the format is unsupported/misaligned.
[[nodiscard]] std::size_t EncodedSizeBytes(core::GgmlType type,
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

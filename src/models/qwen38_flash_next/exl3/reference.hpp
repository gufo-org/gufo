// Scalar host reference of EXL3 linear tensors and n-gram rows. Tests check
// the HIP kernels against it, and the loader uses it where a few decoded
// values are needed on the host.
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_REFERENCE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next::exl3 {

/// Host view of one EXL3 linear: y = H(W^T H(x * suh)) * svh, where W is the
/// decoded [in, out] trellis matrix and H the 128-point Hadamard per block.
struct HostLinear {
  const std::uint16_t* trellis{nullptr};  ///< [in/16][out/16][16 K] uint16
  const std::uint16_t* suh{nullptr};      ///< [in] fp16
  const std::uint16_t* svh{nullptr};      ///< [out] fp16
  std::uint32_t in{0};
  std::uint32_t out{0};
  int k{0};
};

/// All 65536 mul1 codebook values, indexed by trellis state.
[[nodiscard]] const float* Mul1Table();

/// The 256 decoded fp16 values of one tile in row-major [16][16] order.
void DecodeTile(const std::uint16_t* tile, int k, float out[256]);

/// The rotated-basis decoded matrix [in][out] (no transforms).
[[nodiscard]] std::vector<float> ReconstructInner(const HostLinear& w);

/// In-place orthonormal 128-point Hadamard over each 128 block of `v`.
void Hadamard128(std::span<float> v);

/// ExLlamaV3's activation path: x rounded to fp16, times suh (fp16),
/// Hadamard (fp32, rounded to fp16), decoded matvec in fp32, output
/// Hadamard and svh in fp32.
[[nodiscard]] std::vector<float> MatVec(const HostLinear& w,
                                        std::span<const float> x);

/// Decodes one n-gram row: value[i] = mul1(state_i) * scale + bias[i].
/// `bias` may be null.
void DecodeNgramRow(const std::uint16_t* row, int k, const std::uint16_t* bias,
                    float* out);

}  // namespace gufo::models::qwen38_flash_next::exl3

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_REFERENCE_HPP_

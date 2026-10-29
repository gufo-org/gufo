#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

/// Model-private vector products for short chunks (decode and verification),
/// streaming each weight row once for every activation row of the chunk.
namespace gufo::models::nemotron35_bolt::rocm {

/// Most activation rows the vector products take per weight matrix.
inline constexpr int kGemvMaxTokens = 8;

/// Words of the IQ2_XXS grid as BF16: 256 entries of four words, each word
/// two consecutive magnitudes (the lower one in the low half).
inline constexpr int kIq2xxsGridWords = 256 * 4;

/// Expands the GGUF IQ2_XXS grid (eight uint8 magnitudes per entry) into
/// BF16 pairs; every magnitude is exact in BF16.
void Iq2xxsGridBf16(const std::uint64_t* grid, std::uint32_t* words);

/// out = W x for IQ2_XXS weights. Activations are rounded to BF16, which
/// keeps 8 significant bits as Q8_1 does but without a shared block scale;
/// the signed grid values are exact in BF16 and products accumulate in F32.
struct Iq2xxsGemvArgs {
  const void* weights{nullptr};        ///< [experts][rows][cols / 256 blocks]
  const std::uint32_t* grid{nullptr};  ///< kIq2xxsGridWords device words
  const float* x{nullptr};             ///< activation rows of `cols` floats
  /// [n_tokens][used] expert per slot; null for a dense product.
  const std::int32_t* ids{nullptr};
  int used{1};
  /// Routed products read one x row per slot instead of one per token.
  bool x_per_slot{false};
  /// [n_tokens][used][rows] when routed, else [n_tokens][rows].
  float* out{nullptr};
  int rows{0};  ///< Multiple of 32.
  int cols{0};  ///< Multiple of 256.
  int n_tokens{0};
  bool relu2{false};       ///< Stores max(out, 0)^2.
  bool accumulate{false};  ///< Adds into out instead of storing.
};

hipError_t Iq2xxsGemv(const Iq2xxsGemvArgs& args, hipStream_t stream);

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMV_HPP_

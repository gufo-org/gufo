#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMM_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

/// Model-private matrix products for prefill-sized chunks.
namespace gufo::models::nemotron35_bolt::rocm {

/// Weight rows per tile of the NVFP4 product.
inline constexpr int kNvfp4GemmRows = 128;

/// out[t][r] = sum_c W[r][c] x[t][c] for NVFP4 weights and FP16
/// activations, without the per-tensor scale. Each 16-value sub-block is
/// decoded to FP16, which holds every E2M1 value times its UE4M3 scale
/// exactly, and FP16 WMMA accumulates in F32, so no per-block rescaling
/// interrupts the matrix units. rows must be a multiple of kNvfp4GemmRows,
/// cols of 64.
hipError_t Nvfp4Gemm(const void* weights, const __half* x, float* out, int rows,
                     int cols, int n_tokens, hipStream_t stream);

/// Weight rows and token columns per tile of the routed IQ2_XXS product.
inline constexpr int kIq2xxsRoutedRows = 256;
inline constexpr int kIq2xxsRoutedTokens = 32;

/// Slots of a [n_tokens][used] routing grouped by expert, and the work
/// tiles of a routed product: up to kIq2xxsRoutedTokens slots of one expert.
struct Iq2xxsRoute {
  std::int32_t* cursor{nullptr};     ///< [experts] scratch
  std::int32_t* order{nullptr};      ///< [slots] slot indices by expert
  int4* tiles{nullptr};              ///< {expert, first, count, 0}
  std::int32_t* num_tiles{nullptr};  ///< tiles in use
  int max_tiles{0};                  ///< Iq2xxsRouteMaxTiles
};

/// Tiles a routing of `slots` slots over `experts` experts can need.
[[nodiscard]] int Iq2xxsRouteMaxTiles(int slots, int experts);

/// Groups the slots of `ids` ([slots] expert per slot) by expert into
/// `route`. The order within an expert is arbitrary; a product's columns do
/// not depend on their position in a tile.
hipError_t Iq2xxsRouteSlots(const std::int32_t* ids, int slots, int experts,
                            const Iq2xxsRoute& route, hipStream_t stream);

/// Routed out[slot] = W[expert] x for IQ2_XXS experts on BF16 WMMA, for
/// prefill chunks. The signed grid values are exact in BF16 and x rounds to
/// BF16, as in the vector products; each 32-value group sums in F32 and is
/// scaled by its group scale, so only the F32 addition order differs.
struct Iq2xxsGemmArgs {
  const void* weights{nullptr};        ///< [experts][rows][cols / 256 blocks]
  const std::uint32_t* grid{nullptr};  ///< kIq2xxsGridWords device words
  /// One row of `cols` floats per slot with x_per_slot, else per token.
  const float* x{nullptr};
  bool x_per_slot{false};
  float* out{nullptr};  ///< One row of `rows` floats per slot.
  int rows{0};          ///< Multiple of kIq2xxsRoutedRows.
  int cols{0};          ///< Multiple of 256.
  bool relu2{false};    ///< Stores max(out, 0)^2.
  const Iq2xxsRoute* route{nullptr};
  int used{1};  ///< Slots per token.
};

hipError_t Iq2xxsRoutedGemm(const Iq2xxsGemmArgs& args, hipStream_t stream);

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_GEMM_HPP_

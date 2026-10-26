#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_OPS_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::embedding_gemma2::hip {

// Activations stay float32 between operations. Each GEMM consumes BF16
// activations and the checkpoint's BF16 weights and produces float32.
// Buffers typed `void*` hold BF16 values.

/// Gathers embedding rows and applies the embedding scale. Writes both the
/// float32 residual stream and the BF16 copy that feeds the per-layer
/// projection.
void LaunchEmbeddingLookup(const void* table_bf16,
                           const std::uint32_t* token_ids, float* state,
                           void* embeddings_bf16, std::size_t tokens,
                           std::size_t hidden, float scale, hipStream_t stream);

/// output_bf16 = bf16(rmsnorm(input) * weight).
void LaunchRmsNormToBf16(const float* input, const void* weight_bf16,
                         void* output_bf16, std::size_t rows, std::size_t width,
                         float epsilon, hipStream_t stream);

/// state += rmsnorm(update) * weight.
void LaunchRmsNormResidual(const float* update, const void* weight_bf16,
                           float* state, std::size_t rows, std::size_t width,
                           float epsilon, hipStream_t stream);

/// In place: values = rmsnorm(values * scale) * weight over each row.
void LaunchScaledRmsNorm(float* values, const void* weight_bf16,
                         std::size_t rows, std::size_t width, float scale,
                         float epsilon, hipStream_t stream);

void LaunchFloatToBf16(const float* input, void* output_bf16, std::size_t count,
                       hipStream_t stream);

/// Normalizes the fused projection output, applies rotary embeddings to the
/// query and key, and rounds all three to BF16. `qkv` rows hold the query,
/// key and value projections back to back. `rope_cos` and `rope_sin` are
/// [position][head_dim / 2] tables.
struct QkvLayout {
  std::uint32_t num_heads{0};
  std::uint32_t num_key_value_heads{0};
  std::uint32_t head_dim{0};
};
void LaunchQkvNormRope(const float* qkv, const void* query_norm_bf16,
                       const void* key_norm_bf16, void* query_bf16,
                       void* key_bf16, void* value_bf16,
                       const std::uint32_t* positions, std::size_t tokens,
                       const QkvLayout& layout, const float* rope_cos,
                       const float* rope_sin, float epsilon,
                       hipStream_t stream);

/// Tokens per column of the transposed value buffer: `tokens` rounded up to a
/// multiple of 32.
inline constexpr std::size_t ValueStride(std::size_t tokens) {
  return (tokens + 31U) / 32U * 32U;
}

/// Transposes values from [token][column] to [column][token] with
/// `ValueStride(tokens)` tokens per column, zero-filling the tokens past
/// `tokens`. `columns` is a multiple of 32. This is the layout
/// `LaunchAttention` reads.
void LaunchTransposeValues(const void* value_bf16, void* value_t_bf16,
                           std::size_t tokens, std::uint32_t columns,
                           hipStream_t stream);

/// Bidirectional attention with scale 1. A token attends to the keys of its
/// own sequence within `window` positions on either side; `window` equal to
/// zero means the whole sequence. Query and key are [token][head][column];
/// the value is transposed, [kv head][column][token], with
/// `ValueStride(tokens)` entries per column, as `LaunchTransposeValues`
/// writes it. Value reads run up to that stride, so the padding must be
/// finite. Returns false for an unsupported geometry.
[[nodiscard]] bool LaunchAttention(const void* query_bf16, const void* key_bf16,
                                   const void* value_t_bf16, void* context_bf16,
                                   const std::uint32_t* sequence_begin,
                                   const std::uint32_t* sequence_end,
                                   std::size_t tokens, const QkvLayout& layout,
                                   std::uint32_t window, hipStream_t stream);

/// output_bf16 = bf16(gelu_tanh(gate) * up) with gate and up stored back to
/// back in each `2 * width` float32 row.
void LaunchGeglu(const float* gate_up, void* output_bf16, std::size_t rows,
                 std::size_t width, hipStream_t stream);

/// output_bf16 = bf16(gelu_tanh(gate) * per_layer_input).
void LaunchPleGate(const float* gate, const float* per_layer_input,
                   void* output_bf16, std::size_t count, hipStream_t stream);

/// state = (state + rmsnorm(update) * weight) * layer_scalar.
void LaunchPleResidual(const float* update, const void* weight_bf16,
                       const void* layer_scalar_bf16, float* state,
                       std::size_t rows, std::size_t width, float epsilon,
                       hipStream_t stream);

/// Mean-pools each sequence of `projected` rows and L2-normalizes the result.
/// Returns false for a `width` above 1,024.
[[nodiscard]] bool LaunchMeanPoolNormalize(
    const float* projected, const std::uint32_t* sequence_offsets,
    float* output, std::size_t sequences, std::size_t width,
    hipStream_t stream);

}  // namespace gufo::models::embedding_gemma2::hip
#endif

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_HIP_OPS_HPP_

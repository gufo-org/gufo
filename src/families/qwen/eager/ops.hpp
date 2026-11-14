#ifndef GUFO_FAMILIES_QWEN_EAGER_OPS_HPP_
#define GUFO_FAMILIES_QWEN_EAGER_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::qwen::eager {

/// How an RMSNorm applies its reciprocal root mean square. HF Qwen3 multiplies
/// in FP32 (`kFloat`); modules written as
/// `x * rsqrt(mean(x.float() ** 2) + eps).to(x.dtype) * weight` round the
/// factor to BF16 first (`kBfloat16`).
enum class RmsScale : std::uint8_t {
  kFloat,
  kBfloat16,
};

/// Causal attention with the eager checkpoint's BF16 QK, scaling, probability,
/// and output boundaries; the softmax reduction itself remains FP32.
void LaunchBfloat16Attention(const float* query, const float* key,
                             const float* value, float* key_cache,
                             float* value_cache, void* output_bfloat16,
                             std::size_t layer, std::size_t position,
                             std::size_t tokens, std::size_t capacity,
                             std::size_t heads, std::size_t kv_heads,
                             std::size_t head_dim, hipStream_t stream,
                             bool cache_written);

void LaunchBfloat16PerHeadRMSNorm(float* values, const float* weight,
                                  std::size_t batch_size,
                                  std::uint32_t num_heads,
                                  std::uint32_t head_dim, float epsilon,
                                  hipStream_t stream);

void LaunchBfloat16RoPEAndRoundV(float* query, float* key, float* value,
                                 std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t num_key_value_heads,
                                 std::uint32_t head_dim,
                                 std::uint32_t start_position, float rope_theta,
                                 hipStream_t stream);

/// Fuses the q/k per-head RMSNorm, the q/k rotation and the V rounding into one
/// launch. Equivalent to the standalone per-head RMSNorm and RoPE kernels,
/// including every intermediate BF16 rounding.
///
/// When `key_cache` is non-null this also stores the rotated key and the
/// rounded value at `start_position` of the shared attention cache, writing the
/// same bytes as the standalone cache-write kernel. The caller then asks the
/// attention launcher to skip its own write.
///
/// Returns false without launching when `head_dim` exceeds one workgroup, so
/// the caller can fall back to the separate launches.
[[nodiscard]] bool LaunchBfloat16QkNormRoPE(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    float* key_cache, float* value_cache, std::uint32_t layer_index,
    std::uint32_t max_context, hipStream_t stream);

/// Optionally adds `update` into `hidden`, then applies the checkpoint RMSNorm:
/// round normalization to BF16 before multiplying its weight. A null update
/// performs normalization alone, using the same reduction and rounding.
void LaunchBfloat16ResidualAddRMSNorm(
    float* hidden, const float* update, const float* weight,
    void* output_bfloat16, std::size_t batch_size, std::size_t dimension,
    float epsilon, hipStream_t stream, RmsScale scale = RmsScale::kFloat);

/// Per-head q/k RMSNorm, RoPE at `start_position + row` and V rounding, as
/// `LaunchBfloat16QkNormRoPE`, writing BF16 head-major results instead of
/// updating the inputs: query row r of head h goes to
/// `query_out[(h * query_stride + query_row + r) * head_dim]`, key/value rows
/// to `[(h * cache_stride + cache_row + r) * head_dim]`. Inputs are row-major
/// `[rows][heads * head_dim]` FP32. `head_dim` must be at most 128 and even.
void LaunchBfloat16QkNormRoPEHeadMajor(
    const float* query, const float* key, const float* value,
    const float* query_weight, const float* key_weight, std::size_t rows,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta,
    float epsilon, RmsScale scale, void* query_out, std::size_t query_stride,
    std::size_t query_row, void* key_out, void* value_out,
    std::size_t cache_stride, std::size_t cache_row, hipStream_t stream);

/// Scratch floats `LaunchBfloat16DecodeAttention` needs for `capacity` keys.
[[nodiscard]] std::size_t DecodeAttentionScratch(std::size_t capacity,
                                                 std::size_t heads);

/// One query row against a BF16 head-major cache (`[kv_heads][stride][128]`,
/// first `keys` rows visible). Splits the keys across workgroups and merges
/// the partial softmax, so long contexts fill the GPU. Scores and the softmax
/// stay FP32; probabilities are rounded to BF16 before the value product, as
/// PyTorch's fused BF16 attention does. `query` is BF16 `[heads][128]`;
/// `output` is BF16 `[heads * 128]`.
void LaunchBfloat16DecodeAttention(const void* query, const void* key_cache,
                                   const void* value_cache, std::size_t stride,
                                   std::size_t keys, std::size_t heads,
                                   std::size_t kv_heads, float* scratch,
                                   void* output, hipStream_t stream);

void LaunchRoundBfloat16InPlace(float* values, std::size_t count,
                                hipStream_t stream);

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t count, hipStream_t stream);

void LaunchBfloat16ResidualAdd(const float* residual, const float* update,
                               float* output, std::size_t count,
                               hipStream_t stream);

void LaunchBfloat16SwiGlu(const float* gate, const float* up, float* output,
                          void* output_bfloat16, std::size_t count,
                          hipStream_t stream);

void LaunchBfloat16BiasSilu(const float* input, const void* bias_bfloat16,
                            float* output, void* output_bfloat16,
                            std::size_t rows, std::size_t columns,
                            hipStream_t stream);

void LaunchBfloat16Bias(const float* input, const void* bias_bfloat16,
                        float* output, std::size_t rows, std::size_t columns,
                        hipStream_t stream);

/// Looks up every code in a frame-major `[frames, groups]` tensor and writes
/// the BF16-rounded sum of its group-specific codec embeddings as
/// frame-major `[frames, hidden_size]`.
///
/// `text_embedding` is optional. When present it holds one `hidden_size` row
/// that is added after rounding the codec sum to BF16. The final add rounds
/// again, matching the two upstream BF16 operations in one dispatch.
void LaunchBatchedCodecEmbeddingSum(
    const void* const* embedding_tables_bfloat16, const std::uint32_t* codes,
    const float* text_embedding, float* output, std::size_t frames,
    std::size_t groups, std::size_t hidden_size, hipStream_t stream);

/// Computes `output[(b * rows) + m] = sum_k weights[(m * columns) + k] *
/// inputs[(b * columns) + k]` for the decode-time projections, accumulating in
/// float32. Matches the row-major weight layout and column-major output of the
/// batched BF16 GEMM.
///
/// One workgroup reduces one output row, so every row issues eight
/// independently scheduled coalesced streams and the projection reaches the
/// gfx1151 DRAM roofline instead of the tile-shaped batched path. All `batch`
/// columns share a single pass over the weights. The reduction order is fixed,
/// which keeps the result reproducible.
///
/// Returns false without launching when the batch or layout is not supported,
/// so the caller can fall back to the batched GEMM.
[[nodiscard]] bool LaunchBfloat16Gemv(const void* weights_bfloat16,
                                      const void* inputs_bfloat16,
                                      float* output, std::size_t batch,
                                      std::size_t rows, std::size_t columns,
                                      hipStream_t stream);

/// One projection of a grouped GEMV: a `rows x columns` BF16 weight matrix and
/// where its `batch x rows` float32 result goes.
struct Bfloat16GemvGroup {
  const void* weights_bfloat16{nullptr};
  float* output{nullptr};
  std::size_t rows{0};
};

/// Largest number of projections a single grouped launch can cover. Decode
/// needs three (query, key, value) and two (gate, up).
constexpr std::size_t kBfloat16GemvMaxGroups = 3;

/// Runs several projections that share one input in a single dispatch.
///
/// Each workgroup still reduces exactly one output row with the same fixed
/// order as `LaunchBfloat16Gemv`, so the results are bit-identical to invoking
/// that function once per group; only the launch count changes. Decode issues
/// well over a thousand kernels per codec frame, and at the measured ~2.4 us
/// gfx1151 dispatch gap that count, not the arithmetic, is what the projections
/// still have to spare.
///
/// Returns false without launching when the group count, batch, or layout is
/// not supported, so the caller can fall back to per-group launches.
[[nodiscard]] bool LaunchBfloat16GemvGrouped(const Bfloat16GemvGroup* groups,
                                             std::size_t group_count,
                                             const void* inputs_bfloat16,
                                             std::size_t batch,
                                             std::size_t columns,
                                             hipStream_t stream);

}  // namespace gufo::qwen::eager
#endif

#endif  // GUFO_FAMILIES_QWEN_EAGER_OPS_HPP_

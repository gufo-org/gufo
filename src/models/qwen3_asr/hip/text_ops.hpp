#ifndef GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_
#define GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::qwen3_asr::hip {

/// Reproduces the Qwen3 BF16 q/k RMSNorm, RoPE, value rounding, and optional
/// float32 KV-cache write in one dispatch.
[[nodiscard]] bool LaunchTextQkNormRoPE(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    float* key_cache, float* value_cache, std::uint32_t layer_index,
    std::uint32_t max_context, hipStream_t stream);

/// Exact Qwen3 RMSNorm ordering: normalize in float32, round to BF16, multiply
/// by the BF16 scale, and round again.
void LaunchTextRMSNorm(const float* input, const float* weight,
                       void* output_bfloat16, std::size_t batch_size,
                       std::size_t dimension, float epsilon,
                       hipStream_t stream);

/// Adds a BF16-rounded residual and emits the following BF16 RMSNorm in one
/// pass. `hidden` retains the unnormalized layer output as float32 values that
/// are exactly representable as BF16.
void LaunchTextResidualAddRMSNorm(float* hidden, const float* update,
                                  const float* weight, void* output_bfloat16,
                                  std::size_t batch_size, std::size_t dimension,
                                  float epsilon, hipStream_t stream);

void LaunchTextSwiGLU(const float* gate, const float* up, void* output_bfloat16,
                      std::size_t count, hipStream_t stream);

void LaunchTextRoundBfloat16(float* values, std::size_t count,
                             hipStream_t stream);

void LaunchTextBfloat16ToFloat(const void* input_bfloat16, float* output,
                               std::size_t count, hipStream_t stream);

/// BF16 embedding-table row gather. Model-private replacement for the shared
/// qwen lookup, which additionally decodes quantized tables this checkpoint
/// never uses.
void LaunchTextEmbeddingLookup(const void* table_bfloat16,
                               const std::uint32_t* token_ids, float* output,
                               std::size_t batch_size, std::size_t hidden_size,
                               hipStream_t stream);

/// Batch-one BF16 decode GEMV, one wave per row. This is the geometry-agnostic
/// fallback for shapes LaunchTextDecodeGemv() rejects; it reproduces the shared
/// qwen wave32 single-row kernel's accumulation order exactly.
void LaunchTextRowGemv(const void* weights_bfloat16, const float* input,
                       float* output, std::size_t rows, std::size_t columns,
                       hipStream_t stream);

/// Multi-token attention over the float32 KV cache, one block per (head,
/// query).
///
/// QK, scaling, softmax output and final context use the official eager BF16
/// rounding boundaries. q/k/v normalization already appended to the cache.
void LaunchTextBatchedAttention(
    const float* query, const float* key_cache, const float* value_cache,
    float* out_context, void* out_context_bfloat16, std::uint32_t layer_index,
    std::uint32_t start_position, std::size_t tokens, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, hipStream_t stream);

/// Batch-one decode attention over the float32 KV cache.
///
/// Reproduces LaunchTextBatchedAttention arithmetic and raises the number of
/// loads in flight: the generic kernel issues one 16-byte key load
/// per wave per position and reduces it immediately, which leaves it bound by
/// load latency at roughly an eighth of the DRAM read rate. Score reductions
/// per position are independent and the context accumulator stays single and in
/// ascending position order, so results are bit-identical.
///
/// Returns false for any geometry it does not cover -- notably a context
/// shorter than the length at which the batched kernel switches accumulation
/// order, where the caller must keep using the batched kernel to stay
/// bit-identical.
[[nodiscard]] bool LaunchTextDecodeAttention(
    const float* query, const float* key_cache, const float* value_cache,
    float* out_context, void* out_context_bfloat16, std::uint32_t layer_index,
    std::uint32_t position, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    hipStream_t stream);

/// One weight tensor in a batch-one decode GEMV dispatch.
struct TextDecodeGemvTensor {
  const void* weights_bfloat16{nullptr};
  float* output{nullptr};
  std::uint32_t rows{0U};
};

/// Largest number of tensors one decode GEMV dispatch can cover.
constexpr std::uint32_t kTextDecodeGemvMaxTensors = 3;

/// Batch-one BF16 decode GEMV over up to `kTextDecodeGemvMaxTensors` weight
/// tensors that share a single activation vector, so q/k/v and gate/up each run
/// as one dispatch over their concatenated rows.
///
/// Two properties matter and both were measured on gfx1151 against the 240 GB/s
/// DRAM read ceiling (tools/bench/asr_decode_gemv_bench.hip):
///
///   * the activation vector is staged in LDS once per block, which stops it
///     from competing with the weight stream for the vector cache -- that
///     contention alone was a quarter of the kernel's runtime;
///   * concatenating rows keeps every SIMD occupied for the narrow projections,
///     where one wave per row leaves the machine less than half busy.
///
/// Each row's products are still summed in ascending `k` with a single
/// accumulator and reduced by the same butterfly, so results are bit-identical
/// to the unfused route and greedy decode reproduces the official token IDs.
///
/// Returns false when the request is outside the supported geometry -- the
/// caller must then fall back to the general route.
[[nodiscard]] bool LaunchTextDecodeGemv(const TextDecodeGemvTensor* tensors,
                                        std::uint32_t count,
                                        const void* input_bfloat16,
                                        std::size_t columns,
                                        hipStream_t stream);

/// Exact greedy maximum, choosing the lowest index only for equal logits.
/// Any nonfinite logit returns UINT32_MAX so generation fails closed.
///
/// `scratch` must hold at least TextArgmaxScratchElements() floats, which lets
/// the scan run across the whole grid instead of on a single compute unit. Both
/// routes select the same token: a maximum and a minimum index are order
/// independent. Passing nullptr uses the single-block route.
void LaunchTextArgmax(const float* logits, std::uint32_t* output_token,
                      std::size_t count, float* scratch, hipStream_t stream);

/// Float count LaunchTextArgmax() needs for its grid-wide scratch.
[[nodiscard]] std::size_t TextArgmaxScratchElements();

}  // namespace gufo::models::qwen3_asr::hip
#endif

#endif  // GUFO_MODELS_QWEN3_ASR_HIP_TEXT_OPS_HPP_

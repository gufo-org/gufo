#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/sampling.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::kernels {

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream);

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream);

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Thread-per-key reference route of the call above. Exposed so an equivalence
/// test can run both routes in one process; the production entry point picks
/// between them.
void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Wave-per-key route of the call above. Requires `head_dim` to be a multiple
/// of four so a K row can be read as `float4`.
void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Quantizes a BF16 weight matrix into a Q8_0 copy for the draft-only LM head.
/// `total_elements` must be a multiple of the Q8_0 block size.
void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream);

[[nodiscard]] std::size_t DFlashSelectorScratchElements(
    std::uint32_t vocab_size) noexcept;

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream);

struct DFlashSampling {
  float repeat_penalty{1.0F};
  float frequency_penalty{0.0F}, presence_penalty{0.0F};
  std::uint32_t repeat_last_n{0};
  const sampling::TokenPenalty* penalties{nullptr};
  std::uint32_t penalty_count{0};
  const sampling::TokenId* history{nullptr};
  std::uint32_t history_count{0};
};

/// A sequence that scores the draft vocabulary (draft_vocab.hpp) instead of
/// the whole head. Its logits rows hold vocabulary IDs 0..prefix-1 in their
/// first `prefix` columns; `logits` holds `rows` more scores per proposal for
/// the request's table, whose IDs are `ids` (-1 for an empty slot, whose
/// score must be -inf). The selector returns full vocabulary IDs, so the
/// transition codebooks, penalties and sampled candidates are unchanged.
struct DFlashSelectorVocab {
  std::uint32_t prefix{0};  ///< 0: the sequence scores the whole vocabulary
  const float* logits{nullptr};
  const std::int32_t* ids{nullptr};
  std::uint32_t rows{0};
};

struct DFlashSelectorSequence {
  std::uint32_t count;
  float temperature;
  DFlashSampling sampling;
  DFlashSelectorVocab vocab{};
};

// Proposal rows are contiguous by sequence. Each token sequence includes its
// anchor before the proposal slots; all other buffers contain proposals only.
// Logits rows are `logits_stride` apart (0: vocab_size). Partial buffers need
// DFlashSelectorScratchElements(vocab_size) per proposal. Requested
// candidates include one-hot probabilities for greedy sequences.
void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream, std::uint32_t logits_stride = 0);

/// Writes `rows` x `table_rows` draft-table scores: `source[row * stride +
/// ids[slot]]` when `source` is set (a full-vocabulary row), otherwise the
/// table's own projection already in `table_logits`. Empty slots (ID -1)
/// become -inf either way.
/// Copies the `count` rows `ids` names (device IDs, all valid) of a
/// row-major `source` into consecutive rows of `destination`. `row_bytes`
/// must be a multiple of four.
void LaunchDFlashGatherRows(const void* source, std::size_t row_bytes,
                            const std::int32_t* ids, void* destination,
                            std::uint32_t count, hipStream_t stream);

void LaunchDFlashDraftTableLogits(const float* source, std::uint32_t stride,
                                  const std::int32_t* ids, float* table_logits,
                                  std::uint32_t rows, std::uint32_t table_rows,
                                  hipStream_t stream);

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

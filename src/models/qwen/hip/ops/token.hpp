#ifndef GUFO_MODELS_QWEN_HIP_OPS_TOKEN_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_TOKEN_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

struct GpuSamplingWorkspace {
  float* adjusted_logits{nullptr};
  float* sorted_logits{nullptr};
  std::uint32_t* token_ids{nullptr};
  std::uint32_t* sorted_token_ids{nullptr};
  std::uint32_t* penalty_tokens{nullptr};
  std::uint32_t* penalty_counts{nullptr};
  std::uint32_t* draft_candidate_ids{nullptr};
  float* draft_candidate_probabilities{nullptr};
  std::uint32_t* speculative_accepted{nullptr};
  void* sort_temp_storage{nullptr};
  std::size_t sort_temp_storage_bytes{0};
  std::size_t vocab_size{0};
  std::size_t penalty_capacity{0};

  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    if (vocab_size == 0)
      return 0;
    return 2 * vocab_size * (sizeof(float) + sizeof(std::uint32_t)) +
           penalty_capacity * (3 * sizeof(std::uint32_t) + sizeof(float)) +
           sizeof(std::uint32_t) + sort_temp_storage_bytes;
  }
};

struct GpuSamplingParameters {
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  float repeat_penalty{1.0F};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
  float uniform{0.0F};
};

void AllocateGpuSamplingWorkspace(GpuSamplingWorkspace* workspace,
                                  std::size_t vocab_size,
                                  std::size_t penalty_capacity);
[[nodiscard]] std::size_t EstimateGpuSamplingWorkspaceBytes(
    std::size_t vocab_size, std::size_t penalty_capacity);
void FreeGpuSamplingWorkspace(GpuSamplingWorkspace* workspace) noexcept;

/// Samples one device-resident logit row and writes a single device token.
///
/// Penalty token/count spans are host-resident compact unique-token arrays.
void LaunchGPUSampling(const float* logits, std::uint32_t* out_token,
                       std::size_t vocab_size,
                       const GpuSamplingParameters& parameters,
                       const std::uint32_t* penalty_tokens,
                       const std::uint32_t* penalty_counts,
                       std::size_t penalty_count,
                       GpuSamplingWorkspace* workspace,
                       hipStream_t stream = nullptr);

/// Applies the target sampling policy to one device-resident verification row,
/// accepts `draft_token` with the lossless speculative ratio, or samples the
/// residual target-minus-draft distribution after rejection.
///
/// Draft candidate spans and penalty spans are host-resident compact arrays.
void LaunchGPUSpeculativeSampling(
    const float* logits, std::uint32_t* out_token, std::uint32_t* out_accepted,
    std::size_t vocab_size, const GpuSamplingParameters& parameters,
    std::uint32_t draft_token, float draft_token_probability,
    const std::uint32_t* draft_candidate_ids,
    const float* draft_candidate_probabilities,
    std::size_t draft_candidate_count, float acceptance_uniform,
    float residual_uniform, const std::uint32_t* penalty_tokens,
    const std::uint32_t* penalty_counts, std::size_t penalty_count,
    GpuSamplingWorkspace* workspace, hipStream_t stream = nullptr);

/// Asynchronously copies embedding row for token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Asynchronously copies embedding row for *d_token_id into out_hidden
void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// De-interleaves [Q0 (head_dim), Gate0 (head_dim), Q1, Gate1, ...] into
/// separate Q and Gate buffers
void LaunchUnpackQG(const float* qg_interleaved, float* q_out, float* gate_out,
                    std::uint32_t num_heads, std::uint32_t head_dim,
                    hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) on Q and K heads
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream = nullptr);

/// Computes Rotary Position Embedding (RoPE) reading position from device
/// memory
void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                float rope_theta, hipStream_t stream = nullptr);

/// Computes parallel GPU argmax reduction over logits
void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream = nullptr);

/// Computes one argmax per row of a [batch_size, vocab_size] logits matrix.
void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            std::span<float> scratch,
                            hipStream_t stream = nullptr);

/// Batched Embedding lookup for B tokens
void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched Unpack Q and Gate across B tokens
void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream = nullptr);

/// Batched RoPE across B tokens starting at pos
void LaunchBatchedRoPE(float* q, float* k, std::size_t batch_size,
                       std::uint32_t num_heads, std::uint32_t num_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rotary_dim,
                       std::uint32_t start_pos, float rope_theta,
                       hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_TOKEN_HPP_

#ifndef STRIX_MODELS_QWEN_HIP_OPS_TOKEN_HPP_
#define STRIX_MODELS_QWEN_HIP_OPS_TOKEN_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

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

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_OPS_TOKEN_HPP_

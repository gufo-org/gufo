#ifndef STRIX_MODELS_QWEN_DFLASH_REFERENCE_HPP_
#define STRIX_MODELS_QWEN_DFLASH_REFERENCE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace strix::speculative {

/// Configuration for a DFlash-2 block-diffusion drafting model.
struct QwenDFlashConfig {
  /// Zero-based target layer outputs from which hidden representations are
  /// extracted.
  std::vector<std::uint32_t> target_layer_ids;

  /// Total diffusion block size, including the committed anchor token.
  std::uint32_t block_size{0};

  /// Diffusion mask token ID.
  tokenization::TokenId mask_token_id{0};

  /// Vocabulary size of the draft output head.
  std::size_t draft_vocab_size{0};

  /// Dynamic causal-convolution kernel width and channel grouping.
  std::uint32_t conv_kernel_size{0};
  std::uint32_t conv_group_size{0};

  /// Low-rank candidate selector topology.
  std::uint32_t selector_rank{0};
  std::uint32_t selector_top_k{0};

  /// Attention topology. DFlash-2 is non-causal inside the current block.
  std::uint32_t sliding_window{0};
  bool is_causal{false};

  /// Number of transformer blocks in the draft model.
  std::uint32_t num_layers{0};
};

/// DFlash-2 transformer weights plus the layer-local dynamic convolutions.
struct QwenDFlashLayerWeights {
  models::QwenLayerWeights transformer;

  /// [hidden_size, conv_kernel_size, 2]
  models::QwenTensorRef attention_conv_base;
  models::QwenTensorRef ffn_conv_base;

  /// [hidden_size, 2 * conv_kernel_size * (hidden_size / conv_group_size)]
  models::QwenTensorRef attention_conv_projection;
  models::QwenTensorRef ffn_conv_projection;
};

/// Immutable weights for DFlash-2 models.
struct QwenDFlashWeights {
  core::ModelConfig config;
  QwenDFlashConfig dflash_config;

  // Feature fusion encoder: compresses multi-layer target hidden states
  models::QwenTensorRef
      fc_projection;  // [n_target_layers * hidden_size, draft_hidden_size]
  models::QwenTensorRef fc_scale;  // optional scale factor
  models::QwenTensorRef fc_norm;   // [draft_hidden_size] RMSNorm

  // Shared / tied target embeddings and LM head
  models::QwenTensorRef token_embedding;
  models::QwenTensorRef output_norm;
  models::QwenTensorRef output;
  models::QwenTensorRef d2t;  // optional compact-to-target vocab index map

  // Draft transformer decoder layers and layer-local dynamic convolutions.
  std::vector<QwenDFlashLayerWeights> layers;

  // DFlash-2 low-rank candidate path selector.
  models::QwenTensorRef selector_predecessor;  // [selector_rank, vocab_size]
  models::QwenTensorRef selector_successor;    // [selector_rank, vocab_size]
  models::QwenTensorRef selector_hidden;       // [hidden_size, selector_rank]

  [[nodiscard]] static std::optional<QwenDFlashWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Stateful CPU oracle and reference pipeline for DFlash (v1) and DFlash-2
/// drafting.
class QwenDFlashReference final {
public:
  [[nodiscard]] static std::unique_ptr<QwenDFlashReference> Create(
      const std::shared_ptr<const core::GgufReader>& reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashReference>
  CreateWithTiedWeights(
      const std::shared_ptr<const core::GgufReader>& reader,
      const std::shared_ptr<const core::GgufReader>& tied_reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  /// Phase 1: Ingests concatenated multi-layer target hidden states from prompt
  /// prefill or verified steps, projects them via the fusion encoder, and seeds
  /// the draft KV cache. features layout: [num_tokens, target_layer_ids.size()
  /// * target_hidden_size]
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens);

  /// Phase 2: Performs non-causal parallel block diffusion to propose a draft
  /// sequence. Given the committed anchor token at `current_pos - 1`, generates
  /// up to `draft_count` tokens.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, std::vector<float>* out_confidences = nullptr);

  /// Computes proposal logits [draft_count, vocab_size] for a block containing
  /// one anchor followed by draft_count masked proposal positions.
  void ComputeBlockLogits(tokenization::TokenId anchor_token,
                          std::uint32_t current_pos, std::uint32_t draft_count,
                          std::vector<float>& out_logits);

  /// Applies one side of the layer-local grouped dynamic causal convolution.
  static void ApplyGroupedDynamicCausalConv(
      std::span<const float> input, std::span<const float> dynamic_coefficients,
      std::size_t num_tokens, std::size_t hidden_size, std::size_t kernel_size,
      std::size_t group_size, std::size_t side,
      const models::QwenTensorRef& base_kernel, std::span<float> output);

  /// Selects the greedy top-k path conditioned on the anchor and prior choice.
  static void SelectCandidatePath(
      std::span<const float> normalized_hidden,
      std::span<const float> proposal_logits, std::size_t num_tokens,
      std::size_t hidden_size, std::size_t vocab_size,
      std::size_t selector_rank, std::size_t selector_top_k,
      const models::QwenTensorRef& selector_predecessor,
      const models::QwenTensorRef& selector_successor,
      const models::QwenTensorRef& selector_hidden,
      tokenization::TokenId anchor_token,
      std::vector<tokenization::TokenId>& out_tokens,
      std::vector<float>* out_confidences = nullptr);

  [[nodiscard]] const QwenDFlashWeights& GetWeights() const noexcept {
    return weights_;
  }

  [[nodiscard]] const QwenDFlashConfig& GetConfig() const noexcept {
    return weights_.dflash_config;
  }

  [[nodiscard]] std::uint32_t GetInjectedContextLength() const noexcept {
    return injected_context_len_;
  }

private:
  QwenDFlashReference(std::shared_ptr<const core::GgufReader> reader,
                      std::shared_ptr<const core::GgufReader> tied_reader,
                      QwenDFlashWeights weights, std::uint32_t max_context);

  std::shared_ptr<const core::GgufReader> reader_;
  std::shared_ptr<const core::GgufReader> tied_reader_;
  QwenDFlashWeights weights_;
  std::uint32_t max_context_{4096};
  std::uint32_t injected_context_len_{0};

  // Injected target Key and Value states per draft layer [max_context, kv_dim]
  std::vector<std::vector<float>> injected_k_;
  std::vector<std::vector<float>> injected_v_;

  // Scratch buffers
  std::vector<float> fused_enc_scratch_;
  std::vector<float> noise_block_scratch_;
  std::vector<float> layer_scratch_;
  std::vector<float> logits_scratch_;
};

}  // namespace strix::speculative

#endif  // STRIX_MODELS_QWEN_DFLASH_REFERENCE_HPP_

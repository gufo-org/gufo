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

/// Configuration for DFlash (v1) and DFlash-2 block-diffusion drafting models.
struct QwenDFlashConfig {
  /// Target model layer indices from which hidden representations are extracted.
  std::vector<std::uint32_t> target_layer_ids;

  /// Block size (maximum parallel draft length, e.g. 8 or 16).
  std::uint32_t block_size{16};

  /// Whether position 0 is treated as a committed anchor token.
  bool sample_from_anchor{true};

  /// Diffusion mask token ID.
  tokenization::TokenId mask_token_id{0};

  /// Vocabulary size of the draft model (may be smaller than target vocab when d2t is used).
  std::size_t draft_vocab_size{0};

  /// Low-rank dimension for DFlash-2 / DSpark Markov candidate path scoring.
  std::size_t markov_rank{0};

  /// Whether 2-tap dynamic depthwise convolution layers are enabled (DFlash-2).
  bool has_dynamic_conv{false};

  /// Whether candidate path selector (Markov / bilinear head) is enabled (DFlash-2).
  bool has_path_selector{false};

  /// Number of transformer blocks in the draft model.
  std::uint32_t num_layers{1};
};

/// Immutable weights for DFlash and DFlash-2 models.
struct QwenDFlashWeights {
  core::ModelConfig config;
  QwenDFlashConfig dflash_config;

  // Feature fusion encoder: compresses multi-layer target hidden states
  models::QwenTensorRef fc_projection;  // [n_target_layers * hidden_size, draft_hidden_size]
  models::QwenTensorRef fc_scale;       // optional scale factor
  models::QwenTensorRef fc_norm;        // [draft_hidden_size] RMSNorm

  // Shared / tied target embeddings and LM head
  models::QwenTensorRef token_embedding;
  models::QwenTensorRef output_norm;
  models::QwenTensorRef output;
  models::QwenTensorRef d2t;            // optional compact-to-target vocab index map

  // Draft transformer decoder layers
  std::vector<models::QwenLayerWeights> layers;

  // DFlash-2: 2-tap dynamic depthwise convolution parameters
  models::QwenTensorRef in_conv_weight;   // [draft_hidden_size, 2]
  models::QwenTensorRef in_conv_bias;     // [draft_hidden_size]
  models::QwenTensorRef out_conv_weight;  // [draft_hidden_size, 2]
  models::QwenTensorRef out_conv_bias;    // [draft_hidden_size]

  // DFlash-2 / DSpark: Parallel bilinear candidate path selector & confidence gate
  models::QwenTensorRef markov_w1;        // [markov_rank, vocab_size]
  models::QwenTensorRef markov_w2;        // [markov_rank, draft_vocab_size]
  models::QwenTensorRef conf_proj;        // [draft_hidden_size + markov_rank, 1]
  models::QwenTensorRef conf_bias;        // [1]

  [[nodiscard]] static std::optional<QwenDFlashWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Stateful CPU oracle and reference pipeline for DFlash (v1) and DFlash-2 drafting.
class QwenDFlashReference final {
 public:
  [[nodiscard]] static std::unique_ptr<QwenDFlashReference> Create(
      std::shared_ptr<const core::GgufReader> reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashReference> CreateWithTiedWeights(
      std::shared_ptr<const core::GgufReader> reader,
      std::shared_ptr<const core::GgufReader> tied_reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  /// Phase 1: Ingests concatenated multi-layer target hidden states from prompt prefill or
  /// verified steps, projects them via the fusion encoder, and seeds the draft KV cache.
  /// features layout: [num_tokens, target_layer_ids.size() * target_hidden_size]
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens);

  /// Phase 2: Performs non-causal parallel block diffusion to propose a draft sequence.
  /// Given the committed anchor token at `current_pos - 1`, generates up to `draft_count` tokens.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, std::vector<float>* out_confidences = nullptr);

  /// Computes raw output logits [draft_count, vocab_size] for the diffusion block.
  void ComputeBlockLogits(tokenization::TokenId anchor_token,
                          std::uint32_t current_pos, std::uint32_t draft_count,
                          std::vector<float>& out_logits);

  /// Helper: applies 2-tap grouped depthwise convolution along sequence length.
  static void ApplyDynamicConv2Tap(std::span<float> sequence_hidden,
                                   std::size_t num_tokens, std::size_t hidden_size,
                                   const models::QwenTensorRef& weight,
                                   const models::QwenTensorRef& bias);

  /// Helper: applies bilinear / Markov candidate path scoring across block positions.
  static void ApplyMarkovPathScoring(
      std::span<float> block_logits, std::size_t num_tokens,
      std::size_t vocab_size, std::size_t draft_vocab_size,
      const models::QwenTensorRef& w1, const models::QwenTensorRef& w2,
      const models::QwenTensorRef& d2t, tokenization::TokenId anchor_token,
      bool sample_from_anchor, std::vector<tokenization::TokenId>& out_tokens);

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

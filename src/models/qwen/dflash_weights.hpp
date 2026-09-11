#ifndef GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::speculative {

/// Configuration for a DFlash-2 block-diffusion drafting model.
struct QwenDFlashConfig {
  /// Zero-based target layer outputs from which hidden representations are
  /// extracted.
  std::vector<std::uint32_t> target_layer_ids;

  /// Total diffusion block size, including the committed anchor token.
  std::uint32_t block_size{0};

  /// Diffusion mask token ID.
  tokenization::TokenId mask_token_id{0};

  /// Dynamic causal-convolution kernel width and channel grouping.
  std::uint32_t conv_kernel_size{0};
  std::uint32_t conv_group_size{0};

  /// Low-rank candidate selector topology.
  std::uint32_t selector_rank{0};
  std::uint32_t selector_top_k{0};

  /// Attention topology. DFlash-2 is non-causal inside the current block.
  std::uint32_t sliding_window{0};

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
  models::QwenTensorRef fc_norm;  // [draft_hidden_size] RMSNorm

  // Shared / tied target embeddings and LM head
  models::QwenTensorRef token_embedding;
  models::QwenTensorRef output_norm;
  models::QwenTensorRef output;

  // Draft transformer decoder layers and layer-local dynamic convolutions.
  std::vector<QwenDFlashLayerWeights> layers;

  // DFlash-2 low-rank candidate path selector.
  models::QwenTensorRef selector_predecessor;  // [selector_rank, vocab_size]
  models::QwenTensorRef selector_successor;    // [selector_rank, vocab_size]
  models::QwenTensorRef selector_hidden;       // [hidden_size, selector_rank]

  [[nodiscard]] static std::optional<QwenDFlashWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_

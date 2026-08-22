#ifndef STRIX_MODELS_QWEN_STATE_HPP_
#define STRIX_MODELS_QWEN_STATE_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace strix::models {

/// Generic reference to a mapped tensor supporting F32 and BF16 formats.
struct QwenTensorRef {
  const void* data = nullptr;
  core::GgmlType type = core::GgmlType::kF32;
  std::size_t num_elements = 0;

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }

  [[nodiscard]] float Get(std::size_t index) const noexcept {
    if (type == core::GgmlType::kF32) {
      return static_cast<const float*>(data)[index];
    }
    if (type == core::GgmlType::kBF16) {
      const auto u16 = static_cast<const std::uint16_t*>(data)[index];
      const std::uint32_t u32 = static_cast<std::uint32_t>(u16) << 16;
      float f = 0.0F;
      std::memcpy(&f, &u32, sizeof(float));
      return f;
    }
    if (type == core::GgmlType::kQ8_K) {
      constexpr std::size_t kBlockSize = 256;
      struct Q8KBlock {
        float d;
        std::int8_t qs[kBlockSize];
        std::int16_t bsums[16];
      };
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block = reinterpret_cast<const Q8KBlock*>(data) + block_idx;
      return block->d * static_cast<float>(block->qs[pos]);
    }
    if (type == core::GgmlType::kQ8_0) {
      // Q8_0 block: fp16 scale + 32 int8 values (QK=32, 34 bytes).
      constexpr std::size_t kBlockSize = 32;
      struct Q8_0Block {
        std::uint16_t d;
        std::int8_t qs[kBlockSize];
      };
      static_assert(sizeof(Q8_0Block) == 34);
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block = reinterpret_cast<const Q8_0Block*>(data) + block_idx;
      return strix::quant::Fp16ToFloat(block->d) *
             static_cast<float>(block->qs[pos]);
    }
    // kQ5_K / kQ6_K Get() dequant is deferred to the quant support phase (they
    // are matmul weights never read via this per-element accessor).
    return 0.0F;
  }

  [[nodiscard]] std::span<const float> AsFloatSpan() const noexcept {
    if (type == core::GgmlType::kF32 && data != nullptr) {
      return {static_cast<const float*>(data), num_elements};
    }
    return {};
  }
};

/// Tensor weight references for a single transformer layer block (Linear SSM or
/// Full Attention).
struct QwenLayerWeights {
  bool is_full_attention = false;

  // Common Layer Norms
  QwenTensorRef attn_norm;
  QwenTensorRef ffn_norm;

  // Full Attention Weights (every 4th block)
  QwenTensorRef attn_q;
  QwenTensorRef attn_k;
  QwenTensorRef attn_v;
  QwenTensorRef attn_output;
  QwenTensorRef attn_q_norm;
  QwenTensorRef attn_k_norm;

  // Linear Attention / SSM Weights (3 of 4 blocks)
  QwenTensorRef attn_qkv;
  QwenTensorRef attn_gate;
  QwenTensorRef ssm_a;
  QwenTensorRef ssm_conv1d;
  QwenTensorRef ssm_dt;
  QwenTensorRef ssm_alpha;
  QwenTensorRef ssm_beta;
  QwenTensorRef ssm_norm;
  QwenTensorRef ssm_out;

  // Feed Forward Network
  QwenTensorRef ffn_gate;
  QwenTensorRef ffn_up;
  QwenTensorRef ffn_down;
};

/// Full model tensor references mapped directly from GGUF storage.
struct QwenModelWeights {
  core::ModelConfig config;
  QwenTensorRef token_embd;
  std::vector<QwenLayerWeights> layers;
  QwenTensorRef output_norm;
  QwenTensorRef output;

  /// Loads and binds weights from a GgufReader.
  [[nodiscard]] static std::optional<QwenModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Contiguous Key-Value Cache for auto-regressive generation.
class QwenKvCache {
public:
  QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
              std::uint32_t max_context, std::uint32_t head_dim);

  void Reset() noexcept { current_pos_ = 0; }
  [[nodiscard]] std::uint32_t GetCurrentPos() const noexcept {
    return current_pos_;
  }
  void AdvancePos() noexcept { ++current_pos_; }

  [[nodiscard]] std::span<float> GetKeySlice(std::uint32_t layer,
                                             std::uint32_t kv_head,
                                             std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetKeySlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

  [[nodiscard]] std::span<float> GetValueSlice(std::uint32_t layer,
                                               std::uint32_t kv_head,
                                               std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetValueSlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

private:
  std::uint32_t num_layers_;
  std::uint32_t num_kv_heads_;
  std::uint32_t max_context_;
  std::uint32_t head_dim_;
  std::uint32_t current_pos_{0};
  std::vector<float> k_data_;
  std::vector<float> v_data_;
};

/// Preallocated activation scratch arena for zero-heap-allocation decode
/// passes.
struct QwenScratchArena {
  explicit QwenScratchArena(const core::ModelConfig& config);

  std::vector<float> buffer;
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> attn_scores;
  std::span<float> attn_out;
  std::span<float> mlp_gate;
  std::span<float> mlp_up;
  std::span<float> mlp_act;
  std::span<float> mlp_out;
  std::span<float> ssm_qkv;
  std::span<float> ssm_gate;
  std::span<float> ssm_out_buf;
  std::span<float> logits;
};

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_STATE_HPP_

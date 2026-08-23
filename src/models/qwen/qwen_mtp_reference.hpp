#ifndef STRIX_MODELS_QWEN_QWEN_MTP_REFERENCE_HPP_
#define STRIX_MODELS_QWEN_QWEN_MTP_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/qwen_ssm.hpp"
#include "src/models/qwen/qwen_state.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

namespace strix::speculative {

struct QwenMtpWeights {
  core::ModelConfig config;
  models::QwenTensorRef token_embedding;
  models::QwenTensorRef output;
  models::QwenTensorRef embedding_norm;
  models::QwenTensorRef hidden_norm;
  models::QwenTensorRef fusion_projection;
  models::QwenTensorRef shared_head_norm;
  models::QwenLayerWeights layer;

  [[nodiscard]] static std::optional<QwenMtpWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Stateful CPU oracle for the single Qwen3.8 MTP layer. The first call uses a
/// target-model hidden state; subsequent calls may feed LastHidden() back in.
class QwenMtpReference final {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpReference> Create(
      std::shared_ptr<const core::GgufReader> reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenMtpReference> CreateWithTiedWeights(
      std::shared_ptr<const core::GgufReader> reader,
      std::shared_ptr<const core::GgufReader> tied_reader,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  [[nodiscard]] std::span<const float> ForwardHidden(
      tokenization::TokenId input_token,
      std::span<const float> target_or_feedback_hidden, std::uint32_t position);

  [[nodiscard]] float ComputeLogit(std::uint32_t token_id) const noexcept;

  [[nodiscard]] std::span<const float> LastHidden() const noexcept {
    return feedback_hidden_;
  }

  [[nodiscard]] const QwenMtpWeights& GetWeights() const noexcept {
    return weights_;
  }

private:
  QwenMtpReference(std::shared_ptr<const core::GgufReader> reader,
                   std::shared_ptr<const core::GgufReader> tied_reader,
                   QwenMtpWeights weights, std::uint32_t max_context);

  std::shared_ptr<const core::GgufReader> reader_;
  std::shared_ptr<const core::GgufReader> tied_reader_;
  QwenMtpWeights weights_;
  models::QwenKvCache kv_cache_;
  models::QwenSsmCache ssm_cache_;
  models::QwenScratchArena arena_;
  std::uint32_t max_context_;
  std::uint32_t next_position_{0};
  std::vector<float> embedding_;
  std::vector<float> normalized_embedding_;
  std::vector<float> normalized_hidden_;
  std::vector<float> fusion_input_;
  std::vector<float> feedback_hidden_;
};

}  // namespace strix::speculative

#endif  // STRIX_MODELS_QWEN_QWEN_MTP_REFERENCE_HPP_

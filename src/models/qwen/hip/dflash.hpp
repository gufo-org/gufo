#ifndef STRIX_MODELS_QWEN_HIP_DFLASH_HPP_
#define STRIX_MODELS_QWEN_HIP_DFLASH_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/dflash_reference.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

/// Immutable GPU-visible DFlash / DFlash-2 model weights and topology.
class QwenDFlashGpuModel final {
 public:
  ~QwenDFlashGpuModel();

  QwenDFlashGpuModel(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel& operator=(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel(QwenDFlashGpuModel&&) = delete;
  QwenDFlashGpuModel& operator=(QwenDFlashGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenDFlashGpuModel> Create(
      std::shared_ptr<const core::GgufReader> dflash_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenDFlashWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const speculative::QwenDFlashConfig& GetDFlashConfig() const noexcept {
    return weights_.dflash_config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }

 private:
  QwenDFlashGpuModel(std::shared_ptr<const core::GgufReader> dflash_reader,
                     std::shared_ptr<const QwenGpuModel> target_model,
                     speculative::QwenDFlashWeights weights,
                     std::vector<void*> allocations,
                     std::size_t packed_weight_bytes);

  std::shared_ptr<const core::GgufReader> dflash_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenDFlashWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
};

/// Session executor for GPU DFlash / DFlash-2 parallel block drafting.
class QwenDFlashGpuExecutor final {
 public:
  ~QwenDFlashGpuExecutor();

  QwenDFlashGpuExecutor(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor& operator=(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor(QwenDFlashGpuExecutor&&) = delete;
  QwenDFlashGpuExecutor& operator=(QwenDFlashGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuExecutor> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  /// Ingests target multi-layer hidden states and injects K/V into the draft cache.
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens);

  /// Executes non-causal parallel block diffusion drafting.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, std::vector<float>* out_confidences = nullptr);

  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] std::size_t GetTargetFeaturesSize() const noexcept {
    return model_->GetDFlashConfig().target_layer_ids.size() * GetHiddenSize();
  }
  [[nodiscard]] std::uint32_t GetInjectedContextLength() const noexcept {
    return injected_context_len_;
  }
  [[nodiscard]] const QwenDFlashGpuModel& GetModel() const noexcept {
    return *model_;
  }

 private:
  QwenDFlashGpuExecutor(std::shared_ptr<const QwenDFlashGpuModel> model,
                        std::uint32_t max_context);

  void Allocate();
  void Free() noexcept;

  std::shared_ptr<const QwenDFlashGpuModel> model_;
  std::uint32_t max_context_{4096};
  std::uint32_t injected_context_len_{0};
  hipStream_t stream_{nullptr};

  // GPU Allocations for draft KV cache
  std::vector<float*> d_injected_k_;
  std::vector<float*> d_injected_v_;

  // GPU Scratch buffers
  float* d_target_features_{nullptr};
  float* d_fused_features_{nullptr};
  float* d_block_hidden_{nullptr};
  float* d_block_normed_{nullptr};
  float* d_q_{nullptr};
  float* d_k_block_{nullptr};
  float* d_v_block_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_gate_{nullptr};
  float* d_ffn_up_{nullptr};
  float* d_ffn_down_{nullptr};
  float* d_logits_{nullptr};
  std::uint32_t* d_out_token_{nullptr};
  unsigned long long* d_argmax_out_{nullptr};

  // Host buffers for asynchronous synchronization
  std::vector<float> h_target_features_;
  std::vector<float> h_logits_;
};

struct QwenDFlashGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{16};
};

/// Adapts the GPU DFlash executor to the repository's IDraftBackend speculative interface.
class QwenDFlashGpuDraftBackend final : public speculative::IDraftBackend {
 public:
  ~QwenDFlashGpuDraftBackend() override = default;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      QwenDFlashGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend> CreateFromGguf(
      std::string_view dflash_model_path,
      std::shared_ptr<const QwenGpuModel> target_model,
      QwenDFlashGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenDFlashGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

 private:
  QwenDFlashGpuDraftBackend(std::unique_ptr<QwenDFlashGpuExecutor> executor,
                            QwenDFlashGpuDraftConfig config);

  std::unique_ptr<QwenDFlashGpuExecutor> executor_;
  QwenDFlashGpuDraftConfig config_;
  std::vector<float> target_hidden_accumulator_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  std::uint32_t proposal_checkpoint_{0};
  tokenization::TokenId proposal_input_{0};
  bool primed_{false};
  bool proposal_active_{false};
  std::string last_error_;
};

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_DFLASH_HPP_

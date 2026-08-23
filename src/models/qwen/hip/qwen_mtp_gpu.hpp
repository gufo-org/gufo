#ifndef STRIX_MODELS_QWEN_HIP_QWEN_MTP_GPU_HPP_
#define STRIX_MODELS_QWEN_HIP_QWEN_MTP_GPU_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen/hip/qwen_gpu_executor.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/qwen_mtp_reference.hpp"
#include "src/tokenization/qwen_tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#if defined(ENGINE_ENABLE_XRT)
#include "src/core/xdna2/qwen_mtp_eh_proj.h"
#endif

namespace strix::hip {

enum class QwenMtpExecutionMode : std::uint8_t {
  kGpu,
  kHybridNpuEhProj,
};

struct QwenMtpHybridMetrics {
  std::uint64_t projection_count{0};
  double gpu_to_host_us{0.0};
  double activation_pack_us{0.0};
  double npu_command_us{0.0};
  double npu_end_to_end_us{0.0};
  double host_to_gpu_us{0.0};
};

/// Immutable GPU-visible MTP layer weights. Current GGUF K-quant matrices are
/// expanded once to BF16; the tied embedding and LM head remain shared with
/// the target model.
class QwenMtpGpuModel final {
public:
  ~QwenMtpGpuModel();

  QwenMtpGpuModel(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel& operator=(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel(QwenMtpGpuModel&&) = delete;
  QwenMtpGpuModel& operator=(QwenMtpGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel> Create(
      std::shared_ptr<const core::GgufReader> mtp_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenMtpWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }
  [[nodiscard]] double GetPackTimeSeconds() const noexcept {
    return pack_time_seconds_;
  }
  [[nodiscard]] const models::QwenTensorRef& GetRawFusionProjection()
      const noexcept {
    return raw_fusion_projection_;
  }

private:
  QwenMtpGpuModel(std::shared_ptr<const core::GgufReader> mtp_reader,
                  std::shared_ptr<const QwenGpuModel> target_model,
                  speculative::QwenMtpWeights weights,
                  models::QwenTensorRef raw_fusion_projection,
                  std::vector<void*> allocations,
                  std::size_t packed_weight_bytes, double pack_time_seconds);

  std::shared_ptr<const core::GgufReader> mtp_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenMtpWeights weights_;
  models::QwenTensorRef raw_fusion_projection_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  double pack_time_seconds_{0.0};
};

/// Mutable single-session executor for the Qwen3.8 layer-64 MTP graph.
class QwenMtpGpuExecutor final {
public:
  ~QwenMtpGpuExecutor();

  QwenMtpGpuExecutor(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor& operator=(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor(QwenMtpGpuExecutor&&) = delete;
  QwenMtpGpuExecutor& operator=(QwenMtpGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuExecutor> Create(
      std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
      std::string* error_msg = nullptr,
      QwenMtpExecutionMode execution_mode = QwenMtpExecutionMode::kGpu);

  void Reset() noexcept;
  void Rewind(std::uint32_t position);

  [[nodiscard]] tokenization::TokenId ForwardTargetHidden(
      tokenization::TokenId input_token, std::span<const float> target_hidden,
      std::uint32_t position, bool compute_logits = true);

  [[nodiscard]] tokenization::TokenId ForwardFeedback(
      tokenization::TokenId input_token, std::uint32_t position,
      bool compute_logits = true);

  [[nodiscard]] std::span<const float> CopyLastHidden();
  [[nodiscard]] std::span<const float> CopyLastLogits();

  [[nodiscard]] std::uint32_t GetNextPosition() const noexcept {
    return next_position_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] QwenMtpExecutionMode GetExecutionMode() const noexcept {
    return execution_mode_;
  }
  [[nodiscard]] const QwenMtpHybridMetrics& GetHybridMetrics() const noexcept {
    return hybrid_metrics_;
  }
  void ResetHybridMetrics() noexcept { hybrid_metrics_ = {}; }

private:
  QwenMtpGpuExecutor(std::shared_ptr<const QwenMtpGpuModel> model,
                     std::uint32_t max_context,
                     QwenMtpExecutionMode execution_mode);

  [[nodiscard]] tokenization::TokenId Run(tokenization::TokenId input_token,
                                          const float* hidden_input,
                                          std::uint32_t position,
                                          bool compute_logits);
  void Allocate();
  void Free() noexcept;

  std::shared_ptr<const QwenMtpGpuModel> model_;
  std::uint32_t max_context_;
  std::uint32_t next_position_{0};
  QwenMtpExecutionMode execution_mode_{QwenMtpExecutionMode::kGpu};
  QwenMtpHybridMetrics hybrid_metrics_;
  hipStream_t stream_{nullptr};

  float* d_target_hidden_{nullptr};
  float* d_embedding_{nullptr};
  float* d_fusion_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_qg_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_{nullptr};
  float* d_context_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_act_{nullptr};
  float* d_ffn_out_{nullptr};
  float* d_feedback_hidden_{nullptr};
  float* d_logits_{nullptr};
  float* d_kv_cache_{nullptr};
  void* d_kv_cache_f16_{nullptr};
  float* d_split_k_scratch_{nullptr};
  std::uint32_t* d_out_token_{nullptr};

  std::vector<float> h_last_hidden_;
  std::vector<float> h_logits_;
#if defined(ENGINE_ENABLE_XRT)
  std::unique_ptr<xdna2::QwenMtpEhProjSession> npu_eh_proj_;
  std::vector<float> h_npu_input_;
  std::vector<float> h_npu_output_;
#endif
};

struct QwenMtpGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{4};
  QwenMtpExecutionMode execution_mode{QwenMtpExecutionMode::kGpu};
};

/// Stateful MTP draft provider backed by the production HIP executor.
class QwenMtpGpuDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> Create(
      std::shared_ptr<const QwenMtpGpuModel> model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> CreateFromGguf(
      std::string_view model_path,
      std::shared_ptr<const QwenGpuModel> target_model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return config_.execution_mode == QwenMtpExecutionMode::kHybridNpuEhProj
               ? "QwenMtpHybridNpuDraftBackend"
               : "QwenMtpGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;
  void Reset() noexcept override;

  [[nodiscard]] const std::string& GetLastError() const noexcept {
    return last_error_;
  }
  [[nodiscard]] const QwenMtpHybridMetrics& GetHybridMetrics() const noexcept {
    return executor_->GetHybridMetrics();
  }

private:
  QwenMtpGpuDraftBackend(std::unique_ptr<QwenMtpGpuExecutor> executor,
                         QwenMtpGpuDraftConfig config);

  std::unique_ptr<QwenMtpGpuExecutor> executor_;
  QwenMtpGpuDraftConfig config_;
  std::vector<float> target_hidden_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  tokenization::TokenId proposal_input_{0};
  std::uint32_t proposal_checkpoint_{0};
  bool primed_{false};
  bool proposal_active_{false};
  std::string last_error_;
};

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_QWEN_MTP_GPU_HPP_

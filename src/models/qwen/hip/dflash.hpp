#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/dflash_policy.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Optional synchronous host observation for the independent reference test.
/// The span is valid only during the callback; production leaves it empty.
using DFlashTrace =
    std::function<void(std::string_view, std::span<const float>)>;

class QwenDFlashGpuSnapshot final {
public:
  ~QwenDFlashGpuSnapshot();

  QwenDFlashGpuSnapshot(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot& operator=(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot(QwenDFlashGpuSnapshot&&) = delete;
  QwenDFlashGpuSnapshot& operator=(QwenDFlashGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  QwenDFlashGpuSnapshot() = default;

  void* d_k_{nullptr};
  void* d_v_{nullptr};
  std::size_t elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::uint32_t history_capacity_{0};
  std::size_t payload_bytes_{0};

  friend class QwenDFlashGpuExecutor;
};

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

  [[nodiscard]] const speculative::QwenDFlashWeights& GetWeights()
      const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const speculative::QwenDFlashConfig& GetDFlashConfig()
      const noexcept {
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

  /// Ingests target multi-layer hidden states and injects K/V into the draft
  /// cache.
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens,
                           const DFlashTrace& trace = {});

  /// Executes non-causal parallel block diffusion drafting.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, float temperature = 0.0F,
      std::span<const float> sample_uniforms = {},
      std::vector<float>* out_confidences = nullptr,
      std::vector<tokenization::TokenId>* out_candidate_ids = nullptr,
      std::vector<float>* out_candidate_probabilities = nullptr,
      const DFlashTrace& trace = {});

  [[nodiscard]] std::unique_ptr<QwenDFlashGpuSnapshot> SaveSnapshot() const;
  void RestoreSnapshot(const QwenDFlashGpuSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;

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

  bool InjectTargetContextChunk(std::span<const float> target_features,
                                std::uint32_t position,
                                std::uint32_t num_tokens,
                                const DFlashTrace& trace);
  void Allocate();
  void Free() noexcept;
  void PrewarmBlockGemms();
  void RunBlockGemm(const models::QwenTensorRef& weight, const float* input,
                    float* output, std::size_t batch_size,
                    std::size_t output_size, std::size_t input_size);
  void RunInjectGemm(const models::QwenTensorRef& weight, const float* input,
                     float* output, std::size_t num_tokens,
                     std::size_t output_size, std::size_t input_size);

  std::shared_ptr<const QwenDFlashGpuModel> model_;
  std::uint32_t max_context_{4096};
  std::uint32_t injected_context_len_{0};
  std::uint32_t history_capacity_{0};
  std::uint32_t injection_capacity_{0};
  hipStream_t stream_{nullptr};
  hipblasHandle_t hipblas_handle_{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm_;

  // GPU Allocations for draft KV cache
  std::vector<float*> d_injected_k_;
  std::vector<float*> d_injected_v_;

  // GPU Scratch buffers
  float* d_target_features_{nullptr};
  float* d_fused_features_{nullptr};
  float* d_block_hidden_{nullptr};
  float* d_block_normed_{nullptr};
  float* d_conv_hidden_{nullptr};
  float* d_dynamic_coefficients_{nullptr};
  float* d_q_{nullptr};
  float* d_k_block_{nullptr};
  float* d_v_block_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_gate_{nullptr};
  float* d_ffn_up_{nullptr};
  float* d_ffn_down_{nullptr};
  float* d_logits_{nullptr};
  float* d_selector_hidden_{nullptr};
  float* d_selector_partial_scores_{nullptr};
  std::uint32_t* d_selector_partial_ids_{nullptr};
  std::uint32_t* d_selector_candidate_ids_{nullptr};
  float* d_selector_candidate_probabilities_{nullptr};
  float* d_selector_uniforms_{nullptr};
  float* d_confidences_{nullptr};
  std::uint32_t* d_out_token_{nullptr};
  hip_bfloat16* d_bf16_input_{nullptr};
};

struct QwenDFlashGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  speculative::DFlashDraftPolicy policy{
      speculative::DFlashDraftPolicy::kAdaptive};
};

/// Adapts the GPU DFlash executor to the repository's IDraftBackend speculative
/// interface.
class QwenDFlashGpuDraftBackend final : public speculative::IDraftBackend {
public:
  ~QwenDFlashGpuDraftBackend() override = default;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      QwenDFlashGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend>
  CreateFromGguf(std::string_view dflash_model_path,
                 std::shared_ptr<const QwenGpuModel> target_model,
                 QwenDFlashGpuDraftConfig config = {},
                 std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenDFlashGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return executor_->GetModel().GetDFlashConfig().target_layer_ids;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  void Reset() noexcept override;
  void BeginRequest() noexcept override { controller_.Reset(); }

private:
  void InjectPendingFeatures(std::uint32_t position);
  QwenDFlashGpuDraftBackend(std::unique_ptr<QwenDFlashGpuExecutor> executor,
                            QwenDFlashGpuDraftConfig config);
  [[nodiscard]] speculative::DraftProposal ProposeImpl(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state);

  std::unique_ptr<QwenDFlashGpuExecutor> executor_;
  QwenDFlashGpuDraftConfig config_;
  speculative::DFlashLengthController controller_;
  std::vector<float> pending_target_features_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  bool primed_{false};
  bool proposal_active_{false};
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_HPP_

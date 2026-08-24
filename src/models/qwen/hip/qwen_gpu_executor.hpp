#ifndef STRIX_MODELS_QWEN_HIP_QWEN_GPU_EXECUTOR_HPP_
#define STRIX_MODELS_QWEN_HIP_QWEN_GPU_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/qwen_forward.hpp"
#include "src/models/qwen/qwen_generator.hpp"
#include "src/models/qwen/qwen_state.hpp"
#include "src/models/qwen/qwen_tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/models/qwen/hip/qwen_execution_policy.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

class HipblasLtGemm;

struct QwenGpuWeightRegion {
  const void* host_data{nullptr};
  void* device_data{nullptr};
  std::size_t size{0};
  bool owns_device_memory{false};
  bool host_registered{false};
};

/// Immutable GPU-visible Qwen model resources shared by executor sessions.
class QwenGpuModel {
public:
  QwenGpuModel(std::shared_ptr<const core::GgufReader> reader,
               models::QwenModelWeights weights,
               std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
               std::vector<QwenGpuWeightRegion> weight_regions);
  ~QwenGpuModel();

  QwenGpuModel(const QwenGpuModel&) = delete;
  QwenGpuModel& operator=(const QwenGpuModel&) = delete;
  QwenGpuModel(QwenGpuModel&&) = delete;
  QwenGpuModel& operator=(QwenGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenGpuModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const models::QwenModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetWeightRegionCount() const noexcept {
    return weight_regions_.size();
  }

private:
  // Keep the mapped GGUF storage alive until every registered region is
  // released.
  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
};

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
class QwenGpuArena {
public:
  explicit QwenGpuArena(const core::ModelConfig& config,
                        std::uint32_t max_context = 4096);
  ~QwenGpuArena();

  QwenGpuArena(const QwenGpuArena&) = delete;
  QwenGpuArena& operator=(const QwenGpuArena&) = delete;
  QwenGpuArena(QwenGpuArena&&) noexcept;
  QwenGpuArena& operator=(QwenGpuArena&&) noexcept;

  void Reset() noexcept;
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  [[nodiscard]] bool BeginSsmReplayCapture();
  void DisableSsmReplayCapture() noexcept;
  void MarkSsmReplayPosition(std::uint32_t position) noexcept;
  [[nodiscard]] bool CanReplaySsmPosition(
      std::uint32_t position) const noexcept;
  [[nodiscard]] SsmReplayCapture GetSsmReplayCapture() const noexcept;
  [[nodiscard]] const float* GetReplayQkv(std::uint32_t layer,
                                          std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayAlpha(std::uint32_t layer,
                                            std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayBeta(std::uint32_t layer,
                                           std::uint32_t position) const;

  float* d_hidden{nullptr};
  float* d_normed{nullptr};
  float* d_q{nullptr};
  float* d_k{nullptr};
  float* d_v{nullptr};
  float* d_attn_out{nullptr};
  float* d_ffn_gate{nullptr};
  float* d_ffn_up{nullptr};
  float* d_ffn_act{nullptr};
  float* d_ffn_out{nullptr};
  float* d_ssm_qkv{nullptr};
  float* d_conv_out{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_out{nullptr};
  float* d_alpha_buf{nullptr};
  float* d_beta_buf{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  float* d_kv_cache{nullptr};
  float* d_ssm_conv_state{nullptr};
  float* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};

  hipStream_t stream{nullptr};
  hipStream_t prefetch_stream{nullptr};
  hipEvent_t prefetch_event{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};
  hip_bfloat16* d_weights_bf16{nullptr};

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }

private:
  void FreeAll() noexcept;
  void AllocateRecurrentSnapshot();
  [[nodiscard]] bool AllocateSsmReplayLog();

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
  float* d_saved_ssm_conv_state_{nullptr};
  float* d_saved_ssm_deltanet_state_{nullptr};
  float* d_ssm_replay_qkv_{nullptr};
  float* d_ssm_replay_alpha_{nullptr};
  float* d_ssm_replay_beta_{nullptr};
  std::uint32_t* d_ssm_replay_enabled_{nullptr};
  std::uint32_t saved_context_{0};
  std::uint32_t replay_last_position_{0};
  std::size_t replay_captured_positions_{0};
  bool has_saved_state_{false};
  bool replay_capture_active_{false};
};

/// End-to-end GPU model executor running directly on the gfx1151 RDNA 3.5 CUs.
class QwenGpuExecutor {
public:
  explicit QwenGpuExecutor(
      std::shared_ptr<const QwenGpuModel> model,
      std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  ~QwenGpuExecutor();

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> Create(
      std::shared_ptr<const QwenGpuModel> model,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());

  /// Generates tokens auto-regressively on GPU with streaming callback.
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }

  [[nodiscard]] const QwenGpuModel& GetModel() const noexcept {
    return *model_;
  }

  [[nodiscard]] const std::shared_ptr<const QwenGpuModel>& GetSharedModel()
      const noexcept {
    return model_;
  }

  [[nodiscard]] const QwenExecutionPolicy& GetExecutionPolicy() const noexcept {
    return policy_;
  }

  /// Runs one single token forward step on GPU, returning next token ID.
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos = 0, bool compute_logits = true);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  /// Enables host capture of every final-layer prompt hidden state. Disabled
  /// by default so ordinary prefill does not incur device-to-host copies.
  void SetPromptHiddenCapture(bool enabled);

  /// Returns the flattened [prompt_tokens, hidden_size] capture from the most
  /// recent ForwardPromptBatch call.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates() const noexcept {
    return h_prompt_hidden_;
  }

  /// Copies the final-layer hidden state from the most recent forward pass.
  [[nodiscard]] std::span<const float> CopyLastHidden();

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return arena_.GetMaxContext();
  }

  /// Resets GPU cache and recurrent states in the arena.
  void Reset() noexcept;
  void SaveState(std::uint32_t valid_context);
  void RestoreState();

private:
  void ReplaySsmState(std::uint32_t position);

  [[nodiscard]] tokenization::TokenId ForwardPromptChunk(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits);

  std::shared_ptr<const QwenGpuModel> model_;
  const models::QwenModelWeights& weights_;
  const tokenization::QwenTokenizer* tokenizer_;
  const QwenExecutionPolicy policy_;
  QwenGpuArena arena_;
  detail::HipGraphDecodeExecutor graph_executor_;
  std::vector<float> h_logits_;
  std::vector<float> h_prompt_hidden_;
  std::vector<float> h_last_hidden_;
  std::size_t last_hidden_offset_{0};
  bool capture_prompt_hidden_{false};
  bool replaying_ssm_state_{false};
};

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_QWEN_GPU_EXECUTOR_HPP_

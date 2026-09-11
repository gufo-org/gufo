#ifndef GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {

class HipblasLtGemm;
class QwenGpuExecutor;

struct QwenGpuBatchItem {
  QwenGpuExecutor* executor{nullptr};
  tokenization::TokenId token_id{0};
  std::uint32_t position{0};
};

struct QwenVerificationPolicy {
  bool batched_lm_head{false};
  int bf16_from_layer{-1};
  int fp32_from_layer{-1};
};

struct QwenSampledVerificationResult {
  tokenization::TokenId token{0};
  bool accepted{false};
};

struct QwenGpuWeightRegion {
  const void* host_data{nullptr};
  void* device_data{nullptr};
  std::size_t size{0};
  bool owns_device_memory{false};
  bool host_registered{false};
};

struct QwenGpuMemoryUsage {
  std::size_t request_state_bytes{0};
  std::size_t temporary_scratch_bytes{0};

  [[nodiscard]] std::size_t TotalBytes() const noexcept {
    return request_state_bytes + temporary_scratch_bytes;
  }
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
  [[nodiscard]] std::size_t GetResidentBytes() const noexcept;

private:
  // Keep the mapped GGUF storage alive until every registered region is
  // released.
  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
};

class QwenGpuSnapshot final {
public:
  ~QwenGpuSnapshot();

  QwenGpuSnapshot(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot& operator=(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot(QwenGpuSnapshot&&) = delete;
  QwenGpuSnapshot& operator=(QwenGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] QwenKvCacheStorage KvStorage() const noexcept {
    return kv_storage_;
  }
  [[nodiscard]] QwenRecurrentStateStorage RecurrentStateStorage()
      const noexcept {
    return recurrent_state_storage_;
  }
  [[nodiscard]] std::size_t CompactPayloadBytes() const;
  [[nodiscard]] std::size_t SerializeCompact(
      std::span<std::uint8_t> destination) const;

private:
  QwenGpuSnapshot() = default;

  void* d_kv_f32_{nullptr};
  void* d_kv_f16_{nullptr};
  void* d_ssm_conv_{nullptr};
  void* d_ssm_deltanet_{nullptr};
  std::size_t kv_elements_per_plane_{0};
  std::size_t conv_elements_{0};
  std::size_t deltanet_elements_{0};
  std::size_t conv_elements_per_layer_{0};
  std::size_t deltanet_elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t attention_layers_{0};
  std::uint32_t full_attention_interval_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::size_t payload_bytes_{0};
  QwenKvCacheStorage kv_storage_{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage_{
      QwenRecurrentStateStorage::kFp32};

  friend class QwenGpuArena;
};

/// Shared decode-lifetime workspaces over stable arena allocations.
struct QwenDecodeScratch {
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> logits;
  std::span<hip_bfloat16> bf16;
  std::span<hip_bfloat16> weight_bf16;
  std::span<std::uint32_t> prompt_tokens;

  /// Aliases the first element of QwenSsmScratch::alpha. It is legal only
  /// after all layer execution has completed, during the sampling/output epoch.
  std::span<std::uint32_t> sampled_token;
};

struct QwenAttentionScratch {
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> output;
  std::span<float> split_k;
};

struct QwenSsmScratch {
  std::span<float> qkv;
  std::span<float> conv_out;
  std::span<float> gate;
  std::span<float> out;
  std::span<float> alpha;
  std::span<float> beta;
};

struct QwenFfnScratch {
  std::span<float> gate;
  std::span<float> up;
  std::span<float> activation;
  std::span<float> out;
};

/// Typed non-owning capability views over stable Qwen GPU arena allocations.
/// The nested aggregates carry element counts without owning memory or changing
/// any address used by kernels or graph capture.
struct QwenGpuScratchView {
  QwenDecodeScratch decode;
  QwenAttentionScratch attention;
  QwenSsmScratch ssm;
  QwenFfnScratch ffn;
};

static_assert(std::is_trivially_copyable_v<QwenDecodeScratch>);
static_assert(std::is_trivially_copyable_v<QwenAttentionScratch>);
static_assert(std::is_trivially_copyable_v<QwenSsmScratch>);
static_assert(std::is_trivially_copyable_v<QwenFfnScratch>);
static_assert(std::is_trivially_copyable_v<QwenGpuScratchView>);
static_assert(std::is_same_v<decltype(QwenDecodeScratch::sampled_token),
                             std::span<std::uint32_t>>);

template<typename T>
[[nodiscard]] constexpr T* OffsetIfPresent(T* pointer,
                                           std::size_t offset) noexcept {
  return pointer == nullptr ? nullptr : pointer + offset;
}

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
class QwenGpuArena {
public:
  explicit QwenGpuArena(
      const core::ModelConfig& config, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
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
  [[nodiscard]] bool IsSsmReplayCaptureActive() const noexcept {
    return replay_capture_active_;
  }
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
  // opt-c170-deltanet-rowsplit prologue scratch: 3 floats per (token, key head)
  // for the k/q norms and the unnormalized k . q, and 2 floats per (token,
  // value head) for the decay and beta gates.
  float* d_ssm_kq_scales{nullptr};
  float* d_ssm_alpha_beta{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  float* d_kv_cache{nullptr};
  float* d_ssm_conv_state{nullptr};
  void* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};
  float* d_target_layer_features{nullptr};

  hipStream_t stream{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};
  float* d_split_k_attention{nullptr};
  hip_bfloat16* d_weights_bf16{nullptr};
  void* d_scratch_q8_act{nullptr};

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] QwenRecurrentStateStorage GetRecurrentStateStorage()
      const noexcept {
    return policy_.recurrent_state_storage;
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production());
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);
  [[nodiscard]] QwenGpuScratchView GetScratchView(
      std::size_t batch_size = 1) noexcept;
  void SetTargetLayerCapture(std::span<const std::uint32_t> target_layer_ids);
  [[nodiscard]] std::span<const std::uint32_t> GetTargetLayerCapture()
      const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] std::optional<std::size_t> GetTargetLayerCaptureIndex(
      std::uint32_t layer) const noexcept;

private:
  void FreeAll() noexcept;
  void AllocateRecurrentSnapshot();
  [[nodiscard]] bool AllocateSsmReplayLog();

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
  QwenExecutionPolicy policy_;
  std::vector<std::uint32_t> target_layer_ids_;
  float* d_saved_ssm_conv_state_{nullptr};
  void* d_saved_ssm_deltanet_state_{nullptr};
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

  /// Continues generation from an exact retained prompt prefix.
  ///
  /// The executor state must represent prompt_tokens[0:cached_prefix_tokens].
  /// A zero prefix is a cold prefill over the complete prompt.
  std::vector<tokenization::TokenId> GenerateFromPrefix(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::size_t cached_prefix_tokens,
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

  /// Advances two to eight independent retained sessions in one
  /// layer-synchronous decode step. Stateless projections share weight reads;
  /// every row still addresses its own KV cache and recurrent state.
  [[nodiscard]] static std::vector<tokenization::TokenId> ForwardTokenBatch(
      std::span<const QwenGpuBatchItem> items);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos = 0, bool compute_logits = true);

  /// Verifies a chunk of speculative draft candidate tokens in a single
  /// parallel prefill forward pass on the GPU.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits = false);
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos);
  [[nodiscard]] std::span<const float> GetVerificationHiddenStates()
      const noexcept {
    return h_verification_hidden_;
  }
  [[nodiscard]] std::span<const float> GetVerificationLogits() const noexcept {
    return h_verification_logits_;
  }
  [[nodiscard]] std::span<const float> CopyVerificationLogits(std::size_t row);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  /// Samples the most recent device-resident logit row and transfers one token.
  [[nodiscard]] tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler);

  /// Samples one row from the most recent verification batch without copying
  /// its vocabulary-sized logits to the host.
  [[nodiscard]] tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler);

  /// Performs one exact sampled-speculation accept/residual decision on the
  /// GPU, returning only the decision and selected token.
  [[nodiscard]] QwenSampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability, sampling::SamplerState& sampler);

  /// Enables host capture of every final-layer prompt hidden state. Disabled
  /// by default so ordinary prefill does not incur device-to-host copies.
  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids = {});

  /// Returns the flattened [prompt_tokens, hidden_size] capture from the most
  /// recent ForwardPromptBatch call.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates() const noexcept {
    return h_prompt_hidden_;
  }

  /// Copies the final-layer hidden state from the most recent forward pass.
  [[nodiscard]] std::span<const float> CopyLastHidden();

  void SetVerificationPolicy(QwenVerificationPolicy policy) noexcept;

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return arena_.GetMaxContext();
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = QwenExecutionPolicy::Production()) {
    return QwenGpuArena::EstimateMemoryUsage(config, max_context, policy);
  }
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const {
    return arena_.GetMemoryUsage();
  }
  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);

  /// Resets GPU cache and recurrent states in the arena.
  void Reset() noexcept;
  void SaveState(std::uint32_t valid_context);
  void RestoreState();

private:
  void ReplaySsmState(std::uint32_t position, std::uint32_t count = 1);
  void EnsureVerificationLogits(std::size_t batch_size);
  [[nodiscard]] GpuSamplingParameters PrepareGpuSamplingParameters(
      const sampling::SamplerState& sampler);

  [[nodiscard]] tokenization::TokenId ForwardPromptChunk(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits);
  [[nodiscard]] std::vector<tokenization::TokenId>
  ForwardDecodeEquivalentVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits);

  std::shared_ptr<const QwenGpuModel> model_;
  const models::QwenModelWeights& weights_;
  const tokenization::QwenTokenizer* tokenizer_;
  const QwenExecutionPolicy policy_;
  QwenGpuArena arena_;
  const detail::HipGraphCaptureKey graph_key_;
  detail::HipGraphDecodeExecutor graph_executor_;
  std::vector<float> h_logits_;
  std::vector<float> h_prompt_hidden_;
  std::vector<float> h_verification_hidden_;
  std::vector<float> h_verification_logits_;
  std::vector<float> h_last_hidden_;
  float* d_verification_logits_{nullptr};
  std::size_t verification_logits_capacity_{0};
  std::size_t last_verification_rows_{0};
  std::size_t last_hidden_offset_{0};
  float* d_target_layer_features_{nullptr};
  GpuSamplingWorkspace sampling_workspace_;
  std::vector<std::uint32_t> h_penalty_tokens_;
  std::vector<std::uint32_t> h_penalty_counts_;
  QwenVerificationPolicy verification_policy_;
  std::optional<tokenization::TokenId> next_token_;
  bool capture_prompt_hidden_{false};
  bool verification_chunk_active_{false};
  bool replaying_ssm_state_{false};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

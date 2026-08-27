#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/hrx/hrx_backend.hpp"
#include "src/core/hrx/hrx_buffer_binding.hpp"
#include "src/core/hrx/hrx_module_loader.hpp"
#include "src/core/hrx/hrx_utils.hpp"
#include "src/models/qwen/hrx/qwen_hrx_arena.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"
#include "src/models/qwen/hrx/qwen_hrx_manifest.hpp"
#include "src/models/qwen/hrx/qwen_hrx_model.hpp"

namespace gufo::hrx {

/// Native-HRX dispatcher for the strict Qwen3.8-27B Q8_0 sequential MVP.
/// Unsupported models and missing artifacts remain fail closed.
class QwenHrxExecutor {
public:
  explicit QwenHrxExecutor(QwenHrxArtifactContract contract,
                           int device_index = 0);
  ~QwenHrxExecutor();

  /// Creates the strict native Q8_0 model/session boundary. Creation rejects
  /// mixed quantization before any token execution is attempted.
  [[nodiscard]] static std::unique_ptr<QwenHrxExecutor> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
      const std::string& kernels_dir = "share/gufo/kernels",
      std::string* error_msg = nullptr, int device_index = 0);

  bool Initialize(const std::string& loom_artifact_path);
  bool InitializeAllKernels(
      const std::string& kernels_dir = "share/gufo/kernels",
      std::string* error_msg = nullptr);

  [[nodiscard]] const HrxArtifactManifest* Manifest() const noexcept {
    return manifest_.get();
  }

  /// True only when every required runtime artifact has loaded. Model-level
  /// readiness additionally requires valid model bindings and arena state.
  [[nodiscard]] bool PrototypeArtifactsReady() const noexcept {
    return prototype_artifacts_ready_;
  }
  [[nodiscard]] bool IsSwiGLUReady() const noexcept {
    return swiglu_executable_ != nullptr;
  }
  [[nodiscard]] bool Q8MathReady() const noexcept {
    return q8_embedding_executable_ != nullptr &&
           q8_gemv_k5120_executable_ != nullptr &&
           q8_gemv_k6144_executable_ != nullptr &&
           q8_gemv_k17408_executable_ != nullptr;
  }
  [[nodiscard]] bool FfnStageReady() const noexcept {
    return Q8MathReady() && rmsnorm_executable_ != nullptr &&
           swiglu_pointwise_executable_ != nullptr &&
           residual_add_executable_ != nullptr && copy_executable_ != nullptr;
  }
  [[nodiscard]] bool AttentionStageReady() const noexcept {
    return Q8MathReady() && rmsnorm_executable_ != nullptr &&
           split_q_gate_executable_ != nullptr &&
           per_head_rmsnorm_executable_ != nullptr &&
           rope_kv_executable_ != nullptr &&
           attention_decode_executable_ != nullptr &&
           residual_add_executable_ != nullptr && copy_executable_ != nullptr;
  }
  [[nodiscard]] bool SsmStageReady() const noexcept {
    return Q8MathReady() && rmsnorm_executable_ != nullptr &&
           ssm_conv_executable_ != nullptr &&
           deltanet_prepare_executable_ != nullptr &&
           deltanet_recurrence_executable_ != nullptr &&
           residual_add_executable_ != nullptr && copy_executable_ != nullptr;
  }
  [[nodiscard]] bool FinalStageReady() const noexcept {
    return rmsnorm_executable_ != nullptr &&
           q8_vocab_gemv_k5120_executable_ != nullptr &&
           argmax_executable_ != nullptr;
  }
  [[nodiscard]] const std::vector<std::string>& MissingKernelArtifacts()
      const noexcept {
    return missing_kernel_artifacts_;
  }
  [[nodiscard]] const QwenHrxArtifactContract& Contract() const noexcept {
    return contract_;
  }
  [[nodiscard]] HrxBackend& Backend() noexcept { return backend_; }
  [[nodiscard]] HrxModuleLoader& Loader() noexcept { return loader_; }
  [[nodiscard]] bool ModelExecutionReady() const noexcept {
    return model_execution_ready_;
  }
  [[nodiscard]] const std::vector<std::string>& MissingModelCapabilities()
      const noexcept {
    return missing_model_capabilities_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const;
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer() const;
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  /// Exposes stable arena operands for focused primitive and stage parity.
  /// Full-model callers should use ForwardToken.
  [[nodiscard]] std::optional<HrxBufferBinding> GetArenaBinding(
      QwenHrxArenaBuffer buffer) const noexcept {
    if (!arena_.has_value()) {
      return std::nullopt;
    }
    return arena_->Binding(buffer);
  }

  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);
  [[nodiscard]] bool SaveState(std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreState(std::string* error_msg = nullptr);
  [[nodiscard]] std::optional<tokenization::TokenId> ForwardToken(
      tokenization::TokenId token, std::uint32_t position,
      bool compute_logits = true, std::string* error_msg = nullptr);
  [[nodiscard]] std::optional<tokenization::TokenId> ForwardPromptBatch(
      std::span<const tokenization::TokenId> tokens,
      std::uint32_t start_position = 0, bool compute_logits = true,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::vector<float> CopyLastLogits(
      std::string* error_msg = nullptr) const;

  /// Decodes one Q8_0 token-embedding row into the 5120-wide hidden buffer.
  bool DispatchQ8Embedding(const HrxBufferBinding& embedding,
                           tokenization::TokenId token,
                           const HrxBufferBinding& output,
                           std::string* error_msg = nullptr);

  /// Dispatches a native Q8_0 GEMV for one of the production input widths.
  bool DispatchQ8Gemv(const HrxBufferBinding& weight,
                      const HrxBufferBinding& input,
                      const HrxBufferBinding& output, std::uint32_t rows,
                      std::uint32_t input_elements);

  /// Executes one complete production FFN stage in arena storage:
  /// RMSNorm, gate/up Q8_0 GEMVs, SwiGLU, down Q8_0 GEMV, and residual.
  bool DispatchFfnQ8(std::size_t layer_index, std::string* error_msg = nullptr);
  bool DispatchAttentionQ8(std::size_t layer_index, std::uint32_t position,
                           std::string* error_msg = nullptr);
  /// Executes the complete SSM stage, including native alpha/beta preparation.
  bool DispatchSsmQ8(std::size_t layer_index, std::string* error_msg = nullptr);
  bool DispatchFinalQ8(tokenization::TokenId* token,
                       std::string* error_msg = nullptr);

  /// Dispatches the independently tested BF16 SwiGLU projection artifact.
  bool DispatchSwiGLU(const HrxBufferBinding& input,
                      const HrxBufferBinding& gate, const HrxBufferBinding& up,
                      const HrxBufferBinding& output, std::uint32_t num_rows);

  /// Dispatches true RMSNorm followed by the SSM QKV projection. The artifact
  /// accepts the SSM projection layout only; it is not a full-attention QKV
  /// projection.
  bool DispatchRMSNormSsmQKV(const HrxBufferBinding& input,
                             const HrxBufferBinding& gamma,
                             const HrxBufferBinding& qkv_weight,
                             const HrxBufferBinding& output,
                             std::uint32_t num_rows);

  /// Applies RoPE to separate full-attention Q/K projections and stores K/V for
  /// one position. It does not compute attention scores or output projection.
  bool DispatchRoPEKVCache(const HrxBufferBinding& q, const HrxBufferBinding& k,
                           const HrxBufferBinding& v,
                           const HrxBufferBinding& cos,
                           const HrxBufferBinding& sin,
                           const HrxBufferBinding& k_cache,
                           const HrxBufferBinding& v_cache);

  /// Dispatches true F32 RMSNorm for one hidden row.
  bool DispatchRMSNorm(const HrxBufferBinding& input,
                       const HrxBufferBinding& gamma,
                       const HrxBufferBinding& output);

  /// Dispatches an F32 elementwise residual add.
  bool DispatchResidualAdd(const HrxBufferBinding& left,
                           const HrxBufferBinding& right,
                           const HrxBufferBinding& output,
                           std::uint32_t elements);

  /// Dispatches pointwise SiLU(gate) * up over FFN intermediates.
  bool DispatchSwiGLUPointwise(const HrxBufferBinding& gate,
                               const HrxBufferBinding& up,
                               const HrxBufferBinding& output,
                               std::uint32_t elements);

  /// Splits the production 12288-row Q+gate projection and applies sigmoid to
  /// the gate half.
  bool DispatchSplitQGate(const HrxBufferBinding& q_gate,
                          const HrxBufferBinding& query,
                          const HrxBufferBinding& gate);

  /// Dispatches the independently tested BF16 down projection plus residual.
  bool DispatchDownResidual(const HrxBufferBinding& input,
                            const HrxBufferBinding& down_weight,
                            const HrxBufferBinding& residual,
                            const HrxBufferBinding& output);

private:
  void ResetKernelState();

  const QwenHrxArtifactContract contract_;
  HrxBackend& backend_;
  HrxModuleLoader loader_;

  hrx_executable_t swiglu_executable_{nullptr};
  hrx_executable_t rmsnorm_ssm_qkv_executable_{nullptr};
  hrx_executable_t rope_kv_executable_{nullptr};
  hrx_executable_t down_residual_executable_{nullptr};
  hrx_executable_t rmsnorm_executable_{nullptr};
  hrx_executable_t residual_add_executable_{nullptr};
  hrx_executable_t swiglu_pointwise_executable_{nullptr};
  hrx_executable_t split_q_gate_executable_{nullptr};
  hrx_executable_t copy_executable_{nullptr};
  hrx_executable_t q8_embedding_executable_{nullptr};
  hrx_executable_t q8_gemv_k5120_executable_{nullptr};
  hrx_executable_t q8_gemv_k6144_executable_{nullptr};
  hrx_executable_t q8_gemv_k17408_executable_{nullptr};
  hrx_executable_t q8_gemv_k17408_wg256_executable_{nullptr};
  hrx_executable_t q8_vocab_gemv_k5120_executable_{nullptr};
  hrx_executable_t per_head_rmsnorm_executable_{nullptr};
  hrx_executable_t attention_decode_executable_{nullptr};
  hrx_executable_t ssm_conv_executable_{nullptr};
  hrx_executable_t deltanet_prepare_executable_{nullptr};
  hrx_executable_t deltanet_recurrence_executable_{nullptr};
  hrx_executable_t argmax_executable_{nullptr};

  [[nodiscard]] bool DispatchPerHeadRmsNorm(const HrxBufferBinding& input,
                                            const HrxBufferBinding& gamma,
                                            const HrxBufferBinding& output,
                                            std::uint32_t heads);
  [[nodiscard]] bool DispatchAttentionDecode(
      const HrxBufferBinding& query, const HrxBufferBinding& gate,
      const HrxBufferBinding& key_cache, const HrxBufferBinding& value_cache,
      const HrxBufferBinding& output, std::uint32_t position);
  [[nodiscard]] bool DispatchSsmConv(const HrxBufferBinding& input,
                                     const HrxBufferBinding& weights,
                                     const HrxBufferBinding& state,
                                     const HrxBufferBinding& output);
  [[nodiscard]] bool DispatchDeltaNetPrepare(const HrxBufferBinding& alpha,
                                             const HrxBufferBinding& beta,
                                             const HrxBufferBinding& a,
                                             const HrxBufferBinding& dt);
  [[nodiscard]] bool DispatchDeltaNetPrepared(
      const HrxBufferBinding& conv, const HrxBufferBinding& alpha_decay,
      const HrxBufferBinding& beta_correction, const HrxBufferBinding& norm,
      const HrxBufferBinding& gate, const HrxBufferBinding& state,
      const HrxBufferBinding& output);
  [[nodiscard]] bool DispatchArgmax(const HrxBufferBinding& logits,
                                    const HrxBufferBinding& token);
  [[nodiscard]] bool DispatchCopyF32(const HrxBufferBinding& source,
                                     const HrxBufferBinding& destination,
                                     std::uint32_t elements);

  std::vector<std::string> missing_kernel_artifacts_;
  std::vector<std::string> missing_model_capabilities_;
  std::unique_ptr<HrxArtifactManifest> manifest_;
  std::unique_ptr<QwenHrxModel> model_;
  std::optional<QwenHrxArena> arena_;
  std::uint32_t max_context_{0};
  std::uint32_t current_position_{0};
  bool prototype_artifacts_ready_{false};
  bool model_execution_ready_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_

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
#include "src/models/qwen/hrx/qwen_hrx_policy.hpp"

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
  [[nodiscard]] bool UsesDeviceLocalWeights() const noexcept;
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  /// Next sequential position the executor accepts. Rolls back with a failed
  /// operation, so tests and callers can observe transaction outcomes.
  [[nodiscard]] std::uint32_t CurrentPosition() const noexcept {
    return current_position_;
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

  enum class FaultInjectionPoint {
    kNone = 0,
    kFailEmbedding,
    kFailLayer1Stage,
    kFailLayer1Ffn,
    kFailFinalStage,
    kFailRestoreState,
  };

  void SetFaultInjection(FaultInjectionPoint point) noexcept {
    fault_injection_ = point;
  }

  void SetPolicy(const QwenHrxExecutionPolicy& policy) noexcept {
    policy_ = policy;
  }
  [[nodiscard]] const QwenHrxExecutionPolicy& Policy() const noexcept {
    return policy_;
  }

  [[nodiscard]] bool IsPoisoned() const noexcept { return poisoned_; }
  [[nodiscard]] const std::string& PoisonReason() const noexcept {
    return poison_reason_;
  }
  void Poison(std::string reason) noexcept {
    poisoned_ = true;
    poison_reason_ = std::move(reason);
  }

  [[nodiscard]] bool BeginOperationRollback(std::string* error_msg = nullptr);
  void CommitOperation() noexcept;
  [[nodiscard]] bool RollbackOperation(std::string* error_msg = nullptr);

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
  /// Residual-stream accessors for the ping-pong route. CurrentHidden is the
  /// buffer holding the live residual stream; CurrentScratch is the buffer a
  /// stage may overwrite with its normalization output and residual result.
  [[nodiscard]] HrxBufferBinding CurrentHidden() const noexcept;
  [[nodiscard]] HrxBufferBinding CurrentScratch() const noexcept;
  /// Publishes the stage result that was written into the scratch buffer,
  /// either by swapping the roles or by copying it back into kHidden.
  [[nodiscard]] bool PublishStageResult();

  /// Batched Q8_0 projection over a prefill chunk. `input` and `output` are
  /// token-major with `tokens` <= kHrxPrefillChunkTokens rows.
  bool DispatchQ8GemmT8(const HrxBufferBinding& weight,
                        const HrxBufferBinding& input,
                        const HrxBufferBinding& output, std::uint32_t rows,
                        std::uint32_t input_elements, std::uint32_t tokens);
  [[nodiscard]] bool ChunkedPrefillReady() const noexcept {
    return q8_gemm_k5120_t8_executable_ != nullptr &&
           q8_gemm_k6144_t8_executable_ != nullptr &&
           q8_gemm_k17408_t8_executable_ != nullptr &&
           rmsnorm_batch_executable_ != nullptr &&
           residual_add_batch_executable_ != nullptr &&
           swiglu_pointwise_batch_executable_ != nullptr;
  }
  /// Chunk-wide elementwise stages. Each replaces one dispatch per token.
  bool DispatchRMSNormBatch(const HrxBufferBinding& input,
                            const HrxBufferBinding& gamma,
                            const HrxBufferBinding& output,
                            std::uint32_t tokens);
  bool DispatchResidualAddBatch(const HrxBufferBinding& left,
                                const HrxBufferBinding& right,
                                const HrxBufferBinding& output,
                                std::uint32_t elements);
  bool DispatchSwiGLUPointwiseBatch(const HrxBufferBinding& pairs,
                                    const HrxBufferBinding& output,
                                    std::uint32_t elements);
  /// SwiGLU folded into the blocked activation quantizer. The f32 activation
  /// buffer is never written, so the FFN loses one full read/write of the
  /// 17408-wide tile per layer.
  bool DispatchSwiGLUQuantizeBlocked(const HrxBufferBinding& pairs,
                                     std::uint32_t tokens);
  /// True when the fused SwiGLU + quantize artifact is present and the route
  /// that consumes its output (the blocked projection) is active.
  [[nodiscard]] bool FusedSwiGLUQuantizeReady() const noexcept {
    return swiglu_quantize_blocked_k17408_executable_ != nullptr;
  }
  [[nodiscard]] bool UsesFusedSwiGLUQuantize() const noexcept {
    return policy_.fused_swiglu_quantize && UsesBlockedPrefill() &&
           FusedSwiGLUQuantizeReady();
  }
  /// RMSNorm folded into the blocked activation quantizer. The f32 normed
  /// tile has no other consumer on the blocked route.
  bool DispatchRMSNormQuantizeBlocked(const HrxBufferBinding& input,
                                      const HrxBufferBinding& gamma,
                                      std::uint32_t tokens);
  [[nodiscard]] bool FusedNormQuantizeReady() const noexcept {
    return rmsnorm_quantize_blocked_k5120_executable_ != nullptr;
  }
  [[nodiscard]] bool UsesFusedNormQuantize() const noexcept {
    return policy_.fused_norm_quantize && UsesBlockedPrefill() &&
           FusedNormQuantizeReady();
  }
  /// DeltaNet readout with the blocked activation quantizer folded in. The
  /// f32 context tile has no other consumer on the blocked route.
  bool DispatchDeltaNetReadoutQuantizeBatch(const HrxBufferBinding& readout,
                                            const HrxBufferBinding& norm,
                                            const HrxBufferBinding& gate,
                                            std::uint32_t tokens);
  [[nodiscard]] bool FusedReadoutQuantizeReady() const noexcept {
    return deltanet_readout_quantize_batch_executable_ != nullptr;
  }
  [[nodiscard]] bool UsesFusedReadoutQuantize() const noexcept {
    return policy_.fused_readout_quantize && UsesBlockedPrefill() &&
           FusedReadoutQuantizeReady();
  }
  /// True when the int8 projection route has every artifact it needs.
  [[nodiscard]] bool Int8PrefillReady() const noexcept {
    return q8_gemm_i8_k5120_t8_executable_ != nullptr &&
           q8_gemm_i8_k6144_t8_executable_ != nullptr &&
           q8_gemm_i8_k17408_t8_executable_ != nullptr &&
           activation_quantize_k5120_executable_ != nullptr &&
           activation_quantize_k6144_executable_ != nullptr &&
           activation_quantize_k17408_executable_ != nullptr;
  }
  /// True when the blocked 128x128 route has every artifact it needs.
  [[nodiscard]] bool BlockedPrefillReady() const noexcept {
    return q8_gemm_blocked_k5120_executable_ != nullptr &&
           q8_gemm_blocked_k6144_executable_ != nullptr &&
           q8_gemm_blocked_k17408_executable_ != nullptr &&
           activation_quantize_blocked_k5120_executable_ != nullptr &&
           activation_quantize_blocked_k6144_executable_ != nullptr &&
           activation_quantize_blocked_k17408_executable_ != nullptr;
  }
  /// True when the paired-K-staging projection artifacts are all present.
  [[nodiscard]] bool PairedKStageReady() const noexcept {
    return q8_gemm_blocked_bk2_k5120_executable_ != nullptr &&
           q8_gemm_blocked_bk2_k6144_executable_ != nullptr &&
           q8_gemm_blocked_bk2_k17408_executable_ != nullptr;
  }
  /// True when the K-split projection and its reduction are both present.
  [[nodiscard]] bool SplitKReady() const noexcept {
    return q8_gemm_blocked_bk2_splitk_k5120_executable_ != nullptr &&
           split_reduce_executable_ != nullptr;
  }
  /// True when the policy asks for the blocked route and it can run.
  [[nodiscard]] bool UsesBlockedPrefill() const noexcept {
    return policy_.blocked_prefill && BlockedPrefillReady();
  }
  /// Physical tokens per prefill tile for the currently selected route.
  [[nodiscard]] std::size_t PrefillChunkTokens() const noexcept {
    return UsesBlockedPrefill() ? kHrxBlockedPrefillExecutionTokens
                                : kHrxDot4iChunkTokens;
  }
  /// Quantizes one activation tile with whichever operand layout the active
  /// projection route consumes.
  bool DispatchChunkQuantize(const HrxBufferBinding& input,
                             std::uint32_t input_elements,
                             std::uint32_t tokens);
  /// Fragment-ordered quantization for the blocked route.
  bool DispatchActivationQuantizeBlocked(const HrxBufferBinding& input,
                                         std::uint32_t input_elements,
                                         std::uint32_t tokens);
  /// Blocked 128 row x 128 token W8A8 projection.
  bool DispatchSplitReduce(const HrxBufferBinding& output,
                           std::uint32_t elements);
  bool DispatchQ8GemmBlocked(const HrxBufferBinding& weight,
                             const HrxBufferBinding& output,
                             std::uint32_t rows, std::uint32_t input_elements,
                             std::uint32_t tokens);
  /// True when the K=5120 WMMA specialization and its padded quantizer exist.
  [[nodiscard]] bool WmmaPrefillReady() const noexcept {
    return q8_gemm_i8_wmma_k5120_t8_executable_ != nullptr &&
           activation_quantize_wmma_k5120_executable_ != nullptr;
  }
  /// Quantizes one token-major activation chunk into the arena int8 operands.
  bool DispatchActivationQuantize(const HrxBufferBinding& input,
                                  std::uint32_t input_elements,
                                  std::uint32_t tokens);
  /// Batched Q8_0 x int8 projection reading the quantized arena operands.
  bool DispatchQ8GemmInt8(const HrxBufferBinding& weight,
                          const HrxBufferBinding& output, std::uint32_t rows,
                          std::uint32_t input_elements, std::uint32_t tokens);
  /// GFX11 WMMA K=5120 projection with a physical 16-token activation tile.
  bool DispatchQ8GemmWmmaK5120(const HrxBufferBinding& weight,
                               const HrxBufferBinding& output,
                               std::uint32_t rows, std::uint32_t tokens);
  /// Batch-native DeltaNet: one recurrence dispatch per layer for a whole
  /// chunk, then one readout dispatch, instead of three dispatches per token.
  [[nodiscard]] bool BatchedDeltaNetReady() const noexcept {
    return deltanet_recurrence_batch_executable_ != nullptr &&
           deltanet_readout_batch_executable_ != nullptr;
  }
  bool DispatchDeltaNetRecurrenceBatch(const HrxBufferBinding& conv,
                                       const HrxBufferBinding& prepared,
                                       const HrxBufferBinding& state,
                                       const HrxBufferBinding& readout,
                                       std::uint32_t tokens);
  /// Whole-tile SSM front end: one preparation and one convolution dispatch
  /// per layer instead of one of each per token.
  [[nodiscard]] bool BatchedSsmFrontEndReady() const noexcept {
    return deltanet_prepare_batch_executable_ != nullptr &&
           ssm_conv_batch_executable_ != nullptr;
  }
  /// Whole-tile attention front end: one dispatch per stage per layer instead
  /// of one per token.
  [[nodiscard]] bool BatchedAttentionReady() const noexcept {
    return split_q_gate_batch_executable_ != nullptr &&
           per_head_rmsnorm_batch_executable_ != nullptr &&
           rope_kv_batch_executable_ != nullptr &&
           attention_decode_batch_executable_ != nullptr;
  }
  bool DispatchSplitQGateBatch(const HrxBufferBinding& q_gate,
                               const HrxBufferBinding& query,
                               const HrxBufferBinding& gate,
                               std::uint32_t tokens);
  bool DispatchPerHeadRmsNormBatch(const HrxBufferBinding& input,
                                   const HrxBufferBinding& gamma,
                                   const HrxBufferBinding& output,
                                   std::uint32_t heads, std::uint32_t tokens);
  bool DispatchRoPEKVCacheBatch(const HrxBufferBinding& query,
                                const HrxBufferBinding& key,
                                const HrxBufferBinding& value,
                                const HrxBufferBinding& cos,
                                const HrxBufferBinding& sin,
                                const HrxBufferBinding& key_cache,
                                const HrxBufferBinding& value_cache,
                                std::uint32_t start_position,
                                std::uint32_t tokens);
  bool DispatchAttentionDecodeBatch(const HrxBufferBinding& query,
                                    const HrxBufferBinding& gate,
                                    const HrxBufferBinding& key_cache,
                                    const HrxBufferBinding& value_cache,
                                    const HrxBufferBinding& output,
                                    std::uint32_t start_position,
                                    std::uint32_t tokens);
  bool DispatchDeltaNetPrepareBatch(const HrxBufferBinding& prepared,
                                    const HrxBufferBinding& a,
                                    const HrxBufferBinding& dt,
                                    std::uint32_t tokens);
  bool DispatchSsmConvBatch(const HrxBufferBinding& input,
                            const HrxBufferBinding& weights,
                            const HrxBufferBinding& state,
                            const HrxBufferBinding& output,
                            std::uint32_t tokens);
  bool DispatchDeltaNetReadoutBatch(const HrxBufferBinding& readout,
                                    const HrxBufferBinding& norm,
                                    const HrxBufferBinding& gate,
                                    const HrxBufferBinding& output,
                                    std::uint32_t tokens);
  /// Selects the int8 route when the policy and artifacts allow it, otherwise
  /// the f32 route. `input` must already be quantized for the int8 case.
  bool DispatchChunkProjection(const HrxBufferBinding& weight,
                               const HrxBufferBinding& input,
                               const HrxBufferBinding& output,
                               std::uint32_t rows,
                               std::uint32_t input_elements,
                               std::uint32_t tokens);
  [[nodiscard]] bool ForwardPromptChunk(
      std::span<const tokenization::TokenId> tokens,
      std::uint32_t start_position, std::string* error_msg);
  bool DispatchBatchedAttentionQ8(std::size_t layer_index,
                                  std::uint32_t start_position,
                                  std::uint32_t tokens, std::string* error_msg);
  bool DispatchBatchedSsmQ8(std::size_t layer_index, std::uint32_t tokens,
                            std::string* error_msg);
  bool DispatchBatchedFfnQ8(std::size_t layer_index, std::uint32_t tokens,
                            std::string* error_msg);

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
  // Chunked-prefill projections. Optional artifacts: when absent the executor
  // keeps the sequential single-token prefill route.
  hrx_executable_t q8_gemm_k5120_t8_executable_{nullptr};
  hrx_executable_t q8_gemm_k6144_t8_executable_{nullptr};
  hrx_executable_t q8_gemm_k17408_t8_executable_{nullptr};
  hrx_executable_t rmsnorm_batch_executable_{nullptr};
  hrx_executable_t residual_add_batch_executable_{nullptr};
  hrx_executable_t swiglu_pointwise_batch_executable_{nullptr};
  hrx_executable_t q8_gemm_i8_k5120_t8_executable_{nullptr};
  hrx_executable_t q8_gemm_i8_wmma_k5120_t8_executable_{nullptr};
  hrx_executable_t q8_gemm_i8_k6144_t8_executable_{nullptr};
  hrx_executable_t q8_gemm_i8_k17408_t8_executable_{nullptr};
  hrx_executable_t activation_quantize_k5120_executable_{nullptr};
  hrx_executable_t activation_quantize_wmma_k5120_executable_{nullptr};
  hrx_executable_t activation_quantize_k6144_executable_{nullptr};
  hrx_executable_t activation_quantize_k17408_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_k5120_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_k6144_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_k17408_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_bk2_k5120_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_bk2_k6144_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_bk2_k17408_executable_{nullptr};
  hrx_executable_t q8_gemm_blocked_bk2_splitk_k5120_executable_{nullptr};
  hrx_executable_t split_reduce_executable_{nullptr};
  hrx_executable_t activation_quantize_blocked_k5120_executable_{nullptr};
  hrx_executable_t activation_quantize_blocked_k6144_executable_{nullptr};
  hrx_executable_t activation_quantize_blocked_k17408_executable_{nullptr};
  hrx_executable_t swiglu_quantize_blocked_k17408_executable_{nullptr};
  hrx_executable_t rmsnorm_quantize_blocked_k5120_executable_{nullptr};
  hrx_executable_t deltanet_readout_quantize_batch_executable_{nullptr};
  hrx_executable_t deltanet_recurrence_batch_executable_{nullptr};
  hrx_executable_t deltanet_readout_batch_executable_{nullptr};
  hrx_executable_t deltanet_prepare_batch_executable_{nullptr};
  hrx_executable_t ssm_conv_batch_executable_{nullptr};
  hrx_executable_t split_q_gate_batch_executable_{nullptr};
  hrx_executable_t per_head_rmsnorm_batch_executable_{nullptr};
  hrx_executable_t rope_kv_batch_executable_{nullptr};
  hrx_executable_t attention_decode_batch_executable_{nullptr};
  hrx_executable_t attention_tile_batch_executable_{nullptr};
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
  std::uint32_t saved_position_{0};
  bool in_transaction_{false};
  bool poisoned_{false};
  std::string poison_reason_;
  FaultInjectionPoint fault_injection_{FaultInjectionPoint::kNone};
  QwenHrxExecutionPolicy policy_{};
  /// False while the residual stream lives in the scratch buffer. The layer
  /// stages ping-pong between kHidden and kNormed to avoid a copy per stage;
  /// each layer performs two swaps, so kHidden owns the stream at layer
  /// boundaries, at the final stage, and across transactions.
  bool hidden_primary_{true};
  /// Same ping-pong discipline for the chunked-prefill residual stream.
  bool batch_hidden_primary_{true};
  bool prototype_artifacts_ready_{false};
  bool model_execution_ready_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_EXECUTOR_HPP_

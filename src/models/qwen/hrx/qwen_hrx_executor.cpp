#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

namespace gufo::hrx {
namespace {

[[nodiscard]] std::optional<std::size_t> CheckedBytes(
    std::initializer_list<std::size_t> factors) noexcept {
  std::size_t product = 1;
  for (const std::size_t factor : factors) {
    if (factor == 0 ||
        product > std::numeric_limits<std::size_t>::max() / factor) {
      return std::nullopt;
    }
    product *= factor;
  }
  return product;
}

[[nodiscard]] bool TryBindOperand(
    const HrxBufferBinding& binding,
    const std::optional<std::size_t>& required_bytes,
    hrx_buffer_ref_t* buffer_ref) noexcept {
  return required_bytes.has_value() &&
         TryMakeBufferRef(binding, *required_bytes, buffer_ref);
}

[[nodiscard]] std::optional<std::size_t> Q8_0MatrixBytes(
    std::size_t rows, std::size_t columns) noexcept {
  constexpr std::size_t kQ8BlockElements = 32;
  constexpr std::size_t kQ8BlockBytes = 34;
  if (columns == 0 || (columns % kQ8BlockElements) != 0) {
    return std::nullopt;
  }
  return CheckedBytes({rows, columns / kQ8BlockElements, kQ8BlockBytes});
}

bool Reject(std::string_view message, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = std::string(message);
  }
  return false;
}

void ReportHrxStatus(hrx_status_t status, const char* operation,
                     std::string* error_msg) {
  char* message = nullptr;
  std::size_t length = 0;
  const auto format_status = hrx_status_to_string(status, &message, &length);
  if (error_msg != nullptr) {
    *error_msg = std::string("native HRX ") + operation + " failed";
    if (hrx_status_is_ok(format_status) && message != nullptr) {
      error_msg->append(": ");
      error_msg->append(message, length);
    }
  }
  if (message != nullptr) {
    hrx_status_free_message(message);
  }
  if (!hrx_status_is_ok(format_status)) {
    hrx_status_ignore(format_status);
  }
  hrx_status_ignore(status);
}

[[nodiscard]] std::optional<HrxBufferBinding> SliceBinding(
    const HrxBufferBinding& binding, std::size_t offset,
    std::size_t length) noexcept {
  if (!binding.IsValid() || length == 0 || offset > binding.length ||
      length > binding.length - offset ||
      binding.offset > std::numeric_limits<std::size_t>::max() - offset) {
    return std::nullopt;
  }
  return HrxBufferBinding{.buffer = binding.buffer,
                          .offset = binding.offset + offset,
                          .length = length};
}

}  // namespace

QwenHrxExecutor::QwenHrxExecutor(QwenHrxArtifactContract contract,
                                 int device_index)
    : contract_(contract), backend_(HrxBackend::Instance()) {
  (void)backend_.Initialize(device_index);
}

QwenHrxExecutor::~QwenHrxExecutor() {
  arena_.reset();
  model_.reset();
  loader_.UnloadAll();
}

std::unique_ptr<QwenHrxExecutor> QwenHrxExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
    const std::string& kernels_dir, std::string* error_msg, int device_index) {
  if (reader == nullptr || max_context == 0) {
    Reject("native HRX creation requires a GGUF reader and non-zero context",
           error_msg);
    return nullptr;
  }
  const auto config = reader->ExtractModelConfig(error_msg);
  if (!config.has_value()) {
    return nullptr;
  }
  const auto contract = QwenHrxArtifactContract::FromConfig(*config, error_msg);
  if (!contract.has_value()) {
    return nullptr;
  }
  constexpr std::uint32_t kAttentionArtifactContextCapacity = 131'072;
  if (max_context > config->context_length ||
      max_context > kAttentionArtifactContextCapacity) {
    Reject("native HRX context exceeds the model or attention artifact limit",
           error_msg);
    return nullptr;
  }

  auto manifest =
      HrxArtifactManifest::LoadFromDirectory(kernels_dir, error_msg);
  if (manifest == nullptr) {
    return nullptr;
  }
  if (!manifest->ValidateDirectory(kernels_dir, *contract, error_msg)) {
    return nullptr;
  }

  auto executor = std::unique_ptr<QwenHrxExecutor>(
      new QwenHrxExecutor(*contract, device_index));
  if (!executor->backend_.IsInitialized()) {
    Reject("failed to initialize the native HRX backend", error_msg);
    return nullptr;
  }
  executor->manifest_ = std::move(manifest);
  if (!executor->InitializeAllKernels(kernels_dir, error_msg)) {
    Reject("failed to initialize native HRX kernel artifacts", error_msg);
    return nullptr;
  }

  executor->model_ = QwenHrxModel::CreateFromGguf(
      std::move(reader), executor->backend_.Device(), error_msg);
  if (executor->model_ == nullptr) {
    return nullptr;
  }
  auto arena = QwenHrxArena::Create(executor->backend_.Device(),
                                    executor->backend_.Stream(), *contract,
                                    max_context, error_msg);
  if (!arena.has_value()) {
    return nullptr;
  }
  executor->arena_.emplace(std::move(*arena));
  executor->max_context_ = max_context;

  if (!executor->arena_->PrecomputeRope(
          executor->model_->GetConfig().rope_theta, error_msg)) {
    Reject("failed to precompute RoPE tables in native HRX arena", error_msg);
    return nullptr;
  }

  // Capabilities stay explicit so the opt-in CLI cannot silently run an
  // incomplete model.
  if (!executor->Q8MathReady()) {
    executor->missing_model_capabilities_.emplace_back(
        "Q8_0 embedding/GEMV artifacts");
  }
  if (!executor->FfnStageReady()) {
    executor->missing_model_capabilities_.emplace_back(
        "complete native Q8_0 FFN stage");
  }
  if (!executor->AttentionStageReady()) {
    executor->missing_model_capabilities_.emplace_back(
        "complete causal GQA attention stage");
  }
  if (!executor->SsmStageReady()) {
    executor->missing_model_capabilities_.emplace_back(
        "complete SSM convolution/DeltaNet artifacts");
  }
  if (!executor->FinalStageReady()) {
    executor->missing_model_capabilities_.emplace_back(
        "final norm, vocabulary projection, and greedy argmax");
  }
  executor->model_execution_ready_ =
      executor->missing_model_capabilities_.empty();
  if (!executor->Reset(error_msg)) {
    return nullptr;
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return executor;
}

const core::ModelConfig& QwenHrxExecutor::GetConfig() const {
  if (model_ == nullptr) {
    throw std::logic_error("native HRX executor has no model");
  }
  return model_->GetConfig();
}

const tokenization::QwenTokenizer& QwenHrxExecutor::GetTokenizer() const {
  if (model_ == nullptr) {
    throw std::logic_error("native HRX executor has no tokenizer");
  }
  return model_->GetTokenizer();
}

bool QwenHrxExecutor::UsesDeviceLocalWeights() const noexcept {
  return model_ != nullptr && model_->UsesDeviceLocalWeights();
}

bool QwenHrxExecutor::BeginOperationRollback(std::string* error_msg) {
  if (poisoned_) {
    return Reject("native HRX executor is poisoned: " + poison_reason_,
                  error_msg);
  }
  if (!arena_.has_value()) {
    return Reject("native HRX executor has no arena", error_msg);
  }
  saved_position_ = current_position_;
  arena_->SetCurrentPosition(current_position_);
  if (!arena_->EnqueueSaveState(copy_executable_, error_msg)) {
    Poison("failed to snapshot state before operation: " +
           (error_msg ? *error_msg : ""));
    return false;
  }
  in_transaction_ = true;
  return true;
}

void QwenHrxExecutor::CommitOperation() noexcept {
  in_transaction_ = false;
}

bool QwenHrxExecutor::RollbackOperation(std::string* error_msg) {
  if (fault_injection_ == FaultInjectionPoint::kFailRestoreState) {
    Poison("injected rollback failure");
    return Reject("injected rollback failure", error_msg);
  }
  if (!arena_.has_value()) {
    Poison("rollback failed: executor has no arena");
    return Reject("native HRX executor has no arena for rollback", error_msg);
  }
  if (!arena_->RestoreState(copy_executable_, error_msg)) {
    Poison("unrecoverable failure during state rollback: " +
           (error_msg ? *error_msg : ""));
    return false;
  }
  current_position_ = saved_position_;
  arena_->SetCurrentPosition(current_position_);
  in_transaction_ = false;
  return true;
}

bool QwenHrxExecutor::Reset(std::string* error_msg) {
  if (!arena_.has_value()) {
    return Reject("native HRX executor has no arena", error_msg);
  }
  if (!arena_->Reset(error_msg)) {
    Poison("failed to reset arena: " + (error_msg ? *error_msg : ""));
    return false;
  }
  current_position_ = 0;
  saved_position_ = 0;
  in_transaction_ = false;
  poisoned_ = false;
  poison_reason_.clear();
  return true;
}

bool QwenHrxExecutor::SaveState(std::string* error_msg) {
  if (poisoned_) {
    return Reject("native HRX executor is poisoned: " + poison_reason_,
                  error_msg);
  }
  if (!arena_.has_value()) {
    return Reject("native HRX executor has no arena", error_msg);
  }
  saved_position_ = current_position_;
  arena_->SetCurrentPosition(current_position_);
  return arena_->SaveState(copy_executable_, error_msg);
}

bool QwenHrxExecutor::RestoreState(std::string* error_msg) {
  if (poisoned_) {
    return Reject("native HRX executor is poisoned: " + poison_reason_,
                  error_msg);
  }
  if (!arena_.has_value()) {
    return Reject("native HRX executor has no arena", error_msg);
  }
  if (!arena_->RestoreState(copy_executable_, error_msg)) {
    Poison("unrecoverable failure during state restore: " +
           (error_msg ? *error_msg : ""));
    return false;
  }
  current_position_ = arena_->CurrentPosition();
  return true;
}

std::optional<tokenization::TokenId> QwenHrxExecutor::ForwardToken(
    tokenization::TokenId token, std::uint32_t position, bool compute_logits,
    std::string* error_msg) {
  if (poisoned_) {
    Reject("native HRX session is poisoned: " + poison_reason_, error_msg);
    return std::nullopt;
  }
  if (model_ == nullptr || !arena_.has_value()) {
    Reject("native HRX executor is not initialized", error_msg);
    return std::nullopt;
  }
  if (!model_execution_ready_) {
    Reject(
        "native HRX Q8_0 model execution is unavailable because required "
        "artifacts are missing; no HIP or host fallback is permitted",
        error_msg);
    return std::nullopt;
  }
  if (token >= contract_.VocabSize()) {
    Reject("native HRX input token is outside the vocabulary", error_msg);
    return std::nullopt;
  }
  if (position >= max_context_) {
    Reject("native HRX token position exceeds max_context", error_msg);
    return std::nullopt;
  }
  if (position != current_position_) {
    Reject("native HRX token position is not the next sequential position",
           error_msg);
    return std::nullopt;
  }

  const auto& bindings = model_->GetNativeBindings();
  if (bindings.layers.size() != contract_.NumLayers()) {
    Reject("native HRX model does not expose exactly 64 layers", error_msg);
    return std::nullopt;
  }

  if (!BeginOperationRollback(error_msg)) {
    return std::nullopt;
  }

  const auto rollback_and_fail = [&](const std::string& reason) {
    if (error_msg != nullptr && error_msg->empty()) {
      *error_msg = reason;
    }
    std::string rollback_err;
    if (!RollbackOperation(&rollback_err)) {
      Poison("rollback failed: " + rollback_err);
    }
    return std::nullopt;
  };

  if (fault_injection_ == FaultInjectionPoint::kFailEmbedding) {
    return rollback_and_fail("fault injection: embedding dispatch failed");
  }

  const bool trace_stages = std::getenv("GUFO_HRX_TRACE_STAGES") != nullptr;
  const auto token_start = std::chrono::steady_clock::now();
  auto stage_start = token_start;
  double embedding_ms = 0.0;
  double attention_ms = 0.0;
  double ssm_ms = 0.0;
  double ffn_ms = 0.0;
  double final_ms = 0.0;
  const auto synchronize_stage = [&](const char* stage, double* elapsed_ms) {
    if (!trace_stages) {
      return true;
    }
    const auto status = hrx_stream_synchronize(backend_.Stream());
    const auto now = std::chrono::steady_clock::now();
    *elapsed_ms +=
        std::chrono::duration<double, std::milli>(now - stage_start).count();
    stage_start = now;
    if (!hrx_status_is_ok(status)) {
      ReportHrxStatus(status, stage, error_msg);
      return false;
    }
    return true;
  };
  if (!DispatchQ8Embedding(bindings.token_embedding, token,
                           arena_->Binding(QwenHrxArenaBuffer::kHidden),
                           error_msg)) {
    return rollback_and_fail("native HRX token embedding dispatch failed");
  }
  if (!synchronize_stage("embedding", &embedding_ms)) {
    return rollback_and_fail("embedding stage synchronization failed");
  }

  std::size_t attention_layers = 0;
  std::size_t ssm_layers = 0;
  for (std::size_t layer_index = 0; layer_index < bindings.layers.size();
       ++layer_index) {
    const bool expected_attention = ((layer_index + 1) % 4) == 0;
    if (bindings.layers[layer_index].is_full_attention != expected_attention) {
      return rollback_and_fail(
          "native HRX model layer ordering violates the Qwen3.8 contract");
    }

    if (layer_index == 1 &&
        fault_injection_ == FaultInjectionPoint::kFailLayer1Stage) {
      return rollback_and_fail("fault injection: layer 1 stage failed");
    }

    const bool stage_ok =
        expected_attention
            ? DispatchAttentionQ8(layer_index, position, error_msg)
            : DispatchSsmQ8(layer_index, error_msg);
    attention_layers += expected_attention ? 1 : 0;
    ssm_layers += expected_attention ? 0 : 1;
    double* const layer_elapsed = expected_attention ? &attention_ms : &ssm_ms;
    if (!stage_ok ||
        !synchronize_stage(expected_attention ? "attention" : "SSM",
                           layer_elapsed)) {
      return rollback_and_fail("attention/SSM stage dispatch failed");
    }

    if (layer_index == 1 &&
        fault_injection_ == FaultInjectionPoint::kFailLayer1Ffn) {
      return rollback_and_fail("fault injection: layer 1 FFN failed");
    }

    if (!DispatchFfnQ8(layer_index, error_msg) ||
        !synchronize_stage("FFN", &ffn_ms)) {
      return rollback_and_fail("FFN stage dispatch failed");
    }
  }
  if (attention_layers != contract_.FullAttentionLayerCount() ||
      ssm_layers != contract_.SsmLayerCount()) {
    return rollback_and_fail(
        "native HRX model layer counts violate the Qwen3.8 contract");
  }

  if (fault_injection_ == FaultInjectionPoint::kFailFinalStage) {
    return rollback_and_fail("fault injection: final stage failed");
  }

  tokenization::TokenId next_token = 0;
  if (compute_logits && !DispatchFinalQ8(&next_token, error_msg)) {
    return rollback_and_fail("final stage projection failed");
  }
  const auto token_end = std::chrono::steady_clock::now();
  if (trace_stages) {
    final_ms =
        std::chrono::duration<double, std::milli>(token_end - stage_start)
            .count();
    const double token_ms =
        std::chrono::duration<double, std::milli>(token_end - token_start)
            .count();
    std::cerr << "[HRX native profile] position=" << position
              << " embedding_ms=" << embedding_ms
              << " attention_ms=" << attention_ms << " ssm_ms=" << ssm_ms
              << " ffn_ms=" << ffn_ms << " final_ms=" << final_ms
              << " token_ms=" << token_ms << '\n';
  }

  CommitOperation();
  current_position_ = position + 1;
  arena_->SetCurrentPosition(current_position_);
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return next_token;
}

std::optional<tokenization::TokenId> QwenHrxExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> tokens, std::uint32_t start_position,
    bool compute_logits, std::string* error_msg) {
  if (tokens.empty()) {
    Reject("native HRX prompt must contain at least one token", error_msg);
    return std::nullopt;
  }
  if (start_position >= max_context_ ||
      tokens.size() > max_context_ - start_position) {
    Reject("native HRX prompt exceeds max_context", error_msg);
    return std::nullopt;
  }
  if (tokens.size() == 1) {
    return ForwardToken(tokens[0], start_position, compute_logits, error_msg);
  }

  if (poisoned_) {
    Reject("native HRX session is poisoned: " + poison_reason_, error_msg);
    return std::nullopt;
  }
  if (model_ == nullptr || !arena_.has_value()) {
    Reject("native HRX executor is not initialized", error_msg);
    return std::nullopt;
  }
  if (!model_execution_ready_) {
    Reject("native HRX Q8_0 model execution is unavailable", error_msg);
    return std::nullopt;
  }
  if (start_position != current_position_) {
    Reject("native HRX token position is not the next sequential position",
           error_msg);
    return std::nullopt;
  }

  if (!BeginOperationRollback(error_msg)) {
    return std::nullopt;
  }

  const auto rollback_and_fail = [&](const std::string& reason) {
    if (error_msg != nullptr && error_msg->empty()) {
      *error_msg = reason;
    }
    std::string rollback_err;
    if (!RollbackOperation(&rollback_err)) {
      Poison("rollback failed: " + rollback_err);
    }
    return std::nullopt;
  };

  const auto& bindings = model_->GetNativeBindings();
  tokenization::TokenId next_token = 0;

  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const auto token = tokens[index];
    const std::uint32_t position =
        start_position + static_cast<std::uint32_t>(index);
    const bool is_last = (index + 1 == tokens.size());

    if (token >= contract_.VocabSize()) {
      return rollback_and_fail("native HRX input token is outside vocabulary");
    }

    if (!DispatchQ8Embedding(bindings.token_embedding, token,
                             arena_->Binding(QwenHrxArenaBuffer::kHidden),
                             error_msg)) {
      return rollback_and_fail("native HRX token embedding dispatch failed");
    }

    for (std::size_t layer_index = 0; layer_index < bindings.layers.size();
         ++layer_index) {
      const bool expected_attention = ((layer_index + 1) % 4) == 0;
      const bool stage_ok =
          expected_attention
              ? DispatchAttentionQ8(layer_index, position, error_msg)
              : DispatchSsmQ8(layer_index, error_msg);
      if (!stage_ok || !DispatchFfnQ8(layer_index, error_msg)) {
        return rollback_and_fail("layer stage dispatch failed");
      }
    }

    if (is_last && compute_logits) {
      if (!DispatchFinalQ8(&next_token, error_msg)) {
        return rollback_and_fail("final stage projection failed");
      }
    }
  }

  CommitOperation();
  current_position_ =
      start_position + static_cast<std::uint32_t>(tokens.size());
  arena_->SetCurrentPosition(current_position_);
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return next_token;
}

std::vector<float> QwenHrxExecutor::CopyLastLogits(
    std::string* error_msg) const {
  if (!arena_.has_value()) {
    Reject("native HRX executor has no logits arena", error_msg);
    return {};
  }
  auto status = hrx_stream_synchronize(backend_.Stream());
  if (!hrx_status_is_ok(status)) {
    ReportHrxStatus(status, "logit synchronization", error_msg);
    return {};
  }
  std::vector<float> logits(contract_.VocabSize());
  if (!HrxCopyToHost(backend_.Device(),
                     arena_->Binding(QwenHrxArenaBuffer::kLogits),
                     logits.data(), logits.size() * sizeof(float), error_msg)) {
    return {};
  }
  return logits;
}

void QwenHrxExecutor::ResetKernelState() {
  swiglu_executable_ = nullptr;
  rmsnorm_ssm_qkv_executable_ = nullptr;
  rope_kv_executable_ = nullptr;
  down_residual_executable_ = nullptr;
  rmsnorm_executable_ = nullptr;
  residual_add_executable_ = nullptr;
  swiglu_pointwise_executable_ = nullptr;
  split_q_gate_executable_ = nullptr;
  copy_executable_ = nullptr;
  q8_embedding_executable_ = nullptr;
  q8_gemv_k5120_executable_ = nullptr;
  q8_gemv_k6144_executable_ = nullptr;
  q8_gemv_k17408_executable_ = nullptr;
  q8_gemv_k17408_wg256_executable_ = nullptr;
  q8_vocab_gemv_k5120_executable_ = nullptr;
  per_head_rmsnorm_executable_ = nullptr;
  attention_decode_executable_ = nullptr;
  ssm_conv_executable_ = nullptr;
  deltanet_prepare_executable_ = nullptr;
  deltanet_recurrence_executable_ = nullptr;
  argmax_executable_ = nullptr;
  missing_kernel_artifacts_.clear();
  prototype_artifacts_ready_ = false;
}

bool QwenHrxExecutor::Initialize(const std::string& loom_artifact_path) {
  loader_.UnloadAll();
  ResetKernelState();
  if (!backend_.IsInitialized()) {
    return false;
  }

  hrx_status_t status =
      loader_.LoadFromFile(backend_.Device(), "qwen_swiglu", loom_artifact_path,
                           "amdgpu", "gfx1151");
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }

  swiglu_executable_ = loader_.GetExecutable("qwen_swiglu");
  if (swiglu_executable_ == nullptr) {
    return false;
  }

  missing_kernel_artifacts_ = {
      "qwen_fused_rmsnorm_qkv_bf16.fb",
      "qwen_fused_rope_kv_cache_bf16.fb",
      "qwen_fused_down_residual_bf16.fb",
      "qwen_rmsnorm_f32.fb",
      "qwen_residual_add_f32.fb",
      "qwen_swiglu_pointwise_f32.fb",
      "qwen_split_q_gate_f32.fb",
      "qwen_copy_f32.fb",
      "qwen_q8_0_embedding_k5120.fb",
      "qwen_q8_0_gemv_k5120.fb",
      "qwen_q8_0_gemv_k6144.fb",
      "qwen_q8_0_gemv_k17408.fb",
      "qwen_q8_0_vocab_gemv_k5120.fb",
      "qwen_per_head_rmsnorm_f32.fb",
      "qwen_attention_decode_f32.fb",
      "qwen_ssm_conv_silu_f32.fb",
      "qwen_deltanet_prepare_f32.fb",
      "qwen_deltanet_recurrence_f32.fb",
      "qwen_argmax_f32.fb",
  };
  return true;
}

bool QwenHrxExecutor::InitializeAllKernels(const std::string& kernels_dir,
                                           std::string* error_msg) {
  loader_.UnloadAll();
  ResetKernelState();
  if (!backend_.IsInitialized()) {
    if (error_msg != nullptr) {
      *error_msg = "HRX backend is not initialized";
    }
    return false;
  }

  if (manifest_ == nullptr) {
    manifest_ = HrxArtifactManifest::LoadFromDirectory(kernels_dir, error_msg);
    if (manifest_ == nullptr) {
      return false;
    }
  }

  for (const auto& entry : manifest_->Entries()) {
    const std::string found_path =
        (std::filesystem::path(kernels_dir) / entry.filename).string();
    if (!std::filesystem::exists(found_path)) {
      if (!entry.optional) {
        missing_kernel_artifacts_.emplace_back(entry.filename);
      }
      continue;
    }

    hrx_status_t status = loader_.LoadFromFile(backend_.Device(), entry.name,
                                               found_path, "amdgpu", "gfx1151");
    if (!hrx_status_is_ok(status)) {
      hrx_status_ignore(status);
      if (!entry.optional) {
        missing_kernel_artifacts_.emplace_back(entry.filename);
      }
      continue;
    }

    hrx_executable_t exec = loader_.GetExecutable(entry.name);
    if (exec == nullptr) {
      if (!entry.optional) {
        missing_kernel_artifacts_.emplace_back(entry.filename);
      }
      continue;
    }

    if (entry.name == "qwen_swiglu") {
      swiglu_executable_ = exec;
    } else if (entry.name == "qwen_rmsnorm_qkv") {
      rmsnorm_ssm_qkv_executable_ = exec;
    } else if (entry.name == "qwen_rope_kv") {
      rope_kv_executable_ = exec;
    } else if (entry.name == "qwen_down_residual") {
      down_residual_executable_ = exec;
    } else if (entry.name == "qwen_rmsnorm") {
      rmsnorm_executable_ = exec;
    } else if (entry.name == "qwen_residual_add") {
      residual_add_executable_ = exec;
    } else if (entry.name == "qwen_swiglu_pointwise") {
      swiglu_pointwise_executable_ = exec;
    } else if (entry.name == "qwen_split_q_gate") {
      split_q_gate_executable_ = exec;
    } else if (entry.name == "qwen_copy") {
      copy_executable_ = exec;
    } else if (entry.name == "qwen_q8_embedding") {
      q8_embedding_executable_ = exec;
    } else if (entry.name == "qwen_q8_gemv_k5120") {
      q8_gemv_k5120_executable_ = exec;
    } else if (entry.name == "qwen_q8_gemv_k6144") {
      q8_gemv_k6144_executable_ = exec;
    } else if (entry.name == "qwen_q8_gemv_k17408") {
      q8_gemv_k17408_executable_ = exec;
    } else if (entry.name == "qwen_q8_gemv_k17408_wg256") {
      q8_gemv_k17408_wg256_executable_ = exec;
    } else if (entry.name == "qwen_q8_vocab_gemv_k5120") {
      q8_vocab_gemv_k5120_executable_ = exec;
    } else if (entry.name == "qwen_per_head_rmsnorm") {
      per_head_rmsnorm_executable_ = exec;
    } else if (entry.name == "qwen_attention_decode") {
      attention_decode_executable_ = exec;
    } else if (entry.name == "qwen_ssm_conv") {
      ssm_conv_executable_ = exec;
    } else if (entry.name == "qwen_deltanet_prepare") {
      deltanet_prepare_executable_ = exec;
    } else if (entry.name == "qwen_deltanet_recurrence") {
      deltanet_recurrence_executable_ = exec;
    } else if (entry.name == "qwen_argmax") {
      argmax_executable_ = exec;
    }
  }

  prototype_artifacts_ready_ = missing_kernel_artifacts_.empty();
  return prototype_artifacts_ready_;
}

bool QwenHrxExecutor::DispatchQ8Embedding(const HrxBufferBinding& embedding,
                                          tokenization::TokenId token,
                                          const HrxBufferBinding& output,
                                          std::string* error_msg) {
  if (q8_embedding_executable_ == nullptr || token >= contract_.VocabSize()) {
    return false;
  }
  // Only the selected row and preceding storage must be addressable. Production
  // model bindings still expose the full matrix, while focused parity tests can
  // use a bounded prefix without allocating the entire vocabulary.
  const auto embedding_bytes = Q8_0MatrixBytes(
      static_cast<std::size_t>(token) + 1, contract_.HiddenSize());
  const auto output_bytes =
      CheckedBytes({contract_.HiddenSize(), sizeof(float)});
  hrx_buffer_ref_t bindings[2];
  if (!TryBindOperand(embedding, embedding_bytes, &bindings[0]) ||
      !TryBindOperand(output, output_bytes, &bindings[1])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = contract_.HiddenSize() / 32;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const std::array<std::uint32_t, 2> params{static_cast<std::uint32_t>(token),
                                            contract_.VocabSize()};
  auto status = hrx_stream_dispatch(backend_.Stream(), q8_embedding_executable_,
                                    0, &config, params.data(), sizeof(params),
                                    bindings, 2, 0);
  if (!hrx_status_is_ok(status)) {
    ReportHrxStatus(status, "embedding dispatch", error_msg);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchQ8Gemv(const HrxBufferBinding& weight,
                                     const HrxBufferBinding& input,
                                     const HrxBufferBinding& output,
                                     std::uint32_t rows,
                                     std::uint32_t input_elements) {
  hrx_executable_t executable = nullptr;
  std::uint32_t workgroup_size = 0;
  std::uint32_t row_capacity = 0;
  switch (input_elements) {
    case 5120:
      executable = q8_gemv_k5120_executable_;
      workgroup_size = 160;
      row_capacity = contract_.FfnSize();
      break;
    case 6144:
      executable = q8_gemv_k6144_executable_;
      workgroup_size = 192;
      row_capacity = contract_.HiddenSize();
      break;
    case 17408:
      if (std::getenv("GUFO_HRX_Q8_K17408_WG544") == nullptr &&
          q8_gemv_k17408_wg256_executable_ != nullptr) {
        executable = q8_gemv_k17408_wg256_executable_;
        workgroup_size = 256;
      } else {
        executable = q8_gemv_k17408_executable_;
        workgroup_size = 544;
      }
      row_capacity = contract_.HiddenSize();
      break;
    default:
      return false;
  }
  if (executable == nullptr || rows == 0 || rows > row_capacity) {
    return false;
  }
  const auto weight_bytes = Q8_0MatrixBytes(rows, input_elements);
  const auto input_bytes = CheckedBytes({input_elements, sizeof(float)});
  const auto output_bytes = CheckedBytes({rows, sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(weight, weight_bytes, &bindings[0]) ||
      !TryBindOperand(input, input_bytes, &bindings[1]) ||
      !TryBindOperand(output, output_bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = rows;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = workgroup_size;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  auto status = hrx_stream_dispatch(backend_.Stream(), executable, 0, &config,
                                    &rows, sizeof(rows), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchCopyF32(const HrxBufferBinding& source,
                                      const HrxBufferBinding& destination,
                                      std::uint32_t elements) {
  if (copy_executable_ == nullptr || elements < 256 || (elements % 256) != 0) {
    return false;
  }
  const auto bytes = CheckedBytes({elements, sizeof(float)});
  hrx_buffer_ref_t bindings[2];
  if (!TryBindOperand(source, bytes, &bindings[0]) ||
      !TryBindOperand(destination, bytes, &bindings[1])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = elements / 256;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 256;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  auto status =
      hrx_stream_dispatch(backend_.Stream(), copy_executable_, 0, &config,
                          &elements, sizeof(elements), bindings, 2, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchFfnQ8(std::size_t layer_index,
                                    std::string* error_msg) {
  if (model_ == nullptr || !arena_.has_value() || !FfnStageReady()) {
    return Reject("native HRX Q8_0 FFN stage is not initialized", error_msg);
  }
  const auto& bindings = model_->GetNativeBindings();
  if (layer_index >= bindings.layers.size()) {
    return Reject("native HRX FFN layer index is out of range", error_msg);
  }
  const auto& layer = bindings.layers[layer_index];
  const auto hidden = arena_->Binding(QwenHrxArenaBuffer::kHidden);
  const auto normed = arena_->Binding(QwenHrxArenaBuffer::kNormed);
  const auto gate = arena_->Binding(QwenHrxArenaBuffer::kFfnGate);
  const auto up = arena_->Binding(QwenHrxArenaBuffer::kFfnUp);
  const auto activation = arena_->Binding(QwenHrxArenaBuffer::kFfnActivation);
  const auto projected = arena_->Binding(QwenHrxArenaBuffer::kFfnOutput);
  if (!DispatchRMSNorm(hidden, layer.ffn_norm, normed) ||
      !DispatchQ8Gemv(layer.ffn_gate, normed, gate, contract_.FfnSize(),
                      contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.ffn_up, normed, up, contract_.FfnSize(),
                      contract_.HiddenSize()) ||
      !DispatchSwiGLUPointwise(gate, up, activation, contract_.FfnSize()) ||
      !DispatchQ8Gemv(layer.ffn_down, activation, projected,
                      contract_.HiddenSize(), contract_.FfnSize()) ||
      !DispatchResidualAdd(hidden, projected, normed, contract_.HiddenSize()) ||
      !DispatchCopyF32(normed, hidden, contract_.HiddenSize())) {
    return Reject("native HRX Q8_0 FFN dispatch failed", error_msg);
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

bool QwenHrxExecutor::DispatchAttentionQ8(std::size_t layer_index,
                                          std::uint32_t position,
                                          std::string* error_msg) {
  if (model_ == nullptr || !arena_.has_value() || !AttentionStageReady()) {
    return Reject("native HRX attention stage is not initialized", error_msg);
  }
  const auto& config = model_->GetConfig();
  const auto& bindings = model_->GetNativeBindings();
  if (layer_index >= bindings.layers.size() ||
      !bindings.layers[layer_index].is_full_attention ||
      position >= max_context_) {
    return Reject("native HRX attention layer or position is invalid",
                  error_msg);
  }
  const auto& layer = bindings.layers[layer_index];
  const auto hidden = arena_->Binding(QwenHrxArenaBuffer::kHidden);
  const auto normed = arena_->Binding(QwenHrxArenaBuffer::kNormed);
  const auto q_gate = arena_->Binding(QwenHrxArenaBuffer::kAttentionQGate);
  const auto query = arena_->Binding(QwenHrxArenaBuffer::kAttentionQ);
  const auto gate = arena_->Binding(QwenHrxArenaBuffer::kAttentionGate);
  const auto key = arena_->Binding(QwenHrxArenaBuffer::kAttentionK);
  const auto value = arena_->Binding(QwenHrxArenaBuffer::kAttentionV);
  const auto context = arena_->Binding(QwenHrxArenaBuffer::kSsmRecurrentOutput);
  const auto projected = arena_->Binding(QwenHrxArenaBuffer::kAttentionOutput);

  if (!DispatchRMSNorm(hidden, layer.attn_norm, normed) ||
      !DispatchQ8Gemv(layer.attn_q, normed, q_gate,
                      contract_.FullAttentionQGateWidth(),
                      contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.attn_k, normed, key,
                      contract_.FullAttentionKeyWidth(),
                      contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.attn_v, normed, value,
                      contract_.FullAttentionValueWidth(),
                      contract_.HiddenSize()) ||
      !DispatchSplitQGate(q_gate, query, gate) ||
      !DispatchPerHeadRmsNorm(query, layer.attn_q_norm, query,
                              contract_.QHeadCount()) ||
      !DispatchPerHeadRmsNorm(key, layer.attn_k_norm, key,
                              contract_.KvHeadCount())) {
    return Reject("native HRX attention projection/QK normalization failed",
                  error_msg);
  }

  const std::size_t rotary_bytes =
      (static_cast<std::size_t>(contract_.RotaryDim()) / 2) * sizeof(float);
  const auto cos_binding = arena_->Binding(
      QwenHrxArenaBuffer::kRopeCos, position * rotary_bytes, rotary_bytes);
  const auto sin_binding = arena_->Binding(
      QwenHrxArenaBuffer::kRopeSin, position * rotary_bytes, rotary_bytes);
  if (!cos_binding.IsValid() || !sin_binding.IsValid()) {
    return Reject("native HRX RoPE position is outside arena range", error_msg);
  }

  const std::size_t cache_row_bytes =
      static_cast<std::size_t>(contract_.KvHeadCount()) * contract_.HeadDim() *
      sizeof(float);
  const std::size_t cache_layer_bytes =
      cache_row_bytes * static_cast<std::size_t>(max_context_);
  const std::size_t cache_layer_index =
      layer_index / config.full_attention_interval;
  const auto cache = arena_->Binding(QwenHrxArenaBuffer::kKvCache);
  const std::size_t layer_base = cache_layer_index * 2 * cache_layer_bytes;
  const auto key_cache = SliceBinding(cache, layer_base, cache_layer_bytes);
  const auto value_cache =
      SliceBinding(cache, layer_base + cache_layer_bytes, cache_layer_bytes);
  if (!key_cache || !value_cache) {
    return Reject("native HRX attention cache slice is invalid", error_msg);
  }
  const auto key_row = SliceBinding(
      *key_cache, static_cast<std::size_t>(position) * cache_row_bytes,
      cache_row_bytes);
  const auto value_row = SliceBinding(
      *value_cache, static_cast<std::size_t>(position) * cache_row_bytes,
      cache_row_bytes);
  if (!key_row || !value_row ||
      !DispatchRoPEKVCache(query, key, value, cos_binding, sin_binding,
                           *key_row, *value_row) ||
      !DispatchAttentionDecode(query, gate, *key_cache, *value_cache, context,
                               position) ||
      !DispatchQ8Gemv(layer.attn_output, context, projected,
                      contract_.HiddenSize(),
                      contract_.FullAttentionQueryWidth()) ||
      !DispatchResidualAdd(hidden, projected, normed, contract_.HiddenSize()) ||
      !DispatchCopyF32(normed, hidden, contract_.HiddenSize())) {
    return Reject("native HRX causal GQA attention dispatch failed", error_msg);
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

bool QwenHrxExecutor::DispatchSsmQ8(std::size_t layer_index,
                                    std::string* error_msg) {
  if (model_ == nullptr || !arena_.has_value() || !SsmStageReady()) {
    return Reject("native HRX SSM stage is not initialized", error_msg);
  }
  const auto& bindings = model_->GetNativeBindings();
  if (layer_index >= bindings.layers.size() ||
      bindings.layers[layer_index].is_full_attention) {
    return Reject("native HRX SSM layer index is invalid", error_msg);
  }
  const auto& layer = bindings.layers[layer_index];
  const auto hidden = arena_->Binding(QwenHrxArenaBuffer::kHidden);
  const auto normed = arena_->Binding(QwenHrxArenaBuffer::kNormed);
  const auto qkv = arena_->Binding(QwenHrxArenaBuffer::kSsmQkv);
  const auto gate = arena_->Binding(QwenHrxArenaBuffer::kSsmGate);
  const auto alpha = arena_->Binding(QwenHrxArenaBuffer::kSsmAlpha);
  const auto beta = arena_->Binding(QwenHrxArenaBuffer::kSsmBeta);
  const auto conv = arena_->Binding(QwenHrxArenaBuffer::kSsmConvOutput);
  const auto recurrent =
      arena_->Binding(QwenHrxArenaBuffer::kSsmRecurrentOutput);
  const auto projected = arena_->Binding(QwenHrxArenaBuffer::kAttentionOutput);
  const std::size_t conv_state_bytes =
      static_cast<std::size_t>(contract_.SsmQkvWidth()) *
      contract_.SsmConvKernel() * sizeof(float);
  const std::size_t recurrent_state_bytes =
      static_cast<std::size_t>(contract_.SsmValueHeadCount()) *
      contract_.SsmKeyDim() * contract_.SsmValueDim() * sizeof(float);
  const auto conv_state =
      SliceBinding(arena_->Binding(QwenHrxArenaBuffer::kSsmConvState),
                   layer_index * conv_state_bytes, conv_state_bytes);
  const auto recurrent_state =
      SliceBinding(arena_->Binding(QwenHrxArenaBuffer::kSsmRecurrentState),
                   layer_index * recurrent_state_bytes, recurrent_state_bytes);
  if (!conv_state || !recurrent_state ||
      !DispatchRMSNorm(hidden, layer.attn_norm, normed) ||
      !DispatchQ8Gemv(layer.attn_qkv, normed, qkv, contract_.SsmQkvWidth(),
                      contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.attn_gate, normed, gate, contract_.SsmGateWidth(),
                      contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.ssm_alpha, normed, alpha,
                      contract_.SsmAlphaBetaWidth(), contract_.HiddenSize()) ||
      !DispatchQ8Gemv(layer.ssm_beta, normed, beta,
                      contract_.SsmAlphaBetaWidth(), contract_.HiddenSize()) ||
      !DispatchDeltaNetPrepare(alpha, beta, layer.ssm_a, layer.ssm_dt) ||
      !DispatchSsmConv(qkv, layer.ssm_conv1d, *conv_state, conv) ||
      !DispatchDeltaNetPrepared(conv, alpha, beta, layer.ssm_norm, gate,
                                *recurrent_state, recurrent) ||
      !DispatchQ8Gemv(layer.ssm_out, recurrent, projected,
                      contract_.HiddenSize(), contract_.SsmGateWidth()) ||
      !DispatchResidualAdd(hidden, projected, normed, contract_.HiddenSize()) ||
      !DispatchCopyF32(normed, hidden, contract_.HiddenSize())) {
    return Reject("native HRX SSM dispatch failed", error_msg);
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

bool QwenHrxExecutor::DispatchFinalQ8(tokenization::TokenId* token,
                                      std::string* error_msg) {
  if (model_ == nullptr || !arena_.has_value() || !FinalStageReady() ||
      token == nullptr) {
    return Reject("native HRX final stage is not initialized", error_msg);
  }
  const auto& bindings = model_->GetNativeBindings();
  const auto hidden = arena_->Binding(QwenHrxArenaBuffer::kHidden);
  const auto normed = arena_->Binding(QwenHrxArenaBuffer::kNormed);
  const auto logits = arena_->Binding(QwenHrxArenaBuffer::kLogits);
  const auto token_binding = arena_->Binding(QwenHrxArenaBuffer::kToken);
  if (!DispatchRMSNorm(hidden, bindings.output_norm, normed)) {
    return Reject("native HRX final RMSNorm failed", error_msg);
  }
  const auto weight_bytes =
      Q8_0MatrixBytes(contract_.VocabSize(), contract_.HiddenSize());
  const auto input_bytes =
      CheckedBytes({contract_.HiddenSize(), sizeof(float)});
  const auto output_bytes =
      CheckedBytes({contract_.VocabSize(), sizeof(float)});
  hrx_buffer_ref_t gemv_bindings[3];
  if (!TryBindOperand(bindings.output, weight_bytes, &gemv_bindings[0]) ||
      !TryBindOperand(normed, input_bytes, &gemv_bindings[1]) ||
      !TryBindOperand(logits, output_bytes, &gemv_bindings[2])) {
    return Reject("native HRX vocabulary GEMV bindings are invalid", error_msg);
  }
  hrx_dispatch_config_t gemv_config{};
  gemv_config.workgroup_count[0] = contract_.VocabSize();
  gemv_config.workgroup_count[1] = 1;
  gemv_config.workgroup_count[2] = 1;
  gemv_config.workgroup_size[0] = 160;
  gemv_config.workgroup_size[1] = 1;
  gemv_config.workgroup_size[2] = 1;
  gemv_config.subgroup_size = 32;
  const auto rows = contract_.VocabSize();
  auto status = hrx_stream_dispatch(
      backend_.Stream(), q8_vocab_gemv_k5120_executable_, 0, &gemv_config,
      &rows, sizeof(rows), gemv_bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return Reject("native HRX vocabulary GEMV dispatch failed", error_msg);
  }
  if (!DispatchArgmax(logits, token_binding)) {
    return Reject("native HRX argmax dispatch failed", error_msg);
  }
  status = hrx_stream_synchronize(backend_.Stream());
  if (!hrx_status_is_ok(status)) {
    ReportHrxStatus(status, "final-stage synchronization", error_msg);
    return false;
  }
  if (!HrxCopyToHost(backend_.Device(), token_binding, token, sizeof(*token),
                     error_msg)) {
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchSwiGLU(const HrxBufferBinding& input,
                                     const HrxBufferBinding& gate,
                                     const HrxBufferBinding& up,
                                     const HrxBufferBinding& output,
                                     std::uint32_t num_rows) {
  if (swiglu_executable_ == nullptr || num_rows < 2 ||
      num_rows > contract_.FfnSize() || (num_rows % 2) != 0) {
    return false;
  }

  const std::size_t hidden_dim = contract_.HiddenSize();
  const auto input_bytes = CheckedBytes({hidden_dim, sizeof(float)});
  const auto weight_bytes =
      CheckedBytes({num_rows, hidden_dim, sizeof(std::uint16_t)});
  const auto output_bytes = CheckedBytes({num_rows, sizeof(float)});
  hrx_buffer_ref_t bindings[4];
  if (!TryBindOperand(input, input_bytes, &bindings[0]) ||
      !TryBindOperand(gate, weight_bytes, &bindings[1]) ||
      !TryBindOperand(up, weight_bytes, &bindings[2]) ||
      !TryBindOperand(output, output_bytes, &bindings[3])) {
    return false;
  }

  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_rows / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  std::uint32_t rows_param = num_rows;

  hrx_status_t status =
      hrx_stream_dispatch(backend_.Stream(), swiglu_executable_, 0, &config,
                          &rows_param, sizeof(rows_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchRMSNormSsmQKV(const HrxBufferBinding& input,
                                            const HrxBufferBinding& gamma,
                                            const HrxBufferBinding& qkv_weight,
                                            const HrxBufferBinding& output,
                                            std::uint32_t num_rows) {
  if (rmsnorm_ssm_qkv_executable_ == nullptr || num_rows < 2 ||
      num_rows > contract_.SsmQkvWidth() || (num_rows % 2) != 0) {
    return false;
  }
  const std::size_t hidden_dim = contract_.HiddenSize();
  const auto vector_bytes = CheckedBytes({hidden_dim, sizeof(float)});
  const auto weight_bytes =
      CheckedBytes({num_rows, hidden_dim, sizeof(std::uint16_t)});
  const auto output_bytes = CheckedBytes({num_rows, sizeof(float)});
  hrx_buffer_ref_t bindings[4];
  if (!TryBindOperand(input, vector_bytes, &bindings[0]) ||
      !TryBindOperand(gamma, vector_bytes, &bindings[1]) ||
      !TryBindOperand(qkv_weight, weight_bytes, &bindings[2]) ||
      !TryBindOperand(output, output_bytes, &bindings[3])) {
    return false;
  }

  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = num_rows / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  std::uint32_t rows_param = num_rows;
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), rmsnorm_ssm_qkv_executable_, 0, &config, &rows_param,
      sizeof(rows_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchRoPEKVCache(const HrxBufferBinding& q,
                                          const HrxBufferBinding& k,
                                          const HrxBufferBinding& v,
                                          const HrxBufferBinding& cos,
                                          const HrxBufferBinding& sin,
                                          const HrxBufferBinding& k_cache,
                                          const HrxBufferBinding& v_cache) {
  if (rope_kv_executable_ == nullptr) {
    return false;
  }
  const std::uint32_t q_heads = contract_.QHeadCount();
  const std::uint32_t kv_heads = contract_.KvHeadCount();
  const auto q_bytes =
      CheckedBytes({contract_.FullAttentionQueryWidth(), sizeof(float)});
  const auto k_bytes =
      CheckedBytes({contract_.FullAttentionKeyWidth(), sizeof(float)});
  const auto v_bytes =
      CheckedBytes({contract_.FullAttentionValueWidth(), sizeof(float)});
  const auto rotary_bytes =
      CheckedBytes({contract_.RotaryDim() / 2, sizeof(float)});
  hrx_buffer_ref_t bindings[7];
  if (!TryBindOperand(q, q_bytes, &bindings[0]) ||
      !TryBindOperand(k, k_bytes, &bindings[1]) ||
      !TryBindOperand(v, v_bytes, &bindings[2]) ||
      !TryBindOperand(cos, rotary_bytes, &bindings[3]) ||
      !TryBindOperand(sin, rotary_bytes, &bindings[4]) ||
      !TryBindOperand(k_cache, k_bytes, &bindings[5]) ||
      !TryBindOperand(v_cache, v_bytes, &bindings[6])) {
    return false;
  }

  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = q_heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = contract_.RotaryDim() / 2;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  const std::array<std::uint32_t, 2> head_params{q_heads, kv_heads};
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), rope_kv_executable_, 0, &config, head_params.data(),
      sizeof(head_params), bindings, 7, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchRMSNorm(const HrxBufferBinding& input,
                                      const HrxBufferBinding& gamma,
                                      const HrxBufferBinding& output) {
  if (rmsnorm_executable_ == nullptr) {
    return false;
  }
  const auto bytes = CheckedBytes({contract_.HiddenSize(), sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(input, bytes, &bindings[0]) ||
      !TryBindOperand(gamma, bytes, &bindings[1]) ||
      !TryBindOperand(output, bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 160;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const std::uint32_t elements = contract_.HiddenSize();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), rmsnorm_executable_, 0, &config,
                          &elements, sizeof(elements), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchResidualAdd(const HrxBufferBinding& left,
                                          const HrxBufferBinding& right,
                                          const HrxBufferBinding& output,
                                          std::uint32_t elements) {
  if (residual_add_executable_ == nullptr || elements == 0 ||
      elements > contract_.FfnSize() || (elements % 32) != 0) {
    return false;
  }
  const auto bytes = CheckedBytes({elements, sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(left, bytes, &bindings[0]) ||
      !TryBindOperand(right, bytes, &bindings[1]) ||
      !TryBindOperand(output, bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = elements / 32;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  auto status =
      hrx_stream_dispatch(backend_.Stream(), residual_add_executable_, 0,
                          &config, &elements, sizeof(elements), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchSwiGLUPointwise(const HrxBufferBinding& gate,
                                              const HrxBufferBinding& up,
                                              const HrxBufferBinding& output,
                                              std::uint32_t elements) {
  if (swiglu_pointwise_executable_ == nullptr || elements == 0 ||
      elements > contract_.FfnSize() || (elements % 32) != 0) {
    return false;
  }
  const auto bytes = CheckedBytes({elements, sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(gate, bytes, &bindings[0]) ||
      !TryBindOperand(up, bytes, &bindings[1]) ||
      !TryBindOperand(output, bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = elements / 32;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  auto status =
      hrx_stream_dispatch(backend_.Stream(), swiglu_pointwise_executable_, 0,
                          &config, &elements, sizeof(elements), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchSplitQGate(const HrxBufferBinding& q_gate,
                                         const HrxBufferBinding& query,
                                         const HrxBufferBinding& gate) {
  if (split_q_gate_executable_ == nullptr) {
    return false;
  }
  const auto q_gate_bytes =
      CheckedBytes({contract_.FullAttentionQGateWidth(), sizeof(float)});
  const auto query_bytes =
      CheckedBytes({contract_.FullAttentionQueryWidth(), sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(q_gate, q_gate_bytes, &bindings[0]) ||
      !TryBindOperand(query, query_bytes, &bindings[1]) ||
      !TryBindOperand(gate, query_bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = contract_.FullAttentionQueryWidth() / 32;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const std::uint32_t elements = contract_.FullAttentionQGateWidth();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), split_q_gate_executable_, 0,
                          &config, &elements, sizeof(elements), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchPerHeadRmsNorm(const HrxBufferBinding& input,
                                             const HrxBufferBinding& gamma,
                                             const HrxBufferBinding& output,
                                             std::uint32_t heads) {
  if (per_head_rmsnorm_executable_ == nullptr ||
      (heads != contract_.QHeadCount() && heads != contract_.KvHeadCount())) {
    return false;
  }
  const auto vector_bytes =
      CheckedBytes({heads, contract_.HeadDim(), sizeof(float)});
  const auto gamma_bytes = CheckedBytes({contract_.HeadDim(), sizeof(float)});
  hrx_buffer_ref_t bindings[3];
  if (!TryBindOperand(input, vector_bytes, &bindings[0]) ||
      !TryBindOperand(gamma, gamma_bytes, &bindings[1]) ||
      !TryBindOperand(output, vector_bytes, &bindings[2])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = heads;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  auto status =
      hrx_stream_dispatch(backend_.Stream(), per_head_rmsnorm_executable_, 0,
                          &config, &heads, sizeof(heads), bindings, 3, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchAttentionDecode(
    const HrxBufferBinding& query, const HrxBufferBinding& gate,
    const HrxBufferBinding& key_cache, const HrxBufferBinding& value_cache,
    const HrxBufferBinding& output, std::uint32_t position) {
  if (attention_decode_executable_ == nullptr || position >= max_context_) {
    return false;
  }
  const auto query_bytes =
      CheckedBytes({contract_.FullAttentionQueryWidth(), sizeof(float)});
  const auto cache_bytes = CheckedBytes({contract_.KvHeadCount(), max_context_,
                                         contract_.HeadDim(), sizeof(float)});
  hrx_buffer_ref_t bindings[5];
  if (!TryBindOperand(query, query_bytes, &bindings[0]) ||
      !TryBindOperand(gate, query_bytes, &bindings[1]) ||
      !TryBindOperand(key_cache, cache_bytes, &bindings[2]) ||
      !TryBindOperand(value_cache, cache_bytes, &bindings[3]) ||
      !TryBindOperand(output, query_bytes, &bindings[4])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = contract_.QHeadCount();
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const std::array<std::uint32_t, 2> params{position, max_context_};
  auto status = hrx_stream_dispatch(
      backend_.Stream(), attention_decode_executable_, 0, &config,
      params.data(), sizeof(params), bindings, 5, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchSsmConv(const HrxBufferBinding& input,
                                      const HrxBufferBinding& weights,
                                      const HrxBufferBinding& state,
                                      const HrxBufferBinding& output) {
  if (ssm_conv_executable_ == nullptr) {
    return false;
  }
  const auto vector_bytes =
      CheckedBytes({contract_.SsmQkvWidth(), sizeof(float)});
  const auto state_bytes = CheckedBytes(
      {contract_.SsmQkvWidth(), contract_.SsmConvKernel(), sizeof(float)});
  hrx_buffer_ref_t bindings[4];
  if (!TryBindOperand(input, vector_bytes, &bindings[0]) ||
      !TryBindOperand(weights, state_bytes, &bindings[1]) ||
      !TryBindOperand(state, state_bytes, &bindings[2]) ||
      !TryBindOperand(output, vector_bytes, &bindings[3])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = contract_.SsmQkvWidth() / 32;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const auto elements = contract_.SsmQkvWidth();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), ssm_conv_executable_, 0, &config,
                          &elements, sizeof(elements), bindings, 4, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDeltaNetPrepare(const HrxBufferBinding& alpha,
                                              const HrxBufferBinding& beta,
                                              const HrxBufferBinding& a,
                                              const HrxBufferBinding& dt) {
  if (deltanet_prepare_executable_ == nullptr) {
    return false;
  }
  const auto bytes =
      CheckedBytes({contract_.SsmAlphaBetaWidth(), sizeof(float)});
  hrx_buffer_ref_t bindings[4];
  if (!TryBindOperand(alpha, bytes, &bindings[0]) ||
      !TryBindOperand(beta, bytes, &bindings[1]) ||
      !TryBindOperand(a, bytes, &bindings[2]) ||
      !TryBindOperand(dt, bytes, &bindings[3])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = contract_.SsmValueHeadCount();
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const auto heads = contract_.SsmValueHeadCount();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), deltanet_prepare_executable_, 0,
                          &config, &heads, sizeof(heads), bindings, 4, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDeltaNetPrepared(
    const HrxBufferBinding& conv, const HrxBufferBinding& alpha_decay,
    const HrxBufferBinding& beta_correction, const HrxBufferBinding& norm,
    const HrxBufferBinding& gate, const HrxBufferBinding& state,
    const HrxBufferBinding& output) {
  if (deltanet_recurrence_executable_ == nullptr) {
    return false;
  }
  const auto conv_bytes =
      CheckedBytes({contract_.SsmQkvWidth(), sizeof(float)});
  const auto ab_bytes =
      CheckedBytes({contract_.SsmAlphaBetaWidth(), sizeof(float)});
  const auto norm_bytes =
      CheckedBytes({contract_.SsmValueDim(), sizeof(float)});
  const auto gate_bytes =
      CheckedBytes({contract_.SsmGateWidth(), sizeof(float)});
  const auto state_bytes =
      CheckedBytes({contract_.SsmValueHeadCount(), contract_.SsmKeyDim(),
                    contract_.SsmValueDim(), sizeof(float)});
  hrx_buffer_ref_t bindings[7];
  if (!TryBindOperand(conv, conv_bytes, &bindings[0]) ||
      !TryBindOperand(alpha_decay, ab_bytes, &bindings[1]) ||
      !TryBindOperand(beta_correction, ab_bytes, &bindings[2]) ||
      !TryBindOperand(norm, norm_bytes, &bindings[3]) ||
      !TryBindOperand(gate, gate_bytes, &bindings[4]) ||
      !TryBindOperand(state, state_bytes, &bindings[5]) ||
      !TryBindOperand(output, gate_bytes, &bindings[6])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = contract_.SsmValueHeadCount();
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 128;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const auto heads = contract_.SsmValueHeadCount();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), deltanet_recurrence_executable_, 0,
                          &config, &heads, sizeof(heads), bindings, 7, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchArgmax(const HrxBufferBinding& logits,
                                     const HrxBufferBinding& token) {
  if (argmax_executable_ == nullptr) {
    return false;
  }
  const auto logits_bytes =
      CheckedBytes({contract_.VocabSize(), sizeof(float)});
  const auto token_bytes = CheckedBytes({sizeof(std::uint32_t)});
  hrx_buffer_ref_t bindings[2];
  if (!TryBindOperand(logits, logits_bytes, &bindings[0]) ||
      !TryBindOperand(token, token_bytes, &bindings[1])) {
    return false;
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 1;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 1;
  const auto elements = contract_.VocabSize();
  auto status =
      hrx_stream_dispatch(backend_.Stream(), argmax_executable_, 0, &config,
                          &elements, sizeof(elements), bindings, 2, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

bool QwenHrxExecutor::DispatchDownResidual(const HrxBufferBinding& input,
                                           const HrxBufferBinding& down_weight,
                                           const HrxBufferBinding& residual,
                                           const HrxBufferBinding& output) {
  if (down_residual_executable_ == nullptr) {
    return false;
  }
  const std::size_t hidden_dim = contract_.HiddenSize();
  const std::size_t intermediate_dim = contract_.FfnSize();
  const auto input_bytes = CheckedBytes({intermediate_dim, sizeof(float)});
  const auto weight_bytes =
      CheckedBytes({hidden_dim, intermediate_dim, sizeof(std::uint16_t)});
  const auto hidden_bytes = CheckedBytes({hidden_dim, sizeof(float)});
  hrx_buffer_ref_t bindings[4];
  if (!TryBindOperand(input, input_bytes, &bindings[0]) ||
      !TryBindOperand(down_weight, weight_bytes, &bindings[1]) ||
      !TryBindOperand(residual, hidden_bytes, &bindings[2]) ||
      !TryBindOperand(output, hidden_bytes, &bindings[3])) {
    return false;
  }

  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = (hidden_dim + 1) / 2;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 544;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;

  const std::uint32_t hidden_param = contract_.HiddenSize();
  hrx_status_t status = hrx_stream_dispatch(
      backend_.Stream(), down_residual_executable_, 0, &config, &hidden_param,
      sizeof(hidden_param), bindings, 4, 0);

  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }
  return true;
}

}  // namespace gufo::hrx

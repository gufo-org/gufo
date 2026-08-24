#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/mtp.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#if defined(ENGINE_ENABLE_XRT)
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/xdna2/device.h"
#endif

namespace strix::hip {
namespace {

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

}  // namespace

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    QwenMtpExecutionMode execution_mode)
    : model_(std::move(model)),
      max_context_(max_context),
      execution_mode_(execution_mode),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size)
#if defined(ENGINE_ENABLE_XRT)
      ,
      h_npu_input_(execution_mode == QwenMtpExecutionMode::kHybridNpuEhProj
                       ? 2 * model_->GetConfig().hidden_size
                       : 0),
      h_npu_output_(execution_mode == QwenMtpExecutionMode::kHybridNpuEhProj
                        ? model_->GetConfig().hidden_size
                        : 0)
#endif
{
  try {
    Allocate();
#if defined(ENGINE_ENABLE_XRT)
    if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
      const auto inventory = diagnostics::CollectSystemInventory();
      const auto device = xdna2::DiscoverXrtDevice(0, inventory);
      xdna2::QwenMtpEhProjFailure failure;
      npu_eh_proj_ = xdna2::QwenMtpEhProjSession::Create(
          {}, device, model_->GetRawFusionProjection(), &failure);
      if (npu_eh_proj_ == nullptr) {
        throw std::runtime_error("MTP NPU eh_proj initialization failed: " +
                                 failure.category + ": " + failure.message);
      }
    }
#else
    if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
      throw std::runtime_error(
          "MTP NPU execution requested without ENGINE_ENABLE_XRT");
    }
#endif
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::string* error_msg, QwenMtpExecutionMode execution_mode) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context, execution_mode));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  AllocateBuffer(d_target_hidden_, hidden);
  AllocateBuffer(d_embedding_, hidden);
  AllocateBuffer(d_fusion_, 2 * hidden);
  AllocateBuffer(d_hidden_, hidden);
  AllocateBuffer(d_normed_, hidden);
  AllocateBuffer(d_qg_, 2 * attention);
  AllocateBuffer(d_q_, attention);
  AllocateBuffer(d_k_, kv);
  AllocateBuffer(d_v_, kv);
  AllocateBuffer(d_gate_, attention);
  AllocateBuffer(d_context_, attention);
  AllocateBuffer(d_attn_out_, hidden);
  AllocateBuffer(d_ffn_act_, intermediate);
  AllocateBuffer(d_ffn_out_, hidden);
  AllocateBuffer(d_feedback_hidden_, hidden);
  AllocateBuffer(d_logits_, config.vocab_size);
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ = detail::AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  AllocateBuffer(d_split_k_scratch_,
                 detail::DecodeAttentionScratchElements(
                     config.num_attention_heads, config.head_dim));
  AllocateBuffer(d_out_token_, 1);
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_ffn_out_);
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  const auto& config = model_->GetConfig();
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  (void)hipMemsetAsync(d_kv_cache_, 0, 2 * total_kv * sizeof(float), stream_);
  (void)hipMemsetAsync(d_kv_cache_f16_, 0, 2 * total_kv * sizeof(std::uint16_t),
                       stream_);
  next_position_ = 0;
  hybrid_metrics_ = {};
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits) {
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits) {
  return Run(input_token, d_feedback_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  LaunchEmbeddingLookup(weights.token_embedding.data,
                        weights.token_embedding.type, input_token, d_embedding_,
                        hidden, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  if (execution_mode_ == QwenMtpExecutionMode::kHybridNpuEhProj) {
#if defined(ENGINE_ENABLE_XRT)
    const auto gpu_to_host_start = std::chrono::steady_clock::now();
    const auto download_error = hipMemcpyAsync(
        h_npu_input_.data(), d_fusion_, h_npu_input_.size() * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    const auto download_sync_error = hipStreamSynchronize(stream_);
    if (download_error != hipSuccess || download_sync_error != hipSuccess) {
      throw std::runtime_error("MTP GPU-to-NPU input copy failed");
    }
    const auto gpu_to_host_end = std::chrono::steady_clock::now();
    xdna2::QwenMtpEhProjRunMetrics metrics;
    xdna2::QwenMtpEhProjFailure failure;
    if (!npu_eh_proj_->Run(h_npu_input_, h_npu_output_, &metrics, &failure)) {
      throw std::runtime_error("MTP NPU eh_proj failed: " + failure.category +
                               ": " + failure.message);
    }
    const auto host_to_gpu_start = std::chrono::steady_clock::now();
    const auto upload_error = hipMemcpyAsync(
        d_hidden_, h_npu_output_.data(), h_npu_output_.size() * sizeof(float),
        hipMemcpyHostToDevice, stream_);
    const auto upload_sync_error = hipStreamSynchronize(stream_);
    if (upload_error != hipSuccess || upload_sync_error != hipSuccess) {
      throw std::runtime_error("MTP NPU-to-GPU output copy failed");
    }
    const auto host_to_gpu_end = std::chrono::steady_clock::now();
    ++hybrid_metrics_.projection_count;
    hybrid_metrics_.gpu_to_host_us += std::chrono::duration<double, std::micro>(
                                          gpu_to_host_end - gpu_to_host_start)
                                          .count();
    hybrid_metrics_.activation_pack_us += metrics.activation_pack_us;
    hybrid_metrics_.npu_command_us += metrics.command_us;
    hybrid_metrics_.npu_end_to_end_us += metrics.end_to_end_us;
    hybrid_metrics_.host_to_gpu_us += std::chrono::duration<double, std::micro>(
                                          host_to_gpu_end - host_to_gpu_start)
                                          .count();
#else
    throw std::runtime_error(
        "MTP NPU execution requested without ENGINE_ENABLE_XRT");
#endif
  } else {
    LaunchGEMV(weights.fusion_projection.data, core::GgmlType::kBF16, d_fusion_, d_hidden_,
               hidden, 2 * hidden, stream_);
  }

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(layer.attn_q.data, core::GgmlType::kBF16,
                            layer.attn_k.data, core::GgmlType::kBF16,
                            layer.attn_v.data, core::GgmlType::kBF16, d_normed_,
                            d_qg_, d_k_, d_v_, 2 * attention, kv, hidden,
                            stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, core::GgmlType::kBF16, d_context_, d_attn_out_, hidden,
             attention, stream_);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, core::GgmlType::kBF16,
                        layer.ffn_up.data, core::GgmlType::kBF16, d_normed_,
                        d_ffn_act_, config.intermediate_size, hidden, stream_);
  LaunchGEMV(layer.ffn_down.data, core::GgmlType::kBF16, d_ffn_act_, d_ffn_out_, hidden,
             config.intermediate_size, stream_);
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  ++next_position_;

  if (!compute_logits) {
    return 0;
  }
  LaunchGEMV(weights.output.data, weights.output.type,
             d_feedback_hidden_, d_logits_, config.vocab_size, hidden, stream_);
  LaunchGPUArgmax(d_logits_, d_out_token_, config.vocab_size, stream_);
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  return result;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
